#!/usr/bin/env python3
"""
status_panel.py — the STATUS CENTER: MABEL's live terminal panel, drawn in run.sh's own style
(the same rules, spaced title and indented sections). Redrawn IN PLACE twice a second from the
gateway's robot_state: one line per module (the module_status.py roster — base · torso · lift ·
neck · arms · hands · cams · lidar · audio, with WHY a dark one is dark), then the client
roster, the plant and the e-stop state.

    python3 server/status_panel.py                                   # ws://127.0.0.1:9090/teleop
    python3 server/status_panel.py --key "$MABEL_TOKEN"              # a token-gated bridge
    python3 server/status_panel.py --http http://127.0.0.1:8080/api/state   # NOVA-style snapshot

real_run.sh runs this in the foreground, so Ctrl-C on the panel stops the whole stack. When
stdout is not a tty (piped into a log) it prints one plain status line per tick instead.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

# the palette of MABEL's server/console.py + controller/runtime/hud.py
RST, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
GRN, YLW, RED = "\033[38;5;78m", "\033[38;5;221m", "\033[38;5;203m"
CYN, GRY, VIO = "\033[38;5;81m", "\033[38;5;245m", "\033[38;5;141m"

W = 74                                     # rule width — run.sh's, exactly
RULE = "─" * W
TTY = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
ORDER = ("base", "torso", "lift", "neck", "left_arm", "right_arm", "left_hand", "right_hand")


def c(color: str, s: str) -> str:
    return f"{color}{s}{RST}" if TTY else s


def vlen(s: str) -> int:
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and not s[i].isalpha():
                i += 1
            i += 1
            continue
        n, i = n + 1, i + 1
    return n


def bar(frac: float, n: int = 6) -> str:
    f = max(0, min(n, int(round(frac * n))))
    return "█" * f + "░" * (n - f)


# ── where the state comes from ───────────────────────────────────────────────
class HttpSource:
    """The gateway's own snapshot — /api/state (NOVA's teleop_bridge serves it)."""

    def __init__(self, url):
        self.url, self.what = url, url

    def state(self):
        try:
            with urllib.request.urlopen(self.url, timeout=1.0) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None


class WsSource:
    """A read-only tap on ws://…/teleop. robot_state fields that are change-gated
    (the module roster) are merged, not replaced, so the panel keeps the last truth."""

    def __init__(self, url):
        self.url, self.what = url, url
        self._st, self._t = None, 0.0
        threading.Thread(target=self._run, daemon=True, name="status-tap").start()

    def _run(self):
        while True:
            try:
                from websockets.sync.client import connect       # import inside: a missing
                with connect(self.url, max_size=None, open_timeout=3) as ws:   # websockets must
                                                                 # not throw out of the thread
                    for raw in ws:
                        env = json.loads(raw if isinstance(raw, str) else raw.decode())
                        if env.get("type") != "robot_state":
                            continue
                        st = dict(self._st or {})
                        st.update(env.get("payload") or {})
                        self._st, self._t = st, time.time()
            except Exception:                                    # noqa: BLE001
                self._st = None
                time.sleep(1.0)

    def state(self):
        return self._st if (self._st and time.time() - self._t < 3.0) else None


# ── the rows ─────────────────────────────────────────────────────────────────
def module_rows(st):
    """(plain, colored) per module, for both robot_state shapes:
    MABEL's modules [{id, label, group, ok, detail}] and
    NOVA's torque.modules {name: {connected, live, n, torque, rateHz}}."""
    rows = []
    mods = (st.get("torque") or {}).get("modules")
    if isinstance(mods, dict) and mods:
        names = [n for n in ORDER if n in mods] + sorted(k for k in mods if k not in ORDER)
        for name in names:
            m = mods[name] or {}
            n_j, live = int(m.get("n") or 0), int(m.get("live") or 0)
            up = bool(m.get("connected")) and live > 0
            frac = (live / n_j) if n_j else 0.0
            state = f"{live}/{n_j} live" if up else "offline"
            rate = (f"{float(m.get('rateHz') or 0):.0f} Hz" if up else "—")
            torq = ("torque ON" if m.get("torque") else "torque off") if up else "—"
            col = GRN if frac >= 0.999 else (YLW if frac > 0 else GRY)
            plain = f"{name:<14}{state:<11}{rate:>7}   {torq:<12}{bar(frac)}"
            rows.append((plain, f"{c(GRY, f'{name:<14}')}{c(col, f'{state:<11}')}"
                                f"{c(CYN if up else GRY, f'{rate:>7}')}   "
                                f"{c(YLW if m.get('torque') else DIM, f'{torq:<12}')}{c(col, bar(frac))}"))
        return rows

    mods = st.get("modules")
    if isinstance(mods, list) and mods:
        group = mods[0].get("group")
        for m in mods:
            if m.get("group") != group:                          # blank line between groups
                group = m.get("group")
                rows.append(("", ""))
            up = bool(m.get("ok"))
            label, detail = m.get("label") or m.get("id"), (m.get("detail") or "")[:30]
            state, col = ("up", GRN) if up else ("down", GRY)
            plain = f"{label:<14}{state:<11}{detail:<32}{bar(1.0 if up else 0.0)}"
            rows.append((plain, f"{c(GRY, f'{label:<14}')}{c(col, f'{state:<11}')}"
                                f"{c(DIM, f'{detail:<32}')}{c(col, bar(1.0 if up else 0.0))}"))
        return rows
    return [("no module roster yet", c(DIM, "no module roster yet"))]


class Panel:
    def __init__(self, title, studio, source):
        self.title, self.studio, self.src = title, studio, source
        self.n = 0                                               # lines the last redraw left

    def lines(self, st):
        head = "  " + c(DIM, RULE)
        clock = time.strftime("%H:%M:%S")
        t = f"    {self.title}"
        L = [head, f"{c(BOLD + CYN, t)}{' ' * max(1, W + 2 - len(t) - len(clock))}{c(DIM, clock)}", head]
        if st is None:
            L.append("      " + c(YLW, f"⚠ waiting for the gateway — {self.src.what}"))
            L.append(head)
            return L
        for plain, colored in module_rows(st):
            L.append("      " + colored if plain else "")
        d = st.get("driver") or {}
        if st.get("rateHz") is not None:                         # NOVA's gateway reports its own tick
            metric = f"{c(VIO, 'gateway')} {c(CYN, '%.1f Hz' % float(st['rateHz']))}"
        else:                                                    # MABEL's reports the state age instead
            age = st.get("stateAgeMs")
            metric = f"{c(VIO, 'state age')} {c(CYN, ('%.0f ms' % float(age)) if age is not None else '—')}"
        plant = st.get("plant") or (st.get("server") or {}).get("kind") or "—"
        L.append("")
        L.append("      " + f"{metric}     {c(VIO, 'clients')} {d.get('clients', 0)}"
                            f"     {c(VIO, 'holder')} {d.get('holder') or '—'}"
                            f"     {c(VIO, 'plant')} {plant}")
        L.append("    " + (c(RED, "● E-STOP ENGAGED — the robot is LIMP")
                           if st.get("estop") else c(GRN, "● e-stop released — the robot is LIVE (torque on)")))
        L.append("    " + c(DIM, f"studio {self.studio}   ·   Ctrl-C here stops everything"))
        L.append(head)
        return L

    def draw(self, st):
        L = self.lines(st)
        out = "\033[?25l"                                        # hide cursor
        if self.n:
            out += f"\033[{self.n}A"                             # back up over the old panel
        out += "".join(line + "\033[K\n" for line in L)
        out += "\033[?25h"
        sys.stdout.write(out)
        sys.stdout.flush()
        self.n = len(L)

    def log_line(self, st):
        if st is None:
            print("  ▸ [status] gateway offline", flush=True)
            return
        parts = [p.strip() for p, _ in module_rows(st) if p]
        d = st.get("driver") or {}
        print(f"  ▸ [status {time.strftime('%H:%M:%S')}] gw {float(st.get('rateHz') or 0):.1f} Hz "
              f"clients {d.get('clients', 0)} | {' | '.join(' '.join(q.split()) for q in parts)} | "
              f"estop {'ON' if st.get('estop') else 'off'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", default="M A B E L   ·   S T A T U S   C E N T E R")
    ap.add_argument("--studio", default=f"http://127.0.0.1:{os.getenv('MABEL_CC_PORT', '8094')}")
    ap.add_argument("--ws", default=f"ws://127.0.0.1:{os.getenv('MABEL_WS_PORT', '9090')}/teleop")
    ap.add_argument("--key", default=os.getenv("MABEL_TOKEN", ""), help="token for a gated bridge (?key=)")
    ap.add_argument("--http", default="", help="read a /api/state snapshot instead of the websocket")
    ap.add_argument("--hz", type=float, default=2.0)
    a = ap.parse_args()

    ws = a.ws + (f"?key={a.key}" if (a.ws and a.key and "key=" not in a.ws) else "")
    panel = Panel(a.title, a.studio, HttpSource(a.http) if a.http else WsSource(ws))

    def bye(*_):
        if TTY:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)
    period = 1.0 / max(0.2, a.hz)
    while True:
        st = panel.src.state()
        panel.draw(st) if TTY else panel.log_line(st)
        time.sleep(period)


if __name__ == "__main__":
    main()
