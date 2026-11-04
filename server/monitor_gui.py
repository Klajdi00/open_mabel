#!/usr/bin/env python3
"""monitor_gui.py — MABEL Command Monitor.

A clean, iOS-dark desktop window that shows, live, every control command the
teleop bridge receives and from which client — exactly the operator-intent
stream the future hardware bridge has to act on.

It connects (read-only) to the bridge's `/monitor` WebSocket (shares the teleop
port, default :9090), which fans out a derived view of the wire protocol
(see monitor_hub.py). Nothing here can command the robot.

    python monitor_gui.py                      # localhost:9090
    python monitor_gui.py --host mabel.local   # a remote bridge
    python monitor_gui.py --host 1.2.3.4 --port 9090

Layout (top → bottom):
  • Header     — server identity + connection state.
  • Status     — pills/toggles for the ACTIVE driver: teleop mode, control law
                 (impedance / compliance = gravity-comp), body region, head
                 tracking, per-hand tracking, recording, and a reset flash.
  • Body       — left: every connected client + its live rate / role;
                 right: the colour-coded command stream (the protocol trace).

The networking runs in a daemon asyncio thread and hands parsed messages to the
Tk main loop through a queue, so the UI never blocks on the socket (the same
thread-safe pattern the firmware GUIs use for their CAN buses).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import tkinter as tk
from collections import deque

import gui_style as S


# ───────────────────────────── networking ──────────────────────────────────
class MonitorClient:
    """Background WebSocket subscriber for ONE bridge host. Pushes
    (`host_key`, `kind`, dict) onto a shared queue and auto-reconnects with
    linear backoff — the same resilience contract the iPhone link uses. The app
    runs one of these per host so several robots/sims can be watched at once."""

    def __init__(self, host_key: str, host: str, port: int, path: str,
                 token: str, q: "queue.Queue"):
        uri = f"ws://{host}:{port}{path}"
        if token:
            uri += f"?key={token}"
        self.host_key = host_key
        self.uri = uri
        self.q = q
        self._running = True
        self._ws = None
        self._loop_ref = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    def send(self, obj: dict):
        """Send a control message to the bridge (e.g. a rename) from the Tk
        thread, marshalled onto this client's asyncio loop."""
        ws, loop = self._ws, self._loop_ref
        if ws is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(ws.send(json.dumps(obj)), loop)
            except Exception:
                pass

    def _run(self):
        asyncio.run(self._loop())

    async def _loop(self):
        import websockets
        self._loop_ref = asyncio.get_event_loop()
        attempt = 0
        while self._running:
            try:
                async with websockets.connect(self.uri, ping_interval=20,
                                               max_size=2 ** 20) as ws:
                    attempt = 0
                    self._ws = ws
                    self.q.put((self.host_key, "conn", {"state": "online"}))
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        self.q.put((self.host_key, msg.get("kind", "?"), msg))
            except Exception as e:
                self.q.put((self.host_key, "conn", {"state": "offline", "error": str(e)}))
            finally:
                self._ws = None
            if not self._running:
                break
            attempt += 1
            await asyncio.sleep(min(0.5 * attempt, 4.0))


class HostState:
    """Everything the monitor knows about ONE bridge host — its latest global
    state, client list, recent event log, and online flag. The app keeps one per
    host and renders whichever is selected; switching is just pointing the UI at
    a different HostState."""

    def __init__(self, key: str, host: str, port: int, label: str = "",
                 discovered: bool = False):
        self.key = key
        self.host = host
        self.port = port
        self.label = label or host
        self.discovered = discovered
        self.glob: dict = {}
        self.clients: list = []
        self.online = False
        self.events: deque = deque(maxlen=700)
        self.evt_count = 0
        self.evt_window: list[float] = []

    def name(self) -> str:
        """Best-known display name: the robot's user-set NAME (the same one the
        iPhone shows), else the reported hostname, else the configured address."""
        si = self.glob.get("serverInfo") or {}
        return si.get("name") or si.get("host") or self.label or self.host


class BonjourBrowser(threading.Thread):
    """Best-effort discovery of other MABEL bridges on the LAN via `dns-sd`
    (the same `_mabel-teleop._tcp` service the bridge advertises). Pushes
    ('__discover__', {host, port, label}) onto the queue. Fails silently — the
    explicit --hosts list always works regardless."""

    def __init__(self, q: "queue.Queue"):
        super().__init__(daemon=True)
        self.q = q
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        if not shutil.which("dns-sd"):
            return
        while self._running:
            try:
                for name in self._browse(2.5):
                    hp = self._resolve(name, 2.0)
                    if hp:
                        self.q.put(("__discover__", "discover",
                                    {"label": name, "host": hp[0], "port": hp[1]}))
            except Exception:
                pass
            time.sleep(8.0)

    @staticmethod
    def _timed(cmd, t):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        threading.Timer(t, p.terminate).start()
        return p

    def _browse(self, t) -> set:
        names = set()
        p = self._timed(["dns-sd", "-B", "_mabel-teleop._tcp"], t)
        try:
            for line in p.stdout:
                if " Add " in line and "_mabel-teleop._tcp." in line:
                    nm = line.split("_mabel-teleop._tcp.")[-1].strip()
                    if nm:
                        names.add(nm)
        except Exception:
            pass
        return names

    def _resolve(self, name, t):
        p = self._timed(["dns-sd", "-L", name, "_mabel-teleop._tcp"], t)
        try:
            for line in p.stdout:
                if "can be reached at" in line:
                    tok = line.split("can be reached at", 1)[1].strip().split()[0]
                    host, _, port = tok.rstrip(".").rpartition(":")
                    if host and port.isdigit():
                        return (host.rstrip("."), int(port))
        except Exception:
            pass
        return None


# ───────────────────────────── the app ─────────────────────────────────────
class MonitorApp:
    MAX_LINES = 1200

    def __init__(self, root: tk.Tk, hosts: list[tuple[str, int]], path: str,
                 token: str, discover: bool = True):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self._path = path
        self._token = token

        # One HostState + one MonitorClient per bridge host.
        self._hosts: dict[str, HostState] = {}
        self._conns: dict[str, MonitorClient] = {}
        for host, port in hosts:
            self._add_host(host, port, start=False)
        self._active = next(iter(self._hosts), None)

        # Rendering reads these — they alias the ACTIVE host each refresh.
        self._clients: list = []
        self._clients_sig = None          # identity sig of the rendered client rows
        self._client_rate_lbls = {}       # client id -> rate Label, for live in-place updates
        self._global: dict = {}
        self._paused = False
        self._reset_flash_until = 0.0
        # render caches (invalidated on host switch)
        self._server_cache = None
        self._act_cache = None
        self._cmd_sig = None
        self._cmd_w = None

        root.title("MABEL · Command Monitor")
        root.configure(bg=S.BG)
        root.geometry("1480x820")
        root.minsize(1120, 640)

        self._build_header()
        self._build_status_strip()
        self._build_body()

        for c in self._conns.values():
            c.start()
        self._browser = BonjourBrowser(self.q) if discover else None
        if self._browser:
            self._browser.start()
        self._update_host_selector()
        self.root.after(33, self._drain)
        self.root.after(250, self._tick_ui)

    # ── host registry ───────────────────────────────────────────────────────
    def _add_host(self, host: str, port: int, start: bool = True,
                  discovered: bool = False) -> str:
        key = f"{host}:{port}"
        if key in self._hosts:
            return key
        self._hosts[key] = HostState(key, host, port, label=host,
                                     discovered=discovered)
        self._conns[key] = MonitorClient(key, host, port, self._path,
                                         self._token, self.q)
        if start:
            self._conns[key].start()
        return key

    # ── de-dup hosts that are the SAME bridge reached two ways ────────────────
    def _host_identity(self, hs: "HostState") -> set:
        si = hs.glob.get("serverInfo") or {}
        ids = set()
        if si.get("host"):
            ids.add(("h", si["host"].strip().lower()))
        for a in (si.get("addresses") or []):
            if a:
                ids.add(("a", a))
        for k in ("lanIp", "tailscaleIp"):
            if si.get(k):
                ids.add(("a", si[k]))
        return ids

    def _dedup_hosts(self):
        """One bridge reached two ways (e.g. 127.0.0.1 AND its .local name) reports
        the same serverInfo identity — collapse to the preferred entry."""
        keyed = [(k, self._host_identity(self._hosts[k])) for k in list(self._hosts)]
        drop = set()
        for i, (k1, id1) in enumerate(keyed):
            if k1 in drop or not id1:
                continue
            for k2, id2 in keyed[i + 1:]:
                if k2 in drop or not id2:
                    continue
                if id1 & id2:                         # same bridge
                    keep, gone = self._prefer(k1, k2)
                    drop.add(gone)
        for g in drop:
            self._remove_host(g)

    def _prefer(self, a: str, b: str) -> tuple[str, str]:
        def score(k):
            h = self._hosts[k]
            return ((0 if h.discovered else 4)
                    + (2 if h.host in ("127.0.0.1", "localhost") else 0)
                    + (1 if k == self._active else 0))
        return (a, b) if score(a) >= score(b) else (b, a)

    def _remove_host(self, key: str):
        if key not in self._hosts:
            return
        c = self._conns.pop(key, None)
        if c:
            c.stop()
        self._hosts.pop(key, None)
        if self._active == key:
            self._active = next(iter(self._hosts), None)
            if self._active:
                self._reset_render_caches()
                hs = self._hosts[self._active]
                self._render_active()
                self._clear_stream()
                for ev in list(hs.events):
                    self._render_event_line(ev)
        self._update_host_selector()

    def _maybe_add_discovered(self, msg: dict):
        host, port = msg.get("host"), msg.get("port")
        if not host or not port:
            return
        key = f"{host}:{port}"
        if key in self._hosts:
            return
        # Skip the local machine if we already watch it as localhost (the common
        # case); _dedup_hosts collapses any remaining duplicate once it connects
        # and reports the same serverInfo identity.
        local = {socket.gethostname().split(".")[0].lower(),
                 socket.gethostname().lower()}
        if host.split(".")[0].lower() in local and any(
                h.host in ("127.0.0.1", "localhost") for h in self._hosts.values()):
            return
        self._add_host(host, int(port), start=True, discovered=True)
        if self._active is None:
            self._active = key
        self._update_host_selector()

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = tk.Frame(self.root, bg=S.SURFACE, height=74)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Top-left: the HOST SELECTOR — a raised, hover-highlighted BUTTON showing
        # the active host (name + address + ▾). Click it to drop down every bridge
        # being watched and switch the whole page to one.
        left = tk.Frame(bar, bg=S.SURFACE)
        left.pack(side="left", padx=S.PAD_LG, pady=7)

        chip = tk.Frame(left, bg=S.ELEVATED, cursor="hand2",
                        highlightbackground=S.SEP, highlightthickness=1)
        chip.pack(anchor="w")
        self.host_chip = chip
        inner = tk.Frame(chip, bg=S.ELEVATED)
        inner.pack(padx=11, pady=5)
        self.host_dot = S.Dot(inner, color=S.RED, r=4, bg=S.ELEVATED)
        self.host_dot.pack(side="left", padx=(0, 7))
        self.lbl_host = tk.Label(inner, text="—", bg=S.ELEVATED, fg=S.LABEL,
                                 font=S.FONT_H2)
        self.lbl_host.pack(side="left")
        self.lbl_host_addr = tk.Label(inner, text="", bg=S.ELEVATED, fg=S.LABEL3,
                                      font=S.FONT_MONO_S)
        self.lbl_host_addr.pack(side="left", padx=(8, 0))
        self.lbl_caret = tk.Label(inner, text="▾", bg=S.ELEVATED, fg=S.LABEL2,
                                  font=S.FONT_BODY)
        self.lbl_caret.pack(side="left", padx=(9, 0))

        row2 = tk.Frame(left, bg=S.SURFACE)
        row2.pack(anchor="w", pady=(3, 0))
        tk.Label(row2, text="MABEL", bg=S.SURFACE, fg=S.ORANGE,
                 font=S.FONT_TINY).pack(side="left")
        tk.Label(row2, text="Command Monitor", bg=S.SURFACE, fg=S.LABEL2,
                 font=S.FONT_TINY).pack(side="left", padx=(5, 0))
        self.lbl_ver = tk.Label(row2, text="", bg=S.SURFACE, fg=S.LABEL3,
                                font=S.FONT_TINY)
        self.lbl_ver.pack(side="left", padx=(8, 0))

        # click anywhere on the chip opens the menu; hover lightens it.
        self._chip_widgets = [chip, inner, self.lbl_host, self.lbl_host_addr,
                              self.lbl_caret]
        for w in self._chip_widgets:
            try:
                w.configure(cursor="hand2")
            except tk.TclError:
                pass
            w.bind("<Button-1>", self._open_host_menu)
            w.bind("<Enter>", lambda e: self._chip_bg(S.ELEVATED_HOVER))
            w.bind("<Leave>", lambda e: self._chip_leave())

        right = tk.Frame(bar, bg=S.SURFACE)
        right.pack(side="right", padx=S.PAD_LG)
        self.conn_dot = S.Dot(right, color=S.RED, r=5, bg=S.SURFACE)
        self.conn_dot.pack(side="left", padx=(0, 7))
        self.lbl_conn = tk.Label(right, text="connecting…", bg=S.SURFACE,
                                 fg=S.LABEL2, font=S.FONT_BOLD)
        self.lbl_conn.pack(side="left")

        # Server info bar — the bridge's host / IPs / endpoints, in the top bar.
        sbar = tk.Frame(self.root, bg=S.PANEL_BG, height=30)
        sbar.pack(fill="x", side="top")
        sbar.pack_propagate(False)
        self.srv_bar = tk.Frame(sbar, bg=S.PANEL_BG)
        self.srv_bar.pack(fill="x", padx=S.PAD_LG, pady=5)
        tk.Label(self.srv_bar, text="server: waiting…", bg=S.PANEL_BG,
                 fg=S.LABEL3, font=S.FONT_SMALL).pack(side="left")

    def _chip_bg(self, color):
        for w in self._chip_widgets:
            try:
                w.configure(bg=color)
            except tk.TclError:
                pass
        self.host_dot.configure(bg=color)

    def _chip_leave(self):
        def chk():
            try:
                under = self.host_chip.winfo_containing(
                    *self.host_chip.winfo_pointerxy())
            except Exception:
                under = None
            if under not in self._chip_widgets:
                self._chip_bg(S.ELEVATED)
        self.host_chip.after(12, chk)

    # ── host selector (liquid-glass popover) ──────────────────────────────────
    def _open_host_menu(self, _event=None):
        if getattr(self, "_host_menu", None) is not None:   # toggle closed
            self._close_host_menu()
            return
        items = list(self._hosts.items())
        W, pad, rowh = 392, 8, 46

        top = tk.Toplevel(self.root)
        top.withdraw()                  # build + position hidden → no flash-then-jump
        top.overrideredirect(True)
        transparent = False
        try:
            top.attributes("-transparent", True)
            top.configure(bg="systemTransparent")
            transparent = True
        except Exception:
            top.configure(bg=S.BG)
        try:
            top.attributes("-alpha", 0.96)              # the glass translucency
        except Exception:
            pass
        self._host_menu = top
        cbg = "systemTransparent" if transparent else S.BG
        cv = tk.Canvas(top, bg=cbg, highlightthickness=0)
        cv.pack(fill="both", expand=True)

        body = tk.Frame(top, bg=S.GLASS)
        win = cv.create_window(pad, pad, anchor="nw", window=body, width=W - 2 * pad)

        for key, hs in items:
            self._glass_host_row(body, key, hs)
        tk.Frame(body, bg=S.GLASS_BORDER, height=1).pack(fill="x", padx=10, pady=5)
        ahs = self._hosts.get(self._active)
        if ahs is not None and ahs.online:
            self._glass_action_row(
                body, "✎", f"Rename “{ahs.name()}”…", self._prompt_rename)
        self._glass_action_row(body, "＋", "Add host…", self._prompt_add_host)
        if self._browser:
            tk.Label(body, text="discovering bridges on the LAN…", bg=S.GLASS,
                     fg=S.LABEL3, font=S.FONT_TINY).pack(anchor="w", padx=16,
                                                         pady=(2, 6))

        # size to content, position under the host button, draw the glass
        top.update_idletasks()
        H = body.winfo_reqheight() + 2 * pad
        cv.configure(width=W, height=H)
        S.glass_panel(cv, 2, 2, W - 2, H - 2, 18)
        cv.tag_raise(win)                               # rows sit ON the glass panel
        # Expand straight DOWN from the host button.
        try:
            self.root.update_idletasks()
            x = self.host_chip.winfo_rootx()
            y = self.host_chip.winfo_rooty() + self.host_chip.winfo_height() + 6
        except Exception:
            x = self.root.winfo_rootx() + S.PAD_LG
            y = self.root.winfo_rooty() + 70
        top.geometry(f"{W}x{H}+{x}+{y}")

        top.bind("<Escape>", lambda e: self._close_host_menu())
        top.bind("<FocusOut>", lambda e: self._close_host_menu())
        self.root.bind("<Button-1>", self._click_outside_menu, add="+")
        top.deiconify()                 # now show it, already sized + placed
        top.lift()
        top.focus_force()

    def _glass_host_row(self, body, key, hs):
        active = (key == self._active)
        rf = tk.Frame(body, bg=S.GLASS, cursor="hand2")
        rf.pack(fill="x")
        inner = tk.Frame(rf, bg=S.GLASS)
        inner.pack(fill="x", padx=12, pady=7)
        dot = S.Dot(inner, color=S.GREEN if hs.online else S.LABEL3, r=4, bg=S.GLASS)
        dot.pack(side="left", padx=(0, 9))
        txt = tk.Frame(inner, bg=S.GLASS)
        txt.pack(side="left", fill="x", expand=True)
        name = tk.Label(txt, text=hs.name(), bg=S.GLASS,
                        fg=S.LABEL if hs.online else S.LABEL2, font=S.FONT_BOLD,
                        anchor="w")
        name.pack(anchor="w")
        n = hs.glob.get("online", len(hs.clients))
        sub = tk.Label(txt, text=f"{hs.host}:{hs.port}   ·   {n} client(s)",
                       bg=S.GLASS, fg=S.LABEL3, font=S.FONT_MONO_S, anchor="w")
        sub.pack(anchor="w")
        check = tk.Label(inner, text="✓" if active else "", bg=S.GLASS,
                         fg=S.ORANGE, font=S.FONT_H2)
        check.pack(side="right")
        widgets = [rf, inner, txt, name, sub, check]
        self._glass_hover(widgets, dot,
                          lambda k=key: (self._close_host_menu(), self._switch_host(k)))

    def _glass_action_row(self, body, glyph, label, cmd):
        rf = tk.Frame(body, bg=S.GLASS, cursor="hand2")
        rf.pack(fill="x")
        inner = tk.Frame(rf, bg=S.GLASS)
        inner.pack(fill="x", padx=12, pady=7)
        g = tk.Label(inner, text=glyph, bg=S.GLASS, fg=S.BLUE, font=S.FONT_H2)
        g.pack(side="left", padx=(0, 9))
        lb = tk.Label(inner, text=label, bg=S.GLASS, fg=S.BLUE, font=S.FONT_BODY)
        lb.pack(side="left")
        self._glass_hover([rf, inner, g, lb], None,
                          lambda: (self._close_host_menu(), cmd()))

    def _glass_hover(self, widgets, dot, on_click):
        def setbg(c):
            for w in widgets:
                try:
                    w.configure(bg=c)
                except tk.TclError:
                    pass
            if dot is not None:
                dot.configure(bg=c)
        def leave(_e):
            def chk():
                try:
                    under = widgets[0].winfo_containing(*widgets[0].winfo_pointerxy())
                except Exception:
                    under = None
                if under not in widgets:
                    setbg(S.GLASS)
            widgets[0].after(12, chk)
        for w in widgets:
            w.bind("<Enter>", lambda e: setbg(S.GLASS_HOVER), add="+")
            w.bind("<Leave>", leave, add="+")
            w.bind("<Button-1>", lambda e: on_click(), add="+")

    def _click_outside_menu(self, event):
        m = getattr(self, "_host_menu", None)
        if m is None:
            return
        # close if the click landed outside the popover window
        if event.widget.winfo_toplevel() is not m:
            self._close_host_menu()

    def _close_host_menu(self):
        m = getattr(self, "_host_menu", None)
        self._host_menu = None
        try:
            self.root.unbind("<Button-1>")
        except Exception:
            pass
        if m is not None:
            try:
                m.destroy()
            except Exception:
                pass

    def _update_host_selector(self):
        """Refresh the top-left button to the active host's identity + status."""
        hs = self._hosts.get(self._active) if self._active else None
        if hs is None:
            self.lbl_host.configure(text="—")
            self.lbl_host_addr.configure(text="")
            self.host_dot.set(S.LABEL3)
            return
        n_hosts = len(self._hosts)
        self.lbl_host.configure(text=hs.name())
        si = hs.glob.get("serverInfo") or {}
        addr = si.get("lanIp") or si.get("tailscaleIp") or hs.host
        suffix = f"   ·   {n_hosts} hosts" if n_hosts > 1 else ""
        self.lbl_host_addr.configure(text=f"{addr}:{hs.port}{suffix}")
        self.host_dot.set(S.GREEN if hs.online else S.RED)
        self.lbl_caret.configure(text="▾" if n_hosts > 1 else " ")

    def _prompt_rename(self):
        """Rename the active robot from the monitor — sent over its /monitor ws so
        the bridge persists it and syncs the new name to every client."""
        from tkinter import simpledialog
        hs = self._hosts.get(self._active)
        conn = self._conns.get(self._active)
        if hs is None or conn is None:
            return
        cur = hs.name()
        new = simpledialog.askstring("Rename robot", f"New name for “{cur}”:",
                                     initialvalue=cur, parent=self.root)
        if new and new.strip():
            conn.send({"type": "set_name", "name": new.strip()})

    def _prompt_add_host(self):
        from tkinter import simpledialog
        ans = simpledialog.askstring(
            "Add host", "Bridge host (host or host:port):", parent=self.root)
        if not ans:
            return
        ans = ans.strip()
        host, _, port = ans.rpartition(":")
        if host and port.isdigit():
            key = self._add_host(host, int(port), start=True)
        else:
            key = self._add_host(ans, 9090, start=True)
        self._switch_host(key)

    def _switch_host(self, key: str):
        if key not in self._hosts or key == self._active:
            return
        self._active = key
        self._reset_render_caches()
        hs = self._hosts[key]
        self._render_active()
        self._clear_stream()
        for ev in list(hs.events):
            self._render_event_line(ev)
        self._update_host_selector()

    def _reset_render_caches(self):
        self._server_cache = object()    # force the server bar to rebuild
        self._act_cache = None
        self._cmd_sig = None
        self._cmd_w = None
        self._clients_sig = None         # force the client list to rebuild

    def _render_active(self):
        hs = self._hosts.get(self._active)
        if hs is None:
            return
        self._global = hs.glob
        self._set_clients(hs.clients)
        self._refresh_status()
        self._set_conn_indicator(hs.online)

    # ── status strip ──────────────────────────────────────────────────────────
    def _build_status_strip(self):
        wrap = tk.Frame(self.root, bg=S.BG)
        wrap.pack(fill="x", padx=S.PAD, pady=(S.PAD, 0))

        self.p_clients = S.Pill(wrap, "Clients", width=104, bg=S.BG)
        self.p_driver  = S.Pill(wrap, "Active Driver", width=180, bg=S.BG)
        self.p_rate    = S.Pill(wrap, "Cmd Rate", width=110, bg=S.BG)
        self.p_mode    = S.Pill(wrap, "Teleop Mode", width=150, bg=S.BG)
        self.p_control = S.Pill(wrap, "Control Law", width=168, bg=S.BG)
        self.p_region  = S.Pill(wrap, "Region", width=128, bg=S.BG)
        self.p_head    = S.Pill(wrap, "Head Track", width=128, bg=S.BG)
        for p in (self.p_clients, self.p_driver, self.p_rate, self.p_mode,
                  self.p_control, self.p_region, self.p_head):
            p.pack(side="left", padx=(0, S.PAD_XS))

        self.t_handL = S.Toggle(wrap, "Hand L", bg=S.BG)
        self.t_handR = S.Toggle(wrap, "Hand R", bg=S.BG)
        self.t_estop = S.Toggle(wrap, "E-Stop", width=84, bg=S.BG)
        self.t_rec   = S.Toggle(wrap, "Record", width=84, bg=S.BG)
        self.t_reset = S.Toggle(wrap, "Reset", bg=S.BG)
        for t in (self.t_handL, self.t_handR, self.t_estop, self.t_rec, self.t_reset):
            t.pack(side="left", padx=(0, S.PAD_XS))

        # Second status row: the recording task folder picked right now.
        wrap2 = tk.Frame(self.root, bg=S.BG)
        wrap2.pack(fill="x", padx=S.PAD, pady=(S.PAD_XS, 0))
        self.p_task = S.Pill(wrap2, "Recording Task Folder", width=300, bg=S.BG)
        self.p_task.pack(side="left", padx=(0, S.PAD_XS))

    # ── body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=S.BG)
        body.pack(fill="both", expand=True, padx=S.PAD, pady=S.PAD)

        # LEFT — connected clients (scrollable, full height).
        left = S.Card(body, title="Connected Clients")
        left.pack(side="left", fill="y", padx=(0, S.PAD))
        left.configure(width=322)
        left.pack_propagate(False)
        self.clients_box = self._scrollable(left)
        tk.Label(self.clients_box, text="no clients connected",
                 bg=S.SURFACE, fg=S.LABEL3, font=S.FONT_SMALL).pack(pady=S.PAD_LG)

        # RIGHT — the standardized robot controls dispatched to the actuators.
        rightc = S.Card(body, title="Robot Controls  →  actuators")
        rightc.pack(side="right", fill="y", padx=(S.PAD, 0))
        rightc.configure(width=372)
        rightc.pack_propagate(False)
        self.cmd_box = self._scrollable(rightc)
        self._act_cache = None
        self._cmd_w = None          # persistent control widgets (built once)
        self._cmd_sig = None        # structural signature (joint/gripper key set)
        tk.Label(self.cmd_box, text="no command yet", bg=S.SURFACE,
                 fg=S.LABEL3, font=S.FONT_SMALL).pack(pady=S.PAD_LG)

        # CENTER — the command stream (fills the space between).
        center = S.Card(body, title="Command Stream")
        center.pack(side="left", fill="both", expand=True)
        right = center

        tools = tk.Frame(right, bg=S.SURFACE)
        tools.pack(fill="x", padx=S.PAD, pady=(2, 4))
        self.lbl_eps = tk.Label(tools, text="0 events", bg=S.SURFACE,
                                fg=S.LABEL3, font=S.FONT_SMALL)
        self.lbl_eps.pack(side="left")
        self.btn_clear = self._chip(tools, "Clear", self._clear_stream)
        self.btn_clear.pack(side="right")
        self.btn_pause = self._chip(tools, "Pause", self._toggle_pause)
        self.btn_pause.pack(side="right", padx=(0, 6))

        txt_wrap = tk.Frame(right, bg=S.PANEL_BG)
        txt_wrap.pack(fill="both", expand=True, padx=S.PAD_SM,
                      pady=(0, S.PAD_SM))
        self.txt = tk.Text(txt_wrap, bg=S.PANEL_BG, fg=S.LABEL,
                           font=S.FONT_MONO_S, bd=0, highlightthickness=0,
                           wrap="none", padx=10, pady=8, spacing1=1, spacing3=1,
                           insertwidth=0, cursor="arrow")
        sb = tk.Scrollbar(txt_wrap, command=self.txt.yview, width=10)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)
        self.txt.configure(state="disabled")
        # colour tags
        for name, color in S.TYPE_COLOR.items():
            self.txt.tag_configure(f"type_{name}", foreground=color)
        for name, color in S.LEVEL_COLOR.items():
            self.txt.tag_configure(f"lvl_{name}", foreground=color)
        self.txt.tag_configure("dim", foreground=S.LABEL3)
        self.txt.tag_configure("peer", foreground=S.LABEL2)

    def _chip(self, parent, text, cmd) -> tk.Label:
        lbl = tk.Label(parent, text=text, bg=S.ELEVATED, fg=S.LABEL,
                       font=S.FONT_TINY, padx=12, pady=4, cursor="hand2")
        lbl.bind("<Button-1>", lambda _e: cmd())
        return lbl

    def _scrollable(self, parent) -> tk.Frame:
        """A vertically scrollable region inside `parent`; returns the inner
        Frame to pack rows into. Wheel scrolls while the cursor is over it
        (bind_all on enter/leave — works over the child rows too)."""
        canvas = tk.Canvas(parent, bg=S.SURFACE, highlightthickness=0, bd=0)
        sb = tk.Scrollbar(parent, command=canvas.yview, width=10)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True,
                    padx=(S.PAD_SM, 0), pady=S.PAD_SM)
        inner = tk.Frame(canvas, bg=S.SURFACE)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _wheel(e):
            if getattr(e, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(e, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")
        def _bind(_e):
            canvas.bind_all("<MouseWheel>", _wheel)
            canvas.bind_all("<Button-4>", _wheel)
            canvas.bind_all("<Button-5>", _wheel)
        def _unbind(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)
        return inner

    # ── server card ───────────────────────────────────────────────────────────
    def _refresh_server(self, info: dict):
        if info == self._server_cache:
            return
        self._server_cache = info
        bar = self.srv_bar
        for w in bar.winfo_children():
            w.destroy()
        if not info:
            tk.Label(bar, text="server: waiting…", bg=S.PANEL_BG, fg=S.LABEL3,
                     font=S.FONT_SMALL).pack(side="left")
            return

        def item(text, color=S.LABEL2, mono=True):
            tk.Label(bar, text=text, bg=S.PANEL_BG, fg=color,
                     font=S.FONT_MONO_S if mono else S.FONT_SMALL).pack(side="left")

        def sep():
            tk.Label(bar, text="   ·   ", bg=S.PANEL_BG, fg=S.LABEL3,
                     font=S.FONT_TINY).pack(side="left")

        ws = info.get("wsPort", 9090)
        tk.Label(bar, text=info.get("host", "?"), bg=S.PANEL_BG, fg=S.LABEL,
                 font=S.FONT_BOLD).pack(side="left")
        item(f"  {info.get('platform', '')}", S.TEAL, mono=False)
        if info.get("lanIp"):
            sep(); item("LAN ", S.LABEL3); item(f"{info['lanIp']}:{ws}", S.GREEN)
        if info.get("tailscaleIp"):
            sep(); item("TS ", S.LABEL3); item(f"{info['tailscaleIp']}:{ws}", S.PURPLE)
        sep(); item("teleop ", S.LABEL3); item(f":{ws}{info.get('wsPath', '/teleop')}")
        sep(); item("cams ", S.LABEL3); item(f":{info.get('httpPort', 8080)}")
        if info.get("udpPort"):
            sep(); item("udp ", S.LABEL3); item(f":{info['udpPort']}")
        for label, on, col in (("TOKEN", info.get("tokenGated"), S.GREEN),):
            if on:
                tk.Label(bar, text=f" {label} ", bg=col, fg=S.BG,
                         font=S.FONT_TINY).pack(side="left", padx=(8, 0))

        # uptime sits at the far right of the bar
        self.srv_uptime = tk.Label(bar, text="", bg=S.PANEL_BG, fg=S.LABEL3,
                                   font=S.FONT_TINY)
        self.srv_uptime.pack(side="right")

    # ── robot command panel (the standardized actuation) ──────────────────────
    _GROUP_LABEL = {"base": "Base steer", "lift": "Lift", "torso": "Torso",
                    "neck": "Neck", "left_arm": "Left arm", "right_arm": "Right arm",
                    "gripper": "Grippers", "other": "Other"}
    _GROUP_ORDER = ["base", "lift", "torso", "neck", "left_arm", "right_arm",
                    "gripper", "other"]

    # Orca-hand finger joints are "Left Thumb MCP", "Right Pinky Abd", … — none
    # contain "grip"/"finger", so match the finger names explicitly.
    _FINGERS = ("thumb", "index", "middle", "ring", "pinky", "finger", "grip", "hand")

    @staticmethod
    def _group_of(name: str) -> str:
        n = name.lower()
        if "arm" in n or "wrist" in n:
            return "left_arm" if "left" in n else "right_arm" if "right" in n else "other"
        if any(f in n for f in MonitorApp._FINGERS):
            return "gripper"
        for key in ("torso", "neck", "lift"):
            if key in n:
                return key
        if "steer" in n or "drive" in n or "wheel" in n:
            return "base"
        return "other"

    # Base-twist bar specs: (key, label, full-scale, unit, bipolar, colour).
    _BASE_SPECS = [
        ("vx", "vx", 0.8, "m/s", True, S.ORANGE),
        ("vy", "vy", 0.8, "m/s", True, S.ORANGE),
        ("omega", "ω", 2.0, "rad/s", True, S.ORANGE),
        ("lift", "lift", 0.635, "m", False, S.GREEN),
    ]

    def _refresh_actuation(self, act: dict | None):
        if not act:
            if self._cmd_sig is not None or self._cmd_w is None:
                for w in self.cmd_box.winfo_children():
                    w.destroy()
                tk.Label(self.cmd_box, text="no command yet", bg=S.SURFACE,
                         fg=S.LABEL3, font=S.FONT_SMALL).pack(pady=S.PAD_LG)
                self._cmd_sig, self._cmd_w = None, None
            return
        joints = act.get("joints") or {}
        grippers = act.get("grippers") or {}
        sig = (tuple(sorted(joints)), tuple(sorted(grippers)))
        if sig != self._cmd_sig:
            self._build_cmd_widgets(joints, grippers)
            self._cmd_sig = sig
        self._update_cmd_values(act, joints, grippers)

    @staticmethod
    def _short_joint(name: str) -> str:
        """Compact joint label for the narrow columns: "Left Arm 3" → "A3",
        "Left Thumb MCP" → "Thu MCP"."""
        s = name.replace("Left ", "").replace("Right ", "").replace("Arm ", "A")
        parts = s.split()
        if len(parts) == 2 and parts[0].lower() in (
                "thumb", "index", "middle", "ring", "pinky"):
            return f"{parts[0][:3]} {parts[1]}"
        return s[:8]

    def _bar_row(self, parent, label, bipolar=True, label_w=12, val_w=8,
                 prec=3, bar_h=11):
        """A `label  [====bar====]  value` row; returns (bar, value_label)."""
        r = tk.Frame(parent, bg=S.SURFACE)
        r.pack(fill="x", pady=1)
        tk.Label(r, text=label, bg=S.SURFACE, fg=S.LABEL2, font=S.FONT_MONO_S,
                 width=label_w, anchor="w").pack(side="left")
        val = tk.Label(r, text="", bg=S.SURFACE, fg=S.LABEL, font=S.FONT_MONO_S,
                       width=val_w, anchor="e")
        val.pack(side="right")
        bar = S.Bar(r, width=60, height=bar_h, bipolar=bipolar, bg=S.SURFACE)
        bar.pack(side="left", fill="x", expand=True, padx=(3, 4))
        val._prec = prec
        return bar, val

    def _two_col(self, parent, left_title, right_title):
        """A titled two-column row; returns (left_frame, right_frame)."""
        hd = tk.Frame(parent, bg=S.SURFACE)
        hd.pack(fill="x", pady=(6, 1))
        tk.Label(hd, text=left_title, bg=S.SURFACE, fg=S.TEAL, font=S.FONT_SMALL,
                 anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(hd, text=right_title, bg=S.SURFACE, fg=S.TEAL, font=S.FONT_SMALL,
                 anchor="w").pack(side="left", fill="x", expand=True, padx=(6, 0))
        row = tk.Frame(parent, bg=S.SURFACE)
        row.pack(fill="x")
        lf = tk.Frame(row, bg=S.SURFACE)
        lf.pack(side="left", fill="both", expand=True)
        rf = tk.Frame(row, bg=S.SURFACE)
        rf.pack(side="left", fill="both", expand=True, padx=(6, 0))
        return lf, rf

    def _build_cmd_widgets(self, joints: dict, grippers: dict):
        for w in self.cmd_box.winfo_children():
            w.destroy()
        box = self.cmd_box
        W: dict = {"base": {}, "wheel": {}, "grip": {}, "joint": {}}

        def header(text):
            tk.Label(box, text=text, bg=S.SURFACE, fg=S.LABEL2,
                     font=S.FONT_TINY).pack(anchor="w", pady=(9, 2))

        # Dispatch target + sequence + source + mode + estop
        head = tk.Frame(box, bg=S.SURFACE)
        head.pack(fill="x", pady=(0, 3))
        W["target"] = tk.Label(head, text="", fg=S.BG, font=S.FONT_TINY)
        W["target"].pack(side="left")
        W["seq"] = tk.Label(head, text="", bg=S.SURFACE, fg=S.LABEL3,
                            font=S.FONT_MONO_S)
        W["seq"].pack(side="right")
        W["source"] = tk.Label(box, text="", bg=S.SURFACE, fg=S.TEAL,
                               font=S.FONT_SMALL, anchor="w")
        W["source"].pack(fill="x")
        W["mode"] = tk.Label(box, text="", bg=S.SURFACE, fg=S.ORANGE,
                             font=S.FONT_SMALL, anchor="w")
        W["mode"].pack(fill="x", pady=(1, 0))
        W["estop"] = tk.Label(box, text="", bg=S.SURFACE, fg=S.BG,
                              font=S.FONT_TINY)
        W["estop"].pack(anchor="w", pady=(2, 0))

        grouped: dict = {}
        for name in joints:
            grouped.setdefault(self._group_of(name), []).append(name)

        # BASE twist (1 column) → /mabel_cmd
        header("BASE TWIST  →  /mabel_cmd")
        for key, label, _fs, _u, bip, _c in self._BASE_SPECS:
            W["base"][key] = self._bar_row(box, label, bipolar=bip, label_w=6)

        # SWERVE MODULES — 2 columns: left = DRIVE, right = STEER, per module.
        l, r = self._two_col(box, "drive (rad/s)", "steer (rad)")
        for mod in ("FL", "FR", "B"):
            W["wheel"][mod] = {
                "drive": self._bar_row(l, mod, bipolar=True, label_w=3, val_w=7, prec=2),
                "steer": self._bar_row(r, mod, bipolar=True, label_w=3, val_w=7, prec=2),
            }

        # TORSO / NECK — 1 column, full width.
        for g in ("torso", "neck"):
            if g in grouped:
                tk.Label(box, text=self._GROUP_LABEL[g], bg=S.SURFACE, fg=S.TEAL,
                         font=S.FONT_SMALL).pack(anchor="w", pady=(6, 1))
                for name in grouped[g]:
                    W["joint"][name] = self._bar_row(box, name[:12], label_w=10)

        # ARMS — 2 columns: left arm | right arm.
        if "left_arm" in grouped or "right_arm" in grouped:
            l, r = self._two_col(box, "LEFT ARM", "RIGHT ARM")
            for name in grouped.get("left_arm", []):
                W["joint"][name] = self._bar_row(l, self._short_joint(name),
                                                 label_w=4, val_w=7, prec=2)
            for name in grouped.get("right_arm", []):
                W["joint"][name] = self._bar_row(r, self._short_joint(name),
                                                 label_w=4, val_w=7, prec=2)

        # HANDS — 2 columns: left hand | right hand (gripper + any finger joints).
        finger = grouped.get("gripper", [])
        if grippers or finger:
            l, r = self._two_col(box, "LEFT HAND", "RIGHT HAND")
            if "left" in grippers:
                W["grip"]["left"] = self._bar_row(l, "grip", bipolar=False,
                                                  label_w=5, val_w=6, prec=2)
            if "right" in grippers:
                W["grip"]["right"] = self._bar_row(r, "grip", bipolar=False,
                                                   label_w=5, val_w=6, prec=2)
            for name in finger:
                side = l if "left" in name.lower() else r
                W["joint"][name] = self._bar_row(side, self._short_joint(name),
                                                 label_w=7, val_w=6, prec=2)

        # OTHER — anything uncategorised, 1 column.
        if grouped.get("other"):
            header("OTHER")
            for name in grouped["other"]:
                W["joint"][name] = self._bar_row(box, name[:12], label_w=10)

        header("PAYLOAD (JSON)  ·  the literal command dispatched")
        W["raw"] = tk.Label(box, text="", bg=S.PANEL_BG, fg=S.GREEN,
                            font=S.FONT_MONO_S, anchor="w", justify="left",
                            wraplength=336, padx=6, pady=5)
        W["raw"].pack(fill="x", pady=(0, 4))
        self._cmd_w = W

    @staticmethod
    def _joint_frac(value: float, lim) -> float:
        if lim and lim[1] > lim[0]:
            lo, hi = lim
            if lo < 0 < hi:
                return value / max(abs(lo), abs(hi))
            return 2 * (value - lo) / (hi - lo) - 1      # position within range
        return max(-1.0, min(1.0, value / 3.14))         # nominal ± arm range

    def _update_cmd_values(self, act, joints, grippers):
        W = self._cmd_w
        if not W:
            return
        limits = (self._global.get("serverInfo") or {}).get("jointLimits") or {}
        real = str(act.get("target", "")).startswith("real")
        W["target"].configure(text=f" ▶ {'REAL ROBOT' if real else 'MUJOCO SIM'} ",
                               bg=S.RED if real else S.GREEN)
        W["seq"].configure(text=f"#{act.get('seq', 0)}")
        src = act.get("source") or {}
        W["source"].configure(text=f"{src.get('device','—')}   {src.get('peer','')}")
        m = act.get("mode") or {}
        W["mode"].configure(text=" · ".join(str(m.get(k)) for k in
                            ("method", "controlType", "region") if m.get(k)) or "—")
        if act.get("estop"):
            W["estop"].configure(text=" E-STOP ENGAGED ", bg=S.RED)
        else:
            W["estop"].configure(text="", bg=S.SURFACE)

        b = act.get("base") or {}
        for key, _l, fs, unit, bip, col in self._BASE_SPECS:
            bar, val = W["base"][key]
            v = float(b.get(key, 0.0))
            bar.set(v / fs, col)
            val.configure(text=f"{v:+.{getattr(val, '_prec', 3)}f}")

        # Swerve wheels: drive scaled to ±(MAX_LIN/wheel-radius)≈21, steer to ±π.
        wheels = act.get("wheels") or {}
        for mod, refs in W["wheel"].items():
            wm = wheels.get(mod, {})
            dbar, dval = refs["drive"]
            sbar, sval = refs["steer"]
            dv, sv = float(wm.get("drive", 0.0)), float(wm.get("steer", 0.0))
            dbar.set(dv / 21.0, S.ORANGE)
            dval.configure(text=f"{dv:+.{getattr(dval, '_prec', 2)}f}")
            sbar.set(sv / 3.1416, S.PURPLE)
            sval.configure(text=f"{sv:+.{getattr(sval, '_prec', 2)}f}")

        for s, (bar, val) in W["grip"].items():
            v = float(grippers.get(s, 0.0))
            bar.set(v, S.PURPLE)
            val.configure(text=f"{v:.{getattr(val, '_prec', 2)}f}")
        for name, (bar, val) in W["joint"].items():
            v = float(joints.get(name, 0.0))
            bar.set(self._joint_frac(v, limits.get(name)), S.BLUE)
            val.configure(text=f"{v:+.{getattr(val, '_prec', 3)}f}")

        raw = json.dumps(act, separators=(",", ":"))
        W["raw"].configure(text=raw if len(raw) <= 1600 else raw[:1600] + " …")

    # ── stream controls ──────────────────────────────────────────────────────
    def _toggle_pause(self):
        self._paused = not self._paused
        self.btn_pause.configure(
            text="Resume" if self._paused else "Pause",
            bg=S.ORANGE if self._paused else S.ELEVATED,
            fg=S.BG if self._paused else S.LABEL)

    def _clear_stream(self):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    # ── queue drain (Tk main thread) ──────────────────────────────────────────
    def _drain(self):
        try:
            while True:
                host_key, kind, msg = self.q.get_nowait()
                self._handle(host_key, kind, msg)
        except queue.Empty:
            pass
        self.root.after(33, self._drain)

    def _handle(self, host_key: str, kind: str, msg: dict):
        if host_key == "__discover__":
            self._maybe_add_discovered(msg)
            return
        hs = self._hosts.get(host_key)
        if hs is None:
            return
        active = (host_key == self._active)
        if kind == "conn":
            hs.online = msg.get("state") == "online"
            if not hs.online:
                hs.glob, hs.clients = {}, []
            if active:
                self._set_conn_indicator(hs.online)
                if not hs.online:
                    self._global = {}
                    self._set_clients([])
                    self._refresh_status()
            self._update_host_selector()
        elif kind == "snapshot":
            hs.glob = msg.get("global", {})
            hs.clients = msg.get("clients", [])
            for ev in msg.get("log", [])[-150:]:
                hs.events.append(ev)
            if active:
                self._render_active()
                self._clear_stream()
                for ev in list(hs.events):
                    self._render_event_line(ev)
            self._dedup_hosts()         # collapse a same-bridge duplicate now
            self._update_host_selector()
        elif kind == "state":
            hs.glob = msg.get("global", {})
            hs.clients = msg.get("clients", [])
            if active:
                self._render_active()
            self._update_host_selector()
        elif kind == "event":
            ev = msg.get("event", {})
            hs.events.append(ev)
            hs.evt_count += 1
            hs.evt_window.append(time.time())
            if active:
                self._append_event(ev)
                if ev.get("type") == "reset":
                    self._reset_flash_until = time.time() + 1.5

    # ── connection state ──────────────────────────────────────────────────────
    def _set_conn_indicator(self, online: bool):
        hs = self._hosts.get(self._active)
        addr = f"{hs.host}:{hs.port}" if hs else ""
        if online:
            self.conn_dot.set(S.GREEN)
            self.lbl_conn.configure(text=f"connected  {addr}", fg=S.GREEN)
        else:
            self.conn_dot.set(S.RED)
            self.lbl_conn.configure(text=f"offline  {addr}", fg=S.RED)

    # ── status pills ──────────────────────────────────────────────────────────
    def _driver_client(self) -> dict | None:
        did = self._global.get("driverId")
        for c in self._clients.values():
            if c.get("id") == did:
                return c
        return next(iter(self._clients.values()), None)

    def _refresh_status(self):
        g = self._global
        if g.get("version"):
            self.lbl_ver.configure(text=f"bridge {g['version']}")
        n = g.get("online", len(self._clients))
        self.p_clients.set(str(n), accent=S.GREEN if n else S.LABEL3)
        self._refresh_server(g.get("serverInfo") or {})
        self._refresh_actuation(g.get("actuation"))

        # Active-driver pill = arbitration winner (who the robot follows).
        d = self._driver_client()
        if not d:
            self.p_driver.set("—", accent=S.LABEL3)
            self.p_rate.set("0 Hz", accent=S.LABEL3)
        else:
            dev = d.get("device", "—")
            accent, _chip, _glyph = S.device_style(dev)
            self.p_driver.set(dev, accent=accent, sub=d.get("peer", ""))
            rate = d.get("rateHz", 0.0)
            self.p_rate.set(f"{rate:g} Hz",
                            accent=S.GREEN if rate >= 1 else S.LABEL3)

        # Headline intent/state pills = FUSED live status (updates the instant a
        # message arrives, regardless of which client is the arbitration driver).
        st = g.get("status") or {}
        has = bool(self._clients)
        mode = st.get("mode", "—") if has else "—"
        self.p_mode.set(mode, accent=(S.BLUE if "Base" in mode else S.PURPLE)
                        if mode not in ("—", "") else S.LABEL3)
        ctype = st.get("ctype", "—") if has else "—"
        clabel = {"compliance": "Gravity-Comp", "impedance": "Impedance"}.get(ctype, ctype)
        self.p_control.set(clabel,
                           accent=(S.GREEN if ctype == "compliance"
                                   else S.ORANGE if ctype == "impedance" else S.LABEL3),
                           sub=f"K{st.get('stiffness', 0):g}" if ctype == "impedance" else "")
        region = (st.get("region", "—") or "—").replace("_", " ") if has else "—"
        self.p_region.set(region, accent=S.TEAL if region not in ("—", "") else S.LABEL3)
        hs = st.get("headState", "—") if has else "—"
        self.p_head.set(hs, accent=S.GREEN if hs == "normal"
                        else (S.YELLOW if hs == "limited"
                              else S.RED if hs == "notAvailable" else S.LABEL3))
        self.t_handL.set(bool(st.get("handL")) and has, color=S.PURPLE)
        self.t_handR.set(bool(st.get("handR")) and has, color=S.PURPLE)

        # E-stop — engaged is a hard, unmissable red.
        estop = bool(st.get("estop")) and has
        self.t_estop.set(estop, color=S.RED, text="STOP" if estop else "clear")

        # Recording + the task folder picked right now.
        rec = g.get("recording") or {}
        active = bool(rec.get("isRecording", rec.get("active")))
        self.t_rec.set(active, color=S.RED, text="REC" if active else "OFF")
        task = rec.get("task") or st.get("recTask") or "—"
        nep = rec.get("episodeCount")
        sub = f"{nep} eps" if isinstance(nep, int) else ""
        self.p_task.set(task, accent=S.RED if active else
                        (S.YELLOW if task not in ("—", "", None) else S.LABEL3), sub=sub)

    # ── clients list ──────────────────────────────────────────────────────────
    def _set_clients(self, clients: list):
        self._clients = {c["id"]: c for c in clients}
        did = self._global.get("driverId")
        # _render_active runs at the ~5 Hz heartbeat. Destroying + recreating every
        # client row each tick made the list visibly FLICKER ("clients show then
        # disappear"). Rebuild ONLY when the set / identity / driver changes; refresh
        # the live per-client rate in place otherwise.
        def _ident(c):
            net = c.get("net") or {}
            return (c["id"], c.get("device"), c.get("platform"), c.get("session"),
                    net.get("transport"), net.get("ip") or c.get("peer"), bool(net.get("secure")))
        sig = (did, tuple(sorted(_ident(c) for c in clients)))
        if sig == self._clients_sig:
            for c in clients:                       # same rows → just refresh the rate
                lbl = self._client_rate_lbls.get(c["id"])
                if lbl is not None:
                    try:
                        r = c.get("rateHz", 0.0)
                        lbl.configure(text=f"{r:g} Hz", fg=S.GREEN if r >= 1 else S.LABEL3)
                    except tk.TclError:
                        pass
            return
        self._clients_sig = sig
        self._client_rate_lbls = {}
        for w in self.clients_box.winfo_children():
            w.destroy()
        if not clients:
            tk.Label(self.clients_box, text="no clients connected",
                     bg=S.SURFACE, fg=S.LABEL3, font=S.FONT_SMALL).pack(
                pady=S.PAD_LG)
            return
        order = sorted(clients, key=lambda c: (not (c["id"] == did), c["peer"]))
        for c in order:
            self._client_row(c, c["id"] == did)

    def _client_row(self, c: dict, is_driver: bool):
        row = tk.Frame(self.clients_box, bg=S.ELEVATED, highlightthickness=0)
        row.pack(fill="x", pady=3)
        inner = tk.Frame(row, bg=S.ELEVATED)
        inner.pack(fill="x", padx=10, pady=7)

        dev = c.get("device", "Unknown")
        accent, chip, glyph = S.device_style(dev)

        top = tk.Frame(inner, bg=S.ELEVATED)
        top.pack(fill="x")
        dot = S.Dot(top, color=S.ORANGE if is_driver else accent, r=4,
                    bg=S.ELEVATED)
        dot.pack(side="left", padx=(0, 6))
        tk.Label(top, text=dev, bg=S.ELEVATED, fg=S.LABEL,
                 font=S.FONT_BOLD).pack(side="left")
        # a small coloured device chip so the platform reads at a glance
        tk.Label(top, text=f" {chip} ", bg=accent, fg=S.BG,
                 font=S.FONT_TINY).pack(side="left", padx=(7, 0))
        if is_driver:
            tk.Label(top, text="DRIVING", bg=S.ORANGE, fg=S.BG,
                     font=S.FONT_TINY, padx=6).pack(side="right")
        else:
            tk.Label(top, text="observing", bg=S.ELEVATED, fg=S.LABEL3,
                     font=S.FONT_TINY).pack(side="right")

        # Platform line: browser · OS (web) or platform (iOS / visionOS), plus the
        # per-tab session id so two browser openings on one machine read apart.
        plat = c.get("platform") or ""
        sid = c.get("session") or ""
        if plat or sid:
            pl = tk.Frame(inner, bg=S.ELEVATED)
            pl.pack(fill="x", pady=(4, 0))
            if plat:
                tk.Label(pl, text=plat, bg=S.ELEVATED, fg=S.TEAL,
                         font=S.FONT_SMALL).pack(side="left")
            if sid:
                tk.Label(pl, text=f"#{sid}", bg=S.ELEVATED, fg=S.LABEL3,
                         font=S.FONT_MONO_S).pack(side="right")

        # Network line: transport chip + real IP (+ 🔒 if secured) + rate.
        net = c.get("net") or {}
        tname = net.get("transport", "")
        ip = net.get("ip") or c.get("peer", "?").rsplit(":", 1)[0]
        meta = tk.Frame(inner, bg=S.ELEVATED)
        meta.pack(fill="x", pady=(6, 0))
        if tname:
            tcol, tchip = S.transport_style(tname)
            tk.Label(meta, text=f" {tchip} ", bg=tcol, fg=S.BG,
                     font=S.FONT_TINY).pack(side="left", padx=(0, 6))
        lock = " 🔒" if net.get("secure") else ""
        tk.Label(meta, text=f"{ip}{lock}", bg=S.ELEVATED, fg=S.LABEL2,
                 font=S.FONT_MONO_S).pack(side="left")
        rate = c.get("rateHz", 0.0)
        _rl = tk.Label(meta, text=f"{rate:g} Hz", bg=S.ELEVATED,
                       fg=S.GREEN if rate >= 1 else S.LABEL3,
                       font=S.FONT_MONO_S)
        _rl.pack(side="right")
        self._client_rate_lbls[c["id"]] = _rl     # live rate refresh without a rebuild

        # Detail line: the addressed host / port / web origin, when present.
        host = net.get("host") or ""
        origin = net.get("origin") or ""
        port = net.get("port") or c.get("peer", "?").rsplit(":", 1)[-1]
        bits = []
        if host:
            bits.append(host)
        elif port:
            bits.append(f":{port}")
        if origin:
            bits.append(origin.replace("https://", "").replace("http://", ""))
        if bits:
            tk.Label(inner, text="  ·  ".join(bits), bg=S.ELEVATED, fg=S.LABEL3,
                     font=S.FONT_TINY, anchor="w").pack(fill="x", pady=(3, 0))

        sub = tk.Frame(inner, bg=S.ELEVATED)
        sub.pack(fill="x", pady=(3, 0))
        mode = c.get("mode", "—")
        ctype = c.get("ctype", "—")
        tk.Label(sub, text=f"{mode} · {c.get('region','—').replace('_',' ')}",
                 bg=S.ELEVATED, fg=S.LABEL2, font=S.FONT_SMALL).pack(side="left")
        tags = []
        if c.get("handL"):
            tags.append("L")
        if c.get("handR"):
            tags.append("R")
        if tags:
            tk.Label(sub, text="✋" + "".join(tags), bg=S.ELEVATED, fg=S.PURPLE,
                     font=S.FONT_SMALL).pack(side="right")

    # ── command stream ────────────────────────────────────────────────────────
    def _append_event(self, ev: dict):
        """Render a live event for the ACTIVE host (counting happens in _handle)."""
        if ev and not self._paused:
            self._render_event_line(ev)

    def _render_event_line(self, ev: dict):
        if not ev:
            return
        t = ev.get("t", time.time())
        ts = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t%1)*1000):03d}"
        typ = ev.get("type", "?")
        peer = ev.get("peer", "?")
        level = ev.get("level", "info")
        text = ev.get("text", "")

        self.txt.configure(state="normal")
        at_bottom = self.txt.yview()[1] >= 0.999
        self.txt.insert("end", f"{ts}  ", ("dim",))
        self.txt.insert("end", f"{typ:<14}", (f"type_{typ}",))
        self.txt.insert("end", f"{peer:<22}", ("peer",))
        self.txt.insert("end", f"{text}\n", (f"lvl_{level}",))
        # trim
        lines = int(self.txt.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self.txt.delete("1.0", f"{lines - self.MAX_LINES}.0")
        self.txt.configure(state="disabled")
        if at_bottom:
            self.txt.see("end")

    # ── periodic UI tick ──────────────────────────────────────────────────────
    def _tick_ui(self):
        now = time.time()
        hs = self._hosts.get(self._active) if self._active else None
        if hs is not None:
            hs.evt_window = [t for t in hs.evt_window if now - t < 1.0]
            self.lbl_eps.configure(
                text=f"{hs.evt_count} events   ·   {len(hs.evt_window)}/s")
        # reset flash
        if now < self._reset_flash_until:
            on = int(now * 4) % 2 == 0
            self.t_reset.set(on, color=S.RED, text="RESET")
        else:
            self.t_reset.set(False, color=S.RED, text="—")
        # server uptime
        info = self._global.get("serverInfo") or {}
        lbl = getattr(self, "srv_uptime", None)
        if lbl is not None and info.get("startedAt"):
            up = max(0.0, now - info["startedAt"])
            h, rem = divmod(int(up), 3600)
            m, s = divmod(rem, 60)
            txt = f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")
            lbl.configure(text=f"uptime {txt}")
        self.root.after(150, self._tick_ui)


# ───────────────────────────── entry ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="MABEL Command Monitor — live view of every teleop command "
                    "the bridge receives, and from which client.")
    ap.add_argument("--host", default=os.getenv("MABEL_HOST", "localhost"),
                    help="primary bridge host (default: localhost)")
    ap.add_argument("--port", type=int, default=int(os.getenv("MABEL_PORT", "9090")),
                    help="bridge teleop/monitor port (default: 9090)")
    ap.add_argument("--hosts", default=os.getenv("MABEL_HOSTS", ""),
                    help="extra bridge hosts to watch, comma-separated "
                         "(host or host:port) — for multiple robots/sims at once")
    ap.add_argument("--path", default="/monitor", help="monitor ws path")
    ap.add_argument("--token", default=os.getenv("MABEL_TOKEN", ""),
                    help="access code (MABEL_TOKEN), if the bridge requires one")
    ap.add_argument("--no-discover", action="store_true",
                    help="don't auto-discover other bridges via Bonjour")
    args = ap.parse_args()

    # Build the host list: --host:--port first, then any --hosts entries.
    hosts: list[tuple[str, int]] = [(args.host, args.port)]
    for h in args.hosts.split(","):
        h = h.strip()
        if not h:
            continue
        host, _, port = h.rpartition(":")
        hosts.append((host, int(port)) if host and port.isdigit() else (h, 9090))
    # de-dupe, preserve order
    seen, uniq = set(), []
    for hp in hosts:
        k = f"{hp[0]}:{hp[1]}"
        if k not in seen:
            seen.add(k)
            uniq.append(hp)

    root = tk.Tk()
    MonitorApp(root, uniq, args.path, args.token, discover=not args.no_discover)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
