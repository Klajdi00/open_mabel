"""monitor.py — the server Command Monitor, embedded as a Control Center tab.

A 1:1 replica of the standalone ``server/monitor_gui.py``: it runs the EXACT same
``MonitorApp`` (same UI, same /monitor WebSocket client, same auto-discovery) — only it
lives inside a tab instead of its own window. ``MonitorApp`` builds itself into whatever
``root`` it's given and the only top-level window methods it calls on that root are
``title`` / ``geometry`` / ``minsize`` (+ a couple of wm_* niceties), so we hand it a
``_FrameRoot`` — a Frame that quietly absorbs those — and everything else (``after``
loops, child widgets, the Toplevel host menu) works unchanged.

It connects read-only to the teleop bridge on ws://<host>:9090/monitor — independent of
the hardware-bridge DDS the other tabs use. Host/port via MABEL_HOST / MABEL_PORT /
MABEL_HOSTS / MABEL_TOKEN, exactly like the standalone tool. Shows "offline" until a
server is up.
"""

import os
import sys
import tkinter as tk

_SERVER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "server"))


class _FrameRoot(tk.Frame):
    """A Frame that tolerates the top-level-window calls ``MonitorApp`` makes on its
    ``root``, so the monitor embeds unchanged. Everything else (after/bind/winfo/configure)
    a Frame already supports."""

    def title(self, *a, **k):      pass
    def geometry(self, *a, **k):   pass
    def minsize(self, *a, **k):    pass
    def maxsize(self, *a, **k):    pass
    def resizable(self, *a, **k):  pass
    def protocol(self, *a, **k):   pass
    def wm_title(self, *a, **k):   pass
    def iconphoto(self, *a, **k):  pass


def _hosts_from_env():
    host = os.getenv("MABEL_HOST", "localhost")
    port = int(os.getenv("MABEL_PORT", "9090"))
    hosts = [(host, port)]
    for h in (os.getenv("MABEL_HOSTS", "") or "").split(","):
        h = h.strip()
        if not h:
            continue
        a, _, p = h.rpartition(":")
        hosts.append((a, int(p)) if a and p.isdigit() else (h, 9090))
    seen, uniq = set(), []
    for hp in hosts:
        k = f"{hp[0]}:{hp[1]}"
        if k not in seen:
            seen.add(k)
            uniq.append(hp)
    return uniq


class MonitorPanel(tk.Frame):
    def __init__(self, parent, link=None):
        super().__init__(parent, bg="#000000")
        if _SERVER_DIR not in sys.path:
            sys.path.append(_SERVER_DIR)        # append: never shadow bridge modules
        try:
            import monitor_gui as MG
        except Exception as exc:
            tk.Label(self, text=f"Command Monitor unavailable:\n{exc}\n\n"
                                f"(expected server/ at {_SERVER_DIR})",
                     bg="#000000", fg="#FF453A", justify="left",
                     font=("Helvetica", 12)).pack(padx=40, pady=60, anchor="w")
            return
        self.configure(bg=MG.S.BG)
        root = _FrameRoot(self, bg=MG.S.BG)
        root.pack(fill="both", expand=True)
        self._app = MG.MonitorApp(
            root, _hosts_from_env(),
            os.getenv("MABEL_MONITOR_PATH", "/monitor"),
            os.getenv("MABEL_TOKEN", ""),
            discover=os.getenv("MABEL_MONITOR_DISCOVER", "1") != "0")

    def refresh(self, snapshot=None):
        pass        # the embedded MonitorApp drives its own after() loops
