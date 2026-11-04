#!/usr/bin/env python3
"""
Headless-Chrome CDP smoke test for the Data-Curation web app. Launches Chrome,
loads the app, captures console errors, and drives the new library-status marks +
snap-grid geometry. Run:  python -m data_curation.server.tools.cdp_smoke
Assumes the server is already up on $MABEL_STUDIO_PORT (default 8093).
"""
import json, os, subprocess, sys, time, urllib.request
import websocket   # websocket-client

PORT = int(os.environ.get("MABEL_STUDIO_PORT", "8093"))
CDP_PORT = 9223
URL = f"http://127.0.0.1:{PORT}/?demo=1"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE = "/tmp/scratch"


def _http_json(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}{path}", timeout=5))


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30, max_size=None)
        self._id = 0
        self.console = []

    def send(self, method, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.consoleAPICalled":
                a = msg["params"]
                if a["type"] in ("error", "warning"):
                    self.console.append((a["type"], " ".join(str(x.get("value", x.get("description", ""))) for x in a["args"])))
            elif msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                self.console.append(("exception", d.get("exception", {}).get("description", d.get("text", "?"))))
            if msg.get("id") == self._id:
                return msg

    def evaluate(self, expr, awaitp=False):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=awaitp, userGesture=True)
        res = r.get("result", {})
        if "exceptionDetails" in res:
            raise RuntimeError("JS exception: " + json.dumps(res["exceptionDetails"])[:400])
        return res.get("result", {}).get("value")


def main():
    subprocess.run(["pkill", "-f", "remote-debugging-port=%d" % CDP_PORT], capture_output=True)
    time.sleep(0.5)
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
         "--remote-allow-origins=*",
         f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
         "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
         "--window-size=1900,1000", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = []
    try:
        target = None
        for _ in range(40):
            try:
                for t in _http_json("/json/list"):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        target = t; break
                if target: break
            except Exception:
                pass
            time.sleep(0.5)
        if not target:
            print("FAIL: no Chrome page target"); return 1
        c = CDP(target["webSocketDebuggerUrl"])
        c.send("Runtime.enable"); c.send("Page.enable"); c.send("Log.enable")

        # wait for the app to finish booting
        ready = None
        for _ in range(60):
            try:
                ready = c.evaluate("typeof window.__studioReady==='function' ? window.__studioReady() : null")
            except Exception:
                ready = None
            if ready and "STUDIO_READY" in ready:
                break
            time.sleep(0.5)
        print("ready:", ready)
        if not ready or "STUDIO_READY" not in ready:
            fails.append("app never became ready")

        # give the library a moment, then assert cards + marks scaffolding exist
        time.sleep(1.5)
        ncards = c.evaluate("document.querySelectorAll('.ep-card').length")
        nmarks = c.evaluate("document.querySelectorAll('.ep-card .ep-mark').length")
        nprog = c.evaluate("document.querySelectorAll('.lib-ds-prog').length")
        print(f"library: {ncards} cards, {nmarks} marks, {nprog} progress badges")
        if not ncards: fails.append("no episode cards rendered")
        if ncards and nmarks != ncards: fails.append("ep-mark missing on some cards")
        if not nprog: fails.append("no dataset progress badges")

        # drive status: mark the first card's episode SAVED, second OPENED
        info = c.evaluate("""(()=>{const cs=[...document.querySelectorAll('.ep-card')];
          const a=cs[0], b=cs[1]||cs[0];
          window.__studio.status.markSaved(a.dataset.ds,a.dataset.ep);
          window.__studio.status.markOpened(b.dataset.ds,b.dataset.ep);
          return {a:{ds:a.dataset.ds,ep:a.dataset.ep},b:{ds:b.dataset.ds,ep:b.dataset.ep}};})()""")
        time.sleep(0.4)
        saved_mark = c.evaluate(f"""(()=>{{const c=document.querySelector('.ep-card[data-ds="{info['a']['ds']}"][data-ep="{info['a']['ep']}"]');
          return c && c.classList.contains('is-saved') && c.querySelector('.ep-mark').textContent;}})()""")
        opened_mark = c.evaluate(f"""(()=>{{const c=document.querySelector('.ep-card[data-ds="{info['b']['ds']}"][data-ep="{info['b']['ep']}"]');
          return c && (c.classList.contains('is-opened')||c.classList.contains('is-saved')) && c.querySelector('.ep-mark').textContent;}})()""")
        prog_done = c.evaluate("""[...document.querySelectorAll('.lib-ds-prog')].map(b=>b.textContent)""")
        print("saved mark:", repr(saved_mark), "| opened mark:", repr(opened_mark), "| progress:", prog_done)
        if saved_mark != "✓": fails.append(f"saved card not marked ✓ (got {saved_mark!r})")
        if opened_mark not in ("◐", "✓"): fails.append(f"opened card not marked (got {opened_mark!r})")

        # snap geometry through the live PanelManager. Clear docked panels first so
        # the side columns are free (otherwise the left edge is correctly rejected —
        # it's over the docked overview panel). Then: left edge → a left column;
        # dead center → null; over a snapped tile → null (never overlap).
        snap = c.evaluate("""(()=>{const p=window.__studio.panels; p.clearAll();
          const ws=p.wsRect();
          const edge=p._zoneAt(ws.x+8, ws.y+ws.h/2);
          const center=p._zoneAt(ws.x+ws.w/2, ws.y+ws.h/2);
          // now occupy the left column and confirm the bottom band shrinks IN
          let bottom=null, overLeft=null;
          if (edge){ p.addAt('overview', edge);
            bottom=p._zoneAt(ws.x+ws.w/2, ws.y+ws.h-8);
            overLeft=p._zoneAt(ws.x+8, ws.y+ws.h/2); }
          const R=(z)=>z?{x:z.x|0,y:z.y|0,w:z.w|0,h:z.h|0}:null;
          return {edge:R(edge), center, bottom:R(bottom), overLeft, wsx:ws.x|0, wsw:ws.w|0};})()""")
        print("snap:", snap)
        if not snap or not snap.get("edge"):
            fails.append("left-edge snap returned no zone")
        else:
            e = snap["edge"]
            if abs(e["x"] - snap["wsx"]) > 3: fails.append("left zone not at left edge")
            b = snap.get("bottom")
            if not b: fails.append("bottom band gone after occupying left column")
            elif b["x"] <= snap["wsx"] + 3:
                fails.append("bottom band did NOT shrink away from the occupied left column (rule 5)")
            if snap.get("overLeft") is not None:
                fails.append("snap allowed over the now-occupied left column (should be null)")
        if snap and snap.get("center") is not None:
            fails.append("center returned a snap zone (should be null)")

        # export inspection-gate MODAL: renders with two buttons; "Go back"
        # resolves false and tears down.
        modal = c.evaluate("""(()=>{const cm=window.__studio.confirmModal;
          const pr=cm({title:'t',body:'b',okLabel:'Export anyway',cancelLabel:'Go back to process',danger:true});
          const back=document.querySelector('.modal-back');
          const btns=back?[...back.querySelectorAll('.mbtn')].map(x=>x.textContent):[];
          const ok=!!back && back.querySelector('.mbtn.ok.danger');
          window.__modalP=pr; return {shown:!!back, btns, danger:!!ok};})()""")
        print("modal:", modal)
        if not modal or not modal.get("shown"): fails.append("export modal did not render")
        if modal and "Go back to process" not in (modal.get("btns") or []): fails.append("modal missing 'Go back' button")
        if modal and not modal.get("danger"): fails.append("export modal 'Export anyway' not styled danger")
        # click "Go back" → promise resolves false, modal removed
        resolved = c.evaluate("""(async()=>{document.querySelector('.modal-back .mbtn.cancel').click();
          const v=await window.__modalP; return {v, gone: !document.querySelector('.modal-back')};})()""", awaitp=True)
        print("modal resolve:", resolved)
        if not resolved or resolved.get("v") is not False: fails.append("'Go back' did not resolve false")
        if resolved and not resolved.get("gone"): fails.append("modal not removed after choice")

        # console errors (ignore benign favicon / net noise)
        errs = [f"{t}: {m}" for t, m in c.console
                if "favicon" not in m and "ERR_" not in m and m.strip()]
        if errs:
            print("CONSOLE ERRORS:"); [print("  •", e[:200]) for e in errs]
            fails.append(f"{len(errs)} console error(s)")
        else:
            print("console: clean")
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()

    if fails:
        print("\nFAILED:"); [print("  ✗", f) for f in fails]; return 1
    print("\nALL CDP CHECKS PASSED"); return 0


if __name__ == "__main__":
    sys.exit(main())
