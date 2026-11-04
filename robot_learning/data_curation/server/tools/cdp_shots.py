#!/usr/bin/env python3
"""Capture screenshots of the new UI: library curation marks, the export
inspection-gate modal, and a snapped-panel layout. Writes PNGs to OUT."""
import base64, json, os, subprocess, sys, time, urllib.request
import websocket

PORT = int(os.environ.get("MABEL_STUDIO_PORT", "8093"))
CDP_PORT = 9224
URL = f"http://127.0.0.1:{PORT}/?demo=1"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCRATCH = "/tmp/scratch"
OUT = SCRATCH


def _http_json(path): return json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}{path}", timeout=5))


class C:
    def __init__(self, ws): self.ws = websocket.create_connection(ws, timeout=40, max_size=None); self.i = 0
    def cmd(self, m, **p):
        self.i += 1; self.ws.send(json.dumps({"id": self.i, "method": m, "params": p}))
        while True:
            x = json.loads(self.ws.recv())
            if x.get("id") == self.i: return x
    def ev(self, e, awaitp=False):
        r = self.cmd("Runtime.evaluate", expression=e, returnByValue=True, awaitPromise=awaitp, userGesture=True)
        return r.get("result", {}).get("result", {}).get("value")
    def shot(self, path):
        r = self.cmd("Page.captureScreenshot", format="png")
        open(path, "wb").write(base64.b64decode(r["result"]["data"])); print("wrote", path)


def main():
    subprocess.run(["pkill", "-f", f"remote-debugging-port={CDP_PORT}"], capture_output=True); time.sleep(0.5)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
        "--remote-allow-origins=*", f"--user-data-dir={SCRATCH}/chrome-shot", "--no-first-run",
        "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
        "--hide-scrollbars", "--force-device-scale-factor=1.5", "--window-size=1900,1000", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        target = None
        for _ in range(40):
            try:
                for t in _http_json("/json/list"):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"): target = t; break
                if target: break
            except Exception: pass
            time.sleep(0.5)
        c = C(target["webSocketDebuggerUrl"]); c.cmd("Page.enable"); c.cmd("Runtime.enable")
        for _ in range(60):
            if (c.ev("typeof window.__studioReady==='function'?window.__studioReady():''") or "").find("STUDIO_READY") >= 0: break
            time.sleep(0.5)
        time.sleep(2.0)
        # give the first dataset a realistic mix: some saved, one opened
        c.ev("""(()=>{const cs=[...document.querySelectorAll('.ep-card')];
          const S=window.__studio.status;
          const g=(n)=>cs.filter(c=>c.dataset.ds===n);
          const my=g('my_demo'), ss=g('studio_sample');
          if(my[0])S.markSaved(my[0].dataset.ds,my[0].dataset.ep);
          if(my[1])S.markSaved(my[1].dataset.ds,my[1].dataset.ep);
          if(my[2])S.markOpened(my[2].dataset.ds,my[2].dataset.ep);
          if(ss[0])S.markSaved(ss[0].dataset.ds,ss[0].dataset.ep);
          if(ss[1])S.markOpened(ss[1].dataset.ds,ss[1].dataset.ep);})()""")
        time.sleep(0.5)
        c.shot(f"{OUT}/shot_library_marks.png")

        # snapped layout: dock a couple panels so the grid is visible
        c.ev("""(()=>{const p=window.__studio.panels; p.clearAll(); const ws=p.wsRect();
          const Z=p; const l=p._zoneAt(ws.x+8,ws.y+ws.h/2); if(l)p.addAt('signals',l);
          const b=p._zoneAt(ws.x+ws.w/2,ws.y+ws.h-8); if(b)p.addAt('jointcompare',b);})()""")
        time.sleep(1.2); c.shot(f"{OUT}/shot_snap_layout.png")

        # export inspection-gate modal
        c.ev("""window.__studio.confirmModal({title:'⚠ Dataset not fully inspected',
          body:'Some source data hasn\\'t been reviewed & saved before export:'+
          '<ul class=\"modal-list\"><li><b>studio_sample</b> — only <b>1/5</b> episodes saved (1 opened, not saved)</li>'+
          '<li><b>stress_sim</b> — only <b>0/4</b> episodes saved (4 never inspected)</li></ul>'+
          'Exporting now bakes un-inspected episodes into the LeRobot dataset.',
          okLabel:'Export anyway',cancelLabel:'Go back to process',danger:true})""")
        time.sleep(0.5); c.shot(f"{OUT}/shot_export_modal.png")
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()


if __name__ == "__main__": main()
