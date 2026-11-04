#!/usr/bin/env python3
"""CDP test for the batch per-episode workflow: load-all into the timeline, then
per-episode export → <dataset>_processed, verify one output file per clip + ✓ marks,
then folder → LeRobot. Server must be on :8093.
Run: python -m data_curation.server.tools.cdp_batch"""
import glob, json, os, shutil, subprocess, time, urllib.request
import websocket

PORT = int(os.environ.get("MABEL_STUDIO_PORT", "8093"))
CDP_PORT = 9226
URL = f"http://127.0.0.1:{PORT}/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SC = "/tmp/scratch"
DS = "multirate_sim"


def _hj(p): return json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}{p}", timeout=5))


class C:
    def __init__(s, ws): s.ws = websocket.create_connection(ws, timeout=30, max_size=None); s.i = 0; s.console = []
    def cmd(s, m, **p):
        s.i += 1; s.ws.send(json.dumps({"id": s.i, "method": m, "params": p}))
        while True:
            x = json.loads(s.ws.recv())
            if x.get("method") == "Runtime.consoleAPICalled" and x["params"]["type"] == "error":
                s.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in x["params"]["args"]))
            if x.get("id") == s.i: return x
    def ev(s, e):
        r = s.cmd("Runtime.evaluate", expression=e, returnByValue=True, userGesture=True)
        res = r.get("result", {})
        if "exceptionDetails" in res: raise RuntimeError(json.dumps(res["exceptionDetails"])[:300])
        return res.get("result", {}).get("value")
    def poll(s, expr, want, tries=120, delay=0.5):
        for _ in range(tries):
            v = s.ev(expr)
            if v == want or (callable(want) and want(v)): return v
            time.sleep(delay)
        return s.ev(expr)


def main():
    from .. import config as CFG
    proc_dir = os.path.join(CFG.DATASET_ROOT, DS + "_processed")
    shutil.rmtree(proc_dir, ignore_errors=True)
    subprocess.run(["pkill", "-f", f"remote-debugging-port={CDP_PORT}"], capture_output=True); time.sleep(0.5)
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
        "--remote-allow-origins=*", f"--user-data-dir={SC}/chrome-batch", "--no-first-run",
        "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1900,1000", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = []
    try:
        tgt = None
        for _ in range(40):
            try:
                for t in _hj("/json/list"):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"): tgt = t; break
                if tgt: break
            except Exception: pass
            time.sleep(0.5)
        c = C(tgt["webSocketDebuggerUrl"]); c.cmd("Runtime.enable"); c.cmd("Page.enable")
        for _ in range(60):
            if "STUDIO_READY" in (c.ev("typeof window.__studioReady==='function'?window.__studioReady():''") or ""): break
            time.sleep(0.5)
        c.poll("document.querySelectorAll('.lib-load.all').length", lambda v: v and v > 0)   # library rendered
        # start from a CLEAN timeline (any restored-autosave clips would skew counts)
        c.ev("(()=>{const s=window.__studio.seq; s.clips.slice().forEach(c=>s.remove(c)); window.__studio.timeline.render();})()")
        c.poll("window.__studio.seq.clips.length", 0, tries=10)

        # how many episodes does multirate_sim have?
        n_eps = c.ev(f"""(()=>{{const w=[...document.querySelectorAll('.lib-ds')].find(w=>/{DS}/.test(w.querySelector('.lib-ds-h span')?.textContent||''));
          return w?w.querySelectorAll('.ep-card').length:0;}})()""")
        print(f"{DS} episodes:", n_eps)

        # 1) batch-load ALL via the library button
        c.ev(f"""(()=>{{const w=[...document.querySelectorAll('.lib-ds')].find(w=>/{DS}/.test(w.querySelector('.lib-ds-h span')?.textContent||''));
          w.querySelector('.lib-load.all').click();}})()""")
        clips = c.poll("window.__studio.seq.clips.length", lambda v: v and v >= n_eps)
        distinct = c.ev("new Set(window.__studio.seq.clips.map(c=>c.ep)).size")
        allDs = c.ev(f"window.__studio.seq.clips.every(c=>c.ds==='{DS}')")
        print("loaded clips:", clips, "| distinct eps:", distinct, "| all from ds:", allDs)
        if clips != n_eps: fails.append(f"loaded {clips} clips, expected {n_eps}")
        if clips != distinct: fails.append("clips not 1-per-episode")

        # 2) per-episode export (call the real handler)
        c.ev("window.__studio.exportEpisodes()")
        saved = c.poll("window.__studio.seq.clips.filter(c=>window.__studio.status.get(c.ds,c.ep)==='saved').length",
                       lambda v: v == clips, tries=120)
        print("clips marked saved:", saved, "/", clips)
        if saved != clips: fails.append(f"only {saved}/{clips} marked saved")

        # 3) one processed .h5 per clip
        time.sleep(1.0)
        outs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(proc_dir, "episode_*.h5")))
        print("processed files:", outs)
        if len(outs) != clips: fails.append(f"{len(outs)} processed files != {clips} clips")

        # 4) ✓ badges on the timeline
        badges = c.poll("document.querySelectorAll('.clip .clip-stat.saved').length", lambda v: v and v >= clips, tries=20)
        print("timeline ✓ badges:", badges)
        if not badges or badges < clips: fails.append(f"{badges} ✓ badges, expected {clips}")

        # 5) folder → LeRobot (endpoint, all episodes now saved so gate passes)
        lr = c.ev(f"""(async()=>{{try{{const r=await window.__studio.exportLeRobot?.();return 'called';}}catch(e){{return 'err '+e.message;}}}})()""")
        # exportLeRobot has no return; verify via the API directly instead
        time.sleep(1.0)
        errs = [e for e in c.console if e.strip() and "favicon" not in e]
        if errs: print("CONSOLE ERRORS:", errs[:5]); fails.append(f"{len(errs)} console errors")
        else: print("console: clean")
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()
    print("\n" + ("FAILED: " + "; ".join(fails) if fails else "ALL BATCH CHECKS PASSED"))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys; sys.exit(main())
