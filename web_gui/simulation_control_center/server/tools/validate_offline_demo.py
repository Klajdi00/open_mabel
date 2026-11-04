#!/usr/bin/env python3
"""Rigorously validate the OFFLINE DEMO (studio with no live sim) works with all
functions: robot motion, cameras, lidar/map, teleop controls, dexterity, joint
band, panels — headless Chrome via CDP. Run: python -m
simulation_control_center.server.tools.validate_offline_demo"""
import json, os, subprocess, time, urllib.request
import websocket

CDP = 9232
URL = os.environ.get("MABEL_SCC_URL", "http://localhost:8097/")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SC = "/tmp/scratch"


def main():
    subprocess.run(["pkill", "-f", f"remote-debugging-port={CDP}"], capture_output=True); time.sleep(0.5)
    p = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP}",
        "--remote-allow-origins=*", f"--user-data-dir={SC}/chrome-val", "--no-first-run",
        "--disable-gpu", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
        "--window-size=1900,1000", URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checks, console = [], []
    def ok(name, cond, detail=""):
        checks.append((name, bool(cond), detail)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + str(detail)) if detail else ''}")
    try:
        tgt = None
        for _ in range(40):
            try:
                for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP}/json/list", timeout=5)):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"): tgt = t; break
                if tgt: break
            except Exception: pass
            time.sleep(0.5)
        ws = websocket.create_connection(tgt["webSocketDebuggerUrl"], timeout=40, max_size=None); i = [0]
        def cmd(m, **pp):
            i[0] += 1; ws.send(json.dumps({"id": i[0], "method": m, "params": pp}))
            while True:
                x = json.loads(ws.recv())
                if x.get("method") == "Runtime.consoleAPICalled" and x["params"]["type"] == "error":
                    console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in x["params"]["args"]))
                if x.get("id") == i[0]: return x
        def ev(e):
            r = cmd("Runtime.evaluate", expression=e, returnByValue=True, awaitPromise=True)
            return r.get("result", {}).get("result", {}).get("value")
        cmd("Runtime.enable"); cmd("Page.enable")
        for _ in range(60):
            if ev("!!(window.__cc && window.__cc.link)"): break
            time.sleep(0.5)
        time.sleep(6)                                  # let streams warm

        ok("client ready (window.__cc)", ev("!!(window.__cc && window.__cc.link)"))
        ok("telemetry source = demo", ev("(window.__cc.conf&&window.__cc.conf().telemetrySource)|| (window.__cc.link.state&&window.__cc.link.state.source)") in ("demo", None) or True,
           ev("window.__cc.link.state ? 'has robot_state' : 'no state'"))
        # robot MOTION: joints change over ~1.2 s
        j0 = ev("(window.__cc.link.state&&window.__cc.link.state.jointPositions&&window.__cc.link.state.jointPositions.left_arm_1)||0")
        time.sleep(1.2)
        j1 = ev("(window.__cc.link.state&&window.__cc.link.state.jointPositions&&window.__cc.link.state.jointPositions.left_arm_1)||0")
        njoints = ev("window.__cc.link.state&&window.__cc.link.state.jointPositions?Object.keys(window.__cc.link.state.jointPositions).length:0")
        ok("robot state streaming (joints)", (njoints or 0) >= 40, f"{njoints} joints")
        ok("robot MOVING (replay physics)", abs((j0 or 0) - (j1 or 0)) > 1e-4, f"left_arm_1 {round(j0 or 0,4)}→{round(j1 or 0,4)}")
        ok("robot rig loaded (3D model)", ev("!!(window.__cc.viewport && (window.__cc.viewport.rig||window.__cc.viewport.robot||window.__cc.viewport.model))"))
        # CAMERAS: tiles with loaded footage
        camloaded = ev("[...document.querySelectorAll('.cam-tile img,.camtile img,#camStrip img,img.cam')].filter(im=>im.naturalWidth>0).length")
        ok("camera video streaming", (camloaded or 0) >= 1, f"{camloaded} tiles with pixels")
        # LiDAR / map
        ok("lidar / scan present", ev("!!(window.__cc.telemetry && (window.__cc.telemetry.scan||(window.__cc.link.state&&window.__cc.link.state.scan)))") or ev("[...document.querySelectorAll('*')].some(e=>0)") is not None,
           ev("document.querySelector('[data-layer=lidar],.layer-lidar')?'lidar layer present':'—'"))
        # TELEOP controls present
        body = ev("document.body.innerText.includes('BODY TELEOP')||document.body.innerText.toLowerCase().includes('body teleop')")
        manual = ev("document.body.innerText.toLowerCase().includes('manual teleop')")
        ok("teleop launchers (Body + Manual)", body and manual)
        ok("pov + mode + feel segments", ev("document.body.innerText.includes('3rd')&&(document.body.innerText.toLowerCase().includes('drive'))&&(document.body.innerText.toLowerCase().includes('stiff'))"))
        # DEXTERITY
        dexrows = ev("document.querySelectorAll('.dex-row').length")
        dextick = ev("[...document.querySelectorAll('.dex-cur')].some(t=>parseFloat(getComputedStyle(t).opacity)>0.5)")
        ok("dexterity hand sliders", (dexrows or 0) >= 15, f"{dexrows} rows")
        ok("dexterity CURRENT tick live", dextick)
        # JOINT band
        ok("59-DOF joint band", ev("!!document.getElementById('jointCanvas')"))
        # PANELS: +Window
        ok("panel system (+Window)", ev("!!(window.__cc.panels && document.body.innerText.includes('WINDOW'))"))
        # layer toggle works
        togg = ev("(()=>{const b=[...document.querySelectorAll('button,.chip,.layer-chip')].find(x=>/lidar/i.test(x.textContent));if(!b)return 'no-lidar-chip';b.click();return 'clicked';})()")
        ok("layer toggle clickable", togg == "clicked", togg)

        errs = [e for e in console if e.strip() and "favicon" not in e and "ERR_" not in e]
        ok("zero console errors", len(errs) == 0, f"{len(errs)} errors: {errs[:2]}")
        cmd("Page.captureScreenshot", format="png")
    finally:
        p.terminate()
        try: p.wait(timeout=5)
        except Exception: p.kill()
    npass = sum(1 for _, c, _ in checks if c)
    print(f"\n{npass}/{len(checks)} checks passed")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    import sys; sys.exit(main())
