#!/usr/bin/env python3
"""
exp_camera_contention.py — does the camera server's scheduling priority govern
the bridge control-loop rate (and therefore joint-stream lag)?

The physics loop and the camera renderer already run in SEPARATE PROCESSES on
independent cores (server.pdf, "Decoupling physics from rendering"). This harness
shows that independence is NOT sufficient on macOS: at default scheduling priority
the camera box's per-camera GPU-readback threads still contend with the bridge's
real-time control loop and drag its iteration rate (how often a fresh operator
frame is consumed and physics catches up to real time) well below its idle ceiling
— which is what makes a client's joint stream look laggy and out of sync. Lowering
the camera server to `utility` QoS (`taskpolicy -c utility`) lets the scheduler
favour the latency-critical bridge while the cameras keep the performance cores, so
the loop recovers at no video-fps cost.

Method: launch the REAL sim_teleop_bridge.py (patched, sensors-off) headless, then
for each camera-priority config (re)launch sim_camera_server.py, attach 2 synthetic
wrist-teleop clients (WBC impedance mode) and 5 MJPEG pullers (forcing every render
thread to run), and record over a 12 s window:
  * bridge control-loop Hz   (parsed from the bridge's own 2 s status line)
  * delivered video fps      (summed across the 5 pulled streams)
  * server-side state age ms (robot_state.stateAgeMs p50/p95)

Outputs: a table to stdout + figures/camera_contention.pdf (bridge Hz and video
fps per config). Reproducible: fixed scenarios, no hand-entered numbers.

Run:  python3 server/tools/exp_camera_contention.py            # measure + plot
      python3 server/tools/exp_camera_contention.py --plot-only # replot last data
"""
from __future__ import annotations
import asyncio, json, os, re, signal, subprocess, sys, threading, time, statistics as st
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.normpath(os.path.join(_HERE, ".."))
ROOT = os.path.normpath(os.path.join(SERVER, ".."))
FIGS = os.path.join(SERVER, "docs", "figures")
PY = sys.executable
SCENE = os.path.join(ROOT, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
WS = "ws://127.0.0.1:9090/teleop"
BLOG = os.path.join("/tmp", "exp_cc_bridge.log")
CLOG = os.path.join("/tmp", "exp_cc_cam.log")
DATA = os.path.join(FIGS, "camera_contention.json")
HZ_RE = re.compile(r"│\s*(\d+)\s*Hz")
IDENT = [1.0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
CAM_PATHS = ["head_left","head_right","main","wrist_left","wrist_right"]

# (label, launcher-prefix, extra camera flags)
CONFIGS = [
    ("cameras idle\n(no video pulled)", [], [], False),
    ("default priority\n(+video) — old",   [], [], True),
    ("utility QoS\n(+video) — fix",        ["/usr/sbin/taskpolicy","-c","utility"], [], True),
    ("--max-fps 12\n(+video)",             [], ["--max-fps","12"], True),
]


def hand_frame(seq):
    joints=[{"joint":nm,"localTransform":{"matrix":IDENT},"isTracked":True}
            for nm in ("wrist","thumbKnuckle","indexFingerKnuckle","middleFingerKnuckle")]
    g=(seq%60)/60.0
    anchor=[1.0,0,0,0.1*g, 0,1,0,0, 0,0,1,0.1, 0,0,0,1]
    return {"type":"teleop_frame","payload":{"sequence":seq,"timestamp":time.time(),
        "head":{"transform":{"matrix":IDENT},"trackingState":"normal"},
        "navJoystick":{"lx":0,"ly":0,"rx":0,"ry":0},
        "rightHand":{"chirality":"right","anchorTransform":{"matrix":anchor},
                     "joints":joints,"isTracked":True}}}


class WristClient:
    def __init__(s): s.recv=[]; s.ages=[]; s.stop=False; s.task=None
    async def run(s):
        import websockets
        async with websockets.connect(WS, max_size=2**20) as ws:
            await ws.recv()  # hello
            await ws.send(json.dumps({"type":"control_mode","payload":{
                "method":"wrist","controlType":"impedance","region":"arm","stiffness":1.0}}))
            await ws.send(json.dumps({"type":"recalibrate","payload":{}}))
            async def rdr():
                try:
                    async for raw in ws:
                        m=json.loads(raw)
                        if m.get("type")=="robot_state":
                            s.recv.append(time.time())
                            a=m["payload"].get("stateAgeMs")
                            if a is not None: s.ages.append(a)
                except Exception: pass
            rt=asyncio.create_task(rdr()); seq=0
            while not s.stop:
                seq+=1
                await ws.send(json.dumps(hand_frame(seq)))
                await asyncio.sleep(1/30)
            rt.cancel()


class VideoPuller(threading.Thread):
    def __init__(s, path): super().__init__(daemon=True); s.path=path; s.stop=False; s.frames=0
    def run(s):
        url=f"http://127.0.0.1:8080/camera/{s.path}/stream.mjpg"
        try:
            r=urllib.request.urlopen(url, timeout=5); buf=b""
            while not s.stop:
                chunk=r.read(4096)
                if not chunk: break
                buf+=chunk
                while b"\xff\xd9" in buf:
                    i=buf.index(b"\xff\xd9"); buf=buf[i+2:]; s.frames+=1
        except Exception: pass


def parse_hz(t0, t1):
    out=[]
    try: lines=open(BLOG).readlines()
    except FileNotFoundError: return out
    for ln in lines:
        m=HZ_RE.search(ln)
        if not m: continue
        tm=re.match(r"(\d\d):(\d\d):(\d\d)\.(\d)", ln)
        if not tm: continue
        h,mi,sc,d=map(int, tm.groups()); sod=h*3600+mi*60+sc+d/10.0
        if t0<=sod<=t1: out.append(int(m.group(1)))
    return out


def sod_now():
    lt=time.localtime(); return lt.tm_hour*3600+lt.tm_min*60+lt.tm_sec+(time.time()%1)


def launch_cam(prefix, extra):
    cl=open(CLOG,"w")
    env=dict(os.environ); env["MABEL_DISPATCH_TARGET"]="sim:mujoco"; env.pop("MABEL_TOKEN", None)
    cmd=list(prefix)+[PY,"sim_camera_server.py","--scene",SCENE,"--jpeg-quality","85",
                      "--udp-video-port","9091","--stereo"]+list(extra)
    p=subprocess.Popen(cmd, cwd=SERVER, env=env, stdout=cl, stderr=subprocess.STDOUT)
    for _ in range(200):
        time.sleep(0.25)
        try:
            if "render plan" in open(CLOG).read(): break
        except FileNotFoundError: pass
    return p


async def measure():
    env=dict(os.environ); env["MABEL_DISPATCH_TARGET"]="sim:mujoco"; env.pop("MABEL_TOKEN", None)
    bl=open(BLOG,"w")
    bridge=subprocess.Popen([PY,"sim_teleop_bridge.py","--scene",SCENE],
        cwd=SERVER, env=env, stdout=bl, stderr=subprocess.STDOUT)
    for _ in range(200):
        await asyncio.sleep(0.25)
        try:
            if "teleop up" in open(BLOG).read(): break
        except FileNotFoundError: pass
    else:
        bridge.kill(); raise SystemExit("bridge failed to start (see %s)" % BLOG)
    await asyncio.sleep(1.0)
    rows=[]
    print(f"{'config':28s}  loop Hz (med[min-max])     video fps   state age p50/p95 ms")
    print("-"*88)
    for label, prefix, extra, pull in CONFIGS:
        cam=launch_cam(prefix, extra)
        clients=[WristClient(), WristClient()]
        for c in clients: c.task=asyncio.create_task(c.run())
        pullers=[VideoPuller(p) for p in CAM_PATHS] if pull else []
        for p in pullers: p.start()
        await asyncio.sleep(2.0)                       # settle + register video clients
        t0=sod_now(); w0=time.time()
        await asyncio.sleep(12.0)
        t1=sod_now(); w1=time.time()
        hz=parse_hz(t0,t1)
        ages=[a for c in clients for a in c.ages]
        fps=round(sum(p.frames for p in pullers)/(w1-w0),1) if pullers else 0.0
        for c in clients: c.stop=True
        for p in pullers: p.stop=True
        await asyncio.sleep(0.5)
        for c in clients:
            if c.task: c.task.cancel()
        cam.send_signal(signal.SIGINT); await asyncio.sleep(1.0); cam.kill()
        await asyncio.sleep(1.5)
        med=st.median(hz) if hz else 0; lo=min(hz) if hz else 0; hi=max(hz) if hz else 0
        a50=st.median(ages) if ages else 0
        a95=sorted(ages)[int(0.95*len(ages))] if ages else 0
        rows.append(dict(label=label, hz_med=med, hz_lo=lo, hz_hi=hi, fps=fps,
                         age50=round(a50,1), age95=round(a95,1)))
        flat=label.replace("\n"," ")
        print(f"{flat:28s}  {med:4.0f} [{lo:3.0f}-{hi:3.0f}]            {fps:5.1f}      "
              f"{a50:4.1f} / {a95:4.1f}")
    bridge.send_signal(signal.SIGINT); await asyncio.sleep(1.0); bridge.kill()
    os.makedirs(FIGS, exist_ok=True)
    json.dump(rows, open(DATA,"w"), indent=2)
    return rows


def plot(rows):
    labels=[r["label"] for r in rows]
    hz=[r["hz_med"] for r in rows]
    lo=[r["hz_med"]-r["hz_lo"] for r in rows]
    hi=[r["hz_hi"]-r["hz_med"] for r in rows]
    fps=[r["fps"] for r in rows]
    x=range(len(rows))
    fig, ax1=plt.subplots(figsize=(8.2,4.4))
    bars=ax1.bar([i-0.18 for i in x], hz, width=0.36, yerr=[lo,hi], capsize=4,
                 color="#2a6f97", label="bridge control-loop rate")
    ax1.set_ylabel("bridge control-loop rate (Hz)", color="#2a6f97")
    ax1.tick_params(axis="y", labelcolor="#2a6f97")
    ax1.axhline(50, ls="--", lw=1, color="#777")
    ax1.text(len(rows)-0.5, 54, "50 Hz state-stream rate", ha="right", fontsize=8, color="#555")
    for i,v in zip(x,hz):
        ax1.text(i-0.18, v+max(hz)*0.02, f"{v:.0f}", ha="center", fontsize=8, color="#2a6f97")
    ax2=ax1.twinx()
    ax2.bar([i+0.18 for i in x], fps, width=0.36, color="#e09f3e", label="delivered video fps")
    ax2.set_ylabel("delivered video fps (5 streams)", color="#e09f3e")
    ax2.tick_params(axis="y", labelcolor="#e09f3e")
    for i,v in zip(x,fps):
        ax2.text(i+0.18, v+max(max(fps),1)*0.02, f"{v:.0f}", ha="center", fontsize=8, color="#b5791f")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("Camera-server scheduling priority vs. bridge control-loop rate\n"
                  "(M1 Max, mabel_home, stereo+wrists, 2 WBC clients + 5 MJPEG pullers)",
                  fontsize=10)
    fig.tight_layout()
    out=os.path.join(FIGS,"camera_contention.pdf")
    fig.savefig(out); print("wrote", out)


def main():
    if "--plot-only" in sys.argv:
        rows=json.load(open(DATA))
    else:
        rows=asyncio.run(measure())
    plot(rows)


if __name__ == "__main__":
    main()
