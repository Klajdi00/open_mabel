#!/usr/bin/env python3
"""exp_head_stereo_sync.py — are the two halves of `head_stereo` the same instant?

The headset's stereo source is ONE side-by-side frame (left|right) that the sim
camera server renders from ONE physics snapshot (the zed_left worker renders both
eyes in one pass). This drives the REAL `sim_camera_server.py` from a synthetic
state file in which the robot's neck sweeps fast (no bridge needed — so it can
run beside anything else on :9090), pulls `head_stereo` over MJPEG, and measures
per frame the horizontal shift between the halves by phase correlation:

  SYNCED  (left_i, right_i)    — the shift is pure stereo disparity: small, steady
  SKEWED  (left_i, right_i+1)  — what pairing two independent eye streams a frame
                                 apart produces: the neck motion leaks in as shift

    python3 server/tools/exp_head_stereo_sync.py [--seconds 12] [--fig out.pdf]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")


def writer(path, model, stop, amp=0.6, hz=0.5):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q0 = data.qpos.copy()
    yaw = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw")] \
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw") >= 0 else None
    if yaw is None:   # fall back to the actuator's joint
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Neck Yaw")
        yaw = model.jnt_qposadr[model.actuator_trnid[a, 0]]
    mm = np.memmap(path, dtype=np.float64, mode="r+", shape=(1 + model.nq,))
    t0 = time.time()
    while not stop.is_set():
        t = time.time() - t0
        q = q0.copy()
        q[yaw] = amp * np.sin(2 * np.pi * hz * t)
        mm[1:] = q
        mm[0] = t                       # the server re-renders when t changes
        time.sleep(0.005)


def mjpeg_frames(url, n, timeout=60):
    out, buf = [], b""
    deadline = time.time() + timeout
    with urllib.request.urlopen(url, timeout=10) as r:
        while len(out) < n and time.time() < deadline:
            buf += r.read(65536)
            while True:
                a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
                if a < 0 or b < 0:
                    break
                img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_GRAYSCALE)
                buf = buf[b + 2:]
                if img is not None:
                    out.append(img)
    return out


def shift(a, b):
    """Horizontal shift (px) between two equal-size images (phase correlation)."""
    (dx, _dy), _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    return dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--port", type=int, default=8280)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    state = os.path.join(os.environ.get("TMPDIR", "/tmp"), "mabel_stereo_sync_state.bin")
    np.zeros(1 + model.nq).astype(np.float64).tofile(state)
    stop = threading.Event()
    th = threading.Thread(target=writer, args=(state, model, stop), daemon=True)
    th.start()
    env = dict(os.environ, MABEL_DATA_STREAMS="0")          # no 96xx planes for this check
    srv = subprocess.Popen([sys.executable, os.path.join(REPO, "server", "sim_camera_server.py"),
                            "--state-file", state, "--scene", SCENE,
                            "--http-port", str(a.port), "--udp-video-port", "0", "--no-h264",
                            "--head-only"],
                           cwd=os.path.join(REPO, "server"), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{a.port}/camera/head_stereo/stream.mjpg"
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/", timeout=1).close()
                break
            except Exception:
                time.sleep(0.5)
        frames = mjpeg_frames(url, a.frames)
    finally:
        srv.terminate(); stop.set()
        try:
            srv.wait(5)
        except Exception:
            srv.kill()
    if len(frames) < 10:
        print(f"FAIL: only {len(frames)} head_stereo frames"); return 1
    h, w = frames[0].shape
    half = w // 2
    L = [f[:, :half] for f in frames]
    R = [f[:, half:2 * half] for f in frames]
    synced = np.array([shift(l, r) for l, r in zip(L, R)])
    skewed = np.array([shift(L[i], R[i + 1]) for i in range(len(L) - 1)])
    inter = np.array([shift(L[i], L[i + 1]) for i in range(len(L) - 1)])
    print(f"{len(frames)} frames, {w}x{h} (each eye {half}x{h})")
    print(f"frame-to-frame motion (left eye):     |shift| p50 {np.median(np.abs(inter)):6.2f} px  p90 {np.percentile(np.abs(inter), 90):6.2f} px")
    print(f"SYNCED  L_i vs R_i   (disparity only): mean {synced.mean():6.2f} px  std {synced.std():5.2f} px")
    print(f"SKEWED  L_i vs R_i+1 (one frame off):  mean {skewed.mean():6.2f} px  std {skewed.std():5.2f} px")
    ok = (w == 2 * half and synced.std() < 0.25 * max(skewed.std(), 1e-6) + 1.0
          and np.median(np.abs(inter)) > 2.0)
    print("VERDICT:", "halves are the same instant (L/R shift is steady disparity while the view sweeps)"
          if ok else "NOT synced or no motion — inspect")
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(synced, "o-", ms=3, label="synced: left$_i$ vs right$_i$")
        ax.plot(skewed, "s-", ms=3, label="one frame apart: left$_i$ vs right$_{i+1}$")
        ax.set_xlabel("frame"); ax.set_ylabel("L→R horizontal shift (px)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.fig); print("wrote", a.fig)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
