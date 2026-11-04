#!/usr/bin/env python3
"""Scientifically measure the MABEL sensor-bridge render limits on THIS machine.

Run it before tuning the per-group rates (front_rate / aux_rate / lidar_rate) in
``sim.launch.py`` -- it reports what the GPU can actually sustain so the rates are
grounded in measurement, not guesswork.

    MUJOCO_GL=egl /usr/bin/python3 \
        ros/src/mabel_sim/scripts/bench_sensors.py [scene.xml]

NOTE: use the SYSTEM python (the one the ROS node uses). The anaconda python's
PyOpenGL has a broken EGL binding (``EGLDeviceEXT`` AttributeError).

What it measures (the same render path the node uses, via sensor_render + the
real camera sizes from sensor_config, so it can't drift from the node):
  1. per-resolution color/depth single-render time
  2. the FRONT worker's full pipeline (render+noise+cloud+serialize) -> max Hz
  3. the AUX worker's full per-tick render time -> max Hz
  4. a full serial tick of ALL cameras (the OLD architecture) for contrast
  5. LiDAR: vectorised scan vs the old per-beam Python loop
  6. multi-context GPU scaling (why one-thread-per-camera doesn't help)
and prints a recommended, sustainable rate per group.
"""
import os
import sys
import time
import threading

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

# import the package's sensor config/helpers directly (no ROS needed)
_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_PKG))
from mabel_sim import sensor_config as scfg          # noqa: E402
from mabel_sim import sensor_render as sr            # noqa: E402

SCENE = (sys.argv[1] if len(sys.argv) > 1 else
         os.path.expanduser("~/open_mabel/simulation/mabel_mujoco/scenes/mabel_home.xml"))


def timeit(fn, n=60):
    fn()                                   # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0    # ms


def main():
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}  mujoco={mujoco.__version__}")
    print(f"scene={SCENE}\n")
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vopt = sr.clean_render_option(3)
    rng = np.random.default_rng()

    front = [c for c in scfg.DEPTH_CAMERAS if c["ns"] == "front"]
    aux_depth = [c for c in scfg.DEPTH_CAMERAS if c["ns"] != "front"]
    mono = list(scfg.MONO_CAMERAS)
    ir = list(scfg.IR_CAMERAS)

    # ---- 1. per-size single render ---------------------------------------
    print("--- single render time per resolution ---")
    csizes = sorted({(w, h) for _, _, _, w, h in mono + ir}
                    | {(c["w"], c["h"]) for c in scfg.DEPTH_CAMERAS})
    rcache = {}
    for (w, h) in csizes:
        r = mujoco.Renderer(model, height=h, width=w)
        rcache[(w, h)] = r
        ms = timeit(lambda: sr.capture_rgb(r, data, 0, vopt))
        print(f"  color {w}x{h:<4}: {ms:6.2f} ms  ({1000/ms:6.1f} fps)")
    dcache = {}
    for c in scfg.DEPTH_CAMERAS:
        wh = (c["w"], c["h"])
        if wh in dcache:
            continue
        r = mujoco.Renderer(model, height=wh[1], width=wh[0])
        r.enable_depth_rendering()
        dcache[wh] = r
        ms = timeit(lambda: sr.capture_depth(r, data, 0, vopt))
        print(f"  depth {wh[0]}x{wh[1]:<4}: {ms:6.2f} ms  ({1000/ms:6.1f} fps)")

    def depth_unit(c):
        mj, w, h = c["mj"], c["w"], c["h"]
        fx, fy, cx, cy = sr.camera_intrinsics(model, mj, w, h)
        rgb = sr.capture_rgb(rcache[(w, h)], data, mj, vopt)
        d = sr.capture_depth(dcache[(w, h)], data, mj, vopt)
        d = sr.add_depth_noise(d, 0.001, 0.005, rng)
        d = sr.add_depth_outliers(d, 0.03, 0.005, 0.3, scfg.DEPTH_MAX_M, rng)
        np.ascontiguousarray(rgb).tobytes()
        np.ascontiguousarray(d).tobytes()
        sr.depth_to_xyzrgb_bytes(d, rgb, fx, fy, cx, cy, scfg.DEPTH_MAX_M, 2, 0.0005, rng)

    def color_unit(mj, w, h):
        sr.capture_rgb(rcache[(w, h)], data, mj, vopt)

    # ---- 2. FRONT worker full tick ---------------------------------------
    def front_tick():
        mujoco.mj_forward(model, data)
        for c in front:
            depth_unit(c)
        for mj, _, _, w, h in ir:
            color_unit(mj, w, h)
    ms = timeit(front_tick, n=60)
    print(f"\nFRONT worker full tick: {ms:6.2f} ms -> sustainable ~{1000/ms:5.1f} Hz")

    # ---- 3. AUX worker full tick -----------------------------------------
    def aux_tick():
        mujoco.mj_forward(model, data)
        for c in aux_depth:
            depth_unit(c)
        for mj, _, _, w, h in mono:
            color_unit(mj, w, h)
    ms = timeit(aux_tick, n=40)
    print(f"AUX   worker full tick: {ms:6.2f} ms -> sustainable ~{1000/ms:5.1f} Hz")

    # ---- 4. OLD serial all-in-one tick -----------------------------------
    def serial_tick():
        front_tick()
        aux_tick()
    ms = timeit(serial_tick, n=30)
    print(f"OLD all-serial tick   : {ms:6.2f} ms -> ~{1000/ms:5.1f} Hz "
          f"(front cloud was gated behind this)")

    # ---- 5. LiDAR --------------------------------------------------------
    beams, chassis = sr.lidar_beams(model)
    adr, amin, inc, cut = sr.lidar_beam_table(model, data, beams, chassis)

    def lidar_loop():
        sr.lidar_scan(model, data, beams, chassis)

    def lidar_vec():
        sr.lidar_scan_fast(data, adr, cut)
    print(f"\nLiDAR ({len(beams)} beams): python-loop {timeit(lidar_loop):.2f} ms "
          f"vs vectorised {timeit(lidar_vec):.3f} ms")

    # ---- 6. multi-context GPU scaling ------------------------------------
    print("\n--- multi-context GPU scaling (why 1-thread-per-camera doesn't help) ---")
    for N in (1, 2, 4):
        out = [0.0] * N

        def work(idx):
            d = mujoco.MjData(model)
            mujoco.mj_forward(model, d)
            r = mujoco.Renderer(model, height=360, width=640)
            f = lambda: sr.capture_rgb(r, d, "body_front_cam", vopt)
            f()
            t0 = time.perf_counter()
            for _ in range(60):
                f()
            out[idx] = (time.perf_counter() - t0) / 60 * 1000
            r.close()
        ts = [threading.Thread(target=work, args=(i,)) for i in range(N)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        print(f"  N={N}: per-thread {np.mean(out):5.2f} ms | "
              f"aggregate {N*60/wall:6.1f} fps")

    print("\nRecommendation: set front_rate <= front-tick limit, aux_rate <= aux "
          "limit (leave ~30% headroom for the physics + RViz on the same box).")


if __name__ == "__main__":
    main()
