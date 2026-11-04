#!/usr/bin/env python3
"""exp_viewer_declutter.py — verify the "robot only" viewer change is safe.

The desktop viewer (server/view_robot.py) used to draw three things that are not
the robot: the camera marker geoms, the 360-beam LiDAR rangefinder fan, and the
green/orange teleop mocap target spheres. This harness proves the fix removed
exactly those and nothing else.

Two independent checks:

  A. MODEL IDENTITY — for every scene touched by the mocap-target patch, compile
     the pre-change XML (from git HEAD) and the post-change XML and assert the
     compiled models are structurally identical except for `geom_group`. This is
     what guarantees the sim server, the state memmap (nq), the perturbation
     memmap (nbody x 6) and the IL/contract consumers cannot notice the change.

  B. RENDER VISIBILITY — render a scene offscreen under the OLD and NEW MjvOption
     and count pixels, confirming the decoration disappeared while the robot's
     own pixels are untouched.

Usage:
    ~/anaconda3/bin/python3 server/tools/exp_viewer_declutter.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes")
DEFAULT_SCENE = os.path.join(SCENES, "mabel_home.xml")

# Fields that describe compiled model structure. geom_group is deliberately
# absent — it is the ONLY thing we allow to differ.
STRUCT_SIZES = ("nq", "nv", "nu", "na", "nbody", "nmocap", "ngeom", "nsite",
                "njnt", "ncam", "nlight", "nsensor", "nmesh", "ntendon", "neq")
STRUCT_ARRAYS = ("body_pos", "body_quat", "body_mass", "body_inertia", "body_parentid",
                 "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid",
                 "geom_contype", "geom_conaffinity", "geom_rgba",
                 "site_pos", "site_group", "jnt_type", "jnt_range",
                 "qpos0", "body_mocapid")


def _git_show(path: str) -> str | None:
    rel = os.path.relpath(path, REPO)
    r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _names(model: mujoco.MjModel, n: int, adr) -> list[str]:
    return [mujoco.mj_id2name(model, adr, i) for i in range(n)]


def check_model_identity() -> int:
    """A. every patched scene compiles, and differs ONLY in geom_group."""
    changed = subprocess.run(
        ["git", "-C", REPO, "diff", "--name-only", "HEAD", "--",
         "simulation/mabel_mujoco/scenes"],
        capture_output=True, text=True).stdout.split()
    scenes = [os.path.join(REPO, p) for p in changed if p.endswith(".xml")]
    print(f"[A] model identity over {len(scenes)} patched scene(s)\n")

    failures = 0
    for path in sorted(scenes):
        name = os.path.basename(path)
        old_text = _git_show(path)
        if old_text is None:
            print(f"    SKIP {name}: not in HEAD")
            continue
        # Write the old XML beside the new one so relative <include> paths resolve.
        fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="_old_", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(old_text)
            try:
                m_old = mujoco.MjModel.from_xml_path(tmp)
                m_new = mujoco.MjModel.from_xml_path(path)
            except Exception as e:  # a scene that fails to compile is a hard fail
                print(f"    FAIL {name}: compile error {e}")
                failures += 1
                continue
        finally:
            os.unlink(tmp)

        problems = []
        for s in STRUCT_SIZES:
            if getattr(m_old, s) != getattr(m_new, s):
                problems.append(f"{s} {getattr(m_old, s)}->{getattr(m_new, s)}")
        for a in STRUCT_ARRAYS:
            o, n = getattr(m_old, a), getattr(m_new, a)
            if o.shape != n.shape or not np.array_equal(o, n):
                problems.append(f"{a} differs")
        if _names(m_old, m_old.nbody, mujoco.mjtObj.mjOBJ_BODY) != \
           _names(m_new, m_new.nbody, mujoco.mjtObj.mjOBJ_BODY):
            problems.append("body names differ")

        moved = int((m_old.geom_group != m_new.geom_group).sum())
        if problems:
            print(f"    FAIL {name}: {'; '.join(problems)}")
            failures += 1
        else:
            print(f"    ok   {name}: identical except geom_group ({moved} geoms -> group 3)")
    return failures


def _opt(old: bool) -> mujoco.MjvOption:
    """OLD = what view_robot.py used to set; NEW = what it sets now."""
    o = mujoco.MjvOption()
    if old:
        o.geomgroup[3] = 1                                        # the removed line
    else:
        o.geomgroup[3] = 0
        o.sitegroup[3] = 0
        o.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    return o


def _cam() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Mirror view_robot.py's opening framing so the checks measure what the
    # operator actually sees.
    az, el, dist, lz = (float(v) for v in
                        os.getenv("MABEL_VIEWER_CAM", "40,-28,3.0,1.0").split(","))
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    cam.lookat[:] = [0.0, 0.0, lz]
    return cam


def _segment(model, data, opt, w=960, h=720):
    """Per-pixel geom id, so we can count EXACTLY which geoms reach the screen.

    Colour heuristics do not work here — mabel_golf is a green golf course, and
    a global "how many pixels changed" threshold just measures how much of the
    frame the LiDAR fan covers (50%+ in the big outdoor scenes). Segmentation
    rendering answers the real question per-geom and scene-independently.
    """
    with mujoco.Renderer(model, h, w) as r:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=_cam(), scene_option=opt)
        seg = r.render()[:, :, 0].astype(np.int32)   # channel 0 = geom id, -1 = bg
    ids, counts = np.unique(seg, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i >= 0}


def _decor_count(model, data, opt) -> int:
    """Abstract-scene geom count. Rangefinder beams are DECOR, not model geoms,
    so they never show up in segmentation — this is how we count them."""
    scn = mujoco.MjvScene(model, maxgeom=200_000)
    mujoco.mjv_updateScene(model, data, opt, None, _cam(),
                           mujoco.mjtCatBit.mjCAT_ALL, scn)
    return scn.ngeom


def check_render(scene: str) -> int:
    print(f"\n[B] render visibility — {os.path.basename(scene)}")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Isolate ONE switch at a time. Comparing the full old/new option sets
    # directly does not work: with mjVIS_RANGEFINDER on, the 360 beams are decor
    # elements whose segmentation ids overlap the model geom id range, so they
    # corrupt every per-geom pixel count. So group-3 visibility is measured with
    # the beams off, and the beams are counted via the abstract scene instead.
    base_opt = _opt(old=False)                  # everything off  == the new viewer
    g3_opt = _opt(old=False)
    g3_opt.geomgroup[3] = 1                     # + camera markers and targets
    rf_opt = _opt(old=False)
    rf_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 1   # + the LiDAR fan

    new, old = _segment(model, data, base_opt), _segment(model, data, g3_opt)

    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
    targets = {"target_l", "target_r", "gaze_point"}
    tgt_geoms, cam_geoms, robot_geoms = set(), set(), set()
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if bname in targets:
            tgt_geoms.add(g)
            continue
        on_robot = False
        p = b
        while p > 0:
            if p == robot_root:
                on_robot = True
                break
            p = int(model.body_parentid[p])
        if not on_robot:
            continue
        # Camera decoration = group 3 AND on the robot. Name alone is not enough:
        # mabel_golf has world props called bench_marker* that are real scenery.
        if int(model.geom_group[g]) == 3 and ("_marker" in gname or "_lens" in gname):
            cam_geoms.add(g)
        else:
            robot_geoms.add(g)

    px = lambda d, s: sum(d.get(g, 0) for g in s)
    fails = 0

    for label, group in (("mocap target spheres", tgt_geoms),
                         ("camera markers/lenses", cam_geoms)):
        before, after = px(old, group), px(new, group)
        print(f"    {label:22s}: {before:7d} px -> {after:7d} px")
        if before == 0:
            print(f"      note: not on screen from this angle — nothing to prove")
        elif after != 0:
            print(f"      FAIL: still visible")
            fails += 1

    # MuJoCo only draws a rangefinder beam that actually returns a hit inside the
    # 12 m cutoff, so the count is scene-dependent (a wide-open course returns
    # far fewer than an indoor room). Assert the fan is gone, not a fixed number.
    beams = _decor_count(model, data, rf_opt) - _decor_count(model, data, base_opt)
    print(f"    lidar beam decor      : {beams} beams drawn before, 0 after")
    if beams <= 0:
        print("      FAIL: rangefinder was not drawing beams — check is vacuous")
        fails += 1

    # The robot must not lose a single pixel. It may GAIN some, where a camera
    # marker or a target sphere used to sit in front of it.
    lost = {g: (old[g], new.get(g, 0)) for g in robot_geoms
            if new.get(g, 0) < old.get(g, 0)}
    r_before, r_after = px(old, robot_geoms), px(new, robot_geoms)
    print(f"    robot geometry        : {r_before:7d} px -> {r_after:7d} px "
          f"({len(robot_geoms)} geoms, {len(lost)} geoms lost pixels)")
    if lost:
        print(f"      FAIL: robot lost pixels: {list(lost.items())[:5]}")
        fails += 1
    if not fails:
        print("    ok  : decoration gone, robot fully intact")
    return fails


def check_robot_intact(scene: str) -> int:
    """C. structural proof that hiding groups cannot hide the ROBOT.

    The pixel delta in [B] is large (the 360-beam LiDAR fan covers much of the
    frame), so a pixel threshold alone is a weak guarantee. This check is exact:
    it walks the compiled model and asserts (1) no geom on a robot body sits in
    a hidden group, and (2) the only geoms we pushed into group 3 belong to the
    three mocap target bodies.
    """
    print(f"\n[C] robot integrity — {os.path.basename(scene)}")
    model = mujoco.MjModel.from_xml_path(scene)
    # ONLY group 3 is in scope. The viewer's change is geomgroup[3] 1 -> 0, so
    # group 3 is the only visibility that moved. Groups 4-5 are hidden by MuJoCo
    # default and were hidden before this change too — flagging them would be
    # noise (mabel_kitchen_fk authors its collision shells in group 4, which no
    # renderer in this repo has ever drawn).
    hidden = {3}
    targets = {"target_l", "target_r", "gaze_point"}

    # Robot bodies = the MJCF root and everything descending from it. NOTE the
    # root is `mabel_chassis` here — `base_link` is the *derived* URDF/rig name,
    # not the MJCF one.
    robot = set()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
    if base < 0:
        print("    SKIP: no mabel_chassis in this scene")
        return 0
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == base:
                robot.add(b)
                break
            p = int(model.body_parentid[p])

    # A geom only *visibly* disappears if it was drawn in the first place. Many
    # group-3 geoms in these scenes are collision proxies authored at alpha 0
    # (every household prop carries a visual geom in group 2 plus an invisible
    # contact geom in group 3; the chassis_bumper is the same trick). Hiding
    # those is a no-op on screen, and geom group never affects physics at all.
    def visible(g):
        return float(model.geom_rgba[g][3]) > 0.0

    hidden_geoms = [g for g in range(model.ngeom) if int(model.geom_group[g]) in hidden]
    proxies = [g for g in hidden_geoms if not visible(g)]
    drawn = [g for g in hidden_geoms if visible(g)]
    print(f"    robot bodies      : {len(robot)}")
    print(f"    geoms in group 3+ : {len(hidden_geoms)}  "
          f"({len(proxies)} invisible collision proxies, {len(drawn)} actually drawn)")

    fails = 0
    robot_drawn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                   for g in drawn if int(model.geom_bodyid[g]) in robot]
    structural = [n for n in robot_drawn if not (n and ("_marker" in n or "_lens" in n))]
    if structural:
        print(f"    FAIL: visible robot geometry would be hidden: {structural[:8]}")
        fails += 1
    else:
        print(f"    ok  : the only robot geometry hidden is {len(robot_drawn)} "
              f"camera markers/lenses")

    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
             for g in drawn if int(model.geom_bodyid[g]) not in robot}
    print(f"    non-robot hidden  : {sorted(n for n in names if n)}")
    if names - targets:
        print(f"    FAIL: unexpected scene geometry hidden: {sorted(names - targets)}")
        fails += 1
    else:
        print("    ok  : the only scene geometry hidden is the teleop mocap targets")
    return fails


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    scene = args[0] if args else DEFAULT_SCENE
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform != "darwin" else "cgl")
    # --quick skips [A], which recompiles 39 scenes from git and takes ~2 min.
    fails = 0 if "--quick" in sys.argv else check_model_identity()
    fails += check_render(scene) + check_robot_intact(scene)
    print(f"\n{'PASS — viewer decluttered, models unchanged' if not fails else f'FAIL ({fails})'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
