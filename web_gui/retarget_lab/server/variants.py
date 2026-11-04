"""
The retargeters the lab can run side by side.

Each variant is a TeleopEngine configuration. There is no PLANT: The lab drives
`TeleopEngine._retarget` with the frame's own clock and treats the command as
achieved (ideal servo, `worker.KinematicPlant`), so a difference between two
columns is purely the map — no servo lag, no gravity sag, no engagement ease.

Why each variant runs in its OWN PROCESS: half the knobs below
(ARM_WS_ASSIST, ARM_MIN_*_RADIUS, ARM_HEAD_ANCHOR_LP) are module-level globals
that engine.py / arm.py / geometry.py bind at import. Two variants in one
interpreter would share them, and the second `configure()` would silently
rewrite the first's — the trap that once produced a whole ablation of
identical numbers. One process per variant makes that impossible, and buys
real parallelism (no GIL) for free.
"""
from __future__ import annotations

# id -> spec. `arm_map` is bound in TeleopEngine.__init__; `fingers` picks the
# finger solver; `knobs` = "shipped" (our additions on) or "published" (off).
VARIANTS = {
    "ours": dict(
        short="Ours", label="Ours (shipped)", color="#e8a33d", arm_map="gmr", fingers="keypoint",
        knobs="shipped",
        blurb="GMR keypoints + front-workspace posture atlas + standoff clamps + "
              "head-anchor low-pass + workspace re-anchor + per-finger keypoint IK. "
              "The live stack."),
    # SHIPPED SNAPSHOTS (2026-10-18, operator: "keep all the changes and the previous re-targeting ... name them
    # shipped_v1 v2 v3"). "ours" with exactly the arm-map knobs each version shipped with, pinned here so a later
    # default change cannot move them. Numbers: experiments/exp_arm_shape*.py, exp_elbow_tuck.py, README.
    "shipped_v1": dict(
        short="v1", label="shipped_v1 · GMR (before the arm-shape fix)", color="#9aa5b1", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=False, elbow_abduct=0.15,
        blurb="Ours as of the morning of 2026-10-18: GMR direction everywhere. Straight arms splay 16-21 deg "
              "outward (NOVA's shoulders are 3x narrower than a human's)."),
    "shipped_v2": dict(
        short="v2", label="shipped_v2 · + extended-arm shape", color="#61afef", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=True, elbow_abduct=0.15,
        blurb="v1 + ARM_SHAPE_GATE: past 70% extension the arm points along the operator's own shoulder->wrist "
              "direction (1.5-8.8 deg off on straight arms). Extended upper arms come in ~5 deg (elbow tuck)."),
    "shipped_v3": dict(
        short="v3", label="shipped_v3 · + elbow out", color="#e5c07b", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=True, elbow_abduct=0.25,
        blurb="v2 with the elbow hint pushed 0.25 m outward (was 0.15): extended-arm abduction back to v1's "
              "43 deg, the arm shape of v2 kept (5.9 deg at 85-95% extension)."),
    "shipped_v4": dict(
        short="v4", label="shipped_v4 · + NOVA arm speed", color="#98c379", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=True, elbow_abduct=0.25, arm_speed="nova",
        blurb="v3's retargeting with NOVA's per-joint arm servo slews, including when using the MABEL "
              "robot preset. Torso, neck, lift, IK trust and collision limits keep the selected preset."),
    "shipped_v5": dict(
        short="v5", label="shipped_v5 · gripper handoff", color="#56b6c2", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=True, elbow_abduct=0.25,
        arm_speed="nova", gripper_contact=True,
        blurb="v4 plus physical gripper-tip targets and gripper collision geometry for two-hand contact. "
              "Experimental until physical approach and handoff validation is complete."),
    "rel": dict(
        short="rel", label="rel · relative increment (test)", color="#c678dd", arm_map="gmr",
        fingers="keypoint", knobs="shipped", shape_gate=True, elbow_abduct=0.25, arm_speed="nova",
        rel_track=True, gripper_contact=True,
        blurb="What most VR teleop does (DROID, TeleMoMa, HATO, XRoboToolkit): at engage the gripper latches where it is; then it moves by the "
              "operator's wrist increments in the world, metres 1:1, and turns by the wrist's rotation increments. "
              "v4's IK, elbow hint and upper-arm standoff; the guard uses the gripper's own collision shape."),
    "gmr": dict(
        short="GMR", label="GMR", color="#5aa9e6", arm_map="gmr", fingers="keypoint",
        knobs="published",
        blurb="Root-uniform scaled keypoints about the mid-shoulder "
              "(Retargeting Matters, arXiv:2510.02252) with every MABEL addition off."),
    "ext": dict(
        short="Ext", label="Extension", color="#7bc47f", arm_map="extension", fingers="keypoint",
        knobs="published",
        blurb="Per-shoulder extension fraction rho carried across. Extension-exact, "
              "shape-distorting (robot shoulders ~3x narrower)."),
    "abs": dict(
        short="Abs", label="Abs. scaled", color="#c678dd", arm_map="head", fingers="keypoint",
        knobs="published",
        blurb="Head-anchored hand offset x one scale — what most VR teleop ships."),
    "wb_ik": dict(
        short="WB-IK", label="WB IK", color="#56c2c6", arm_map="gmr", fingers="keypoint",
        knobs="shipped", profile="whole_body",
        blurb="Whole-body differential-IK profile (relaxed-IK class): atlas off, "
              "heavy elbow objective, slow distal trust region."),
    "direct": dict(
        short="Direct", label="Direct copy", color="#e06c75", arm_map="gmr", fingers="keypoint",
        knobs="published", direct=True,
        blurb="Operator shoulder/elbow ANGLES written onto the robot joints. No IK, "
              "no scaling — the strawman the morphology gap defeats."),
    "ours_fast": dict(
        short="Ours·fast", label="Ours · fast trust 6 rad/s", color="#f5d76e", arm_map="gmr",
        fingers="keypoint", knobs="shipped", trust_rate=6.0,
        blurb="Ours with the proximal arm trust region at 6 rad/s instead of the "
              "shipped 2 rad/s (chosen on the robot to cut overshoot). On fast "
              "operator motion the 2 rad/s cap is where the lag spikes come from."),
    "ours_anat": dict(
        short="Ours·anat", label="Ours · anatomical fingers", color="#d19a66", arm_map="gmr",
        fingers="anatomical", knobs="shipped",
        blurb="Ours with the per-joint-angle finger decomposition instead of "
              "keypoint IK (arm identical)."),
    "ours_vec": dict(
        short="Ours·vec", label="Ours · vector fingers", color="#98c379", arm_map="gmr",
        fingers="vector", knobs="shipped",
        blurb="Ours with the AnyTeleop-style keyvector finger objective (arm identical)."),
    "ours_noatlas": dict(
        short="Ours−atlas", label="Ours − posture atlas", color="#abb2bf", arm_map="gmr",
        fingers="keypoint", knobs="shipped", no_table=True,
        blurb="Ours without the offline front-workspace posture atlas — the "
              "palm-flip fix removed."),
}


# ── NOVA's learned whole-body policies: EVERY exported checkpoint in NOVA/controller/wbc_rl/checkpoints
# (`v<N>_policy.npz`) becomes a variant, so a new policy pulled from NOVA shows up in the list by itself.
# Policies from v28 on were trained with the exact wrist solver (deploy.Adapter.solve_wrist) and run with it;
# earlier ones command the wrist themselves (NOVA's sim2sim.py --no-wrist-ik) — NOVA's README rule.
RL_COLORS = ["#ff79c6", "#bd93f9", "#8be9fd", "#ffb86c", "#50fa7b", "#f1fa8c", "#ff5555", "#6272a4"]
RL_WRIST_IK_FROM = 28


def _rl_variants():
    import os
    import re
    from pathlib import Path
    nova = Path(os.environ.get("MABEL_RTL_NOVA", str(Path(__file__).resolve().parents[4] / "NOVA")))
    ck = nova / "controller" / "wbc_rl" / "checkpoints"
    found = sorted(((int(m.group(1)), f.name) for f in ck.glob("v*_policy.npz")
                    if (m := re.fullmatch(r"v(\d+)_policy\.npz", f.name))), reverse=True)
    out = {}
    for i, (v, fn) in enumerate(found):
        ik = v >= RL_WRIST_IK_FROM
        out[f"rl_v{v}"] = dict(
            short=f"RL v{v}", label=f"RL policy v{v}" + (" (latest)" if i == 0 else ""),
            color=RL_COLORS[i % len(RL_COLORS)], arm_map="gmr", fingers="keypoint", knobs="shipped",
            rl=fn, rl_version=v, rl_wrist_ik=ik,
            blurb=f"NOVA's learned whole-body controller v{v} (NOVA/controller/wbc_rl, PPO on EgoDex + LAFAN1): "
                  "head + wrists -> base, lift, torso, neck, arms in one policy, run through NOVA's own deploy "
                  "adapter (numpy)" + (", wrist from its exact solver" if ik else ", wrist from the policy") +
                  ". Same robot for MABEL and NOVA. Fingers: the lab's solver.")
    return out


VARIANTS.update(_rl_variants())
RL_VARIANTS = [k for k in VARIANTS if VARIANTS[k].get("rl")]

# DEPLOYED (2026-10-23, operator: "always have one called deployed ... others are just tests"): the variant NOVA's
# robot runs. ../NOVA/real_run.sh and controller/lab_retarget.py default to NOVA_LAB_VARIANT=deployed, so this one
# line IS the robot's retargeter. Promote a test variant by changing DEPLOYED, nothing else.
DEPLOYED = "shipped_v4"
VARIANTS = {"deployed": dict(VARIANTS[DEPLOYED], short="Deployed", color="#ffffff",
                             label=f"deployed (= {DEPLOYED}, what NOVA runs)",
                             blurb=f"What the robot runs: {DEPLOYED}. " + VARIANTS[DEPLOYED]["blurb"]),
            **VARIANTS}

DEFAULT_ACTIVE = ["deployed", "rel", "shipped_v3"]   # [0] is mirrored to the headset
assert DEPLOYED in VARIANTS, DEPLOYED


def public():
    """The catalogue the UI renders (ids, labels, colours, one-line blurbs)."""
    return [{"id": k, "label": v["label"], "short": v.get("short", v["label"].split(" ")[0]),
             "color": v["color"], "blurb": v["blurb"],
             "arm_map": v["arm_map"], "fingers": v["fingers"]}
            for k, v in VARIANTS.items()]


def reload():
    """Re-read this file in place (the sidebar's refresh button): variants added or edited since the lab
    started show up without a restart. Workers are spawned processes, so a (re)started one imports the
    file fresh. A file that fails to load raises before anything changes. Returns the catalogue diff."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(__name__, __file__)
    new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new)
    old = {k: repr(v) for k, v in VARIANTS.items()}
    cur = {k: repr(v) for k, v in new.VARIANTS.items()}
    vars(sys.modules[__name__]).update({k: v for k, v in vars(new).items() if not k.startswith("__")})
    return {"added": [k for k in cur if k not in old], "removed": [k for k in old if k not in cur],
            "changed": [k for k in cur if k in old and cur[k] != old[k]]}


def configure(eng, model, vid):
    """Put a freshly built engine into variant `vid`. Runs inside the worker
    process, so module-level knobs are private to this variant."""
    import mabel.teleop_engine.arm as arm_mod
    import mabel.teleop_engine.engine as ENG
    import mabel.teleop_engine.geometry as GEO
    import mabel.teleop_engine.hyperparams as HP
    import mabel.whole_body_retargeting.workspace_assist as WA
    spec = VARIANTS[vid]

    def setall(name, val):
        for m in (HP, GEO, ENG, WA, arm_mod):
            if hasattr(m, name):
                setattr(m, name, val)

    # the shipped configuration, asserted explicitly (not inherited from env)
    setall("ARM_MIN_SHOULDER_RADIUS", 0.20)
    setall("ARM_MIN_TORSO_RADIUS", 0.10)
    setall("ARM_HEAD_ANCHOR_LP", 0.05)
    setall("ARM_WS_ASSIST", True)
    arm_mod.ARM_W_ORIENT_FWD = 60.0
    setall("ARM_TRUST_RATE", float(spec.get("trust_rate", 2.0)))
    # the arm-map knobs a shipped snapshot pins (others keep the hyperparams defaults)
    if "shape_gate" in spec:
        setall("ARM_SHAPE_GATE", bool(spec["shape_gate"]))
    if "elbow_abduct" in spec:
        setall("ARM_ELBOW_ABDUCT", float(spec["elbow_abduct"]))
    eng.rel_track = bool(spec.get("rel_track", False))
    profile = spec.get("profile", "wrist")
    for side in ("left", "right"):
        eng.arm[side].set_profile(profile)

    if spec["knobs"] == "published":
        setall("ARM_WS_ASSIST", False)
        setall("ARM_MIN_SHOULDER_RADIUS", 0.0)
        setall("ARM_MIN_TORSO_RADIUS", 0.0)
        setall("ARM_HEAD_ANCHOR_LP", 1.0)          # 1.0 == no filter
        for side in ("left", "right"):
            eng.arm[side].table = None
    if spec.get("no_table"):
        for side in ("left", "right"):
            eng.arm[side].table = None

    if spec["fingers"] == "vector":
        import mujoco
        from mabel.hand_retargeting.vector import HandVectorRetargeter
        for s in ("right", "left"):
            h = eng.hand.get(s)
            if h is not None and hasattr(h, "_ik"):
                h._ik = HandVectorRetargeter(model, mujoco.MjData(model))
                h._hand = h._ik.hands.get(s.capitalize())
    return spec
