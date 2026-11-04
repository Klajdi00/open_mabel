"""GMR scaled-keypoint arm map (mabel/teleop_engine, 2026-07-24) contract.

The live engine retargets the arms with the root-uniform scaled-keypoint map of
Retargeting Matters / GMR (arXiv:2510.02252): every keypoint relative to the
mid-shoulder ROOT, scaled by sigma = L_rob / L_hum, re-anchored at the robot's
root, so the head-wrist-wrist pose TRIANGLE is preserved. These tests lock in:

  * the map is the shipped default,
  * the ShoulderEstimator exposes the mid-shoulder root,
  * a symmetric bimanual reach lands the robot's inter-wrist separation at the
    scaled human separation (scale tracking — requirement 1), materially closer
    than the per-shoulder "extension" map it replaced,
  * full human extension is still full robot extension, and
  * held poses across the workspace are command-stable — the trust-region
    limit-cycle fix (arm.py solve()) that resolved the "very unstable" report
    (16/36 workspace hold poses oscillated ±0.1 rad before it).

These reuse the faithful synthetic operator + engine rig from the exhaustive
study (simulation/.../exp_retarget_fidelity.py: E8 scale, E10 stability) so the
CI guard exercises the same closed-loop behaviour the study measures. The study
itself (sweeps, figures, all three maps) remains the primary validation."""
import os
import numpy as np
import mujoco
import pytest

# The GMR map needs the full articulated arm/shoulder geometry + the faithful
# synthetic operator. Both live under simulation/; skip cleanly if absent.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL = os.path.join(_REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
_TOOLS = os.path.join(_REPO, "simulation", "mabel_mujoco", "scripts", "tools")
_TELEOP = os.path.join(_REPO, "simulation", "mabel_mujoco", "scripts")

_harness = None
if os.path.isfile(_MODEL) and os.path.isdir(_TOOLS):
    import sys
    for p in (_TELEOP, _TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:                                            # exp_retarget_fidelity imports mabel
        import exp_retarget_fidelity as _harness    # noqa: E402
    except Exception:                               # pragma: no cover
        _harness = None

needs_harness = pytest.mark.skipif(
    _harness is None, reason="exp_retarget_fidelity harness / MJCF not importable")

AVP_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _model():
    return mujoco.MjModel.from_xml_path(_MODEL)


def test_gmr_is_default_map():
    from mabel.teleop_engine import hyperparams as HP
    assert HP.ARM_MAP == "gmr", "GMR must be the shipped default arm map"


@needs_harness
def test_shoulder_root_is_midpoint():
    from mabel.teleop_engine.geometry import ShoulderEstimator
    est = ShoulderEstimator()
    assert est.root_world() is None                 # not ready before an update
    op = _harness.SyntheticOperator()
    head = np.eye(4); head[:3, 3] = op.head_pos
    est.update(head, None, {"right": op.wrist("right", 0.5, _harness.FWD),
                            "left": op.wrist("left", 0.5, _harness.FWD)})
    root = est.root_world()
    assert root is not None
    mid = 0.5 * (est.shoulder_world("right") + est.shoulder_world("left"))
    assert np.allclose(root, mid, atol=1e-9)


@needs_harness
def test_gmr_tracks_human_scale_better_than_extension():
    """Symmetric wide forward reach: the robot inter-wrist gap must approach the
    SCALED human gap (sigma = L_rob / L_hum), and GMR must beat the per-shoulder
    extension map on that ratio (the study measures 0.965 vs 0.541)."""
    model = _model()
    op = _harness.SyntheticOperator()
    # wide reach so the bimanual proximity blend / floor are inactive
    tgt = (0.75, np.array([0.45, 0.05, 1.0]))
    other = (0.75, np.array([-0.45, 0.05, 1.0]))
    ratios = {}
    for arm_map in ("gmr", "extension"):
        rig = _harness.EngineRig(model, arm_map)
        _harness.warmup(rig, op)
        _harness.drive_to(rig, op, tgt, other=other)
        _, _, _, wr = rig.arm_state("right")
        _, _, _, wl = rig.arm_state("left")
        sep_r = float(np.linalg.norm(wr - wl))
        sep_h = float(np.linalg.norm(op.wrist("right", *tgt) - op.wrist("left", *other)))
        sigma = rig.eng.arm_reach["right"] / op.L
        ratios[arm_map] = sep_r / (sigma * sep_h)
    assert abs(ratios["gmr"] - 1.0) < 0.12, f"gmr ratio {ratios['gmr']:.3f}"
    assert abs(ratios["gmr"] - 1.0) < abs(ratios["extension"] - 1.0), \
        f"gmr {ratios['gmr']:.3f} not better than extension {ratios['extension']:.3f}"


@needs_harness
def test_gmr_full_extension_is_full_extension():
    """The extension map's guarantee, kept: fully extending the human arm drives
    the robot elbow to near its joint limit (rho_r near 1)."""
    model = _model()
    op = _harness.SyntheticOperator()
    rig = _harness.EngineRig(model, "gmr")
    _harness.warmup(rig, op)
    _harness.drive_to(rig, op, (1.0, _harness.FWD))
    rho_r, _, _, _ = rig.arm_state("right")
    assert rho_r > 0.92, f"full human extension gave rho_r={rho_r:.3f}"


@needs_harness
def test_gmr_held_poses_are_stable_across_workspace():
    """Held reaches must produce dead-still commands. The 'very unstable' report
    was a trust-region LIMIT CYCLE a single hold pose could miss by landing in a
    clean basin — 16/36 workspace poses oscillated ±0.1 rad before the fix
    (arm.py solve() now clips the basin-local optimum instead of solving inside
    a re-centring box). Sweep a grid so a latent cycle cannot hide."""
    model = _model()
    op = _harness.SyntheticOperator()
    worst = 0.0
    for rho in (0.6, 0.75):
        for dr in (-0.15, 0.1, 0.35):
            d = np.array([dr, 0.1, 1.0])
            rig = _harness.EngineRig(model, "gmr")
            _harness.warmup(rig, op)
            tgt = (rho, d)
            other = (rho, np.array([-dr, 0.1, 1.0]))
            _harness.drive_to(rig, op, tgt, other=other)
            # Hold and measure the TAIL: after the trust-region walk converges,
            # a stable pose has ~0 command variance; a limit cycle keeps a
            # constant-amplitude oscillation no settling can remove (this is
            # what the pre-fix map did — 50 mrad indefinitely).
            hist = []
            for _ in range(120):
                rig.step(op.pose(rig.t, {"right": tgt, "left": other}))
                hist.append(np.concatenate([rig.arm_ctrl("right"),
                                            rig.arm_ctrl("left")]))
            hist = np.stack(hist)
            std = float(np.max(np.std(hist[-40:], axis=0)))
            worst = max(worst, std)
            assert std < 3e-3, \
                f"held pose rho={rho} dir={d.tolist()} oscillates: {std*1e3:.1f} mrad"
