"""The front-workspace posture table: the contract the live arm IK relies on.

These are the properties that make the table safe to interpolate and safe to
steer to. They are cheap (no table BUILD — that is the experiment harness's job,
~75 s/side); they check the built artefact and the frame math around it.
"""
import numpy as np
import mujoco
import pytest

from mabel.teleop_engine.workspace_table import (
    ArmChain, PostureSolver, WorkspaceTable, palm_target, natural_elbow,
    sew_reference, rot_log, orient_err, PALM_ROLLS)
from mabel.teleop_engine.hyperparams import XML_FILE

SIDES = ("right", "left")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML_FILE)


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.nkey - 1)
    mujoco.mj_forward(model, d)
    return d


# ── the palm-target convention ───────────────────────────────────────────────
def test_palm_roll_means_the_same_anatomical_pose_on_both_hands():
    """roll = 0 is palm DOWN, 180 deg is palm UP, and +90 deg is INWARD toward
    the robot's midline for EITHER hand. The chirality lives here, once, so the
    table and the live lookup cannot disagree about what "inward" means."""
    for side, inward_y in (("right", +1.0), ("left", -1.0)):
        down = palm_target([1, 0, 0], PALM_ROLLS["down"], side)
        up = palm_target([1, 0, 0], PALM_ROLLS["up"], side)
        inw = palm_target([1, 0, 0], PALM_ROLLS["inward"], side)
        assert down[:, 2] == pytest.approx([0, 0, -1], abs=1e-6)
        assert up[:, 2] == pytest.approx([0, 0, 1], abs=1e-6)
        assert inw[:, 2] == pytest.approx([0, inward_y, 0], abs=1e-6)
        for R in (down, up, inw):                       # right-handed, orthonormal
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
            assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)


def test_rot_log_survives_the_antipode():
    """The flip is a 180 deg rotation, and a cosine cost is STATIONARY there —
    which is how the flip got lost. The axis-angle residual must report pi."""
    for axis in ([1, 0, 0], [0, 1, 0], [0.3, -0.5, 0.81]):
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(np.pi - 1e-7) * K + (1 - np.cos(np.pi - 1e-7)) * (K @ K)
        assert np.linalg.norm(rot_log(R)) == pytest.approx(np.pi, abs=1e-3)
    assert orient_err(np.eye(3), np.eye(3)) == pytest.approx(0.0, abs=1e-12)


# ── the posture prior ────────────────────────────────────────────────────────
def test_sew_reference_is_the_gravity_one_where_that_is_well_posed():
    """A level forward reach must still mean "elbow straight down" — the whole
    prior was tuned against that convention."""
    assert sew_reference([1, 0, 0]) == pytest.approx([0, 0, -1], abs=1e-9)


def test_sew_reference_has_no_singularity_below_the_shoulder():
    """The conventional (gravity-down) reference blows up as the arm approaches
    vertical: the elbow sweeps a full circle for an arbitrarily small wrist
    circle. MABEL's front box reaches to within 58 deg of straight down, so that
    is inside the workspace, not a corner case."""
    R, L1, L2 = 0.45, 0.222, 0.433

    def elbow_gain(tilt_deg, stereo):
        e = np.radians(tilt_deg)
        prev, tot = None, 0.0
        for phi in np.linspace(0, 2 * np.pi, 721):
            u = np.array([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), -np.cos(e)])
            p = natural_elbow(np.zeros(3), R * u, L1, L2, "right", stereographic=stereo)
            if p is not None and prev is not None:
                tot += float(np.linalg.norm(p - prev))
            prev = p
        return tot / (2 * np.pi * R * np.sin(e))

    for tilt in (8.0, 4.0, 2.0):
        assert elbow_gain(tilt, stereo=True) < 1.0          # follows the wrist
        assert elbow_gain(tilt, stereo=False) > 2.5         # the singularity


def test_natural_elbow_declines_an_unreachable_pair():
    assert natural_elbow(np.zeros(3), [2.0, 0, 0], 0.222, 0.433, "right") is None
    assert natural_elbow(np.zeros(3), [0.01, 0, 0], 0.222, 0.433, "right") is None


# ── the shoulder frame ───────────────────────────────────────────────────────
def test_shoulder_frame_is_symmetric_and_round_trips(model, data):
    """The table is indexed in this frame, so a frame error reads as a
    retargeting defect. The two arms must see mirror-image coordinates (the
    MJCF's r_arm_mount BODY origin is offset 6 cm from its Arm 1 anchor, which is
    exactly the trap this guards)."""
    seen = {}
    for side in SIDES:
        ch = ArmChain(model, side)
        ch.sync(data)
        seen[side] = ch.pose_s(ch.q_nom)[0]
        assert ch.frame.point(ch.shoulder(), ch.mount_pos, ch.mount_R) == \
            pytest.approx(np.zeros(3), abs=1e-9)
        p = np.array([0.4, -0.1, -0.15])
        w = ch.frame.to_world_point(p, ch.mount_pos, ch.mount_R)
        assert ch.frame.point(w, ch.mount_pos, ch.mount_R) == pytest.approx(p, abs=1e-9)
    r, l = seen["right"], seen["left"]
    assert r[0] == pytest.approx(l[0], abs=2e-3)
    assert r[1] == pytest.approx(-l[1], abs=2e-3)
    assert r[2] == pytest.approx(l[2], abs=2e-3)


def test_shoulder_frame_is_invariant_to_the_lift(model, data):
    """The lift is upstream of the mount, so raising it must not move a pose
    expressed in this frame — that invariance is why the table needs no lift
    axis."""
    ch = ArmChain(model, "right")
    ch.sync(data)
    p0 = ch.pose_s(ch.q_nom)[0]
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos
    lj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Lift Lower")
    d2.qpos[model.jnt_qposadr[lj]] += 0.08
    mujoco.mj_forward(model, d2)
    ch.sync(d2)
    assert ch.pose_s(ch.q_nom)[0] == pytest.approx(p0, abs=1e-6)


# ── the built table ──────────────────────────────────────────────────────────
def _table(side):
    t = WorkspaceTable.load_for(side)
    if t is None:
        pytest.skip(f"no built table for {side} — run exp_front_workspace.py "
                    f"--phase W3 --build")
    return t


@pytest.mark.parametrize("side", SIDES)
def test_table_solves_most_of_the_front_box(side):
    rp = _table(side).res_pos.ravel()
    ro = np.degrees(_table(side).res_ori.ravel())
    solved = float(np.mean((rp < 0.015) & (ro < 12.0)))
    assert solved > 0.75, f"{side} table only solves {solved*100:.0f}% of its cells"
    assert np.median(rp) < 0.005
    assert np.median(ro) < 1.0


@pytest.mark.parametrize("side", SIDES)
def test_table_is_coherent_in_position(side):
    """Adjacent POSITION cells must hold neighbouring postures, or interpolating
    them is fiction. The ROLL axis is exempt: palm-up and palm-down sit at
    opposite ends of the one forearm roll, so a branch boundary there is real."""
    seam = _table(side).seam()
    for ax in ("x", "y", "z"):
        assert seam[f"{ax}_p95"] < 2.5, f"{side}: {ax} seams at {seam[f'{ax}_p95']:.2f} rad"


@pytest.mark.parametrize("side", SIDES)
def test_query_never_blends_across_a_branch(side, model, data):
    """A blended posture must stay close to the cell it came from — that is the
    seam guard doing its job. Without it the query averages two branches and
    returns a posture that solves neither."""
    tbl = _table(side)
    ch = ArmChain(model, side)
    ch.sync(data)
    g = tbl.grid
    rs = np.random.default_rng(4)
    for _ in range(120):
        p = np.array([rs.uniform(g.x[0], g.x[-1]), rs.uniform(g.y[0], g.y[-1]),
                      rs.uniform(g.z[0], g.z[-1])])
        R = palm_target(g.canonical_fwd(p), rs.uniform(0, 2 * np.pi), side)
        q = tbl.query(p, R)
        anchor = max(tbl.cells_for(p, R), key=lambda c: c[1])[0]
        assert np.max(np.abs(q - tbl.q[anchor])) <= tbl.SEAM_TOL + 1e-9


@pytest.mark.parametrize("side", SIDES)
def test_query_is_a_usable_seed_between_cells(side, model, data):
    """Between cells the raw lookup is a SEED, not an answer — but a seed that
    is already in the right configuration, which is the whole point."""
    tbl = _table(side)
    ch = ArmChain(model, side)
    ch.sync(data)
    sol = PostureSolver(ch)
    g = tbl.grid
    rs = np.random.default_rng(9)
    pos, tilt = [], []
    for _ in range(80):
        p = np.array([rs.uniform(g.x[0], g.x[-1]), rs.uniform(g.y[0], g.y[-1]),
                      rs.uniform(g.z[0], g.z[-1])])
        R = palm_target(g.canonical_fwd(p), rs.uniform(0, 2 * np.pi), side)
        ep, et, _ = sol.errors(np.clip(tbl.query(p, R), ch.lo, ch.hi), p, R)
        pos.append(ep)
        tilt.append(np.degrees(et))
    assert np.median(pos) < 0.030
    assert np.median(tilt) < 20.0


@pytest.mark.parametrize("side", SIDES)
def test_feasibility_is_low_where_the_table_failed(side):
    """The live cost reads this to decide how hard to chase the palm angle, so a
    feasibility that does not track the table's own residual would push the
    wrist off the target for a palm direction the arm cannot deliver anyway."""
    tbl = _table(side)
    f = tbl._feas_field()
    bad = (tbl.res_pos > 0.045) | (tbl.res_ori > np.radians(16.0))
    good = (tbl.res_pos < 0.010) & (tbl.res_ori < np.radians(4.0))
    assert f[bad].max() < 1e-9
    assert f[good].min() > 0.999
