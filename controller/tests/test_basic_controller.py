"""Contract for the rebuilt controller (`mabel.basic`): PD + task-space wrist.

Written alongside the rewrite, not after it. The numbers come from
`controller/experiments/exp_basic_controller.py`; the bounds here are loose
enough to survive solver noise and tight enough that the failures we care about
(a droop that stops scaling as 1/kp, an IK that diverges past reach, two
controllers fighting over one joint) cannot pass.
"""
import pathlib

import numpy as np
import mujoco
import pytest

from conftest import _load, needs_model

from mabel.basic import TaskSpaceArm, JointPD

SIDE = "right"


def _settle(model, data, arm, seconds=4.0):
    for _ in range(int(seconds / model.opt.timestep)):
        arm.step()
        mujoco.mj_step(model, data)


def _arm(gravity=False, kp=400.0):
    model, data = _load()
    arm = TaskSpaceArm(model, data, side=SIDE, kp=kp,
                       kd=2.0 * np.sqrt(kp), gravity=gravity).attach()
    return model, data, arm


# ── the PD owns its joints ───────────────────────────────────────────────────

@needs_model
def test_attach_silences_the_mjcf_servos_and_detach_restores_them():
    """Two control laws on one joint means every later measurement is a lie."""
    model, data = _load()
    aid = model.actuator(f"{SIDE.capitalize()} Arm 2").id
    gain0 = float(model.actuator_gainprm[aid, 0])
    assert gain0 != 0.0, "the MJCF should ship a live position servo here"

    pd = JointPD(model, data, [f"{SIDE.capitalize()} Arm 2"]).attach()
    assert float(model.actuator_gainprm[aid, 0]) == 0.0, "servo not neutralised"
    pd.detach()
    assert float(model.actuator_gainprm[aid, 0]) == gain0, "servo not restored"


@needs_model
def test_hold_here_is_bumpless():
    model, data, arm = _arm()
    before = {j: arm.pd.q(j) for j in arm.joints}
    arm.hold_here()
    for j in arm.joints:
        assert abs(arm.pd.q_des[j] - before[j]) < 1e-12


@needs_model
def test_torque_is_clamped_to_the_model_limits():
    model, data, arm = _arm(kp=1e6)             # absurd gain on purpose
    arm.pd.set_targets({j: arm.pd.q(j) + 1.0 for j in arm.joints})
    arm.pd.step()
    for j, tau in arm.pd.last_tau.items():
        lim = arm.pd.tau_max[j]
        if np.isfinite(lim):
            assert abs(tau) <= lim + 1e-6, f"{j} exceeded its force range"


# ── the baseline behaves like a PD ───────────────────────────────────────────

@needs_model
def test_pure_pd_droop_scales_as_one_over_kp():
    """The signature of an uncompensated PD. If this stops holding, something
    else is quietly contributing torque."""
    errs = {}
    for kp in (100.0, 400.0):
        model, data, arm = _arm(gravity=False, kp=kp)
        _settle(model, data, arm)
        errs[kp] = max(abs(v) for v in arm.pd.error().values())
    ratio = errs[100.0] / errs[400.0]
    assert 2.5 < ratio < 5.5, (
        f"4x the gain should give ~4x less droop, got {ratio:.2f}x "
        f"({np.degrees(errs[100.0]):.2f} deg -> {np.degrees(errs[400.0]):.2f} deg)"
    )


@needs_model
def test_pure_pd_really_does_droop():
    """Guards the guard: if this passes with ~0 error the model is not loaded
    and every gravity measurement below is meaningless."""
    model, data, arm = _arm(gravity=False, kp=100.0)
    _settle(model, data, arm)
    worst = np.degrees(max(abs(v) for v in arm.pd.error().values()))
    assert worst > 1.0, f"expected a real droop under pure PD, got {worst:.3f} deg"


# ── gravity feed-forward ─────────────────────────────────────────────────────

@needs_model
def test_gravity_feedforward_collapses_the_droop():
    model, data, a0 = _arm(gravity=False, kp=100.0)
    _settle(model, data, a0)
    pure = max(abs(v) for v in a0.pd.error().values())

    model, data, a1 = _arm(gravity=True, kp=100.0)
    _settle(model, data, a1)
    comp = max(abs(v) for v in a1.pd.error().values())

    assert comp < pure / 5.0, (
        f"gravity FF should cut the droop by >5x: "
        f"{np.degrees(pure):.3f} deg -> {np.degrees(comp):.3f} deg"
    )


@needs_model
def test_gravity_term_is_velocity_independent():
    """The model-based term is gravity ALONE — no Coriolis by construction,
    because it reads only angles, never qvel. Feeding velocity-dependent terms
    forward would cancel the damping and the arm would ring."""
    model, data, arm = _arm(gravity=True)
    g0 = arm.pd.gravity_torque()
    for j in arm.joints:
        data.qvel[int(model.joint(j).dofadr[0])] = 2.0
    g1 = arm.pd.gravity_torque()               # same angles, big velocity
    for j in arm.joints:
        assert abs(g1[j] - g0[j]) < 1e-12, f"{j}: gravity moved with qvel"


@needs_model
def test_gravity_model_uses_no_simulator_state():
    """The whole point: it runs on hardware. It must not touch mj_forward,
    qfrc_bias, or the live MjData — only the model constants and the angles."""
    import ast
    import mabel.basic.gravity as gmod
    tree = ast.parse(pathlib.Path(gmod.__file__).read_text())
    # only IMPORTS + attribute access, not comments/docstrings, are the contract
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "mujoco" not in imported, "gravity.py imports mujoco — not hardware-portable"
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for banned in ("qfrc_bias", "mj_forward"):
        assert banned not in attrs, f"gravity.py uses {banned} — simulator-privileged"
    # numpy is the only third-party import it may have
    assert imported - {"numpy", "__future__"} <= {"arm_params"}, (
        f"gravity.py imports more than numpy: {imported}")


@needs_model
def test_analytic_gravity_matches_mujoco_ground_truth():
    """Licenses the swap from qfrc_bias: the model-only computation must equal
    MuJoCo's own gravity torque across the workspace (arm poses)."""
    from mabel.basic.gravity import ArmGravity
    model, data = _load()
    jn = [f"{SIDE.capitalize()} Arm {i}" for i in range(1, 8)]
    qadr = [int(model.joint(n).qposadr[0]) for n in jn]
    dof = [int(model.joint(n).dofadr[0]) for n in jn]
    lo = np.array([model.joint(n).range[0] for n in jn])
    hi = np.array([model.joint(n).range[1] for n in jn])
    parent = int(model.body_parentid[model.body(model.joint(jn[0]).bodyid[0]).id])
    G = ArmGravity(SIDE)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(lo, hi)
        data.qpos[:] = model.qpos0
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        truth = np.array([data.qfrc_bias[i] for i in dof])
        base_R = np.array(data.body(parent).xmat).reshape(3, 3)
        worst = max(worst, float(np.max(np.abs(G.torque_array(q, base_R=base_R) - truth))))
    assert worst < 1e-6, f"analytic gravity diverged from qfrc_bias by {worst:.2e} N*m"


@needs_model
def test_baked_params_still_match_the_model():
    """arm_params.py is generated; if the MJCF's arm masses/geometry changed and
    it was not regenerated, the total mass drifts. Catch that here."""
    from mabel.basic.arm_params import ARM
    model, _ = _load()
    for side, spec in ARM.items():
        jn = [f"{side.capitalize()} Arm {i}" for i in range(1, 8)]
        root = model.body(model.joint(jn[0]).bodyid[0]).id
        mjcf_mass = float(model.body_subtreemass[root])
        assert abs(spec["total_mass"] - mjcf_mass) < 1e-3, (
            f"{side}: baked mass {spec['total_mass']:.4f} != MJCF "
            f"{mjcf_mass:.4f} — re-run extract_arm_params.py")


# ── task space ───────────────────────────────────────────────────────────────

@needs_model
def test_ik_solves_reachable_targets_to_sub_millimetre():
    model, data, arm = _arm()
    home = arm.palm().copy()
    for off in ((0.08, 0, 0), (0, 0.08, 0), (0, 0, 0.08), (0.05, 0.05, 0.05)):
        q, err = arm.ik.solve(home + np.array(off))
        assert err < 2e-3, f"offset {off}: IK residual {err*1000:.2f} mm"


@needs_model
def test_ik_saturates_past_reach_instead_of_diverging():
    """Damping is a SAFETY parameter: an unreachable target must not produce a
    wild joint step."""
    model, data, arm = _arm()
    home = arm.palm().copy()
    q, err = arm.ik.solve(home + np.array([0.0, 0.0, 2.0]))   # far out of reach
    assert np.all(np.isfinite(q)), "IK produced non-finite joints"
    lo = np.array([model.joint(j).range[0] for j in arm.joints])
    hi = np.array([model.joint(j).range[1] for j in arm.joints])
    assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6), "IK left the limits"
    assert err > 0.5, "an out-of-reach target should report a LARGE residual"


@needs_model
def test_wrist_tracking_reaches_the_target_with_gravity_ff():
    model, data, arm = _arm(gravity=True, kp=400.0)
    tgt = arm.palm().copy() + np.array([0.08, 0.08, 0.08])
    arm.set_wrist_target(tgt)
    _settle(model, data, arm)
    assert arm.task_error() < 5e-3, (
        f"palm settled {arm.task_error()*1000:.1f} mm from its target")


@needs_model
def test_the_error_split_is_observable():
    """The point of the rewrite: you can tell WHICH stage failed."""
    model, data, arm = _arm(gravity=False, kp=100.0)
    arm.set_wrist_target(arm.palm().copy() + np.array([0.08, 0.08, 0.08]))
    _settle(model, data, arm)
    assert arm.ik_err < 2e-3, "the solver should have found the pose"
    assert arm.task_error() > arm.ik_err * 5, (
        "under pure PD the palm error must be dominated by the DROOP, not the "
        "solver — that is the diagnostic this controller exists to give")


@needs_model
def test_clearing_the_target_returns_to_joint_hold():
    model, data, arm = _arm(gravity=True)
    arm.set_wrist_target(arm.palm().copy() + np.array([0.05, 0, 0]))
    _settle(model, data, arm, 1.0)
    arm.set_wrist_target(None)
    assert arm.target is None
    assert np.isnan(arm.task_error())
    held = {j: arm.pd.q_des[j] for j in arm.joints}
    _settle(model, data, arm, 1.0)
    for j in arm.joints:                        # targets must stop moving
        assert abs(arm.pd.q_des[j] - held[j]) < 1e-12
