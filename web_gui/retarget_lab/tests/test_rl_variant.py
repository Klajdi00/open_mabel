"""NOVA's RL whole-body policy as a lab variant (server/rl_variant.py).

Pins: the variant loads NOVA's own deploy adapter + exported policy; NOVA's training model is
the lab model with the lift zero half a stroke up (head and hand mounts agree to < 0.1 mm with
that offset, not without); the policy engages and drives the lab robot with finite commands
inside NOVA's slews; upper-body mode holds the base; on NOVA's own sim2sim clip the lab plant
reproduces NOVA's sim2sim tracking (the integration is not where any difference comes from).
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from retarget_lab.server import config as CFG
from retarget_lab.server import rl_variant as RV
from retarget_lab.server import variants as V

def test_every_exported_policy_is_listed():
    """A policy pulled from NOVA shows up in the lab's list by itself (variants._rl_variants)."""
    files = sorted(f.name for f in (RV.WBC_RL / "checkpoints").glob("v*_policy.npz"))
    assert sorted(V.VARIANTS[k]["rl"] for k in V.RL_VARIANTS) == files
    for k in V.RL_VARIANTS:
        assert V.VARIANTS[k]["rl_wrist_ik"] == (V.VARIANTS[k]["rl_version"] >= V.RL_WRIST_IK_FROM)


needs_nova = pytest.mark.skipif(not (RV.WBC_RL / "checkpoints" / "v25_policy.npz").exists(),
                                reason="NOVA checkout (controller/wbc_rl) not next to MABEL")
CLIPS = sorted(CFG.EGODEX_CLIPS.glob("*.json.gz"))


@needs_nova
def test_nova_model_is_the_lab_model_with_a_lift_offset():
    import mujoco
    D, R = RV._import_deploy()
    fk = R.FK()
    m = mujoco.MjModel.from_xml_path(CFG.SCENE); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qa = {m.joint(j).name.lower().replace(" ", "_"): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    rng = np.random.default_rng(3)
    for off, ok in ((RV.LIFT_OFF, True), (0.0, False)):
        worst = 0.0
        for _ in range(20):
            q = {n: float(rng.uniform(*fk.J[fk.joint_child[n]]["lim"])) for n in R.ACTUATED}
            for n, v in q.items():
                d.qpos[qa[n]] = v + (off if n in R.LIFT else 0.0)
            mujoco.mj_kinematics(m, d)
            for b in ("head", "l_hand_mount", "r_hand_mount"):
                Ra, pa = fk.pose("mabel_chassis", q); _, pb = fk.pose(b, q)
                A, B = m.body("mabel_chassis").id, m.body(b).id
                e = np.linalg.norm(Ra.T @ (pb - pa) - d.xmat[A].reshape(3, 3).T @ (d.xpos[B] - d.xpos[A]))
                worst = max(worst, e)
        assert (worst < 1e-4) == ok, (off, worst)


@needs_nova
@pytest.mark.skipif(not CLIPS, reason="EgoDex clips not present")
@pytest.mark.parametrize("vid", V.RL_VARIANTS)
@pytest.mark.parametrize("scope", ["global", "local"])
def test_policy_engages_and_drives_the_lab_robot(scope, vid):
    from retarget_lab.server.worker import make_retargeter, pose_from_frame
    kr = make_retargeter(vid)
    assert type(kr).__name__ == "RLRetargeter"
    kr.apply_settings({"scope": scope, "robot": "nova"})
    pl = json.load(gzip.open(CLIPS[0]))["payloads"]
    recs = [kr.step(pose_from_frame({"kind": "avp", "p": p}), p["timestamp"], collide=False) for p in pl]
    assert recs[-1]["rl"]["engaged"] and recs[-1]["rl"]["upper"] == (scope == "local")
    names = sorted(recs[0]["q"]); q = np.array([[r["q"][n] for n in names] for r in recs])
    assert np.isfinite(q).all()
    arm = [i for i, n in enumerate(names) if "_arm_" in n]
    t = np.array([r["t"] for r in recs]); dt = np.maximum(np.diff(t), 1e-3)
    assert (np.abs(np.diff(q[:, arm], axis=0)) / dt[:, None]).max() <= 4.0 + 1e-6      # NOVA's fastest arm slew
    moved = np.abs(q[-1, arm] - q[0, arm]).max()
    assert moved > 0.05                                                                # it does retarget
    if scope == "local":
        assert abs(recs[-1]["base"]["x"]) < 1e-9 and abs(recs[-1]["base"]["y"]) < 1e-9
    assert any(r["anchored"] for r in recs) and not recs[0]["anchored"]                # glide excluded, then settled


@needs_nova
@pytest.mark.parametrize("vid", V.RL_VARIANTS)
def test_lab_plant_reproduces_novas_sim2sim_on_its_own_clip(vid):
    """Clip 2 (insert_remove_usb), EVERY exported policy: the lab's kinematic plant driven by the
    variant's own tick (RLRetargeter._tick: wrist solver per version) lands within a few cm of NOVA's
    sim2sim (MuJoCo, NOVA's control chain) run here on the same clip with no added latency."""
    import sys
    from retarget_lab.server.worker import make_retargeter
    kr = make_retargeter(vid)
    D, R = kr.D, kr.R
    sys.path.insert(0, str(RV.WBC_RL))
    import sim2sim as S2
    ref = S2.run(str(kr.policy_path), 2, "whole", 0, False, kr.wrist_ik)
    z = np.load(RV.WBC_RL / "checkpoints" / "clips.npz")
    op_all, arm_k = S2.load_clip(z, 2)
    kr.ad = D.Adapter(kr._rl_policy, upper=False, arm_k=arm_k)
    fr = lambda t: {k: v[t] for k, v in op_all.items()}
    kr._op = fr(0)
    kr.ad.engage(kr._op, kr._q20(), kr._odom())
    qprev = kr._q20(); eh, ew = [], []
    for t in range(len(op_all["head_p"])):
        q = kr._q20(); qd = (q - qprev) / D.DT; qprev = q
        kr._op = fr(t)
        tg = kr.ad.targets(kr._op, step=False)
        cmd = kr._tick(q, qd, np.zeros(3))
        for n, v in cmd["joints"].items():
            if n in kr._act:
                kr.data.ctrl[kr._act[n]] = v
        kr.data.ctrl[kr._lift_act] = 2 * (cmd["lift"] + RV.LIFT_OFF)
        vx, vy, w = cmd["twist"]; kr._apply_base((-vx, -vy, w), D.DT)
        kr._servo_ideal(D.DT); kr.mj.mj_kinematics(kr.model, kr.data)
        if t > 1.5 / D.DT:
            cp, _ = kr.ad.tracked(kr._q20()); x, y, yw = kr._odom(); Rb = D.rot_z(yw); o = np.array([x, y, R.CHASSIS_Z])
            cur = [Rb @ c + o for c in cp]
            eh.append(np.linalg.norm(cur[0] - tg["head_p"]))
            ew += [np.linalg.norm(cur[1] - tg["lw_p"]), np.linalg.norm(cur[2] - tg["rw_p"])]
    h, w = 100 * np.mean(eh), 100 * np.mean(ew)
    # The head is one-sided: the lab plant is an IDEAL servo, NOVA's sim2sim has NOVA's neck dynamics (since
    # NOVA 0d73319 its head error for v24/v25 reads 10-11 cm on this clip where the ideal plant gives 4-6 cm).
    # The wrists are the integration check (same targets, same arm commands): two-sided.
    assert h < ref["head_cm"] + 3.0 and abs(w - ref["wrist_cm"]) < 4.0, (vid, h, w, ref)
