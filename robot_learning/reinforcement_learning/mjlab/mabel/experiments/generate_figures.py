"""Reproducible Phase-0/1 validation figures for the MABEL manipulation env.

Regenerates every figure in ``mabel/docs/figures`` from scratch and prints a
metrics summary table. Fixed seed; no hand-tuned numbers -- everything is
measured from the compiled MuJoCo model / mjlab Entity.

    uv run python mabel/experiments/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.mabel import get_mabel_robot_cfg
from mjlab.asset_zoo.robots.mabel.derive_model import (
  ARM_JOINTS,
  build_manip_spec,
  hand_joints,
)
from mjlab.asset_zoo.robots.mabel.mabel_constants import (
  MABEL_ACTION_SCALE,
  RIGHT_HOME_JOINT_POS,
)
from mjlab.entity.entity import Entity
from mjlab.tasks.manipulation.config.mabel.env_cfgs import (
  CUBE_SIZE,
  TABLE_CENTER,
  TABLE_TOP_Z,
)

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 0
rng = np.random.default_rng(SEED)


def _site(m, d, name):
  return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def reach_envelope():
  """Sample the right-arm grasp-site workspace; overlay table + cube box."""
  spec = build_manip_spec(sides=("right",))
  m = spec.compile()
  d = mujoco.MjData(m)
  arm = ARM_JOINTS["right"]
  jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm]
  qadr = [m.jnt_qposadr[i] for i in jids]
  lo = np.array([m.jnt_range[i][0] for i in jids])
  hi = np.array([m.jnt_range[i][1] for i in jids])
  n = 6000
  pts = np.empty((n, 3))
  for k in range(n):
    q = rng.uniform(lo, hi)
    for a, v in zip(qadr, q, strict=True):
      d.qpos[a] = v
    mujoco.mj_forward(m, d)
    pts[k] = _site(m, d, "right_grasp_site")

  # home hand position
  for name, v in RIGHT_HOME_JOINT_POS.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
  mujoco.mj_forward(m, d)
  home = _site(m, d, "right_grasp_site")

  cx, cy = TABLE_CENTER
  cube_box = dict(x=(cx - 0.10, cx + 0.10), y=(cy - 0.10, cy + 0.10))
  panels = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
  for ax, (a, b, ia, ib) in zip(axes, panels, strict=True):
    ax.scatter(
      pts[:, ia], pts[:, ib], s=2, alpha=0.15, color="#4C72B0", rasterized=True
    )
    ax.scatter(
      [home[ia]], [home[ib]], s=90, color="#C44E52", marker="*", zorder=5, label="home"
    )
    if a == "x" and b == "y":
      ax.add_patch(
        plt.Rectangle(
          (cube_box["x"][0], cube_box["y"][0]),
          0.20,
          0.20,
          fill=False,
          ec="#DD8452",
          lw=2,
          label="cube spawn",
        )
      )
    if b == "z":
      ax.axhline(TABLE_TOP_Z, color="#55A868", lw=2, ls="--", label="table top")
    ax.set_xlabel(f"{a} [m]")
    ax.set_ylabel(f"{b} [m]")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
  axes[0].legend(loc="upper left", fontsize=8)
  fig.suptitle("Right arm grasp-site reachable workspace (n=6000 joint samples)")
  fig.tight_layout()
  fig.savefig(FIG / "reach_envelope.pdf", dpi=150)
  plt.close(fig)
  return dict(
    reach_x=(pts[:, 0].min(), pts[:, 0].max()),
    reach_y=(pts[:, 1].min(), pts[:, 1].max()),
    reach_z=(pts[:, 2].min(), pts[:, 2].max()),
    home_hand=home,
  )


def home_stability():
  """Hold the home pose under gravity with mjlab actuators; log joint drift."""
  robot = Entity(get_mabel_robot_cfg(sides=("right",)))
  m = robot.spec.compile()
  d = mujoco.MjData(m)
  jp = {k: v for k, v in RIGHT_HOME_JOINT_POS.items()}
  for name, v in jp.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
  mujoco.mj_forward(m, d)
  q_home = d.qpos.copy()
  for i in range(m.nu):
    d.ctrl[i] = q_home[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
  T = 750
  drift = np.empty((T, m.nq))
  for t in range(T):
    mujoco.mj_step(m, d)
    drift[t] = np.abs(d.qpos - q_home)
  ts = np.arange(T) * m.opt.timestep

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(ts, np.degrees(drift.max(axis=1)), label="max joint")
  ax.plot(ts, np.degrees(drift.mean(axis=1)), label="mean joint")
  ax.set_xlabel("time [s]")
  ax.set_ylabel("|drift from home| [deg]")
  ax.set_title("Home-pose hold under gravity (kp=100 arm, kp=2 hand)")
  ax.legend()
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(FIG / "home_stability.pdf")
  plt.close(fig)
  return dict(
    max_drift_deg=float(np.degrees(drift[-1].max())),
    mean_drift_deg=float(np.degrees(drift[-1].mean())),
    nan=bool(np.isnan(d.qpos).any()),
  )


def action_scale_fig():
  names = list(MABEL_ACTION_SCALE.keys())
  vals = np.array([MABEL_ACTION_SCALE[n] for n in names])
  fig, ax = plt.subplots(figsize=(10, 3.6))
  colors = ["#4C72B0" if "Arm" in n else "#DD8452" for n in names]
  ax.bar(range(len(names)), vals, color=colors)
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(
    [n.replace("Right ", "") for n in names], rotation=60, ha="right", fontsize=7
  )
  ax.set_ylabel("action scale [rad]")
  ax.set_title(
    "Per-actuator action scale (0.25 * effort / kp): arm (blue) vs hand (orange)"
  )
  fig.tight_layout()
  fig.savefig(FIG / "action_scale.pdf")
  plt.close(fig)


def main():
  print("Generating MABEL manipulation validation figures (seed=%d) ..." % SEED)
  reach = reach_envelope()
  stab = home_stability()
  action_scale_fig()

  spec = build_manip_spec(sides=("right",))
  m = spec.compile()
  print("\n===== METRICS =====")
  print(f"{'DOF (nq/nv)':30s}: {m.nq} / {m.nv}")
  print(f"{'action dim':30s}: {len(MABEL_ACTION_SCALE)}")
  print(
    f"{'arm / hand joints':30s}: {len(ARM_JOINTS['right'])} / {len(hand_joints('right'))}"
  )
  print(
    f"{'reach x [m]':30s}: [{reach['reach_x'][0]:+.3f}, {reach['reach_x'][1]:+.3f}]"
  )
  print(
    f"{'reach y [m]':30s}: [{reach['reach_y'][0]:+.3f}, {reach['reach_y'][1]:+.3f}]"
  )
  print(
    f"{'reach z [m]':30s}: [{reach['reach_z'][0]:+.3f}, {reach['reach_z'][1]:+.3f}]"
  )
  print(f"{'home hand xyz [m]':30s}: {np.round(reach['home_hand'], 3).tolist()}")
  print(f"{'table top z [m]':30s}: {TABLE_TOP_Z}")
  print(f"{'cube size [m]':30s}: {CUBE_SIZE}")
  print(f"{'home hold max drift [deg]':30s}: {stab['max_drift_deg']:.2f}")
  print(f"{'home hold mean drift [deg]':30s}: {stab['mean_drift_deg']:.2f}")
  print(f"{'home hold NaN':30s}: {stab['nan']}")
  print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
  main()
