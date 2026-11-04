"""
visualize.py
============

Rendering / plotting helpers:

* :func:`render_excitation` -- render the arm following a Fourier excitation
  trajectory to an animated GIF (offscreen, so it works headless).  This is the
  motion the identification "measures".
* :func:`plot_trajectory`  -- the commanded q / qdot / qddot vs time.
* :func:`plot_torque_fit`  -- measured vs predicted joint torque for the prior
  model and the identified model, showing the fit improvement.

NOTE on rendering: MuJoCo needs an OpenGL backend.  For headless/offscreen use
set ``MUJOCO_GL=osmesa`` (software) *before* importing mujoco.  The calling
script does this; :func:`render_excitation` degrades gracefully (returns None
with a message) if no backend is available.
"""
from __future__ import annotations

import numpy as np

from .arm_model import ArmModel
from . import trajectory as T


# --------------------------------------------------------------------------- #
#  MuJoCo excitation movie                                                     #
# --------------------------------------------------------------------------- #
def render_excitation(arm: ArmModel, out_path: str, *, duration: float = 10.0,
                      dt: float = 0.05, width: int = 480, height: int = 360,
                      seed: int = 0) -> str | None:
    """Render the arm tracking a Fourier trajectory to ``out_path`` (GIF).

    Returns the path on success, or None if rendering is unavailable.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    _, Q, _, _ = T.fourier_trajectory(arm, duration=duration, dt=dt, rng=rng)
    m, d = arm.model, arm.data

    try:
        renderer = mujoco.Renderer(m, height=height, width=width)
    except Exception as exc:  # no GL backend
        print(f"  [viz] skipping movie ({type(exc).__name__}: {exc}). "
              f"Try: MUJOCO_GL=osmesa")
        return None

    # auto-frame: sweep the trajectory, collect every body's world position,
    # and aim the camera at the centre of that swept volume.
    pts = []
    for q in Q[:: max(1, len(Q) // 40)]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        pts.append(d.xpos[1:].copy())          # skip world body
    pts = np.concatenate(pts, axis=0)
    center = 0.5 * (pts.max(0) + pts.min(0))
    radius = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.5

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = max(0.6, 2.4 * radius)
    cam.azimuth = 130.0
    cam.elevation = -15.0

    frames = []
    for q in Q:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()

    try:
        from PIL import Image
    except ImportError:
        print("  [viz] PIL not available; cannot write GIF.")
        return None

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(dt * 1000), loop=0, optimize=True)
    return out_path


# --------------------------------------------------------------------------- #
#  Trajectory and torque-fit plots                                             #
# --------------------------------------------------------------------------- #
def plot_trajectory(arm: ArmModel, out_path: str, *, duration: float = 10.0,
                    dt: float = 0.02, seed: int = 0) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    t, Q, QD, QDD = T.fourier_trajectory(arm, duration=duration, dt=dt, rng=rng)
    names = arm.link_names()

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for j in range(arm.nv):
        axes[0].plot(t, Q[:, j], lw=1, label=names[j])
        axes[1].plot(t, QD[:, j], lw=1)
        axes[2].plot(t, QDD[:, j], lw=1)
    axes[0].set_ylabel("q (rad)")
    axes[1].set_ylabel("q̇ (rad/s)")
    axes[2].set_ylabel("q̈ (rad/s²)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title("Fourier excitation trajectory (what the identification measures)")
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_torque_fit(truth: ArmModel, prior_arm: ArmModel, est_arm: ArmModel,
                    out_path: str, *, duration: float = 6.0, dt: float = 0.02,
                    seed: int = 5, n_joints: int = 4) -> str:
    """Measured (truth) vs prior vs identified torque over a fresh trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    t, Q, QD, QDD = T.fourier_trajectory(truth, duration=duration, dt=dt, rng=rng)
    tau_true = truth.batch_torque(Q, QD, QDD)
    tau_prior = prior_arm.batch_torque(Q, QD, QDD)
    tau_est = est_arm.batch_torque(Q, QD, QDD)
    names = truth.link_names()

    n = min(n_joints, truth.nv)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for j in range(n):
        ax = axes[j]
        ax.plot(t, tau_true[:, j], "k-", lw=2.0, label="truth (measured)")
        ax.plot(t, tau_prior[:, j], "--", color="#999999", lw=1.3,
                label="prior model")
        ax.plot(t, tau_est[:, j], "-", color="#d1495b", lw=1.3,
                label="identified")
        ax.set_ylabel(f"{names[j]}\nτ (N·m)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Joint torque: truth vs prior vs identified "
                      "(identified overlaps truth)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
