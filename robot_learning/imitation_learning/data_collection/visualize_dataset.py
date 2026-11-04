"""
Dataset diagnostics — regenerate vector-PDF figures for a recorded episode (or a
whole dataset folder) into docs/figures/, per the lab-notebook workflow.

Figures (headless, matplotlib Agg, saved as PDF):
  * <stem>_joints.pdf  — measured vs. target joint angles for the arms + neck,
                         lift/torso, with the per-actuator command (data.ctrl)
                         overlaid so tracking error is visible.
  * <stem>_ee.pdf      — left/right palm + target palm 3D paths and z-height.
  * <stem>_base.pdf    — base twist (vx,vy,wz) and lidar min-range over time.

Usage
  python visualize_dataset.py datasets/pick_cube_v1/episode_0000.h5
  python visualize_dataset.py datasets/pick_cube_v1          # all episodes
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import h5py                       # noqa: E402
import numpy as np                # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_HERE, "docs", "figures")


def _schema(f):
    return json.loads(f.attrs["schema_json"]) if "schema_json" in f.attrs else {}


def _idx(names, group):
    """Indices of a named actuator group within the full actuator list."""
    return [names.index(n) for n in group if n in names]


def plot_episode(path):
    os.makedirs(FIG_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        schema = _schema(f)
        names = schema.get("actuator_names", [])
        groups = schema.get("actuator_groups", {})
        t = f["time/sim"][:] if "time/sim" in f else np.arange(len(f["action/ctrl"]))
        t = t - t[0]
        ctrl = f["action/ctrl"][:]          # [T, nu] target
        jpos = f["obs/joint_pos"][:] if "obs/joint_pos" in f else None

        # ── 1. joint tracking (arms + neck + lift/torso) ─────────────────────
        sel = [("left_arm", groups.get("left_arm", [])),
               ("right_arm", groups.get("right_arm", [])),
               ("neck", groups.get("neck", [])),
               ("lift_torso", groups.get("lift_torso", []))]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (label, grp) in zip(axes.flat, sel):
            ii = _idx(names, grp)
            for k, idx in enumerate(ii):
                c = plt.cm.tab10(k % 10)
                ax.plot(t, ctrl[:, idx], "--", color=c, lw=1.0, alpha=0.9)
                if jpos is not None:
                    ax.plot(t, jpos[:, idx], "-", color=c, lw=1.4,
                            label=grp[k].replace(label.split("_")[0].title() + " ", ""))
            ax.set_title(f"{label}  (— measured, -- target)")
            ax.set_xlabel("time (s)"); ax.set_ylabel("angle (rad)")
            ax.grid(alpha=0.3)
            if ii:
                ax.legend(fontsize=6, ncol=2, loc="best")
        fig.suptitle(f"{stem} — joint tracking")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{stem}_joints.pdf"))
        plt.close(fig)

        # ── 2. end-effector paths ────────────────────────────────────────────
        if "obs/pose/left_palm" in f:
            lp = f["obs/pose/left_palm"][:][:, :3]
            rp = f["obs/pose/right_palm"][:][:, :3]
            tl = f["teleop/left_palm_target"][:][:, :3] if "teleop/left_palm_target" in f else None
            tr = f["teleop/right_palm_target"][:][:, :3] if "teleop/right_palm_target" in f else None
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(121, projection="3d")
            ax.plot(*lp.T, color="tab:blue", label="left palm")
            ax.plot(*rp.T, color="tab:red", label="right palm")
            if tl is not None: ax.plot(*tl.T, "--", color="tab:cyan", lw=1, label="left target")
            if tr is not None: ax.plot(*tr.T, "--", color="tab:orange", lw=1, label="right target")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.legend(fontsize=7)
            ax.set_title("palm paths (world)")
            ax2 = fig.add_subplot(122)
            ax2.plot(t, lp[:, 2], color="tab:blue", label="left z")
            ax2.plot(t, rp[:, 2], color="tab:red", label="right z")
            ax2.set_xlabel("time (s)"); ax2.set_ylabel("height (m)")
            ax2.grid(alpha=0.3); ax2.legend(fontsize=7); ax2.set_title("palm height")
            fig.suptitle(f"{stem} — end-effectors")
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, f"{stem}_ee.pdf"))
            plt.close(fig)

        # ── 3. base twist + lidar ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "obs/base_twist" in f:
            bt = f["obs/base_twist"][:]
            for k, lab in enumerate(["vx", "vy", "wz"]):
                axes[0].plot(t, bt[:, k], label=lab)
            axes[0].set_title("base twist (world)"); axes[0].set_xlabel("time (s)")
            axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
        if "obs/lidar" in f:
            lid = f["obs/lidar"][:]
            valid = np.where(lid > 0, lid, np.nan)
            axes[1].plot(t, np.nanmin(valid, axis=1), color="tab:purple")
            axes[1].set_title("LiDAR nearest range"); axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("range (m)"); axes[1].grid(alpha=0.3)
        fig.suptitle(f"{stem} — base & lidar")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{stem}_base.pdf"))
        plt.close(fig)
    print(f"[viz] wrote {stem}_{{joints,ee,base}}.pdf -> {FIG_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".h5 episode or a dataset folder")
    args = ap.parse_args()
    files = ([args.path] if args.path.endswith(".h5")
             else sorted(glob.glob(os.path.join(args.path, "*.h5"))))
    if not files:
        print("no .h5 files found."); return
    for p in files:
        plot_episode(p)


if __name__ == "__main__":
    main()
