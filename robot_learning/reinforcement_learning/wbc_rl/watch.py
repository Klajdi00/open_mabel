"""watch.py — every checkpoint a training run saves gets scored and filmed, while training continues.

    cd robot_learning/reinforcement_learning/wbc_rl
    nohup $ISAACLAB_PY watch.py logs/rsl_rl/nova_wbc/<run> &        # until the run's last model
    $ISAACLAB_PY watch.py logs/rsl_rl/nova_wbc/<run> --final 30      # 30 videos of the last model
    $ISAACLAB_PY watch.py logs/rsl_rl/nova_wbc/<run> --compare v16   # the site's 3D comparison set

Per checkpoint model_N.pt: play.py eval (256 envs x 20 s) + one EgoDex and one LAFAN1 video (12 s) into
runs/<run>/ckpt/. --final: eval (512 envs x 30 s), a playback-speed sweep (0.5-2x), and many videos of the newest
checkpoint (alternating sources, new seeds), an upper-body-mode eval (base and lift locked) and the neck / twist / walk scenarios into runs/<run>/final/.
Every video comes with its 3D trajectory (<tag>_0.traj.bin, play.py) for the site's viewer.
"""
import argparse, glob, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
ENV.pop("DISPLAY", None)


def play(*args):
    r = subprocess.run([PY, os.path.join(HERE, "play.py"), *args], env=ENV, cwd=HERE, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
    return r.returncode


def it(p):
    return int(re.search(r"model_(\d+)\.pt", p).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--final", type=int, default=0)
    ap.add_argument("--max_iter", type=int, default=6000)
    ap.add_argument("--progress", nargs="*", default=[], help="with --final: checkpoints to re-film for the progression montage")
    ap.add_argument("--compare", default="", help="version tag: the standard 3D rollout set of the run's last checkpoint")
    a = ap.parse_args()
    if a.compare:                                    # same seeds for every policy: the site toggles policies on one motion
        ck = max(glob.glob(os.path.join(a.run, "model_*.pt")), key=it)
        d = os.path.join(HERE, "runs", "compare", a.compare)
        for src, n in (("egodex", 6), ("lafan", 2)):
            play("--checkpoint", ck, "--traj", "--videos", str(n), "--seconds", "12", "--source", src, "--seed", "11",
                 "--out", d, "--tag", src)
        play("--checkpoint", ck, "--traj", "--videos", "6", "--seconds", "12", "--source", "egodex", "--seed", "11",
             "--mode", "upper", "--out", d, "--tag", "upper")              # same EgoDex motions, base and lift locked
        for sc in ("neck", "twist", "walk"):
            play("--checkpoint", ck, "--traj", "--videos", "1", "--seconds", "12", "--scenario", sc, "--seed", "11",
                 "--out", d, "--tag", sc)
        json.dump({"version": a.compare, "checkpoint": os.path.relpath(ck, HERE)}, open(os.path.join(d, "policy.json"), "w"))
        return
    name = os.path.basename(a.run.rstrip("/"))
    out = os.path.join(HERE, "runs", name)
    if a.final:
        ck = max(glob.glob(os.path.join(a.run, "model_*.pt")), key=it)
        d = os.path.join(out, "final")
        play("--checkpoint", ck, "--videos", "0", "--eval_envs", "512", "--eval_seconds", "30", "--out", d, "--tag", "final")
        play("--checkpoint", ck, "--videos", "0", "--eval_envs", "512", "--eval_seconds", "30", "--out", d, "--tag", "upper",
             "--source", "egodex", "--mode", "upper")                       # upper-body mode: base and lift locked
        play("--checkpoint", ck, "--videos", "0", "--eval_envs", "512", "--eval_seconds", "30", "--out", d, "--tag", "faults",
             "--faults")                                                    # Vision Pro drops, occlusion, stalls, latency
        for r in (0.5, 1.0, 1.5, 2.0):                                    # how fast can the operator move?
            play("--checkpoint", ck, "--videos", "0", "--eval_envs", "256", "--eval_seconds", "20", "--out", d,
                 "--tag", f"speed_{r:.1f}", "--rate", str(r))
        for sc, sec in (("neck", 12), ("twist", 12), ("walk", 16)):          # body twist vs neck turn vs walking
            play("--checkpoint", ck, "--videos", "1", "--seconds", str(sec), "--scenario", sc, "--view", "front",
                 "--seed", "3", "--out", d, "--tag", f"scenario_{sc}")
        for ck_p in a.progress:                                           # one camera + seed across the lineage
            n = it(ck_p)
            play("--checkpoint", ck_p, "--videos", "1", "--seconds", "12", "--source", "lafan", "--seed", "7",
                 "--out", os.path.join(out, "progress"), "--tag", f"it{n:05d}")
        for k in range(a.final):
            src = ("egodex", "lafan", "mix")[k % 3]
            play("--checkpoint", ck, "--videos", "1", "--seconds", "15", "--out", d, "--tag", f"final_{k:02d}_{src}",
                 "--source", src, "--seed", str(100 + k))
        play("--checkpoint", ck, "--clean", "--videos", "1", "--seconds", "16", "--source", "egodex", "--rate", "1.0",
             "--width", "1600", "--height", "900", "--out", d, "--tag", "hero")         # the site's hero: no caption, no markers
        return
    done = set()
    while True:
        cks = sorted(glob.glob(os.path.join(a.run, "model_*.pt")), key=it)
        new = [c for c in cks if c not in done]
        for c in new[:-1] + new[-1:]:
            n = it(c)
            if n % 250 and n < a.max_iter - 2: done.add(c); continue           # every 250 + the last (rsl-rl ends on N-1 or N-2)
            d = os.path.join(out, "ckpt")
            tag = f"it{n:05d}"
            if not os.path.exists(os.path.join(d, f"{tag}_metrics.json")):
                play("--checkpoint", c, "--videos", "0", "--eval_envs", "256", "--eval_seconds", "20", "--out", d, "--tag", tag)
                for src in ("egodex", "lafan"):
                    play("--checkpoint", c, "--videos", "1", "--seconds", "12", "--out", d, "--tag", f"{tag}_{src}", "--source", src, "--seed", "7")
                print("scored", c, flush=True)
            done.add(c)
        if cks and it(cks[-1]) >= a.max_iter - 2 and not new:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
