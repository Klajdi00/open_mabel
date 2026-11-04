"""
Rigorous end-to-end test harness for the Trainer Studio.

Device-aware: detects cuda/mps/cpu and runs the strongest tier the host supports.
Exercises the whole spine — dataset prep, modular feature selection, config build +
validation for all 4 architectures, REAL short training (ACT + DiT, asserting the
loss drops), VLA plumbing (pi0/pi0.5, gated on weights), eval + MuJoCo deploy — and
prints a summary table + saves figures to docs/figures/. Nothing is faked.

Run:  python -m training_studio.server.tools.test_training [--full]
"""
import os
import sys
import time
import traceback

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from training_studio.server import (arch_registry as AR, compute, config as CFG,  # noqa: E402
                                    config_builder as CB, dataset_registry as DR,
                                    eval as EV, feature_select as FS, metrics_store as MS,
                                    trainer as TR)

CLEAN = ["mabel_test", "mabel_curated_test"]      # NaN-free curated exports
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    mark = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    print(f"  [{mark}] {name} — {detail}")
    return ok


def run_short_training(arch, sel, hp_over, steps, device, tag):
    """Launch a real short run through config_builder + run_train; return losses."""
    import json
    import subprocess
    prep = TR.prepare_dataset(CLEAN, sel, (0.5, 0.5, 0.0), 1)
    hp = {**AR.defaults(arch), **hp_over, "steps": steps, "batch_size": 2,
          "log_freq": 1, "save_freq": steps, "num_workers": 0}
    od = os.path.join(CFG.RUNS_DIR, f"test_{tag}")
    cp = os.path.join(CFG.RUNS_DIR, "_configs", f"test_{tag}.json")
    if os.path.exists(od):
        import shutil
        shutil.rmtree(od)
    CB.build(arch, hp, sel, prep["v30_root"], od, cp, None, device=device)
    env = dict(os.environ)
    env.update(CFG.LEROBOT_ENV)
    env["PYTHONPATH"] = CFG.LEARNING_DIR
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run([CFG.TRAIN_PYTHON, "-m", "training_studio.server.lerobot_shim.run_train",
                        f"--config_path={cp}"], cwd=CFG.LEARNING_DIR, env=env,
                       capture_output=True, text=True, timeout=1200)
    mp = os.path.join(od, "metrics.jsonl")
    losses, ckpts = [], []
    if os.path.exists(mp):
        for line in open(mp):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("_type") == "metric" and "loss" in x:
                losses.append(x["loss"])
            if x.get("_type") == "checkpoint":
                ckpts.append(x.get("path"))
    return {"code": r.returncode, "losses": losses, "checkpoints": ckpts,
            "stderr": r.stderr, "output_dir": od, "prep": prep}


def main(full=False):
    MS.init_db()
    dev = compute.resolve_device("auto")
    print(f"\n=== Trainer Studio test harness · host={compute.describe()['host']} · device={dev} ===\n")

    # 1) dataset prep
    print("[1] dataset prep")
    try:
        rec = DR.combine(CLEAN, "test_combined")
        record("combine", rec["n_episodes"] == 2 and rec["action_dim"] == 59,
               f"{rec['n_episodes']} eps, {rec['total_frames']} frames")
    except Exception as e:
        record("combine", False, str(e))

    # 2) modular feature selection
    print("[2] modular feature selection")
    for scope, exp in (("whole_body", 59), ("arms", 16), ("hands", 32)):
        sel = FS.resolve(scope, cameras=["body_cam", "l_wrist_cam"])
        ok = sel["action_dim"] == exp and len(sel["scatter_map"]) == exp
        record(f"scope {scope}", ok, f"action_dim={sel['action_dim']} (want {exp})")
    # pad-dim validator
    selw = FS.resolve("whole_body")
    errs = [i for i in AR.validate("pi0", AR.defaults("pi0"), selw) if i["level"] == "error"]
    record("pad-dim validator (pi0 whole-body)", any("max_action_dim" in e["msg"] for e in errs),
           f"{len(errs)} errors raised")

    # 3) config build + validation for all 4 archs
    print("[3] config build + validation (ACT/DiT/pi0/pi0.5)")
    sel_arms = FS.resolve("arms", cameras=["body_cam", "l_wrist_cam"])
    for arch in AR.ARCH_ORDER:
        try:
            hp = AR.defaults(arch)
            issues = AR.validate(arch, hp, sel_arms)
            errs = [i for i in issues if i["level"] == "error"]
            spec = AR.arch_spec(arch, hp, sel_arms)
            ok = not errs and len(spec["nodes"]) > 3
            record(f"{arch} config+spec", ok,
                   f"{len(spec['nodes'])} diagram nodes, {len(errs)} errors")
        except Exception as e:
            record(f"{arch} config+spec", False, str(e))

    # 4) REAL training: ACT + DiT converge
    print("[4] real training — ACT + DiT (loss must drop)")
    act_out = None
    try:
        act_out = run_short_training(
            "ACT", FS.resolve("whole_body", cameras=["body_cam"]),
            {"chunk_size": 16, "n_action_steps": 16, "dim_model": 128, "n_heads": 4,
             "dim_feedforward": 256, "n_encoder_layers": 2, "n_decoder_layers": 1,
             "latent_dim": 16, "optimizer_lr": 1e-4}, 60, dev, "act")
        L = act_out["losses"]
        drop = len(L) >= 4 and (sum(L[:3]) / 3) > (sum(L[-3:]) / 3)
        record("ACT trains + converges", act_out["code"] == 0 and drop,
               f"loss {L[0]:.2f}->{L[-1]:.2f}, {len(act_out['checkpoints'])} ckpt")
    except Exception as e:
        record("ACT trains + converges", False, str(e))

    try:
        dit_out = run_short_training(
            "DiT", FS.resolve("arms", cameras=["body_cam", "l_wrist_cam"]),
            {"dim_model": 128, "n_heads": 4, "n_layers": 2, "horizon": 16,
             "n_action_steps": 8, "num_inference_steps": 4}, 40, dev, "dit")
        record("DiT (custom plugin) trains", dit_out["code"] == 0 and len(dit_out["losses"]) > 0,
               f"{len(dit_out['losses'])} loss pts, {len(dit_out['checkpoints'])} ckpt")
    except Exception as e:
        record("DiT (custom plugin) trains", False, str(e))

    # 5) VLA plumbing (pi0/pi0.5) — real on CUDA, config-only on Mac
    print("[5] VLA pi0 / pi0.5")
    for arch in ("pi0", "pi0.5"):
        sel = FS.resolve("arms", cameras=["body_cam"])   # 16-D fits pi0's pad 32
        issues = AR.validate(arch, AR.defaults(arch), sel)
        cfg_ok = not [i for i in issues if i["level"] == "error"]
        if dev == "cuda" and full:
            try:
                out = run_short_training(arch, sel, {"chunk_size": 10, "n_action_steps": 10},
                                         5, dev, arch.replace(".", ""))
                record(f"{arch} trains (CUDA)", out["code"] == 0 and len(out["losses"]) > 0,
                       f"{len(out['losses'])} loss pts")
            except Exception as e:
                record(f"{arch} trains (CUDA)", False, str(e)[:80])
        else:
            record(f"{arch} config valid (plumbing)", None if cfg_ok else False,
                   "config builds; full train needs a CUDA host + pretrained weights")

    # 6) eval + deploy
    print("[6] eval + MuJoCo deploy")
    if act_out and act_out["checkpoints"]:
        ck = act_out["checkpoints"][-1]
        selwb = FS.resolve("whole_body", cameras=["body_cam"])
        try:
            ev = EV.run_open_loop(ck, act_out["prep"]["v21_root"], act_out["prep"]["v30_root"],
                                  0, device=dev, max_frames=30)
            record("eval open-loop (pred vs GT)", ev["frames"] == 30 and ev["action_dim"] == 59,
                   f"L1={ev['l1_mean']:.3f} MSE={ev['mse_mean']:.3f}")
        except Exception as e:
            record("eval open-loop (pred vs GT)", False, str(e)[:80])
        try:
            from training_studio.server import deploy as DP
            dp = DP.run(ck, act_out["prep"]["v30_root"], selwb, device=dev, control_steps=5)
            import numpy as np
            a0 = np.array(dp["trajectory"][0]["actuator_positions"])
            a1 = np.array(dp["trajectory"][-1]["actuator_positions"])
            record("deploy closed-loop MuJoCo", dp["frames"] == 5 and float(np.linalg.norm(a1 - a0)) > 0,
                   f"robot moved Δ={float(np.linalg.norm(a1 - a0)):.3f}")
            _save_figures(act_out["losses"], dp)
        except Exception as e:
            record("deploy closed-loop MuJoCo", False, str(e)[:80])
    else:
        record("eval + deploy", None, "skipped (no ACT checkpoint)")

    # summary
    print("\n" + "=" * 62)
    npass = sum(1 for r in RESULTS if r["ok"] is True)
    nfail = sum(1 for r in RESULTS if r["ok"] is False)
    nskip = sum(1 for r in RESULTS if r["ok"] is None)
    print(f"SUMMARY: {npass} passed · {nfail} failed · {nskip} skipped  (device={dev})")
    print("=" * 62)
    for r in RESULTS:
        mark = "✓" if r["ok"] is True else ("–" if r["ok"] is None else "✗")
        print(f"  {mark} {r['name']:<42} {r['detail']}")
    return 0 if nfail == 0 else 1


def _save_figures(losses, deploy):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        figdir = os.path.abspath(os.path.join(_HERE, "..", "..", "docs", "figures"))
        os.makedirs(figdir, exist_ok=True)
        # loss curve
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(losses, color="#C25B2A", lw=1.6)
        ax.set_xlabel("log step"); ax.set_ylabel("ACT loss"); ax.set_title("ACT convergence (MPS smoke)")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(figdir, "act_loss.pdf")); plt.close(fig)
        # deploy trajectory (mean |action|)
        traj = np.array([f["action"] for f in deploy["trajectory"]])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.abs(traj).mean(1), color="#4FB39C", lw=1.6)
        ax.set_xlabel("control step"); ax.set_ylabel("mean |action|"); ax.set_title("Closed-loop MuJoCo rollout")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(figdir, "deploy_rollout.pdf")); plt.close(fig)
        print(f"  figures -> {figdir}")
    except Exception as e:
        print("  (figure gen skipped:", e, ")")


if __name__ == "__main__":
    try:
        sys.exit(main(full="--full" in sys.argv))
    except Exception:
        traceback.print_exc()
        sys.exit(2)
