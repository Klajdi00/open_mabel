"""mabel_train.py — headless CLI over the training/eval experiment manager.

Thin wrapper around experiment.ExperimentManager so the whole train→resume→eval
lifecycle is scriptable without the GUI. Runs in the studio (mabel) env; spawns
the LeRobot env (MABEL_LEROBOT_PYTHON) for the actual training.

    python -m training_studio.server.mabel_train list
    python -m training_studio.server.mabel_train train --arch ACT --dataset <id> \
        --steps 20000 --batch 8 [--device cuda]
    python -m training_studio.server.mabel_train resume --run <run_id> [--steps N]
    python -m training_studio.server.mabel_train eval   --run <run_id> [--checkpoint P] [--episodes 1]
    python -m training_studio.server.mabel_train show   --run <run_id>

--dataset accepts a registry dataset_id (see `list-datasets`) OR an absolute
LeRobot dataset path.
"""
from __future__ import annotations
import argparse, json, os

from . import dataset_registry as DR
from .experiment import ExperimentManager


def _resolve_dataset(ds):
    if os.path.isdir(ds) and os.path.exists(os.path.join(ds, "meta", "info.json")):
        return os.path.basename(ds.rstrip("/")), os.path.abspath(ds)
    return ds, DR.find(ds)


def main():
    ap = argparse.ArgumentParser(prog="mabel_train")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--arch", required=True, help="ACT | DP | pi0 | pi05")
    p.add_argument("--dataset", required=True, help="dataset_id or path")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default=None)
    p.add_argument("--wait", action="store_true", help="block until done (else detach)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="raw LeRobot passthrough, e.g. --extra --policy.chunk_size=50")

    r = sub.add_parser("resume")
    r.add_argument("--run", required=True); r.add_argument("--steps", type=int, default=None)
    r.add_argument("--checkpoint", default=None); r.add_argument("--wait", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--run", required=True); e.add_argument("--checkpoint", default=None)
    e.add_argument("--episodes", type=int, default=1); e.add_argument("--kind", default="open_loop")

    s = sub.add_parser("show"); s.add_argument("--run", required=True)
    sub.add_parser("list")
    sub.add_parser("list-datasets")

    args = ap.parse_args()
    mgr = ExperimentManager()

    if args.cmd == "list":
        for r in mgr.list_runs():
            ms = r.get("metrics_summary") or {}
            print(f"{r['arch_slug']:6s} {r['run_id']:22s} {r['status']:8s} "
                  f"ckpts={r.get('n_checkpoints',0):<3d} "
                  f"loss↓={ms.get('loss_improvement_pct','—')}%  {r['dataset']}")
    elif args.cmd == "list-datasets":
        for d in DR.list_datasets()["datasets"]:
            print(f"{d.get('source','?'):9s} {d.get('dataset_id')}  ({d.get('num_episodes','?')} eps)")
    elif args.cmd == "train":
        ds_id, root = _resolve_dataset(args.dataset)
        hp = {"steps": args.steps, "batch": args.batch}
        if args.save_freq:
            hp["save_freq"] = args.save_freq
        res = mgr.train(args.arch, ds_id, root, repo_id=f"mabel/{ds_id}", hp=hp,
                        device=args.device, task=args.task,
                        extra=[a for a in args.extra if a != "--"], detach=not args.wait)
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "resume":
        print(json.dumps(mgr.resume(args.run, steps=args.steps, checkpoint=args.checkpoint,
                                    detach=not args.wait), indent=2, default=str))
    elif args.cmd == "eval":
        print(json.dumps(mgr.evaluate(args.run, checkpoint=args.checkpoint,
                                      kind=args.kind, episodes=args.episodes), indent=2, default=str))
    elif args.cmd == "show":
        print(json.dumps(mgr.get_run(args.run), indent=2, default=str))


if __name__ == "__main__":
    main()
