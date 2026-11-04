"""
Launcher for a LeRobot training run under the Trainer Studio.

Run as:  python -m training_studio.server.lerobot_shim.run_train --config_path=<json>

It (1) ensures the studio's DiT plugin is registered, (2) monkeypatches
``lerobot.scripts.lerobot_train.WandBLogger`` with the StudioLogger (which writes
metrics.jsonl), then (3) hands off to LeRobot's own ``main()`` — draccus parses
``--config_path`` from argv exactly as a normal ``lerobot-train`` invocation.

The training config sets ``wandb.enable=True`` so LeRobot instantiates the
(patched) logger; nothing ever calls wandb.
"""
import os
import sys

# Belt-and-suspenders: these must be set before transformers imports (else a
# numpy-1.x TensorFlow crashes the whole lerobot.policies import under numpy 2.x).
os.environ.setdefault("USE_TF", "0")

# In-process decoded-frame cache (per worker). The curated datasets are small and
# repeat every epoch; re-decoding video each time is the #1 data-loading cost, so
# memoizing decoded frames turns data loading to ~0 after the first pass.
_FRAME_CACHE = {}
_FRAME_CACHE_CAP = 80000


def apply_decode_cache(worker_id=None):
    """Monkeypatch LeRobot's decode_video_frames with a memoizing wrapper. Applied
    in the main process (num_workers=0) and in each DataLoader worker via
    worker_init_fn (workers are separate processes with their own cache)."""
    try:
        import lerobot.datasets.video_utils as vu
        import lerobot.datasets.lerobot_dataset as ld
    except Exception:
        return
    if getattr(vu, "_studio_cached", False):
        return
    _orig = vu.decode_video_frames

    def _cached(video_path, timestamps, tolerance_s, backend):
        try:
            ts = timestamps.tolist() if hasattr(timestamps, "tolist") else list(timestamps)
            key = (str(video_path), tuple(round(float(t), 4) for t in ts))
        except Exception:
            return _orig(video_path, timestamps, tolerance_s, backend)
        c = _FRAME_CACHE.get(key)
        if c is not None:
            return c.clone() if hasattr(c, "clone") else c
        out = _orig(video_path, timestamps, tolerance_s, backend)
        if len(_FRAME_CACHE) < _FRAME_CACHE_CAP:
            _FRAME_CACHE[key] = out
        return out.clone() if hasattr(out, "clone") else out

    vu.decode_video_frames = _cached
    ld.decode_video_frames = _cached          # the dataset imported it by name
    vu._studio_cached = True

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _setup_validation(T):
    """If the run has an eval-split dataset (sidecar next to the config), compute a
    real validation loss (same loss metric as training) every ``val_freq`` steps and
    log it as ``eval_loss`` so the plot shows train + eval loss together. Fully
    guarded — any failure just disables the eval curve, never the training."""
    import json
    import os
    try:
        cfg_path = None
        for a in sys.argv:
            if a.startswith("--config_path"):
                cfg_path = a.split("=", 1)[1] if "=" in a else None
        if not cfg_path:
            i = sys.argv.index("--config_path") if "--config_path" in sys.argv else -1
            cfg_path = sys.argv[i + 1] if i >= 0 and i + 1 < len(sys.argv) else None
        side = (cfg_path or "") + ".studio.json"
        if not cfg_path or not os.path.isfile(side):
            return
        meta = json.load(open(side))
        eval_root, val_freq = meta.get("eval_root"), int(meta.get("val_freq", 10))
        if not eval_root or not os.path.isdir(eval_root):
            return
    except Exception:
        return

    state = {"pre": None, "loader": None, "it": None, "step": 0, "err": False}

    # capture the preprocessor LeRobot builds (needed to normalise val batches)
    if hasattr(T, "make_pre_post_processors"):
        _orig_mpp = T.make_pre_post_processors

        def _mpp(*a, **k):
            pre, post = _orig_mpp(*a, **k)
            state["pre"] = pre
            return pre, post
        T.make_pre_post_processors = _mpp

    _orig_update = T.update_policy

    def _update(train_metrics, policy, batch, *a, **k):
        train_metrics, output_dict = _orig_update(train_metrics, policy, batch, *a, **k)
        state["step"] += 1
        if state["err"] or state["pre"] is None or state["step"] % max(1, val_freq) != 0:
            return train_metrics, output_dict
        try:
            import torch
            if state["loader"] is None:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
                from lerobot.datasets.factory import resolve_delta_timestamps
                name = os.path.basename(eval_root.rstrip("/"))
                vmeta = LeRobotDatasetMetadata(name, root=eval_root)
                # SAME delta_timestamps as training so val batches are action CHUNKS
                delta_ts = resolve_delta_timestamps(policy.config, vmeta)
                ds = LeRobotDataset(name, root=eval_root, delta_timestamps=delta_ts,
                                    video_backend="pyav", tolerance_s=0.02)
                state["loader"] = torch.utils.data.DataLoader(
                    ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=0, drop_last=False)
                state["it"] = iter(state["loader"])
            try:
                vb = next(state["it"])
            except StopIteration:
                state["it"] = iter(state["loader"]); vb = next(state["it"])
            dev = next(policy.parameters()).device
            vb = state["pre"](vb)
            vb = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in vb.items()}
            # keep the policy in TRAIN mode (ACT's VAE encoder only runs then, and
            # this makes the val loss directly comparable to the train loss);
            # no_grad ensures the validation pass never updates weights.
            with torch.no_grad():
                out = policy.forward(vb)
                vloss = out[0] if isinstance(out, tuple) else out
            output_dict = dict(output_dict or {})
            output_dict["eval_loss"] = float(vloss.item())
        except Exception as e:
            state["err"] = True
            import traceback
            print(f"[trainer] eval-loss disabled: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return train_metrics, output_dict

    T.update_policy = _update


def main():
    # make `training_studio` importable when launched as a bare script
    here = os.path.dirname(os.path.abspath(__file__))
    learning = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if learning not in sys.path:
        sys.path.insert(0, learning)

    # 1) register the custom DiT policy so policy.type=dit resolves
    try:
        from training_studio.server.lerobot_shim import dit_policy  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[run_train] DiT plugin not registered: {e}", file=sys.stderr)

    # 2) patch the metric logger
    from training_studio.server.lerobot_shim.studio_logger import StudioLogger
    import lerobot.scripts.lerobot_train as T
    T.WandBLogger = StudioLogger

    # 3) training-speed patches (LeRobot's train script doesn't expose these):
    _optimize_training(T)

    # 3b) live validation-loss curve (train + eval loss on one plot)
    _setup_validation(T)

    # 4) hand off to LeRobot's draccus-wrapped entry point
    print("[trainer] launching lerobot-train — the first epoch warms the frame "
          "cache, then data loading drops to ~0 and it speeds up.", flush=True)
    T.main()


def _optimize_training(T):
    """Two throughput wins applied by monkeypatch:
    (a) DataLoader with persistent_workers + pin_memory — LeRobot respawns workers
        every epoch, which on small datasets stalls each epoch and re-imports torch
        in every worker; persistent workers remove that overhead.
    (b) bf16 mixed precision on CUDA — a large speedup on the 4090/5080 training
        hosts (LeRobot builds the Accelerator without mixed_precision). Left as
        fp32 on MPS/CPU where autocast is unreliable."""
    import torch
    import torch.utils.data as tud

    _OrigDL = tud.DataLoader

    # MUST be a subCLASS (not a function): Accelerate does isinstance(obj, DataLoader),
    # which requires DataLoader to remain a type.
    class _FastDataLoader(_OrigDL):
        def __init__(self, *args, **kwargs):
            if kwargs.get("num_workers", 0) and kwargs["num_workers"] > 0:
                kwargs.setdefault("persistent_workers", True)
                kwargs.setdefault("pin_memory", True)
                kwargs.setdefault("prefetch_factor", 6)
                kwargs.setdefault("worker_init_fn", apply_decode_cache)  # cache in each worker
            super().__init__(*args, **kwargs)

    tud.DataLoader = _FastDataLoader
    apply_decode_cache()   # also cache in the main process (num_workers=0 path)

    if hasattr(T, "Accelerator"):
        _OrigAcc = T.Accelerator

        def _Acc(*a, **k):
            try:
                if torch.cuda.is_available() and "mixed_precision" not in k:
                    k["mixed_precision"] = "bf16"
            except Exception:
                pass
            return _OrigAcc(*a, **k)

        T.Accelerator = _Acc


if __name__ == "__main__":
    main()
