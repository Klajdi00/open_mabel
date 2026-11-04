"""Per-stage training benchmark for ACT on the resolved device — quantifies where
time goes (video decode → dataloader → forward → backward → optimizer), so the
training pipeline can be optimized. Run: python -m training_studio.server.tools.bench_train"""
import os
import random
import statistics as st
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def main():
    import torch
    from training_studio.server import feature_select as FS, trainer as TR, metrics_store as MS, compute
    MS.init_db()
    dev = torch.device(compute.resolve_device())

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    sel = FS.resolve("whole_body", cameras=["body_cam", "l_wrist_cam"])
    prep = TR.prepare_dataset(["mabel_test", "mabel_curated_test"], sel, (0.5, 0.5, 0.0), 1)
    v30 = prep["v30_root"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig

    meta = LeRobotDatasetMetadata(os.path.basename(v30), root=v30)
    ds = LeRobotDataset(os.path.basename(v30), root=v30, video_backend="pyav", tolerance_s=0.02)

    print(f"\n=== ACT training benchmark · device={dev} · {len(ds)} frames ===\n")

    idxs = [random.randrange(len(ds)) for _ in range(20)]
    tt = []
    for i in idxs:
        s = time.perf_counter(); _ = ds[i]; tt.append((time.perf_counter() - s) * 1000)
    print(f"[1] dataset[i] incl. video decode : {st.mean(tt):6.1f} ms/sample (min {min(tt):.0f}, max {max(tt):.0f})")

    from torch.utils.data import DataLoader
    for nw in (0, 4):
        dl = DataLoader(ds, batch_size=8, num_workers=nw, shuffle=True,
                        persistent_workers=nw > 0, pin_memory=True,
                        prefetch_factor=4 if nw > 0 else None)
        it = iter(dl); next(it)
        bt = []
        for _ in range(12):
            s = time.perf_counter(); next(it); bt.append((time.perf_counter() - s) * 1000)
        print(f"[2] dataloader batch=8 workers={nw:<2}: {st.mean(bt):6.0f} ms/batch = {st.mean(bt) / 8:.1f} ms/sample")
        del dl, it

    cfg = ACTConfig(chunk_size=50, n_action_steps=50, dim_model=256, n_heads=8,
                    dim_feedforward=1024, n_encoder_layers=4, n_decoder_layers=1)
    cfg.device = dev.type; cfg.push_to_hub = False
    dl = DataLoader(ds, batch_size=8, num_workers=0, shuffle=True)
    it = iter(dl)
    policy = make_policy(cfg, ds_meta=meta); policy.train(); policy.to(dev)
    pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    batch = pre(next(it)); batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for _ in range(3):
        loss, _ = policy.forward(batch); opt.zero_grad(); loss.backward(); opt.step()
    sync()
    ft, bt2, ot = [], [], []
    for _ in range(15):
        sync(); s = time.perf_counter(); loss, _ = policy.forward(batch); sync(); ft.append((time.perf_counter() - s) * 1000)
        opt.zero_grad(); s = time.perf_counter(); loss.backward(); sync(); bt2.append((time.perf_counter() - s) * 1000)
        s = time.perf_counter(); opt.step(); sync(); ot.append((time.perf_counter() - s) * 1000)
    comp = st.mean(ft) + st.mean(bt2) + st.mean(ot)
    npar = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[3] forward  : {st.mean(ft):6.0f} ms")
    print(f"[4] backward : {st.mean(bt2):6.0f} ms")
    print(f"[5] opt.step : {st.mean(ot):6.0f} ms")
    print(f"    ---------------------------------")
    print(f"    compute/iter ≈ {comp:.0f} ms  ({npar:.0f}M params)")
    print(f"    with 4 workers overlapping data, throughput ≈ {1000 / max(comp, 1):.1f} iter/s (compute-bound)")


if __name__ == "__main__":
    main()
