"""train_act_minimal.py — a compact, dependency-light ACT trainer that consumes a
Data-Curation **LeRobot v2.1** export (the curated artifact the studio produces).

This is deliberately small (one file, torch only) so it can train on a CPU/Jetson
without the full LeRobot runtime — its job is to PROVE the collected → curated
data is trainable end to end, not to be a production trainer. It is an
Action-Chunking-Transformer core: an image + proprioceptive state are encoded and
a transformer decoder with learned queries regresses the next K actions (L1). The
CVAE latent of the full ACT paper is omitted for simplicity (deterministic ACT).

Reads:  <lerobot_dir>/meta/info.json, data/chunk-000/episode_*.parquet,
        videos/chunk-000/observation.images.<cam>/episode_*.mp4
Writes: <out>/act_ckpt.pt  +  <out>/loss_curve.csv  +  <out>/train_report.json
"""
from __future__ import annotations
import argparse, glob, json, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── dataset loading ──────────────────────────────────────────────────────────
def _decode_video(path, img_size):
    """Return list of HxWx3 uint8 frames (resized). cv2 first, imageio fallback."""
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(fr, (img_size, img_size)))
        cap.release()
        if frames:
            return frames
    except Exception as e:
        print("  cv2 decode failed:", e)
    import imageio.v3 as iio
    import cv2
    for fr in iio.imiter(path):
        frames.append(cv2.resize(np.asarray(fr)[..., :3], (img_size, img_size)))
    return frames


def load_lerobot(root, cam, img_size, chunk):
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    cams = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
    if cam not in cams:
        cam = cams[0]
    print(f"  camera='{cam}'  fps={info.get('fps')}  cams={cams}")
    parquets = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
    X_img, X_state, Y = [], [], []
    for pq in parquets:
        df = pd.read_parquet(pq)
        ep = int(df["episode_index"].iloc[0])
        vid = os.path.join(root, "videos", "chunk-000",
                           f"observation.images.{cam}", f"episode_{ep:06d}.mp4")
        frames = _decode_video(vid, img_size) if os.path.exists(vid) else []
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        T = len(df)
        for t in range(T - chunk):
            if frames:
                X_img.append(frames[min(t, len(frames) - 1)])
            X_state.append(state[t])
            Y.append(action[t:t + chunk])
    img = (np.stack(X_img).astype(np.float32) / 255.0).transpose(0, 3, 1, 2) if X_img else None
    return (torch.tensor(img) if img is not None else None,
            torch.tensor(np.stack(X_state), dtype=torch.float32),
            torch.tensor(np.stack(Y), dtype=torch.float32))


# ── model ────────────────────────────────────────────────────────────────────
class ACT(nn.Module):
    def __init__(self, state_dim, act_dim, chunk, d=256, use_img=True):
        super().__init__()
        self.chunk, self.use_img = chunk, use_img
        if use_img:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, d))
        self.state_enc = nn.Sequential(nn.Linear(state_dim, d), nn.ReLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 3)
        self.queries = nn.Parameter(torch.randn(chunk, d))
        self.head = nn.Linear(d, act_dim)

    def forward(self, img, state):
        toks = [self.state_enc(state).unsqueeze(1)]
        if self.use_img and img is not None:
            toks.append(self.cnn(img).unsqueeze(1))
        ctx = torch.cat(toks, dim=1)                       # (B, n_ctx, d)
        B = state.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)    # (B, chunk, d)
        seq = torch.cat([ctx, q], dim=1)
        out = self.tr(seq)[:, -self.chunk:]                # (B, chunk, d)
        return self.head(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot export dir (has meta/ data/ videos/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="zed_left")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-img", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    print("loading curated LeRobot dataset:", args.dataset)
    img, state, Y = load_lerobot(args.dataset, args.cam, args.img_size, args.chunk)
    use_img = (img is not None) and not args.no_img
    N = state.shape[0]
    print(f"  samples={N}  state_dim={state.shape[1]}  act_dim={Y.shape[2]}  chunk={args.chunk}  images={use_img}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACT(state.shape[1], Y.shape[2], args.chunk, use_img=use_img).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  device={dev}  params={n_params/1e6:.2f}M")

    # 90/10 train/val split
    idx = torch.randperm(N); ntr = int(0.9 * N)
    tr, va = idx[:ntr], idx[ntr:]
    curve = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = tr[torch.randint(0, len(tr), (args.batch,))]
        im = img[b].to(dev) if use_img else None
        pred = model(im, state[b].to(dev))
        loss = nn.functional.l1_loss(pred, Y[b].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                vb = va[torch.randint(0, len(va), (min(args.batch, len(va)),))]
                vim = img[vb].to(dev) if use_img else None
                vloss = nn.functional.l1_loss(model(vim, state[vb].to(dev)), Y[vb].to(dev)).item()
            curve.append((step, loss.item(), vloss))
            print(f"  step {step:4d}/{args.steps}  train_L1={loss.item():.4f}  val_L1={vloss:.4f}")

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.out, "act_ckpt.pt"))
    with open(os.path.join(args.out, "loss_curve.csv"), "w") as f:
        f.write("step,train_l1,val_l1\n")
        for s, tl, vl in curve:
            f.write(f"{s},{tl:.5f},{vl:.5f}\n")
    report = {"dataset": args.dataset, "samples": int(N), "params_M": round(n_params/1e6, 3),
              "device": dev, "images": bool(use_img), "chunk": args.chunk,
              "first_loss": curve[0][1], "last_train_l1": curve[-1][1],
              "last_val_l1": curve[-1][2],
              "improved_pct": round(100*(1 - curve[-1][1]/curve[0][1]), 1),
              "train_seconds": round(time.time()-t0, 1)}
    json.dump(report, open(os.path.join(args.out, "train_report.json"), "w"), indent=2)
    print("REPORT:", json.dumps(report, indent=2))
    print("saved:", os.path.join(args.out, "act_ckpt.pt"))


if __name__ == "__main__":
    main()
