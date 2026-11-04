# policy_training — ACT, Diffusion Policy and π0.5 on a curated export

```text
curated LeRobot export ─▶ prepare (frame cache) ─▶ train act | dp | pi05 ─▶ eval (test split) ─▶ compare ─▶ deploy (train.Runtime)
```

**Requirements:** Python ≥ 3.10, torch + CUDA (MPS/CPU work for ACT/DP), `lerobot==0.4.4`, `av`, `opencv-python`, `pyarrow`,
`matplotlib`; π0.5 also needs `sentencepiece`, `bitsandbytes` and LeRobot's transformers fork. `bash setup_cuda_env.sh`
installs all of it into a conda env (`ENV_NAME`, default `policy-train`). Checkpoints and datasets are not included.

```bash
python train.py prepare --dataset <name|path> [--robot nova|mabel] [--report]   # decode videos once, cleaning report
python train.py train --policy act  --name act_v1  --set dataset=<name> [--robot mabel]
python train.py train --policy dp   --name dp_v1   --set dataset=<name> dp.lr=3e-4 steps=20000  # keys: train.py defaults
python train.py train --policy pi05 --name pi05_v1 --set dataset=<name>     # CUDA, 16 GB is enough
python train.py train --policy act --name act_v1 --resume --set steps=16000  # continue a run
python train.py eval --run runs/act_v1 [--ckpt best] [--split test]
python train.py compare runs/act_v1 runs/dp_v1 runs/pi05_v1 [--out DIR]
torchrun --standalone --nproc_per_node=4 train.py train --policy pi05 --name pi05_4g --set dataset=<name>  # batch_size is per GPU
sbatch --export=ALL,POLICY=pi05,NAME=pi05_v1,DATASET=<name>,VENV=/path/to/venv slurm_train.sh   # add your --account/--partition
python studio.py                      # web UI on http://127.0.0.1:8260 over the same runs/
```

Datasets: a path, or a name under `POLICY_DATASET_ROOT` (default `<repo>/curated_dataset`, the output of
`../../curate_export.py`). `--robot nova` is the bimanual NOVA configuration (head + two wrist cameras, named joints, 30 Hz);
`--robot mabel` takes every camera and unnamed columns. ACT/DP see 1 s of joint history and predict a 4 s chunk.

**π0.5** fine-tunes `lerobot/pi05_base` (PaliGemma 3B + 300M action expert, flow matching). The VLM and vision tower stay
frozen in bf16; the action expert and the action/time projections train on fp32 master weights with 8-bit AdamW. The
32-dim pretrained action layers are widened to the robot's action size. State is quantile-normalized and binned into the
prompt; chunk = 50 steps. `--set pi05.train_expert_only=false pi05.adam_8bit=false` is the full fine-tune (80 GB GPU).

**Outputs** `runs/<name>/`: `config.json split.json stats.json data_report.json metrics.jsonl status.json best.json`,
`plots/` (training curves, test horizon, rollouts), `eval/test.json`, `checkpoints/{best,last}/pretrained_model/`
(LeRobot `save_pretrained` + `nova_policy.json`, the I/O contract: cameras, history, chunk, names, normalization stats).
Touch `runs/<name>/STOP` to save and stop. `POLICY_RUNS_ROOT` and `POLICY_CACHE` move runs and the frame cache.

**Deploy:** `train.Runtime(run_dir)` is the per-step actor (obs → action, background inference, chunk crossfade). No
robot-side driver ships here; every command must go through the controller's safety gate. `start_pose.py` writes the
run's in-distribution start pose; `deploy_report.py` scores rollout logs against the demos.
