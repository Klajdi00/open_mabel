# Learning pipeline: collect, curate, train, evaluate

```text
robot sensors ─▶ episode_XXXX.h5 ─▶ LeRobot dataset ─▶ checkpoints + plots ─▶ action error
 (collector)      (real/…)            (curated_dataset/…)  (robot_learning/training/…)  (robot_learning/eval/…)
```

All commands below are run from the repository root unless a `cd` is shown. Paths such as
`real/` and `curated_dataset/` are created at the repository root on first use.

## Environments

Two Python environments are kept separate on purpose:

| Environment | Used for | Contents |
|---|---|---|
| `mabel` (conda) | collection, curation, both studio servers | numpy, h5py, mujoco, FastAPI; no torch |
| LeRobot environment (any venv/conda) | training and evaluation only | torch (CUDA recommended) + LeRobot |

The training CLI and the studio's Experiments tab launch training in the LeRobot environment
through `MABEL_LEROBOT_PYTHON`:

```bash
export MABEL_LEROBOT_PYTHON=/path/to/lerobot-env/bin/python
```

## 1. Collect

Real-robot collection writes curation-format HDF5 through `server/real_curation_collector.py`
(sensing bring-up and the full channel list: [`server/docs/data_collection.md`](../server/docs/data_collection.md)).

```bash
python3 server/real_curation_collector.py --client visionos --control-port 9095
```

The Vision Pro app's "Start collecting data" control connects to `ws://<robot-host>:9095`.
A dry run without an operator or moving hardware:

```bash
python3 server/tools/mock_control_publisher.py &
python3 server/tools/mock_vp_collect.py --client visionos --task pick_place --duration 60 --count 10
```

Episodes land in `real/{vision_os,ios,real}/<task>_<NNN>_<date>/episode_XXXX.h5` (JPEG RGB,
uint16 depth, all proprioceptive and command channels at 15 Hz by default; roughly 38 MB per
minute). `--root` (or `MABEL_REAL_ROOT`) moves the output folder.

Simulated, ROS and scripted collection: [`imitation_learning/data_collection/`](imitation_learning/data_collection/).

## 2. Curate

`curate_export.py` runs the curation quality scan over a set of sessions and writes a
LeRobot v2.1 dataset plus a per-episode quality report:

```bash
cd learning
HDF5_USE_FILE_LOCKING=FALSE python3 curate_export.py \
    --sessions '../real/vision_os/pick_place_*' --out-name pick_place --realm real
```

Output: `curated_dataset/<realm>/<name>_<NNN>_<date>/` containing `mabel_<name>/` (the LeRobot
dataset) and `curation_report.json`. `--curated-root` changes the destination.

For interactive review and non-destructive editing, open the Data Curation studio
([`data_curation/`](data_curation/)) on the same sessions:

```bash
cd learning
MABEL_DATASET_ROOT=../real/vision_os python3 -m data_curation.server     # http://127.0.0.1:8093
```

### LeRobot v3.0

Current LeRobot releases train on dataset format v3.0. The training manager builds a v3.0
copy automatically when it needs one; to build it explicitly (run in the LeRobot
environment):

```bash
cd learning
$MABEL_LEROBOT_PYTHON h5_to_lerobot_v30.py \
    --sessions '../real/vision_os/pick_place_*' --repo-id mabel/pick_place \
    --out training_studio/workspace/datasets/.v30/mabel_pick_place
```

Options: `--fps` (default 10), `--cams` (comma list to keep), `--limit`, `--task`.

## 3. Train

The headless CLI and the studio's Experiments tab share one manager
(`training_studio/server/experiment.py`) and write to the same tree. The CLI itself runs
in the `mabel` environment and launches the LeRobot environment as a subprocess.

```bash
cd learning
python3 -m training_studio.server.mabel_train list-datasets
python3 -m training_studio.server.mabel_train train --arch ACT \
    --dataset pick_place_001_<date> --steps 20000 --batch 8 --device cuda
```

`--arch` is one of `ACT`, `DP` (LeRobot Diffusion Policy), `pi0`, `pi05`. Other options:
`--save-freq`, `--task`, `--wait` (block until finished), `--extra ...` (passed through to
LeRobot).

π0 and π0.5 load the gated `google/paligemma-3b-pt-224` backbone. Log in to Hugging Face
from the LeRobot environment and accept that model's license once before training them.

### CLI pipeline with data cleaning (ACT / DP / π0.5)

[`imitation_learning/policy_training/`](imitation_learning/policy_training/README.md) trains directly in the LeRobot
environment: stale-frame / interpolation cleaning, a take-level train/val/test split, ACT, Diffusion Policy and a
π0.5 fine-tune (frozen VLM, trained action expert), with torchrun multi-GPU and a Slurm script.

```bash
cd robot_learning/imitation_learning/policy_training
$MABEL_LEROBOT_PYTHON train.py train --policy act --name act_v1 --robot mabel --set dataset=mabel_<name>
```

## 4. Resume and evaluate

```bash
cd learning
python3 -m training_studio.server.mabel_train resume --run <run_id> --steps 40000
python3 -m training_studio.server.mabel_train eval   --run <run_id> --episodes 3
python3 -m training_studio.server.mabel_train eval   --run <run_id> \
    --checkpoint training/act/<dataset>/<run_id>/train/checkpoints/<step>
python3 -m training_studio.server.mabel_train show   --run <run_id>
python3 -m training_studio.server.mabel_train list
```

`resume` continues the optimizer and step counter from the last (or a given) checkpoint.
`eval` defaults to the latest checkpoint and reports open-loop action error over logged
episodes (`action_l1_mean` and per-joint L1).

## Output layout

| Artifact | Location | Written by |
|---|---|---|
| raw episodes | `real/{vision_os,ios,real}/<task>_<NNN>_<date>/` | collector |
| curated datasets | `curated_dataset/{real,simulation}/<name>_<NNN>_<date>/` | `curate_export.py` |
| training runs | `robot_learning/training/<arch>/<dataset>/<run_id>/` | `runstore.py` |
| evaluations | `robot_learning/eval/<arch>/<dataset>/<run_id>/<eval_id>/` | `runstore.py` |
| index | `robot_learning/training_studio/workspace/studio.db` | `metrics_store.py` |

Each run directory holds `run.json` (the manifest and source of truth), `config/`,
`logs/train.log`, `metrics/metrics.jsonl`, `plots/` and LeRobot's own `train/` output with
`checkpoints/<step>/pretrained_model/`. Because the manifest is authoritative, the index can be
rebuilt by scanning the tree. `MABEL_TRAINING_ROOT`, `MABEL_EVAL_ROOT` and `MABEL_CURATED_ROOT`
override the locations. Internals: [`training_studio/TRAINING.md`](training_studio/TRAINING.md).

## Minimal baseline

`imitation_learning/act/train_act_minimal.py` is a single-file deterministic ACT (torch
only) for checking that a LeRobot export trains at all:

```bash
$MABEL_LEROBOT_PYTHON robot_learning/imitation_learning/act/train_act_minimal.py \
    --dataset curated_dataset/real/<name>/mabel_<name> --out /tmp/act_smoke --steps 400
```
