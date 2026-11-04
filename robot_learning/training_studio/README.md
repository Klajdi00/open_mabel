# Training studio

A browser front end to LeRobot training for MABEL datasets. Pick an architecture, a camera
subset and a robot-module scope, fill a validated hyperparameter form, launch training, and
follow live loss, sub-losses, logs and GPU/CPU load. The centre panel draws the selected
network with its tensor shapes; trained policies can be evaluated open-loop or deployed
closed-loop in MuJoCo. Metrics are collected locally (no external experiment tracker).

```text
training_studio/
├── server/             FastAPI backend (port 8095)
│   ├── lerobot_shim/   LeRobot entry point with the studio logger + the DiT policy plugin
│   ├── backends/       subprocess wrappers for LeRobot training and evaluation
│   └── tools/          setup, tests, benchmarks
├── web/                ES-module frontend: index.html (studio), experiments.html, community.html
├── workspace/          runs, derived datasets, SQLite index (git-ignored, created on first run)
└── docs/               trainer_studio.pdf + figures/
```

## Setup

Install LeRobot into the Python environment that will run training (CUDA recommended; Apple
MPS and CPU work for small models):

```bash
cd learning
MABEL_TRAIN_PYTHON=/path/to/lerobot-env/bin/python \
    bash training_studio/server/tools/install_lerobot.sh
```

The script snapshots the environment first, installs `lerobot`, `psutil` and `pynvml`, and
verifies the training entry point, the registered policy types, the DiT plugin and the
resolved device (`cuda`, then `mps`, then `cpu`). LeRobot subprocesses run with `USE_TF=0`
and the `pyav` video backend.

## Run

```bash
cd learning
export MABEL_TRAIN_PYTHON=/path/to/lerobot-env/bin/python    # studio GUI training
export MABEL_LEROBOT_PYTHON=/path/to/lerobot-env/bin/python  # Experiments tab / CLI
export MABEL_TRAIN_USER=<user> MABEL_TRAIN_KEY=<passkey>     # required for any mutating action
python3 -m training_studio.server                            # http://127.0.0.1:8095
```

Launching, pausing, resuming, evaluating and deploying require a login with
`MABEL_TRAIN_USER` / `MABEL_TRAIN_KEY`; if they are unset every login is refused. Other
variables: `MABEL_TRAINER_PORT`, `MABEL_TRAINER_HOST`, `MABEL_DATASET_ROOT`,
`MABEL_CURATED_ROOT` (see `server/config.py`).

| Region | Content |
|---|---|
| Left rail | dataset library (curated exports and studio datasets) with episode, frame, duration, camera, DOF and fps statistics; train/eval/test split |
| Centre | the selected architecture as an animated block diagram, re-wired when the camera subset or scope changes |
| Right dock | Config (architecture + hyperparameters), Inputs (cameras + module scope, live obs/action dimensions), Info (network I/O contract and validation), Runs |
| Top bar | Train, Pause (checkpoint and stop), Retrain from a checkpoint, extra windows (system load, Eval, Deploy in MuJoCo, I/O plots) |
| Experiments tab | every run grouped by architecture and dataset, with config, checkpoints, plots and evaluations; one-click resume and eval |

## Architectures

| UI name | LeRobot `policy.type` | Notes |
|---|---|---|
| ACT | `act` | native |
| DiT | `dit` | plugin in `server/lerobot_shim/dit_policy/`: a diffusion-transformer denoiser on the Diffusion Policy pipeline |
| pi0 | `pi0` | vision-language-action, flow matching; whole-body 59-D output needs `max_action_dim >= 59` (validated) |
| pi0.5 | `pi05` | vision-language-action |

The headless CLI additionally accepts `DP` (LeRobot Diffusion Policy). π0 and π0.5 need
multi-GB pretrained weights from a gated Hugging Face repository; see [`../PIPELINE.md`](../PIPELINE.md).
Validators check action-dimension padding, attention-head divisibility and chunk size vs.
action steps before a run starts.

## Input and output selection

Choose a subset of the seven cameras and a module scope: `whole_body`, `upper_body`, `arms`,
`hands`, `arms_hands`, `left_arm`, `right_arm`, `base`, or a custom group selection. The
observation and action scopes can differ. The selection slices `observation.state` and
`action` to the chosen joint groups and keeps only the chosen camera streams, producing a real
LeRobot dataset view, so training is unmodified LeRobot. At deployment a scatter map writes the
policy's scoped action back into the full 59-actuator command, holding unselected joints.

## Data flow

```text
curated exports ─▶ combine ─▶ scope view ─▶ LeRobot v3.0 ─▶ split ─▶ train_config.json
   ─▶ lerobot-train (studio logger) ─▶ metrics.jsonl ─▶ SQLite ─▶ WebSocket ─▶ UI
```

The studio logger replaces LeRobot's experiment-tracker logger in-process, so LeRobot writes
`metrics.jsonl` in the run directory; the server tails it into SQLite and pushes updates to
the browser. Run layout and the manager classes: [`TRAINING.md`](TRAINING.md).

## Test

```bash
cd learning
python3 -m training_studio.server.tools.test_training          # add --full for CUDA VLA training
```

Runs dataset preparation, feature selection, configuration and validation for all four
architectures, short real ACT and DiT trainings (asserting the loss decreases), VLA plumbing,
evaluation and MuJoCo deployment; prints a pass/skip table and writes figures to
`docs/figures/`.
