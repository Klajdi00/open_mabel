# Training studio: runs, checkpoints and evaluations

How the training CLI and the Experiments tab organize and manage runs. The user-facing
walkthrough is [`../PIPELINE.md`](../PIPELINE.md).

## Directory layout

```text
robot_learning/training/<arch>/<dataset>/<run_id>/      # arch ∈ act, dp, dit, pi0, pi05
├── run.json                  manifest, the source of truth (fields below)
├── config/train_config.json  resolved training configuration
├── logs/train.log            raw backend stdout (appended across resumes)
├── metrics/metrics.jsonl     studio metric records: config | start | metric | checkpoint | end
├── plots/*.png               server-rendered curves
└── train/                    LeRobot's output_dir
    └── checkpoints/<step>/pretrained_model/ ,  checkpoints/last -> …

robot_learning/eval/<arch>/<dataset>/<run_id>/<eval_id>/
├── eval.json   what was evaluated (checkpoint, episodes, kind)
├── results.json
└── plots/  videos/  logs/
```

Adding an architecture, dataset or run adds a directory at the right level; no code changes.
Metrics, logs and plots live outside `train/` because LeRobot refuses to start in an existing
`output_dir`: the run root and its managed siblings are created up front, `train/` is not.

## Classes (`server/`)

- **`runstore.py`**: paths and manifests only, no backend.
  `RunLayout` (all paths for a run), `RunManifest` (atomic `run.json` read/write),
  `TrainingRun` (layout + manifest + `list_checkpoints`), `RunRegistry`
  (`create`/`all`/`find`/`filter`/`group_by_arch`; the filesystem is authoritative),
  `EvalLayout`/`EvalRun`/`EvalRegistry`, and `arch_slug()` mapping UI names to directory names.
- **`experiment.py`**: `ExperimentManager`, the workflow. `train` (create run, ensure a v3.0
  dataset, spawn the backend, finalize), `resume` (from any checkpoint via LeRobot
  `--config_path` + `--resume`), `evaluate` (any checkpoint into the eval tree), `finalize`
  (render plots, record checkpoints and a metric summary in the manifest), `list_runs`,
  `group_by_arch`, `get_run`.
- **`backends/run_lerobot.py`**: runs `lerobot-train` in the LeRobot environment and translates
  its stdout into `metrics.jsonl`; handles fresh vs. resumed runs.
- **`backends/run_eval.py`**: loads a checkpoint's `pretrained_model/`, rolls it open-loop over
  logged observations and reports `action_l1_mean` and per-joint L1.
- **`plots.py`**: matplotlib (Agg) loss, learning-rate, gradient-norm, throughput and system plots.

## Manifest fields (`run.json`)

`run_id, arch, arch_slug, dataset, dataset_slug, task, backend, status, created, updated,
device, git_sha, hyperparams, selection, dataset_root, dataset_hash, steps_target,
best_checkpoint, last_checkpoint, metrics_summary{first_loss, last_loss,
loss_improvement_pct, metric_points}, plots[], eval_runs[]`.

## HTTP API (`app.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/experiments` | runs grouped by architecture |
| `GET /api/experiments/{run_id}` | detail: checkpoints, plots, evaluations |
| `GET /api/experiments/{run_id}/plot/{name}` | a rendered plot |
| `POST /api/experiments/train` | start a run (authenticated) |
| `POST /api/experiments/{run_id}/resume` | resume a run (authenticated) |
| `POST /api/experiments/{run_id}/eval` | evaluate a checkpoint (authenticated) |

The GUI for these is `web/experiments.html`.

## Environments

The studio server runs in the `mabel` environment and never imports torch or LeRobot. The
manager spawns the LeRobot environment given by `MABEL_LEROBOT_PYTHON`. LeRobot trains on
dataset v3.0 only; the manager builds one from the curation HDF5 with
[`../h5_to_lerobot_v30.py`](../h5_to_lerobot_v30.py) (a native writer; the stock v2.1→v3.0
converter mis-muxes MABEL's MP4 streams) and caches it under `workspace/datasets/.v30/`.

## Validation

On a CUDA host, ACT (52 M parameters) and Diffusion Policy (296 M) train into this tree;
resume continues the optimizer and step counter from a checkpoint; evaluation from a
checkpoint produces open-loop L1 over 600 frames; plots, manifests and the Experiments tab
populate. π0 and π0.5 configure and launch through the same path and additionally require
access to the gated `google/paligemma-3b-pt-224` weights.
