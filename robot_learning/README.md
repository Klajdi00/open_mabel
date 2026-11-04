# robot_learning — imitation learning and reinforcement learning

```text
imitation_learning/      recording (data_collection/), a minimal single-file ACT baseline (act/) and the
                         ACT / DP / π0.5 training CLI (policy_training/)
data_curation/           curation studio (:8093): quality scoring, drift checks, trimming, LeRobot export
training_studio/         training studio (:8095): ACT / Diffusion Policy / π0 / π0.5 on LeRobot, sim deployment
curate_export.py         headless curate → LeRobot export
h5_to_lerobot_v30.py     HDF5 episodes → LeRobot v3.0
reinforcement_learning/  GPU RL on mjlab, incl. whole-body control; wbc_rl/: whole-body teleop RL (NOVA)
```

## Imitation learning

```text
collect ─▶ episode_XXXX.h5 ─▶ curate ─▶ LeRobot dataset ─▶ train ─▶ evaluate / deploy
```

Sim and real episodes share one HDF5 format (every camera, joint state, base twist and the
command sent to each actuator). Collection and curation run in the `mabel` environment;
training uses a separate torch + CUDA + LeRobot environment:
`export MABEL_LEROBOT_PYTHON=/path/to/lerobot-env/bin/python`.

Full walkthrough: [`PIPELINE.md`](PIPELINE.md) · training details:
[`training_studio/TRAINING.md`](training_studio/TRAINING.md).
The training studio wraps LeRobot's own trainers behind a web UI; [`imitation_learning/policy_training/`](imitation_learning/policy_training/)
is the scriptable CLI pipeline (data cleaning, ACT / Diffusion Policy / π0.5 fine-tuning, torchrun + Slurm, test-split
evaluation and comparison).

## Reinforcement learning

See [`reinforcement_learning/README.md`](reinforcement_learning/README.md).
