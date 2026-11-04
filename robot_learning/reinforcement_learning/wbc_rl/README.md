# wbc_rl — whole-body teleoperation controller for NOVA (RL)

One RL policy tracks the three frames a Vision Pro operator streams (head + both wrists/palms) and outputs
**every joint**: swerve base twist (vx, vy, wz), lift, torso pitch, neck (yaw/pitch/roll) and the 14 arm joints.
The plant (setpoint shaper, datasheet speed/torque limits, gravity feed-forward, 3 cm/s lift, swerve base with the
tip-over governor, Vision Pro stream faults) is shared by both tracks in `wbc_core.py`; `robot.py` holds the model.

| Track | Files | Stack |
|---|---|---|
| A: direct PPO, ORCA v1 hands (`data/nova_w4.xml`) | `env.py`, `tasks.py` (`NOVA-WBC-v0`), `play.py`, `watch.py`, `kin.py` | Isaac Lab 3 + Newton (MuJoCo-Warp), rsl_rl |
| B: SONIC-style teacher -> student, OpenArm grippers (`data/nova_grip.xml`) | `sonic.py` | mjlab (`../mjlab`), rsl_rl |

Deployment side (numpy only): `deploy.py` (policy export + adapter + sim parity), `sim2sim.py` (deploy path in plain MuJoCo, CPU).
The training models ship in `data/`; meshes resolve from `simulation/mabel_mujoco/assets/meshes/` and `assets/`.
`python robot.py --rebuild` regenerates them from a robot description (`NOVA_DESC=<dir>`, optional).

**Data (not included).** EgoDex (Apple) and LAFAN1 (Ubisoft) are under non-commercial, no-derivatives licences, so
neither they nor anything derived from them (`data/motions_v4.npz`, `data/clips.npz`) is redistributed. Fetch them yourself:
```bash
cd robot_learning/reinforcement_learning/wbc_rl          # PY = your Isaac Lab python, MJ = your mjlab python
python fetch_egodex.py test && python fetch_egodex.py part1 --max 8000   # -> data/raw/egodex/ (pose HDF5 only)
# LAFAN1: clone github.com/ubisoft/ubisoft-laforge-animation-dataset, unzip lafan1.zip into data/raw/lafan1/*.bvh
python motion_lib.py --out data/motions_v4.npz                            # 50 Hz operator frames + body labels
python motion_lib.py --avp <episodes_dir>     # optional: your own Retarget Lab takes -> data/avp_takes.npz
```
**Train / evaluate.** Trained checkpoints are not included.
```bash
$PY robot.py && $PY robot.py gripper                                       # MJCF -> USD + collision model (once)
PYTHONPATH=$PWD $PY <IsaacLab>/scripts/reinforcement_learning/train.py --rl_library rsl_rl \
    --task NOVA-WBC-v0 --num_envs 4096 --external_callback tasks.register --run_name v1
$PY play.py --checkpoint logs/rsl_rl/nova_wbc/<run>/model_<it>.pt --videos 4 [--mode upper] [--faults]
$PY watch.py logs/rsl_rl/nova_wbc/<run>                                    # score + film every checkpoint
$MJ sonic.py train --run s3 --iters 6000                                   # track B: teacher (PPO)
$MJ sonic.py distil --run s3s --teacher <teacher.pt>                       # student (DAgger on the token)
$MJ sonic.py eval <model.pt> [--student] [--clips mix|egodex|lafan|avp]
$PY deploy.py export <model.pt> policy.npz        # or: sexport <student.pt> sonic_policy.npz
$PY deploy.py parity <model.pt>                   # adapter vs env, tick by tick (sparity: student vs sonic.py)
python sim2sim.py --make-clips && python sim2sim.py policy.npz [--clip 0 --mode upper]   # pip install mujoco numpy
```
`deploy.Adapter` is not wired to any robot transport here; it only maps operator frames to commands.
