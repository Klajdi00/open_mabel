# mabel_mujoco scripts

Interactive entry points for the MuJoCo model, plus offline scene and figure tools. Run from `simulation/mabel_mujoco/`, using `mjpython` on macOS for anything that opens the viewer. Control code comes from the `mabel` library in [`controller/`](../../../controller/) (`teleop/` is an import alias for `mabel.teleop_engine`).

## Run

```bash
mjpython scripts/viewer.py --model scenes/mabel_home.xml    # H home, G/F open/close hands, L/D lift, Backspace reset
mjpython scripts/keyboard_teleop.py                         # --source keyboard|mock|avp; Tab or 1-4 = nav/arm/upper/whole body
mjpython scripts/compliance.py --region whole               # Ctrl-drag bodies; --estimator torque = hardware path
mjpython scripts/visualize_sensors.py                       # camera grid + LiDAR; --backend osmesa headless

python3  scripts/vision_pro_stream.py --ip <headset-ip>     # Vision Pro -> ZMQ tcp://*:5556 (--check first)
mjpython scripts/avp_teleop.py                              # deployed retargeting engine
mjpython scripts/retarget_teleop.py --source mock           # single-file reference pipeline (no repo imports)
```

## Tools (`tools/`)

| Script | Purpose |
|---|---|
| `doc_experiments.py`, `tip_stability_sweep.py`, `compliance_study.py`, `self_collision_study.py` | report figures → `docs/figures/` |
| `teleop_benchmark.py`, `test_retargeting.py`, `tune_retargeting.py`, `exp_retarget_fidelity.py`, `exp_vp_app_teleop.py` | teleop / retargeting checks |
| `gen_extra_scenes.py`, `scenegen/`, `gen_*_objects.py`, `gen_*_textures*.py`, `gen_kitchen_fk.py` | scene library builders |
| `verify_scenes.py`, `gen_scene_thumbs.py` | scene acceptance checks (nonzero exit on failure), thumbnails |

```bash
python3 scripts/tools/gen_extra_scenes.py && python3 scripts/tools/verify_scenes.py && python3 scripts/tools/gen_scene_thumbs.py
```

Reports: [`../docs/`](../docs/). Asset credits: [`../third_party/ATTRIBUTION.md`](../third_party/ATTRIBUTION.md).
