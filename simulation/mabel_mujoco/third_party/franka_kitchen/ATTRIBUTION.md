# Franka Kitchen environment assets

- **Source:** https://github.com/Farama-Foundation/Gymnasium-Robotics
  (`gymnasium_robotics/envs/assets/kitchen_franka/kitchen_assets/`, main branch,
  fetched 2026-07-02). These assets originate from Google Research's
  relay-policy-learning "Adept envs" (https://github.com/google-research/relay-policy-learning),
  i.e. the classic D4RL / Franka Kitchen environment.
- **License:** Apache License 2.0 — see `LICENSE.md` in this directory
  (Copyright 2019 The DSuite Authors / 2020 Google LLC). The Gymnasium-Robotics
  repackaging is MIT.
- **What was taken:** the `kitchen_assets` directory only — kitchen furniture MJCF
  (`item_assets/*.xml`: oven, counters, backwall, slide cabinet, hinge cabinet,
  microwave, kettle), their STL meshes (`meshes/`), textures (`textures/`),
  `basic_scene.xml`, and the original `kitchen_env_model.xml` for reference.
  The Franka robot (`franka_assets/`) was deliberately NOT taken — MABEL replaces it.
- **Local modifications:** none in this directory. `scripts/tools/gen_kitchen_fk.py`
  reads these files and emits `scenes/kitchen_fk_env.xml` with rewritten asset paths
  and a global geometric rescale so counter height suits the MABEL mobile manipulator;
  the generated file is clearly marked as derived.
