# Third-party MuJoCo environment assets

Everything under `third_party/` is downloaded from open-source, permissively-licensed
projects. Each subdirectory carries its own `LICENSE` and `ATTRIBUTION.md` describing
exactly what was taken. Total size is kept small on purpose (~35 MB); large datasets
were subset to representative assets only.

## Included sources

| Directory | Upstream | License | Size | Used by |
|---|---|---|---|---|
| `franka_kitchen/` | Farama-Foundation/Gymnasium-Robotics (`kitchen_franka/kitchen_assets`, originally google-research/relay-policy-learning "Adept envs" / D4RL kitchen) | Apache-2.0 (DSuite Authors / Google LLC 2020) | 8.2 MB | `scenes/mabel_kitchen_fk.xml` via `scripts/tools/gen_kitchen_fk.py` |
| `robosuite_arenas/` | ARISE-Initiative/robosuite (`models/assets/arenas` + only the textures those arenas reference) | MIT (Stanford Vision and Learning Lab & UT RPL) | 5.3 MB | reference arenas / textures for future scenes |
| `mujoco_scanned_objects/` | kevinzakka/mujoco_scanned_objects (11 of 1030 objects) | MIT (conversion) over CC-BY 4.0 (Google Scanned Objects data) | 21 MB | clutter in office / factory / bedroom / warehouse scenes |

## Evaluated and skipped (and why)

- **RoboCasa** (robocasa.ai) — MJCF kitchen scenes, but the asset download is
  multi-GB (fetched via `robocasa.scripts.download_kitchen_assets`, >5 GB) and parts
  of the object set are AI-generated with unclear per-asset licensing. Too large for
  the ~300 MB budget; skipped.
- **MuJoCo Menagerie** (google-deepmind/mujoco_menagerie) — high-quality MJCF, but it
  is a *robot* library, not environments; nothing to take for scenes.
- **dm_control** — environments are procedurally composed in Python (composer), no
  standalone room-scene MJCF files to extract; skipped.
- **Meta-World / robosuite task envs** — MJCF but single-table manipulation rigs,
  redundant with the arenas already taken; skipped.
- **AI2-THOR / Habitat / iGibson scenes** — rich home environments but Unity /
  glTF-based, not MJCF; skipped.
- **Original Google Scanned Objects (Ignition Fuel)** — full set is ~9 GB; instead we
  keep the 7 objects already in `assets/objects/` (CC-BY 4.0) plus the 11-object MJCF
  subset in `mujoco_scanned_objects/`.
