# MuJoCo Scanned Objects (subset)

- **Source:** https://github.com/kevinzakka/mujoco_scanned_objects (main branch,
  fetched 2026-07-02) — MJCF conversions (via obj2mjcf) of Google's Scanned Objects
  Dataset (https://ai.googleblog.com/2022/06/scanned-objects-by-google-research.html).
- **License:** MIT for the MJCF conversion (see `LICENSE`, Copyright (c) 2022 Kevin
  Zakka); the underlying scanned-object data is CC-BY 4.0, Google LLC ("Google
  Scanned Objects" / "Scanned Objects by Google Research").
- **What was taken:** 11 of the 1030 objects (~21 MB), chosen as clutter for the
  MABEL office / factory / bedroom / warehouse scenes:
  `3M_Vinyl_Tape_Green_1_x_36_yd`, `Cole_Hardware_Hammer_Black`,
  `Craftsman_Grip_Screwdriver_Phillips_Cushion`, `Crosley_Alarm_Clock_Vintage_Metal`,
  `Curver_Storage_Bin_Black_Small`, `JBL_Charge_Speaker_portable_wireless_wired_Green`,
  `Kanex_MultiSync_Wireless_Keyboard`, `Pony_C_Clamp_1440`,
  `Razer_Abyssus_Ambidextrous_Gaming_Mouse`, `Room_Essentials_Mug_White_Yellow`,
  `Travel_Mate_P_series_Notebook`.
  Each object dir contains `model.xml`, `model.obj`, `texture.png`, and convex
  collision pieces (`model_collision_*.obj`) — the same `model.obj` + `texture.png`
  layout as the existing `assets/objects/` GSO clutter, so the same wiring pattern
  (cf. `scripts/tools/gen_home_objects.py`) applies.
- **Local modifications:** none.
