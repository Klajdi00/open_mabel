# calibration_data — as-deployed calibration

What every `mabel_hw` driver reads calibration from first. The firmware packages' `configs/` folders are only fallbacks.

| File | Contents |
|---|---|
| `mabel.json` | unified tree by slice: `arm/openarm-dual`, `torso/damiao`, `neck/teensy`, `neck/serial`, `neck/rssr`, `hand/orca-<side>`, `base/maxswerve`, `lift/pico` |
| `hands/<side>_calibration.json` | ORCA raw-tick end-stops per servo ID |

Schema: `mabel_hw/calibration/schema.py`. `CalibrationStore` (`store.slice("arm/openarm-dual")`) hot-reloads running drivers on change; `MABEL_HW_CALIB_ROOT` points it elsewhere. Don't hand-edit while the robot moves.

Writers: hand calibration via the hand RPC (Control Center Hands tab), ORCA Hand Studio (`firmware/orca_hand/feetech/gui`), `CalibrationStore.save_slice()`, and `python3 tools/sync_calibration.py [--watch|--check]`.
