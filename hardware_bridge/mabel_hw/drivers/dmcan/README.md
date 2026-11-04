# dmcan — DaMiao DM-USB2FDCAN binding

ctypes binding to DaMiao's closed-source `libdm_device`, used by `drivers/damiao_fdcan.py` for the DM-USB2FDCAN dongle (USB `34b7:6877`). This is a bench backend (`MABEL_HW_NECK_YAW_BACKEND=fdcan`); the default wiring doesn't use it.

| File | Origin |
|---|---|
| `lib/linux-aarch64/libdm_device.so` | unmodified, DaMiao `dm-device-sdk` v1.1.0 (`C&C++/lib/v1.1.0/linux/arm64/`) |
| `lib/darwin-arm64/libdm_device.dylib` | unmodified, same release (`.../macos/arm64/`), sha256 `39fa19eeb285134047722e57f05b91c212b1de3c323d47b1e7f627069463402d` |
| `__init__.py` | the binding; structs from the SDK's `dmcan.h` / `dmcan_def.py` |

SDK: <https://github.com/dmBots/dm-device-sdk>. `MABEL_DMCAN_LIB` overrides the library path. Linux needs `libusb-1.0` plus `hardware_bridge/install/99-mabel-dmcan.rules`. macOS needs nothing extra.
