# mabel_sensors

Camera launches and calibration for the real robot. On the robot the HAL owns the cameras and publishes the same `/camera/*` topics as `mabel_sim`; this package wraps that, offers standalone drivers for benches without the HAL, and handles calibration. ROS-free bring-up lives in [`sensors/`](../../../sensors/).

Only one process may own a camera: never run `realsense.launch.py`, `zed.launch.py` or the standalone node while the HAL's camera subsystem is running.

| Launch | Starts |
|---|---|
| `sensors.launch.py` | included by `hardware.launch.py`; optional standalone camera node + head-stereo rectification |
| `realsense.launch.py` | front D435i on `/camera/front/*` + mount TF |
| `zed.launch.py` | head ZED Mini on `/camera/zed/*` |
| `calibrate.launch.py`, `calibrate_stereo.launch.py` | intrinsic / stereo calibration GUI |

## Calibrate (after moving or changing a lens)

```bash
ls -l /dev/v4l/by-path/                       # stable path ending in -video-index0
ros2 launch mabel_sensors calibrate.launch.py camera:=head device:=/dev/v4l/by-path/<path>   # head | left_wrist | right_wrist
# CALIBRATE, then SAVE (-> /tmp/calibrationdata.tar.gz)
ros/src/mabel_sensors/scripts/save_calib.sh head   # installs YAMLs into config/
```

`save_calib.sh` assumes the repo is cloned at `~/open_mabel`. The head-stereo pipeline only starts if the `/dev/v4l/by-path/...` device hard-coded in `sensors.launch.py` exists; edit it for your USB layout.

USB permissions: `sudo ./ros/scripts/install_camera_udev.sh`, then replug each camera.
