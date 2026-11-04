# sensors

ROS-free camera bring-up and viewers. `camera_hal` wraps RealSense (`pyrealsense2`), ZED (`pyzed`) and UVC webcams behind one `CameraFrame` (RGB, metric depth, point cloud, IMU).

On the robot the HAL owns the cameras (`hardware_bridge/mabel_hw/cameras/`). Use these tools for bench bring-up only, and never while the HAL's camera subsystem holds a device (stop it or set `MABEL_HW_CAMERAS=false`).

| Camera | Role | Link |
|---|---|---|
| RealSense D435i | front (base) | USB 3 |
| ZED Mini | head (stereo) | USB 3 only (on USB 2 it shows up as IMU-only; use a SuperSpeed Micro-B cable) |
| RealSense D405 ×2 | wrists | USB 3 |

## Run

```bash
bash install_deps.sh                          # sensors/.venv + SDK bindings (auto on first bring-up)
bash usb_check.sh                             # negotiated USB speed per camera, pass/fail
bash bring_up_cameras.sh                      # dashboard + RGB/depth windows + point cloud
bash bring_up_cameras.sh --cameras zed --no-cloud   # all | realsense | zed; --no-window = headless
bash run_camera_node.sh                       # standalone ROS/MJPEG camera node (debug alternative to the HAL)
python view.py --host <robot-ip> --window     # view a running camera server
```

```python
from camera_hal import RealSenseCamera        # also ZedCamera, UvcCamera, CameraManager
with RealSenseCamera(fps=30) as cam:
    f = cam.read()                            # f.rgb, f.depth_m, f.points_xyz, f.intrinsics.K, f.imu
```
