# ros — setup

Two ways to install the workspace. The conda path is recommended and is what `dev_up.sh`
automates; the apt path is for machines that already run a system ROS 2 Jazzy install.

## Recommended: conda (RoboStack), Ubuntu 24.04

```bash
cd ros
./dev_up.sh
```

`dev_up.sh`:

1. finds conda, or installs Miniforge if none is present;
2. creates or updates the `mabel` env from `environment.yml` — ROS 2 Jazzy from RoboStack,
   the colcon toolchain, the RealSense driver (`ros-jazzy-realsense2-camera`, with bundled
   librealsense) and every Python dependency, with no apt and no system ROS;
3. installs the camera Python bindings (`pyrealsense2`; `pyzed` if the ZED SDK is present);
4. runs `colcon build`;
5. opens the `mabel` tmux session (see [README.md](README.md#setup)).

Enter the environment in any new shell with:

```bash
source scripts/mabel_env.sh
```

The environment is hermetic even if your login profile sources `/opt/ros/jazzy` or
another workspace: `dev_up.sh` and `scripts/mabel_env.sh` both call
`scripts/_scrub_ros_env.sh`, which removes foreign ROS entries from `PATH`,
`LD_LIBRARY_PATH`, `PYTHONPATH` and the `*_PREFIX_PATH` variables.

Mixing a conda Python with an apt ROS install breaks `colcon`/`catkin_pkg`; RoboStack
avoids this by keeping ROS, Python and the compilers in one env.

## Native components (not in conda)

The default LiDAR-SLAM workflow needs none of these.

| Component | Needed for | Install |
|---|---|---|
| ZED SDK (CUDA 12 build) | head ZED Mini on the real robot | Stereolabs installer, to `/usr/local/zed`; then `./dev_up.sh --with-sdk` |
| Isaac ROS (cuVSLAM, nvblox) | `mabel_slam visual_slam.launch.py` | NVIDIA apt packages under `/opt/ros/jazzy`; used through `scripts/slam_env.sh` |
| `sllidar_ros2` | RPLIDAR driver (`hardware.launch.py lidar:=true`) | `sudo apt install ros-jazzy-sllidar-ros2` |
| Camera udev rules | RealSense/ZED access without root | `sudo ./scripts/install_camera_udev.sh`, then replug the cameras |

`./dev_up.sh --with-sdk` runs the source build for the ZED wrapper:

```bash
vcs import src < thirdparty.repos     # zed-ros2-wrapper, zed-ros2-interfaces, cob_common
./scripts/build_sdk_cams.sh           # patches the wrapper for Jazzy, builds it against CUDA 12
```

The imported packages are machine-specific and ignored by version control.

### Visual SLAM environment

The Isaac ROS packages exist only as apt binaries, so they cannot be loaded from the conda
env. `scripts/slam_env.sh` sets up the apt ROS underlay with the `mabel_slam` package
overlaid, and pins CycloneDDS on domain 0 so it shares the graph with the conda processes:

```bash
source scripts/slam_env.sh
ros2 launch mabel_slam visual_slam.launch.py
```

## Alternative: system ROS 2 Jazzy (apt)

```bash
sudo apt install -y python3-rosdep python3-vcstool python3-colcon-common-extensions
sudo rosdep init 2>/dev/null; rosdep update
source /opt/ros/jazzy/setup.bash

cd ros
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "librealsense2 realsense2_camera zed_components zed_interfaces \
isaac_ros_visual_slam isaac_ros_nvblox isaac_ros_image_proc nvblox_msgs gazebo_ros"
pip install -r requirements.txt
```

Skipped keys are installed out of band: the RealSense driver from apt or source (do not
mix an apt `librealsense2` with a source build), the ZED packages as described above, and
the Isaac ROS packages from NVIDIA. Gazebo is not used; `gazebo_ros` is only referenced by
the optional Gazebo Classic spawn launch in `mabel_description`.

Build and verify:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch mabel_description display.launch.py
```
