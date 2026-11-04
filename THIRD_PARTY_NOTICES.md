# Third-party notices

MABEL builds on the following open designs and software. Each is used under
its own licence; the licences below are those published by the upstream
projects. Where MABEL modifies an
upstream design, the modified files carry the licence the upstream requires,
noted in the last column.

| Component | Where it is used | Upstream | Licence | MABEL's obligation |
|---|---|---|---|---|
| **OpenArm** hardware (CAD, structural set) | The two 7-DOF arms (`hardware/`, the arm MJCF/URDF bodies) | [enactic/openarm_hardware](https://github.com/enactic/openarm_hardware) | **CERN-OHL-S-2.0** (strongly reciprocal) | Modified arm design files stay under CERN-OHL-S-2.0 with source available; products carry the source location and notices (§3.3, §4). Attribution to Enactic, Inc. |
| **OpenArm** software (drivers, CAN, description, teleop) | `controller/`, `hardware_bridge/`, `ros/` where derived | [enactic/openarm](https://github.com/enactic/openarm), openarm_can, openarm_description | Apache-2.0 | Keep the Apache notices and NOTICE attributions. |
| **ORCA Hand** hardware and code | The two 17-DOF hands (`firmware/orca_hand/`, hand CAD and MJCF) | [orcahand/orcahand_hardware](https://github.com/orcahand/orcahand_hardware), [orca_core](https://github.com/orcahand/orca_core) (MIT), [orcahand_description](https://github.com/orcahand/orcahand_description) (MIT) | **CC BY 4.0** (hardware and code in orcahand_hardware); MIT (orca_core, description) | Attribution to ORCA Dexterity, Inc. and the ETH Zurich Soft Robotics Lab; indicate changes made. No patent or trademark rights are granted by upstream; "ORCA" is their mark. |
| MuJoCo | Simulation, whole-body controller | Google DeepMind | Apache-2.0 | Notices. |
| LeRobot | Datasets, policy training studios | Hugging Face | Apache-2.0 | Notices. |
| mjlab | RL environments (`robot_learning/reinforcement_learning/mjlab/`, vendored) | mujocolab/mjlab | Apache-2.0 | Upstream LICENSE retained. |
| sparkcan | REV SPARK CAN driver (`firmware/swerve_drive/sparkcan_py/`, vendored) | upstream sparkcan | see vendored LICENSE | Notices retained. |
| MuJoCo scanned objects, Franka kitchen, robosuite arenas | Scene assets (`simulation/mabel_mujoco/third_party/`) | see `ATTRIBUTION.md` in each folder | see each folder | Attribution files retained. |
| ROS 2, Nav2, nvblox / cuVSLAM (NVIDIA Isaac ROS) | Navigation, mapping | OSRF / NVIDIA | Apache-2.0 (Isaac ROS: NVIDIA licence terms) | Notices; Isaac ROS components are used unmodified under NVIDIA's terms. |
| three.js, meshoptimizer, MediaPipe | Browser studio 3-D viewers, webcam hand tracking | mrdoob; Arseny Kapoulkine | MIT | Notices retained in the vendored files. |
| OpenArm v1 gripper meshes | NOVA's gripper training model (`robot_learning/reinforcement_learning/wbc_rl/assets/meshes/openarm_gripper_v1/`) | enactic OpenArm v1 gripper | as the OpenArm hardware row (CERN-OHL-S-2.0) | Modified: re-expressed in the gripper/finger frames and decimated; source available here. |
| Isaac Lab, Newton, rsl_rl | `wbc_rl/` training (imported, not vendored) | isaac-sim/IsaacLab, newton-physics/newton, leggedrobotics/rsl_rl | BSD-3-Clause; Apache-2.0; BSD-3-Clause | None beyond their own install. |
| EgoDex, LAFAN1 motion datasets | `wbc_rl/` motion library (referenced only) | Apple (EgoDex); Ubisoft La Forge (LAFAN1) | CC BY-NC-ND 4.0 (check the upstream terms) | **Not redistributed**, nor any file derived from them; users download them (`fetch_egodex.py`, `motion_lib.py`). |
| SONIC, Dyna, CAPS (published methods) | `wbc_rl/sonic.py`, `wbc_rl/tasks.py` | the respective papers | n/a | Re-implemented from the papers and cited; no upstream code copied. |
| REV MAXSwerve modules, DAMIAO actuators, Feetech and Dynamixel servos, FlexiSpot column, NVIDIA Jetson, Raspberry Pi, cameras and lidars | Purchased components | their manufacturers | Commercial products, no licence obligations on the design | Named in the bill of materials. |

If a component is missing from this list, please open an issue.
