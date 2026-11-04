"""
collect_ros.py — launch the MABEL simulated robot *in ROS* and collect imitation
-learning episodes as **rosbags** (MCAP), headlessly and reproducibly.

This is the ROS-native counterpart to ``collect_sim.py``. Where that script logs
straight to HDF5 from a live Vision-Pro operator, this one:

  1. loads the MuJoCo robot (default scene: the kitchen/dining ``mabel_home.xml``)
     and steps physics + the shared whole-body controller — i.e. it *is* the
     simulated robot, running as a real ``rclpy`` node;
  2. publishes the per-frame robot state on standard ROS topics
     (``/joint_states``, ``/scan``, ``/odom``, ``/tf``, the wrist cameras, plus
     the target command on ``/action/joint_targets``) so you can ``ros2 topic
     echo`` / RViz it live;
  3. records each episode to its own **MCAP rosbag** via ``rosbag2_py`` — the
     same on-disk format ``ros2 bag record`` produces, readable by ``ros2 bag``
     and by the Data Studio's drag-and-drop importer (``bag_to_studio.py``).

Motion is *scripted* (synthetic-AVP trajectories with a per-(task, episode) phase
offset), so 5 tasks × 10 episodes can be collected unattended. The demos are
plausible task-shaped motion — they exercise the whole logging/processing/viz
pipeline end-to-end; they are not guaranteed successful grasps.

Usage (run inside the `mabel` conda env; `source ros/scripts/mabel_env.sh`):

    python collect_ros.py                          # 5 tasks × 10 eps -> recordings/sim_home
    python collect_ros.py --tasks 1 --episodes 2   # quick smoke test
    python collect_ros.py --out /path/to/recordings/sim_home --seconds 8
    python collect_ros.py --no-publish             # write bags only, no live ROS graph

Each episode lands at ``<out>/<task_slug>/episode_XXXX`` (a rosbag directory with
``metadata.yaml`` + ``*.mcap`` + a ``session.json`` sidecar).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")          # headless GPU render

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_IL = os.path.dirname(_THIS)
if _IL not in sys.path:
    sys.path.insert(0, _IL)

import numpy as np  # noqa: E402

from data_collection import config as CFG          # noqa: E402 (bootstraps `mabel`)

DEFAULT_SCENE = os.path.abspath(os.path.join(
    _THIS, "..", "..", "..", "simulation", "mabel_mujoco", "scenes", "mabel_home.xml"))

# Home-scene-appropriate task labels (text metadata only; motion is scripted).
TASKS = [
    "pick the red mug",
    "place the bowl on the counter",
    "open the cabinet",
    "wipe the kitchen island",
    "hand over the bottle",
]

# Lean recording scope (per the dataset spec): proprioception + lidar + the two
# wrist cameras + tf + commands. Keeps 50 episodes to a few hundred MB.
WRIST_CAMS = ("l_wrist_cam", "r_wrist_cam")

# Topics written into every episode bag.
TOPICS = [
    ("/joint_states",                       "sensor_msgs/msg/JointState"),
    ("/action/joint_targets",               "sensor_msgs/msg/JointState"),
    ("/scan",                               "sensor_msgs/msg/LaserScan"),
    ("/odom",                               "nav_msgs/msg/Odometry"),
    ("/tf",                                 "tf2_msgs/msg/TFMessage"),
    ("/l_wrist_cam/image_raw/compressed",   "sensor_msgs/msg/CompressedImage"),
    ("/r_wrist_cam/image_raw/compressed",   "sensor_msgs/msg/CompressedImage"),
    ("/episode/meta",                       "std_msgs/msg/String"),
]


def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "task"


def _stamp(stamp_msg, t):
    stamp_msg.sec = int(t)
    stamp_msg.nanosec = int(round((t - int(t)) * 1e9))
    return stamp_msg


def build_sim(scene):
    """Build the MuJoCo robot + shared controller + state provider for `scene`,
    restricted to the two wrist cameras (RGB only) for lean recording."""
    import mujoco
    from mabel.teleop_controller import TeleopController
    from data_collection.providers import SimStateProvider

    # restrict cameras BEFORE the provider builds its renderers
    CFG.CAMERAS = [dict(c, depth=False) for c in CFG.CAMERAS if c["name"] in WRIST_CAMS]

    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctrl = TeleopController(model, data)
    prov = SimStateProvider(model, data, ctrl)
    return model, data, ctrl, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_THIS, "..", "..", "..", "recordings", "sim_home"),
                    help="root folder for episode bags (default: repo recordings/sim_home)")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=8.0, help="episode duration")
    ap.add_argument("--no-publish", action="store_true",
                    help="don't spin a live ROS node (write bags only)")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    n_tasks = max(1, min(args.tasks, len(TASKS)))
    n_frames = int(round(args.seconds * CFG.LOG_FPS))

    import mujoco
    from mabel import mock_avp
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import JointState, LaserScan, CompressedImage
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from std_msgs.msg import String

    print(f"[collect_ros] scene: {args.scene}")
    model, data, ctrl, prov = build_sim(args.scene)
    substeps = max(1, int(round((1.0 / CFG.LOG_FPS) / model.opt.timestep)))
    schema = prov.schema_meta()
    act_names = schema["actuator_names"]
    lidar_meta = schema["lidar"]
    print(f"[collect_ros] nu={model.nu} cams={[c['name'] for c in schema['cameras']]} "
          f"substeps={substeps} frames/ep={n_frames}")

    # ── optional live ROS node (so it's a real running robot you can echo/RViz) ─
    node = pubs = None
    if not args.no_publish:
        try:
            import rclpy
            rclpy.init(args=None)
            node = rclpy.create_node("mabel_sim_collector")
            from rclpy.qos import QoSProfile
            qos = QoSProfile(depth=10)
            type_map = {"sensor_msgs/msg/JointState": JointState,
                        "sensor_msgs/msg/LaserScan": LaserScan,
                        "nav_msgs/msg/Odometry": Odometry,
                        "tf2_msgs/msg/TFMessage": TFMessage,
                        "sensor_msgs/msg/CompressedImage": CompressedImage,
                        "std_msgs/msg/String": String}
            pubs = {name: node.create_publisher(type_map[typ], name, qos)
                    for name, typ in TOPICS}
            print("[collect_ros] live ROS node up (topics advertised)")
        except Exception as e:
            print(f"[collect_ros] live publish disabled ({e})")
            node = pubs = None

    def make_msgs(obs, act, sim_t, wall_t):
        """Build the ROS message set for one frame, all sharing one timestamp."""
        msgs = {}
        js = JointState(); _stamp(js.header.stamp, sim_t); js.header.frame_id = "base_link"
        js.name = list(act_names)
        js.position = obs["obs/joint_pos"].astype(float).tolist()
        js.velocity = obs["obs/joint_vel"].astype(float).tolist()
        js.effort = obs["obs/joint_torque"].astype(float).tolist()
        msgs["/joint_states"] = js

        at = JointState(); _stamp(at.header.stamp, sim_t); at.header.frame_id = "base_link"
        at.name = list(act_names)
        at.position = act["action/ctrl"].astype(float).tolist()
        msgs["/action/joint_targets"] = at

        scan = LaserScan(); _stamp(scan.header.stamp, sim_t); scan.header.frame_id = "lidar"
        n = int(lidar_meta["n_rays"])
        scan.angle_min = 0.0
        scan.angle_increment = np.deg2rad(float(lidar_meta["angle_res_deg"]))
        scan.angle_max = scan.angle_increment * (n - 1)
        scan.range_min = 0.0
        scan.range_max = float(CFG.DEPTH_MAX_M) * 4
        scan.ranges = obs["obs/lidar"].astype(float).tolist()
        msgs["/scan"] = scan

        # poses: chassis on /odom, all four on /tf (quaternion wxyz -> xyzw)
        ch = obs["obs/pose/chassis"]
        bt = obs.get("obs/base_twist", np.zeros(3, np.float32))
        odom = Odometry(); _stamp(odom.header.stamp, sim_t)
        odom.header.frame_id = "world"; odom.child_frame_id = "chassis"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, ch[:3])
        odom.pose.pose.orientation.w = float(ch[3]); odom.pose.pose.orientation.x = float(ch[4])
        odom.pose.pose.orientation.y = float(ch[5]); odom.pose.pose.orientation.z = float(ch[6])
        odom.twist.twist.linear.x = float(bt[0]); odom.twist.twist.linear.y = float(bt[1])
        odom.twist.twist.angular.z = float(bt[2])
        msgs["/odom"] = odom

        tfm = TFMessage()
        for key in ("chassis", "head", "left_palm", "right_palm"):
            p = obs.get(f"obs/pose/{key}")
            if p is None:
                continue
            ts = TransformStamped(); _stamp(ts.header.stamp, sim_t)
            ts.header.frame_id = "world"; ts.child_frame_id = key
            ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = map(float, p[:3])
            ts.transform.rotation.w = float(p[3]); ts.transform.rotation.x = float(p[4])
            ts.transform.rotation.y = float(p[5]); ts.transform.rotation.z = float(p[6])
            tfm.transforms.append(ts)
        msgs["/tf"] = tfm

        for cam, topic in (("l_wrist_cam", "/l_wrist_cam/image_raw/compressed"),
                           ("r_wrist_cam", "/r_wrist_cam/image_raw/compressed")):
            blob = obs.get(f"obs/cam/{cam}/rgb")
            if blob is None:
                continue
            ci = CompressedImage(); _stamp(ci.header.stamp, sim_t); ci.header.frame_id = cam
            ci.format = "jpeg"
            ci.data = np.asarray(blob, dtype=np.uint8).tobytes()
            msgs[topic] = ci
        return msgs

    meta_payload = String()
    total_eps = 0
    t_start = time.time()
    for ti in range(n_tasks):
        task = TASKS[ti]
        task_dir = os.path.join(out_root, _slug(task))
        os.makedirs(task_dir, exist_ok=True)
        existing = len([d for d in os.listdir(task_dir)
                        if os.path.isdir(os.path.join(task_dir, d))])
        for ei in range(args.episodes):
            idx = existing + ei
            bag_uri = os.path.join(task_dir, f"episode_{idx:04d}")
            if os.path.exists(bag_uri):
                import shutil; shutil.rmtree(bag_uri)

            writer = rosbag2_py.SequentialWriter()
            writer.open(rosbag2_py.StorageOptions(uri=bag_uri, storage_id="mcap"),
                        rosbag2_py.ConverterOptions("cdr", "cdr"))
            for i, (name, typ) in enumerate(TOPICS):
                writer.create_topic(rosbag2_py.TopicMetadata(
                    id=i, name=name, type=typ, serialization_format="cdr"))

            # per-episode reset + meta (once, at log-time 0)
            mujoco.mj_resetData(model, data)
            mujoco.mj_step(model, data)
            schema_ep = dict(schema, task=task, fps=CFG.LOG_FPS,
                             episode_index=idx, scene=os.path.basename(args.scene))
            meta_payload.data = json.dumps({"task": task, "fps": CFG.LOG_FPS,
                                            "actuator_names": act_names,
                                            "schema_json": json.dumps(schema_ep)})
            writer.write("/episode/meta", serialize_message(meta_payload), 0)

            phase = 7.3 * (ti * 13 + ei + 1)        # distinct trajectory per (task, ep)
            ep_t0 = time.time()
            for fi in range(n_frames):
                tm = phase + fi / CFG.LOG_FPS
                frame = mock_avp.mock_frame(tm)
                if fi == 0:
                    ctrl.calibrate(frame)
                for _ in range(substeps):
                    ctrl.step(frame, None, model.opt.timestep)
                    mujoco.mj_step(model, data)
                obs = prov.observation(record_images=True)
                act = prov.action()
                sim_t = prov.sim_time()
                wall_t = time.time()
                stamp_ns = int(round(sim_t * 1e9))
                msgs = make_msgs(obs, act, sim_t, wall_t)
                for topic, msg in msgs.items():
                    writer.write(topic, serialize_message(msg), stamp_ns)
                    if pubs is not None:
                        try:
                            pubs[topic].publish(msg)
                        except Exception:
                            pass
            del writer                               # finalize the bag

            # session sidecar
            with open(os.path.join(bag_uri, "session.json"), "w") as fp:
                json.dump({"task": task, "episode_index": idx, "fps": CFG.LOG_FPS,
                           "num_frames": n_frames, "scene": os.path.basename(args.scene),
                           "topics": [t for t, _ in TOPICS],
                           "duration_sim_s": round(n_frames / CFG.LOG_FPS, 3)}, fp, indent=2)
            total_eps += 1
            print(f"[collect_ros] [{total_eps}] {task!r} ep{idx:04d} "
                  f"({n_frames} frames, {time.time()-ep_t0:.1f}s) -> {bag_uri}")

    if node is not None:
        import rclpy
        node.destroy_node(); rclpy.shutdown()
    print(f"[collect_ros] DONE: {total_eps} episodes in {time.time()-t_start:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
