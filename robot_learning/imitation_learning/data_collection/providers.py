"""
State providers: turn a live robot (sim or real) into the *uniform* per-frame
observation/action dict that the EpisodeRecorder logs.

The whole point of this abstraction is the lab-notebook requirement that sim and
real share logic: ``collect_sim.py`` and ``collect_real.py`` build different
providers but log through the identical recorder + schema, so a model trained on
sim data sees the same key layout when deployed on hardware.

A provider implements:
    observation(record_images: bool) -> dict[str, np.ndarray | bytes]
    action()                         -> dict[str, np.ndarray]
    schema_meta()                    -> dict   (names/ordering written once per ep)
    sim_time()                       -> float
"""
import numpy as np

from . import config as CFG


def _mat2quat(mat9):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=float).flatten())
    return q


class SimStateProvider:
    """Reads MuJoCo MjData + the shared TeleopController. Renders the on-board
    cameras on demand (RGB JPEG-encoded, depth as uint16 millimetres)."""

    def __init__(self, model, data, controller):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.data = data
        self.controller = controller

        # ── camera renderers, reused across same-resolution cameras ──────────
        self._rgb_ren = {}
        self._depth_ren = {}
        self._cams = []
        for cam in CFG.CAMERAS:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"])
            if cid < 0:
                print(f"[provider] WARNING: camera '{cam['name']}' not in model — skipped")
                continue
            wh = (cam["w"], cam["h"])
            if cam["rgb"] and wh not in self._rgb_ren:
                self._rgb_ren[wh] = mujoco.Renderer(model, height=cam["h"], width=cam["w"])
            if cam["depth"] and wh not in self._depth_ren:
                r = mujoco.Renderer(model, height=cam["h"], width=cam["w"])
                r.enable_depth_rendering()
                self._depth_ren[wh] = r
            self._cams.append((cam, wh))

        # ── actuator -> measured joint qpos/qvel addresses (per-joint state) ──
        self._act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                           for i in range(model.nu)]
        self._qpos_adr, self._qvel_adr = [], []
        for i in range(model.nu):
            jid = self._joint_for_actuator(i)
            self._qpos_adr.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
            self._qvel_adr.append(int(model.jnt_dofadr[jid]) if jid >= 0 else -1)

        # ── lidar sensor address block ───────────────────────────────────────
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{CFG.LIDAR_PREFIX}000")
        if first >= 0:
            self._lidar_adr = int(model.sensor_adr[first])
            self._lidar_n = CFG.LIDAR_N_RAYS
        else:
            self._lidar_adr, self._lidar_n = -1, 0

        # ── tracked bodies ───────────────────────────────────────────────────
        self._bodies = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v)
                        for k, v in CFG.TRACKED_BODIES.items()}

    # ── helpers ───────────────────────────────────────────────────────────────
    def _joint_for_actuator(self, i):
        """Joint id that an actuator's state should be read from. Position/velocity
        actuators map straight to their joint; the Lift tendon maps to Lift Lower."""
        mj, m = self._mj, self.model
        trntype = m.actuator_trntype[i]
        if trntype == mj.mjtTrn.mjTRN_JOINT:
            return int(m.actuator_trnid[i, 0])
        if self._act_names[i] == "Lift":
            return mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, CFG.LIFT_JOINT)
        return -1

    def _render_rgb(self, cam, wh):
        ren = self._rgb_ren[wh]
        ren.update_scene(self.data, camera=cam["name"])
        return ren.render()                       # HxWx3 uint8 RGB

    def _render_depth(self, cam, wh):
        ren = self._depth_ren[wh]
        ren.update_scene(self.data, camera=cam["name"])
        d = ren.render()                          # HxW float32 metres
        d = np.clip(d, 0.0, CFG.DEPTH_MAX_M) * CFG.DEPTH_SCALE
        return d.astype(np.uint16)

    # ── observation ─────────────────────────────────────────────────────────-
    def observation(self, record_images=True):
        import cv2
        d, m = self.data, self.model
        obs = {}

        if record_images:
            for cam, wh in self._cams:
                if cam["rgb"]:
                    rgb = self._render_rgb(cam, wh)
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                           [int(cv2.IMWRITE_JPEG_QUALITY), CFG.JPEG_QUALITY])
                    obs[f"obs/cam/{cam['name']}/rgb"] = jpg.flatten()
                if cam["depth"]:
                    obs[f"obs/cam/{cam['name']}/depth"] = self._render_depth(cam, wh)

        # lidar (metres; cutoff/no-hit reads as the sensor cutoff value)
        if self._lidar_adr >= 0:
            obs["obs/lidar"] = d.sensordata[self._lidar_adr:self._lidar_adr + self._lidar_n].astype(np.float32)

        # per-actuated-joint state (59-vec, aligned with data.ctrl / action/ctrl)
        jpos = np.array([d.qpos[a] if a >= 0 else np.nan for a in self._qpos_adr], dtype=np.float32)
        jvel = np.array([d.qvel[a] if a >= 0 else np.nan for a in self._qvel_adr], dtype=np.float32)
        obs["obs/joint_pos"] = jpos
        obs["obs/joint_vel"] = jvel
        obs["obs/joint_torque"] = d.actuator_force.astype(np.float32).copy()   # applied actuator force/torque

        # full generalized state (unambiguous, sim ground truth)
        obs["obs/qpos"] = d.qpos.astype(np.float32).copy()
        obs["obs/qvel"] = d.qvel.astype(np.float32).copy()

        # tracked body world poses (pos[3] + quat[4])
        for key, bid in self._bodies.items():
            if bid >= 0:
                obs[f"obs/pose/{key}"] = np.concatenate(
                    [d.xpos[bid], d.xquat[bid]]).astype(np.float32)

        # measured base twist (chassis spatial velocity, world frame): [vx,vy,wz]
        cid = self._bodies.get("chassis", -1)
        if cid >= 0:
            cvel = d.cvel[cid]            # [ang(3), lin(3)] in world
            obs["obs/base_twist"] = np.array([cvel[3], cvel[4], cvel[2]], dtype=np.float32)
        return obs

    # ── action / operator intent ───────────────────────────────────────────--
    def action(self):
        d = self.data
        ctrl = d.ctrl.astype(np.float32).copy()    # target for EVERY actuator
        act = {"action/ctrl": ctrl}

        # operator teleop intent from the shared controller (sim-only richness)
        ik = self.controller.ik
        act["teleop/left_palm_target"] = np.concatenate(
            [ik.target_l_pos, _mat2quat(ik.target_l_mat)]).astype(np.float32)
        act["teleop/right_palm_target"] = np.concatenate(
            [ik.target_r_pos, _mat2quat(ik.target_r_mat)]).astype(np.float32)
        act["teleop/grasp"] = np.array([self.controller.grasp_l,
                                        self.controller.grasp_r], dtype=np.float32)
        act["teleop/base_cmd"] = np.array(
            [self.controller.vx, self.controller.vy, self.controller.omega], dtype=np.float32)
        gaze = getattr(self.controller.neck, "look_at", None)
        if gaze is not None:
            act["teleop/look_at"] = np.asarray(gaze, dtype=np.float32)
        act["teleop/mode"] = np.array(
            [0 if self.controller.mode == self.controller.MODE_MANIP else 1], dtype=np.int8)
        return act

    def sim_time(self):
        return float(self.data.time)

    # ── schema written once per episode (names + ordering for downstream) ─────
    def schema_meta(self):
        return {
            "source": "sim",
            "actuator_names": self._act_names,                # ctrl / joint_* ordering
            "actuator_groups": {
                "base": CFG.BASE_ACTUATORS, "lift_torso": CFG.LIFT_TORSO,
                "neck": CFG.NECK, "left_arm": CFG.LEFT_ARM, "left_hand": CFG.LEFT_HAND,
                "right_arm": CFG.RIGHT_ARM, "right_hand": CFG.RIGHT_HAND,
            },
            "cameras": [{"name": c["name"], "rgb": c["rgb"], "depth": c["depth"],
                         "w": c["w"], "h": c["h"]} for c, _ in self._cams],
            "lidar": {"n_rays": self._lidar_n, "angle_res_deg": CFG.LIDAR_ANGLE_RES_DEG,
                      "units": "metres", "cutoff_is_no_hit": True},
            "depth_encoding": {"dtype": "uint16", "scale": CFG.DEPTH_SCALE,
                               "max_m": CFG.DEPTH_MAX_M, "units": "millimetres"},
            "nq": int(self.model.nq), "nv": int(self.model.nv), "nu": int(self.model.nu),
        }


class RealStateProvider:
    """Hardware counterpart — same observation/action keys, filled from the real
    robot's sensor streams instead of MuJoCo.

    This is a documented skeleton: MABEL's real-robot I/O (motor bus, ZED stereo,
    RealSense wrist cams, RPLIDAR A3, IMU) is not present in this repo, so wire
    each ``# TODO`` to your driver / ROS subscriber. Keeping the SAME keys as
    SimStateProvider is what lets a policy trained in sim deploy unchanged.

    Expected wiring:
      * cameras  : grab latest frame from each device, JPEG-encode RGB, scale
                   depth to uint16 mm exactly as the sim provider does.
      * joints   : read pos/vel/torque from the motor bus in CFG actuator order.
      * lidar    : latest 360-ray scan (metres).
      * action   : the *commanded* target sent to the motors this tick, plus the
                   teleop intent from the shared TeleopController (real teleop
                   drives the very same controller class).
    """

    def __init__(self, controller=None, hw=None):
        self.controller = controller
        self.hw = hw                      # your hardware interface handle
        self._t0 = None

    def observation(self, record_images=True):
        raise NotImplementedError(
            "Wire RealStateProvider to MABEL hardware drivers — see the docstring. "
            "It must return the same keys as SimStateProvider.observation().")

    def action(self):
        raise NotImplementedError("Return action/ctrl + teleop/* from the real controller.")

    def sim_time(self):
        import time
        if self._t0 is None:
            self._t0 = time.time()
        return time.time() - self._t0

    def schema_meta(self):
        return {"source": "real"}
