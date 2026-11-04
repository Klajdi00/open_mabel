"""drivers — the ROS-free device layer of mabel_hw.

One module per transport/protocol; every actuator driver implements the
:class:`mabel_hw.drivers.seam.JointGroupDriver` seam (or its own typed surface
for base/hands/lift/lidar). Drivers speak RAW motor units internally; the
calibrated model-frame view lives one layer up (mabel_hw.subsystems).

Nothing in this package may import rclpy — enforced by tests/test_mabel_hw_pure.py.
"""
from .seam import GroupFeedback, JointGroupDriver, MockJointGroupDriver

__all__ = ["JointGroupDriver", "GroupFeedback", "MockJointGroupDriver",
           "build_joint_driver"]


def build_joint_driver(cfg: dict, links: dict, logger=None) -> JointGroupDriver:
    """The one factory for MIT joint-group backends (arms / neck / torso).

    cfg is a new-taxonomy config dict (config.LEFT_ARM etc.): ``backend`` picks
    the transport, ``joints`` the canonical names. Backends import lazily so a
    CAN-less (or protobuf-less) host can still build the other kind.

    SAFETY: backend="teensy" ENABLES the motors on link connect (damping hold)
    — only call for subsystems the config enables. backend="openarm" and
    backend="dynamixel" do NOT energize on connect (they stream nothing until
    set_torque). backend="mock" returns the deterministic test driver.

    backend="dynamixel" (the serial neck): ROBOTIS Protocol 2.0 X-series servos on
    a shared TTL bus (drivers/dynamixel.py); ``dxl_ids`` = "<joint>:<id>, ...",
    ``device``/``device_serial``/``baud`` pick the adapter.
    """
    backend = cfg.get("backend", "can")
    if backend == "openarm":
        from .openarm_link import OpenArmLink
        link_cfg = links[cfg["link"]]
        link = OpenArmLink.get(link_cfg["ip"], link_cfg["port"],
                               link_cfg["stream_hz"], logger=logger)
        link.connect()                       # socket only; motors stay limp
        # wire_t_max: one register T_MAX for every joint (torso), or a per-joint
        # map — the neck checks only its DaMiao yaw; its Dynamixels advertise 0 by
        # design and must not be held to a limit they cannot describe.
        wtm = cfg.get("wire_t_max")
        if isinstance(wtm, dict):
            wtm = {j: float(v) for j, v in wtm.items() if j in cfg["joints"]} or None
        elif wtm:
            wtm = {j: float(wtm) for j in cfg["joints"]}
        else:
            wtm = None
        return link.group(cfg["joints"], kp=cfg.get("kp", 30.0),
                          kd=cfg.get("kd", 1.0), wire_t_max=wtm)
    if backend == "teensy":
        from ..errors import SubsystemUnavailable
        from .teensy_link import TeensyLink
        link_cfg = links[cfg["link"]]
        link = TeensyLink.get(link_cfg["ip"], link_cfg["port"],
                              link_cfg["stream_hz"], logger=logger)
        try:
            link.connect()
        except SubsystemUnavailable as e:
            # Board absent right now: hand back an UNRESOLVED group — it
            # hot-joins via service() the moment the Teensy appears.
            if logger:
                logger(f"[hw] {e} — group created unresolved, will hot-join")
        return link.group(cfg["joints"], kp=cfg.get("kp", 30.0),
                          kd=cfg.get("kd", 1.0), strict=False)
    if backend == "dynamixel":
        # ROBOTIS Protocol 2.0 servos on a shared TTL bus (MABEL's serial neck:
        # X-series XC330 pitch/roll). Does NOT energize on connect. The id map is
        # "<joint>:<id>, ..."; falls back to joints enumerated from id 1.
        from .dynamixel import DynamixelGroup
        spec = cfg.get("dxl_ids", "")
        id_map = {}
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            name, _, sid = tok.partition(":")
            id_map[name.strip()] = int(sid.strip())
        if not id_map:
            id_map = {j: k + 1 for k, j in enumerate(cfg["joints"])}
        group = DynamixelGroup(id_map, device=cfg.get("device", ""),
                               baud=int(cfg.get("baud", 1000000)),
                               serial_id=cfg.get("device_serial", ""),
                               logger=logger)
        # The neck spans TWO buses: pitch/roll Dynamixels here + the yaw DM-J4310
        # on a second bus. When the config declares that yaw joint (NECK_YAW),
        # merge both behind one CompositeJointGroup so the subsystem stays a
        # single calibrated group. Either bus may be absent (hot-join).
        # yaw_backend selects the yaw wire: "openarm" (2026-07-28 wiring — the
        # motor hangs on the arms Teensy's torso bus, advertised as `neck_yaw`
        # in the SD-card robot.json manifest; does NOT energize) or "fdcan"
        # (the DM-USB2FDCAN dongle).
        if "neck_yaw" in cfg.get("joints", ()):
            yaw = None
            if cfg.get("yaw_backend") == "openarm":
                from .openarm_link import OpenArmLink
                link_cfg = links[cfg.get("link", "arms_openarm")]
                link = OpenArmLink.get(link_cfg["ip"], link_cfg["port"],
                                       link_cfg["stream_hz"], logger=logger)
                try:
                    link.connect()               # socket only; motors stay limp
                except Exception:                # absent board: group hot-joins
                    pass
                yaw = link.group(
                    ["neck_yaw"], kp=cfg.get("yaw_kp", 20.0),
                    kd=cfg.get("yaw_kd", 1.0),
                    wire_t_max={"neck_yaw": float(cfg.get("yaw_wire_t_max", 10.0))})
            elif cfg.get("yaw_backend") == "fdcan":
                from .damiao_fdcan import DamiaoFdcanDriver
                from .solo import DamiaoSoloJoint
                yaw = DamiaoSoloJoint(
                    DamiaoFdcanDriver(can_id=int(cfg.get("yaw_can_id", 2)),
                                      kp=cfg.get("yaw_kp", 20.0),
                                      kd=cfg.get("yaw_kd", 1.0),
                                      p_max=cfg.get("yaw_p_max", 12.5),
                                      v_max=cfg.get("yaw_v_max", 30.0),
                                      t_max=cfg.get("yaw_t_max", 10.0),
                                      logger=logger),
                    "neck_yaw", slew=cfg.get("yaw_slew", 1.5), logger=logger)
            if yaw is not None:
                from .solo import CompositeJointGroup
                return CompositeJointGroup([yaw, group], logger=logger)
        return group
    if backend == "serial":
        # One DaMiao on the HDSC USB-CAN *serial* dongle (MABEL's torso). Does NOT
        # energize on connect (drivers/damiao_serial.py safety contract); the solo
        # adapter caches feedback so seam read() never blocks on the wire.
        from .damiao_serial import DamiaoSerialDriver
        from .solo import DamiaoSoloJoint
        return DamiaoSoloJoint(
            DamiaoSerialDriver(port=cfg.get("port", "auto"),
                               baud=int(cfg.get("serial_baud", 921600)),
                               can_id=int(cfg.get("can_id", 1)),
                               kp=cfg.get("kp", 30.0), kd=cfg.get("kd", 1.0),
                               p_max=cfg.get("p_max", 12.5),
                               v_max=cfg.get("v_max", 30.0),
                               t_max=cfg.get("t_max", 10.0), logger=logger),
            cfg["joints"][0], slew=cfg.get("slew", 1.0), logger=logger)
    if backend == "fdcan":
        # One DaMiao on the DM-USB2FDCAN dongle (libusb backend). Non-energizing.
        from .damiao_fdcan import DamiaoFdcanDriver
        from .solo import DamiaoSoloJoint
        return DamiaoSoloJoint(
            DamiaoFdcanDriver(can_id=int(cfg.get("can_id", 2)),
                              kp=cfg.get("kp", 20.0), kd=cfg.get("kd", 1.0),
                              p_max=cfg.get("p_max", 12.5),
                              v_max=cfg.get("v_max", 30.0),
                              t_max=cfg.get("t_max", 10.0), logger=logger),
            cfg["joints"][0], slew=cfg.get("slew", 1.0), logger=logger)
    if backend == "can":
        from .damiao_can import CanLink
        return CanLink(cfg["joint_motors"], kp=cfg.get("kp", 30.0),
                       kd=cfg.get("kd", 1.0), p_max=cfg.get("p_max", 12.5),
                       v_max=cfg.get("v_max", 30.0), t_max=cfg.get("t_max", 10.0),
                       bustype=cfg.get("bustype", "socketcan"), logger=logger)
    if backend == "mock":
        return MockJointGroupDriver(cfg["joints"])
    raise ValueError(f"unknown joint-group backend {backend!r} (expected 'openarm', "
                     "'dynamixel', 'serial', 'fdcan', 'teensy', 'can', or 'mock')")
