"""
ZMQ subscriber for the Apple Vision Pro stream.

The publisher (scripts/vision_pro_stream.py, or dexterous_hand/vision_pro/
connect_full_avp.py) broadcasts a JSON payload over a PUB socket every frame:
    {
      "head_mat":        [16 floats]          # 4x4 row-major, AVP/ARKit frame
      "right_hand_mats": [[16], ... x21],      # MediaPipe 21-joint, [0]=wrist
      "left_hand_mats":  [[16], ... x21],      # optional
      "right_joints":    {wire_name: [16], ...} # full named ARKit skeleton (optional)
      "left_joints":     {wire_name: [16], ...} # for per-finger retargeting
      "timestamp":       float
    }

`right_joints` / `left_joints` carry the full named hand skeleton (wire names
"wrist", "index_knuckle", "thumb_intermediate_base", "little_tip", ...) that the
per-finger retargeters (hand_ik_retarget.HandIKRetargeter) consume; they are
absent when the source streams only wrist poses, in which case the consumer
falls back to the pinch-grasp scalar derived from the indexed mats.

This client keeps only the latest frame (conflate), so the control loop never
lags behind the stream.
"""
import json
import numpy as np
import zmq

from .. import config as C


class AVPFrame:
    """Parsed Vision Pro frame with convenience accessors (AVP frame, 4x4)."""

    def __init__(self, payload):
        self.timestamp = payload.get("timestamp", 0.0)
        self.head = self._mat(payload.get("head_mat"))
        self.right = self._mats(payload.get("right_hand_mats"))
        self.left = self._mats(payload.get("left_hand_mats"))
        # Full named skeletons for per-finger retargeting (absent on wrist-only
        # sources). TeleopController reads frame.right_joints / left_joints.
        self.right_joints = self._joints(payload.get("right_joints"))
        self.left_joints = self._joints(payload.get("left_joints"))

    @staticmethod
    def _mat(flat):
        if flat is None:
            return None
        return np.asarray(flat, dtype=float).reshape(4, 4)

    @staticmethod
    def _mats(list_of_flat):
        if not list_of_flat:
            return None
        return [np.asarray(m, dtype=float).reshape(4, 4) for m in list_of_flat]

    @staticmethod
    def _joints(name_to_flat):
        """{wire_name: [16]} -> {wire_name: 4x4 np.ndarray}; None if absent."""
        if not name_to_flat:
            return None
        return {k: np.asarray(v, dtype=float).reshape(4, 4)
                for k, v in name_to_flat.items()}


class AVPReceiver:
    def __init__(self, addr=C.AVP_ZMQ_ADDR):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)        # keep only the newest message
        self.sock.setsockopt_string(zmq.SUBSCRIBE, C.AVP_ZMQ_TOPIC)
        self.sock.connect(addr)
        self.addr = addr
        self.latest = None

    def poll(self):
        """Drain the socket; return the newest AVPFrame or None if nothing new."""
        got = None
        while True:
            try:
                msg = self.sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            got = msg
        if got is not None:
            self.latest = AVPFrame(json.loads(got))
        return self.latest

    def close(self):
        self.sock.close(0)
