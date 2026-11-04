#!/usr/bin/env python3
"""Teleop-loop odometry evaluation for MABEL.

Drives a scripted, collision-safe closed loop on /cmd_vel (so it goes through the
tip-safe cmd_vel_bridge -> /mabel_cmd, exactly like Nav2/keyboard teleop), while
recording every available pose source:

  * /odom_ground_truth        -- MuJoCo truth (reference)
  * /odom                     -- swerve wheel odometry (dead reckoning)
  * /visual_slam/tracking/odometry  -- cuVSLAM visual odometry (if running)

At the end it aligns each trajectory to its own start pose and reports, against
ground truth: absolute trajectory error (ATE RMS) and final-pose drift. A closed
loop returns to the start, so final drift ~= accumulated error. This quantifies
the claim that fused SLAM beats raw dead reckoning or pure visual odometry.

    ros2 run mabel_navigation odom_eval        # (or: python3 odom_eval.py)
"""
import math
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomEval(Node):
    def __init__(self):
        super().__init__("odom_eval")
        self.declare_parameter("linspeed", 0.18)
        self.declare_parameter("rotspeed", 0.5)
        self.declare_parameter("settle", 3.0)
        self.v = float(self.get_parameter("linspeed").value)
        self.w = float(self.get_parameter("rotspeed").value)
        self.settle = float(self.get_parameter("settle").value)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tracks = {"truth": [], "wheel": [], "vslam": []}
        self.create_subscription(Odometry, "/odom_ground_truth",
                                 lambda m: self._rec("truth", m), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom",
                                 lambda m: self._rec("wheel", m), 10)
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry",
                                 lambda m: self._rec("vslam", m), qos_profile_sensor_data)

        # Collision-safe closed loop near the spawn (island front 0.44 m ahead at
        # +x, dining table 1.0 m behind at -x, walls far in +-y): a holonomic box
        # via STRAFE (heading fixed so 'forward' never drives into the island) +
        # an in-place 360 spin for rotational drift and cuVSLAM parallax. Box is
        # 0.8 m (y) x 0.4 m (x), staying in x in [-0.4, 0], y in [0, 0.8].
        # The kitchen island is 0.44 m ahead (+x), so the loop lives in the open
        # -x / +y quadrant. Body yaw starts ~0, so body +x == world +x. Durations
        # are generous (the swerve cruises ~0.15 m/s after the steering settles);
        # exact closure isn't needed -- drift is measured against ground truth.
        # Clean, sustained motions (few direction changes so the swerve cruises):
        # a long back-drive (translation drift) then a 360 in-place spin
        # (rotational drift) -- the two ways odometry drifts. The island is +x so
        # we drive -x (open). ATE compares full trajectories; the path needn't close.
        def leg(dist, v):
            return abs(dist) / v
        spin = 2 * math.pi / self.w
        self.segments = [
            ("settle", 0, 0, 0, self.settle),
            ("back -x 0.5", -self.v, 0, 0, leg(0.5, self.v)),
            ("strafe +y 0.5", 0, self.v, 0, leg(0.5, self.v)),
            ("fwd +x 0.5", self.v, 0, 0, leg(0.5, self.v)),
            ("strafe -y 0.5", 0, -self.v, 0, leg(0.5, self.v)),
            ("spin 360", 0, 0, self.w, spin),
            ("settle", 0, 0, 0, self.settle),
        ]
        self.t0 = self.get_clock().now()
        self.seg_i = 0
        self.seg_t0 = self.t0
        self.create_timer(0.05, self._tick)
        self.get_logger().info("odom_eval: driving the loop...")

    def _rec(self, key, msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        p = msg.pose.pose.position
        self.tracks[key].append((t, p.x, p.y, yaw_of(msg.pose.pose.orientation)))

    def _tick(self):
        now = self.get_clock().now()
        name, vx, vy, wz, dur = self.segments[self.seg_i]
        if (now - self.seg_t0).nanoseconds * 1e-9 >= dur:
            self.seg_i += 1
            self.seg_t0 = now
            if self.seg_i >= len(self.segments):
                self.pub.publish(Twist())
                self._report()
                rclpy.shutdown()
                return
            name = self.segments[self.seg_i][0]
            self.get_logger().info(f"segment: {name}")
        _, vx, vy, wz, _ = self.segments[self.seg_i]
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.pub.publish(m)

    # -------- evaluation --------
    @staticmethod
    def _relativize(track):
        """Express a trajectory relative to its first pose (start at identity)."""
        t0, x0, y0, th0 = track[0]
        c, s = math.cos(-th0), math.sin(-th0)
        out = []
        for (t, x, y, th) in track:
            dx, dy = x - x0, y - y0
            out.append((t, c * dx - s * dy, s * dx + c * dy, th - th0))
        return out

    def _report(self):
        truth = self.tracks["truth"]
        if len(truth) < 10:
            print("\n[odom_eval] not enough ground-truth samples; aborting.")
            return
        rt = self._relativize(truth)
        tt = np.array([p[0] for p in rt])
        tx = np.array([p[1] for p in rt])
        ty = np.array([p[2] for p in rt])

        print("\n================ ODOMETRY ACCURACY vs GROUND TRUTH ================")
        print(f"{'source':<22}{'ATE RMS (m)':>14}{'final drift (m)':>18}{'samples':>10}")
        path_len = float(np.sum(np.hypot(np.diff(tx), np.diff(ty))))
        for key, label in [("wheel", "wheel dead-reckoning"),
                           ("vslam", "cuVSLAM visual odom")]:
            tr = self.tracks[key]
            if len(tr) < 10:
                print(f"{label:<22}{'(not running)':>14}")
                continue
            r = self._relativize(tr)
            et = np.array([p[0] for p in r])
            ex = np.interp(tt, et, [p[1] for p in r])
            ey = np.interp(tt, et, [p[2] for p in r])
            err = np.hypot(ex - tx, ey - ty)
            ate = float(np.sqrt(np.mean(err ** 2)))
            final = float(err[-1])
            print(f"{label:<22}{ate:>14.4f}{final:>18.4f}{len(tr):>10}")
        print(f"\npath length ~ {path_len:.2f} m   (lower error = better)")
        print("==================================================================\n")
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = OdomEval()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
