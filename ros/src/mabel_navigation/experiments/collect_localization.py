#!/usr/bin/env python3
"""Drive an adaptive trajectory and record GT + wheel + visual odometry simultaneously.

Saves /tmp/loc_data.npz with synchronized (t, x, y, yaw) tracks for:
  gt     = /odom_ground_truth          (truth)
  wheel  = /odom                       (swerve wheel odometry)
  vo     = /visual_slam/tracking/odometry   (cuVSLAM visual odometry)
plus the raw wheel twist (vx,vy,w) so an offline EKF can re-predict from it.
"""
import math, time, numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from mabel_interfaces.msg import MabelSwerveCmd


def yaw_of(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


class Collector(Node):
    def __init__(self):
        super().__init__("loc_collector")
        self.gt = self.wh = self.vo = None
        self.wh_tw = (0.0, 0.0, 0.0)
        self.create_subscription(Odometry, "/odom_ground_truth", self._gt, 20)
        self.create_subscription(Odometry, "/odom", self._wh, 20)
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._vo, 20)
        self.pub = self.create_publisher(MabelSwerveCmd, "/mabel_cmd", 10)

    def _gt(self, m):
        p = m.pose.pose.position; self.gt = (p.x, p.y, yaw_of(m.pose.pose.orientation))
    def _wh(self, m):
        p = m.pose.pose.position; self.wh = (p.x, p.y, yaw_of(m.pose.pose.orientation))
        t = m.twist.twist; self.wh_tw = (t.linear.x, t.linear.y, t.angular.z)
    def _vo(self, m):
        p = m.pose.pose.position; self.vo = (p.x, p.y, yaw_of(m.pose.pose.orientation))

    def drive(self, vx, vy=0.0, w=0.0):
        c = MabelSwerveCmd()
        c.vx = float(vx); c.vy = float(vy); c.omega = float(w)
        c.lift_pos = 0.0; c.hardware_state = 0
        self.pub.publish(c)

    def settle(self, s):
        t = time.time()
        while time.time()-t < s:
            rclpy.spin_once(self, timeout_sec=0.02)


def main():
    rclpy.init()
    n = Collector()
    # wait for all three streams
    t0 = time.time()
    while (n.gt is None or n.wh is None or n.vo is None) and time.time()-t0 < 30:
        n.settle(0.2)
    if n.gt is None or n.wh is None or n.vo is None:
        print("missing streams:", "gt" if n.gt is None else "", "wh" if n.wh is None else "",
              "vo" if n.vo is None else ""); return
    print("all streams up; ROAMING with stuck-recovery for ~70s")

    # Roam: drive the robot FORWARD (vx<0); when ground truth stalls (hit furniture)
    # rotate in place to a new heading and continue. This keeps it moving through the
    # rooms -> several metres of translation + plenty of heading change, which is what
    # differentiates the estimators.
    # Sustained forward/reverse legs (no rotation -> the swerve aligns ONCE per leg
    # and then drives; rotation would force re-alignment and stall). The base slips
    # heavily, so legs are long; the back-and-forth still accumulates enough true
    # motion to expose wheel-odom over-read vs visual tracking. A couple of arc legs
    # add heading change.
    legs = [(-0.5, 0.0, 0.0, 14), (+0.5, 0.0, 0.0, 14),     # fwd / rev
            (-0.4, 0.0, 0.30, 12),                          # arc (adds yaw)
            (+0.4, 0.0, -0.30, 12),                         # arc back
            (-0.5, 0.0, 0.0, 14), (+0.5, 0.0, 0.0, 14)]     # fwd / rev
    rec = []  # (t, gt(3), wh(3), vo(3), twist(3))
    start = time.time()
    for vx, vy, w, dur in legs:
        tseg = time.time()
        while time.time()-tseg < dur:
            n.drive(vx, vy, w)
            n.settle(0.06)
            rec.append((time.time()-start, *n.gt, *n.wh, *n.vo, *n.wh_tw))
    n.drive(0.0); n.settle(0.5)
    rec.append((time.time()-start, *n.gt, *n.wh, *n.vo, *n.wh_tw))
    n.drive(0.0); n.settle(0.5)
    rec.append((time.time()-start, *n.gt, *n.wh, *n.vo, *n.wh_tw))
    arr = np.array(rec)
    np.savez("/tmp/loc_data.npz", t=arr[:,0], gt=arr[:,1:4], wh=arr[:,4:7],
             vo=arr[:,7:10], tw=arr[:,10:13])
    # quick GT path length so we know the robot actually moved
    gtxy = arr[:,1:3]
    plen = np.sum(np.linalg.norm(np.diff(gtxy, axis=0), axis=1))
    print(f"recorded {len(rec)} samples over {arr[-1,0]:.1f}s; GT path length={plen:.2f} m")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
