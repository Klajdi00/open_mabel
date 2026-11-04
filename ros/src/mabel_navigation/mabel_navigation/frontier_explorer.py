"""Frontier-based autonomous exploration for MABEL (Yamauchi-style).

Drives the robot to map an unknown space END-TO-END with no human input: it
watches the live SLAM occupancy grid, finds the boundaries between KNOWN-FREE and
UNKNOWN space ("frontiers"), repeatedly sends the best one to Nav2 as a
``NavigateToPose`` goal, and stops when no reachable frontier remains -- i.e. the
environment is fully mapped.

It is deliberately SLAM- and hardware-agnostic.  Its only inputs are:
  * ``/map``                 (nav_msgs/OccupancyGrid) -- from EITHER SLAM backend
                             (slam_toolbox directly, or nvblox via slice_to_occupancy)
  * the ``map -> base_link`` TF  -- the robot's pose (from SLAM + wheel odometry)
and its only output is the ``navigate_to_pose`` action it shares with the RViz
goal tool, so the exact same Nav2 stack serves manual driving and exploration.
It never talks to a camera, a LiDAR, the base, or any SLAM node directly.

Algorithm per cycle (``planner_frequency`` Hz):
  1. If a goal is in flight, let it run (cancel + blacklist only on timeout).
  2. Otherwise classify every grid cell free / occupied / unknown, find frontier
     cells (free with an unknown neighbour), connected-component cluster them,
     drop clusters smaller than ``min_frontier_size``.
  3. Score each surviving cluster by travel cost minus an information-gain bonus
     (bigger frontier = more new map), skip any near a blacklisted point, and
     send the best centroid as a goal (oriented from the robot toward it).
  4. When N consecutive cycles find nothing new, declare exploration COMPLETE
     (latched on ``~/exploration_complete``) and optionally return home.

Robustness: goals that abort or time out are blacklisted (within
``blacklist_radius``) so the robot never loops on an unreachable frontier; the
search resumes the moment the map reveals a fresh one.
"""

import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, \
    ExtrapolationException


def _yaw_to_quat(yaw):
    """Quaternion (x, y, z, w) for a planar yaw."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class Frontier:
    """One connected cluster of frontier cells."""
    __slots__ = ("size", "cx", "cy", "wx", "wy")

    def __init__(self, size, cx, cy, wx, wy):
        self.size = size        # number of frontier cells (information-gain proxy)
        self.cx = cx            # centroid, grid cells
        self.cy = cy
        self.wx = wx            # centroid, world metres (map frame)
        self.wy = wy


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        # ---- parameters -----------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("nav_action", "navigate_to_pose")
        self.declare_parameter("planner_frequency", 0.5)     # exploration cycles / s
        self.declare_parameter("min_frontier_size", 12)      # cells; smaller = noise
        self.declare_parameter("occupied_threshold", 50)     # >=this (0-100) = obstacle
        self.declare_parameter("gain_scale", 0.5)            # weight of frontier size vs distance
        self.declare_parameter("blacklist_radius", 0.45)     # m; failed-goal exclusion radius
        self.declare_parameter("goal_timeout", 25.0)         # s; cancel + blacklist if exceeded
        self.declare_parameter("progress_timeout", 8.0)      # s without getting closer -> abort
        #   ^ kept SHORT so an unreachable frontier (planner can't path to it) is
        #     abandoned + blacklisted in ~8 s and the robot moves to the next reachable
        #     one, instead of idling ~50 s while the BT churns on a dead goal.
        self.declare_parameter("min_goal_distance", 0.30)    # m; ignore frontiers basically underfoot
        self.declare_parameter("return_to_start", False)     # drive home when done
        self.declare_parameter("visualize", True)
        # Auto-save the finished map (PGM + YAML, nav2 map_server format) when
        # exploration completes -- no map_saver_cli/service needed, written straight
        # from the last /map this node already holds. map_save_path is the basename
        # (no extension); '~' is expanded. Set save_map_on_complete False to skip.
        self.declare_parameter("save_map_on_complete", True)
        self.declare_parameter("map_save_path", "~/mabel_map")
        self.declare_parameter("map_free_thresh", 0.25)      # <= -> free (white), nav2 default
        self.declare_parameter("map_occupied_thresh", 0.65)  # >= -> occupied (black), nav2 default
        # Also save the nvblox 3D reconstruction alongside the 2D map: <base>.ply
        # (mesh point cloud, loaded by the /map stream as a known room) + <base>.nvblx
        # (serialized map for relocalization). nvblox's save services live in the APT
        # ROS env (nvblox_msgs not importable here), so we shell out through slam_env.
        self.declare_parameter("save_mesh_on_complete", True)
        self.declare_parameter("nvblox_save_env",            # sourced before ros2 service call
                               os.path.expanduser("~/open_mabel/ros/scripts/slam_env.sh"))
        # Hold ALL exploration goals until the body is in its navigation pose (arms
        # tucked down) -- nav_pose_setter latches /mabel/body_ready True when the
        # arms are down. Set wait_for_body_ready False to explore immediately.
        self.declare_parameter("wait_for_body_ready", True)
        self.declare_parameter("body_ready_topic", "/mabel/body_ready")
        # Anti-premature-completion: slam_toolbox needs a few seconds to turn the
        # first scans into a map, during which there are momentarily no frontiers.
        # Never declare "done" before exploring: require this many consecutive empty
        # cycles AND either a goal already pursued or a startup grace elapsed.
        self.declare_parameter("complete_patience", 4)       # empty cycles before done
        self.declare_parameter("startup_grace", 20.0)        # s to wait for the first map
        # Before believing "nothing to explore", the map must have STOPPED growing
        # for this long. slam_toolbox keeps expanding the occupancy grid for a while
        # after start; completing on a time-grace alone declared victory while the
        # first frontiers were still appearing. Map-still-growing => keep waiting.
        self.declare_parameter("map_settle_time", 10.0)      # s of no map growth

        self.map_topic = self.get_parameter("map_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("robot_base_frame").value
        nav_action = self.get_parameter("nav_action").value
        self.period = 1.0 / max(float(self.get_parameter("planner_frequency").value), 0.05)
        self.min_size = int(self.get_parameter("min_frontier_size").value)
        self.occ_thr = int(self.get_parameter("occupied_threshold").value)
        self.gain_scale = float(self.get_parameter("gain_scale").value)
        self.blacklist_radius = float(self.get_parameter("blacklist_radius").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)
        self.progress_timeout = float(self.get_parameter("progress_timeout").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.return_to_start = bool(self.get_parameter("return_to_start").value)
        self.visualize = bool(self.get_parameter("visualize").value)
        self.save_map_on_complete = bool(self.get_parameter("save_map_on_complete").value)
        self.map_save_path = str(self.get_parameter("map_save_path").value)
        self.map_free_thresh = float(self.get_parameter("map_free_thresh").value)
        self.map_occupied_thresh = float(self.get_parameter("map_occupied_thresh").value)
        self.save_mesh_on_complete = bool(self.get_parameter("save_mesh_on_complete").value)
        self.nvblox_save_env = str(self.get_parameter("nvblox_save_env").value)
        self.complete_patience = int(self.get_parameter("complete_patience").value)
        self.startup_grace = float(self.get_parameter("startup_grace").value)
        self.map_settle_time = float(self.get_parameter("map_settle_time").value)
        self.wait_for_body_ready = bool(self.get_parameter("wait_for_body_ready").value)
        body_ready_topic = self.get_parameter("body_ready_topic").value

        # ---- state ----------------------------------------------------------
        self._map = None
        self._blacklist = []            # [(wx, wy), ...] world points to avoid
        self._goal_handle = None        # active NavigateToPose goal handle
        self._goal_xy = None            # (wx, wy) of the active goal
        self._goal_sent_t = None        # wall time the goal was accepted
        self._goal_best_dist = None     # closest range to the goal seen so far
        self._goal_best_t = None        # wall time of that closest approach
        self._empty_cycles = 0
        self._complete = False
        self._home = None               # robot pose at first successful pose read
        self._goals_sent = 0            # frontiers actually pursued (gate completion)
        self._start_t = None            # set on the first real explore cycle
        self._free_count = 0            # free cells in the latest map (growth gauge)
        self._max_free = 0              # peak free-cell count seen
        self._last_growth_t = None      # last time the map grew (settle detector)
        # Gate: do not chase any frontier until the body is in its navigation pose.
        self._body_ready = not self.wait_for_body_ready

        # ---- ROS I/O --------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # SLAM latches /map with TRANSIENT_LOCAL durability -- match it or we get nothing.
        map_qos = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)

        # nav_pose_setter latches /mabel/body_ready (TRANSIENT_LOCAL) True once the
        # arms are tucked into the navigation pose; match its durability so we get
        # the retained value even though it was published before we subscribed.
        if self.wait_for_body_ready:
            ready_qos = QoSProfile(
                depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST)
            self.create_subscription(Bool, body_ready_topic, self._on_body_ready, ready_qos)

        self._done_pub = self.create_publisher(
            Bool, "~/exploration_complete",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       history=QoSHistoryPolicy.KEEP_LAST))
        if self.visualize:
            self._marker_pub = self.create_publisher(MarkerArray, "~/frontiers", 1)

        self._nav = ActionClient(self, NavigateToPose, nav_action)

        self.get_logger().info(
            f"Frontier explorer up: map={self.map_topic} action={nav_action} "
            f"min_frontier={self.min_size} cells, gain_scale={self.gain_scale}. "
            f"Waiting for the Nav2 '{nav_action}' action server...")
        # Kick the loop off on a timer; the first cycles just wait for map + server.
        self.create_timer(self.period, self._explore)
        self._monitor_timer = self.create_timer(1.0, self._monitor_goal)

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid):
        self._map = msg

    def _on_body_ready(self, msg: Bool):
        if msg.data and not self._body_ready:
            self.get_logger().info("body ready (arms in navigation pose) -- exploring.")
        self._body_ready = bool(msg.data)

    def _robot_xy(self):
        """Robot (x, y) in the global frame from TF, or None if unavailable."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    # --------------------------------------------------------------- main loop
    def _explore(self):
        if self._complete:
            return
        if not self._body_ready:                # arms not yet tucked into the nav pose
            self.get_logger().info(
                "holding -- waiting for the body to reach its navigation pose "
                "(/mabel/body_ready) before sending any frontier goal.",
                throttle_duration_sec=5.0)
            return
        if self._goal_handle is not None:       # a goal is in flight; _monitor_goal owns it
            return
        if self._map is None:
            return
        if not self._nav.server_is_ready():
            return

        robot = self._robot_xy()
        if robot is None:
            return
        if self._home is None:
            self._home = robot
        if self._start_t is None:
            self._start_t = self._now()

        frontiers = self._find_frontiers(self._map, robot)  # refreshes self._free_count
        goal = self._select(frontiers, robot)
        if self.visualize:
            self._publish_markers(frontiers, goal)

        # Track map growth: while slam_toolbox is still expanding the grid, the map
        # is NOT settled and we must not conclude there is nothing to explore.
        now = self._now()
        if self._free_count > self._max_free:
            self._max_free = self._free_count
            self._last_growth_t = now
        if self._last_growth_t is None:
            self._last_growth_t = now

        if goal is None:
            self._empty_cycles += 1
            elapsed = now - self._start_t
            explored = self._goals_sent > 0
            map_settled = (now - self._last_growth_t) >= self.map_settle_time
            # Declare DONE only when frontiers are genuinely exhausted:
            #   * enough consecutive empty cycles (not a one-off map gap), AND
            #   * EITHER we've already chased a frontier (real exhaustion after
            #     exploring) OR the startup grace passed AND the map has stopped
            #     growing (it had time to build and there is simply nothing
            #     reachable -- e.g. a closed room). The map-settled gate is what
            #     stops the old bug of announcing "complete" while the very first
            #     frontiers were still materialising.
            if self._empty_cycles >= self.complete_patience and (
                    explored or (elapsed >= self.startup_grace and map_settled)):
                self._on_complete(robot)
            elif not explored:
                self.get_logger().info(
                    f"no frontiers yet -- map building (free={self._free_count} cells, "
                    f"{elapsed:.0f}s, settled={map_settled})",
                    throttle_duration_sec=4.0)
            return
        self._empty_cycles = 0
        self._send_goal(goal, robot)

    # -------------------------------------------------------- frontier finding
    def _find_frontiers(self, grid_msg: OccupancyGrid, robot):
        rx, ry = robot
        info = grid_msg.info
        w, h = info.width, info.height
        if w == 0 or h == 0:
            return []
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        g = np.asarray(grid_msg.data, dtype=np.int16).reshape(h, w)

        free = (g >= 0) & (g < self.occ_thr)
        unknown = (g < 0)
        self._free_count = int(free.sum())      # map-growth gauge (see _explore)

        # A frontier cell is FREE and 4-adjacent to UNKNOWN. Shift the unknown mask
        # in each cardinal direction and OR -- fully vectorised, no per-cell loop.
        adj = np.zeros((h, w), dtype=bool)
        adj[1:, :] |= unknown[:-1, :]
        adj[:-1, :] |= unknown[1:, :]
        adj[:, 1:] |= unknown[:, :-1]
        adj[:, :-1] |= unknown[:, 1:]
        frontier_mask = free & adj
        if not frontier_mask.any():
            return []

        # Connected-component cluster the frontier cells (8-connectivity) with an
        # iterative flood fill -- no scipy dependency.
        clusters = []
        visited = np.zeros((h, w), dtype=bool)
        ys, xs = np.nonzero(frontier_mask)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and not visited[ny, nx]
                                and frontier_mask[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < self.min_size:
                continue
            arr = np.asarray(cells)
            # The goal must be an actual REACHABLE frontier cell -- NOT the cluster
            # centroid. The dominant case early on is a RING of frontier cells around
            # the robot (the edge of the disc it has mapped): the centroid sits AT the
            # robot, and even the innermost ring cell is within min_goal_distance, so
            # the old code rejected it and falsely declared "exploration complete".
            # Instead, pick the cluster cell NEAREST the robot that is still beyond
            # min_goal_distance -- the closest boundary we can actually drive to.
            cwx = ox + (arr[:, 1].astype(float) + 0.5) * res
            cwy = oy + (arr[:, 0].astype(float) + 0.5) * res
            d = np.hypot(cwx - rx, cwy - ry)
            # Drive to the FARTHEST cell of the frontier (deepest toward unknown),
            # not the nearest: nearest-cell goals are 0.3 m hops that stay INSIDE
            # the disc already mapped, so the robot never crosses to the boundary
            # and the map stops growing. The farthest cell is the disc edge / into
            # the opening, which actually reveals new space.
            i = int(d.argmax())
            clusters.append(Frontier(len(cells), float(arr[i, 1]), float(arr[i, 0]),
                                     float(cwx[i]), float(cwy[i])))
        return clusters

    def _select(self, frontiers, robot):
        """Pick the best non-blacklisted frontier: nearest, with a size bonus."""
        rx, ry = robot
        best, best_cost = None, math.inf
        for f in frontiers:
            d = math.hypot(f.wx - rx, f.wy - ry)
            if d < self.min_goal_distance:
                continue
            if self._is_blacklisted(f.wx, f.wy):
                continue
            # explore_lite-style: travel cost minus information-gain reward.
            cost = d - self.gain_scale * f.size * 0.05
            if cost < best_cost:
                best, best_cost = f, cost
        return best

    def _is_blacklisted(self, wx, wy):
        r2 = self.blacklist_radius ** 2
        return any((wx - bx) ** 2 + (wy - by) ** 2 < r2 for bx, by in self._blacklist)

    # ------------------------------------------------------------- goal sending
    def _send_goal(self, frontier, robot):
        rx, ry = robot
        yaw = math.atan2(frontier.wy - ry, frontier.wx - rx)
        qx, qy, qz, qw = _yaw_to_quat(yaw)

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = frontier.wx
        pose.pose.position.y = frontier.wy
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goals_sent += 1
        self._goal_xy = (frontier.wx, frontier.wy)
        self._goal_sent_t = self._now()
        self._goal_best_dist = math.hypot(frontier.wx - rx, frontier.wy - ry)
        self._goal_best_t = self._now()
        self.get_logger().info(
            f"-> frontier ({frontier.wx:+.2f}, {frontier.wy:+.2f}) m, "
            f"{frontier.size} cells, {self._goal_best_dist:.1f} m away")
        send = self._nav.send_goal_async(goal)
        send.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal; blacklisting it.")
            if self._goal_xy:
                self._blacklist.append(self._goal_xy)
            self._clear_goal()
            return
        self._goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        # status: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (action_msgs/GoalStatus)
        status = future.result().status
        if status == 4:
            self.get_logger().info("reached frontier; re-scanning for more.")
        else:
            self.get_logger().warn(f"goal ended status={status}; blacklisting it.")
            if self._goal_xy:
                self._blacklist.append(self._goal_xy)
        self._clear_goal()

    def _monitor_goal(self):
        """Watchdog: cancel + blacklist a goal that stalls or runs too long."""
        if self._goal_handle is None or self._goal_xy is None:
            return
        now = self._now()
        if now - self._goal_sent_t > self.goal_timeout:
            self.get_logger().warn("goal timed out; cancelling + blacklisting.")
            self._cancel_and_blacklist()
            return
        robot = self._robot_xy()
        if robot is None:
            return
        d = math.hypot(self._goal_xy[0] - robot[0], self._goal_xy[1] - robot[1])
        if d < self._goal_best_dist - 0.10:     # made >10 cm of progress
            self._goal_best_dist = d
            self._goal_best_t = now
        elif now - self._goal_best_t > self.progress_timeout:
            self.get_logger().warn("no progress toward goal; cancelling + blacklisting.")
            self._cancel_and_blacklist()

    def _cancel_and_blacklist(self):
        if self._goal_xy:
            self._blacklist.append(self._goal_xy)
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:        # noqa: BLE001
                pass
        self._clear_goal()

    def _clear_goal(self):
        self._goal_handle = None
        self._goal_xy = None
        self._goal_sent_t = None

    # ---------------------------------------------------------------- complete
    def _on_complete(self, robot):
        self._complete = True
        msg = Bool()
        msg.data = True
        self._done_pub.publish(msg)
        # Save BEFORE any homing drive: the map is final now, and self._map still
        # holds the last grid even once /map goes quiet.
        saved = self._save_map() if self.save_map_on_complete else None
        if saved and self.save_mesh_on_complete:
            self._save_mesh(saved)                # <base>.ply + <base>.nvblx (nvblox)
        tail = (f"Map saved to {saved}.pgm + {saved}.yaml." if saved else
                "Save the map with:  ros2 run nav2_map_server map_saver_cli -f <name>")
        self.get_logger().info(
            "================  EXPLORATION COMPLETE  ================\n"
            "No reachable frontiers remain -- the environment is fully mapped. " + tail)
        if self.return_to_start and self._home is not None and self._nav.server_is_ready():
            self.get_logger().info("returning to start pose.")
            home = Frontier(0, 0, 0, self._home[0], self._home[1])
            self._complete = True                 # stay complete; just one homing goal
            self._send_goal(home, robot)

    # ------------------------------------------------------------- map saving
    def _save_map(self):
        """Write self._map as a nav2 map_server pair (<path>.pgm + <path>.yaml).

        Replicates nav2 map_saver's TRINARY PGM encoding (free=254, occupied=0,
        unknown=205) so map_server / slam_toolbox can load it back unchanged. The
        PGM is flipped vertically because the occupancy grid's row 0 is the BOTTOM
        (origin) row while PGM stores the top row first. Returns the basename on
        success, or None on failure (logged, never raises -- a save error must not
        crash the explorer)."""
        if self._map is None:
            self.get_logger().warn("no map to save yet -- skipping auto-save.")
            return None
        try:
            grid = self._map
            info = grid.info
            w, h = info.width, info.height
            if w == 0 or h == 0:
                self.get_logger().warn("map is empty -- skipping auto-save.")
                return None
            g = np.asarray(grid.data, dtype=np.int16).reshape(h, w)

            free_i = int(round(self.map_free_thresh * 100.0))
            occ_i = int(round(self.map_occupied_thresh * 100.0))
            img = np.full((h, w), 205, dtype=np.uint8)          # unknown / in-between
            known = (g >= 0) & (g <= 100)
            img[known & (g <= free_i)] = 254                    # free  -> white
            img[known & (g >= occ_i)] = 0                       # occupied -> black
            img = np.flipud(img)                                # grid bottom-up -> PGM top-down

            base = os.path.abspath(os.path.expanduser(self.map_save_path))
            os.makedirs(os.path.dirname(base), exist_ok=True)
            pgm_path, yaml_path = base + ".pgm", base + ".yaml"
            with open(pgm_path, "wb") as f:                     # binary PGM (P5)
                f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
                f.write(img.tobytes())

            o = info.origin
            yaw = math.atan2(2.0 * (o.orientation.w * o.orientation.z +
                                    o.orientation.x * o.orientation.y),
                             1.0 - 2.0 * (o.orientation.y ** 2 + o.orientation.z ** 2))
            with open(yaml_path, "w") as f:
                f.write(
                    f"image: {os.path.basename(pgm_path)}\n"
                    f"mode: trinary\n"
                    f"resolution: {info.resolution:.6f}\n"
                    f"origin: [{o.position.x:.6f}, {o.position.y:.6f}, {yaw:.6f}]\n"
                    f"negate: 0\n"
                    f"occupied_thresh: {self.map_occupied_thresh}\n"
                    f"free_thresh: {self.map_free_thresh}\n")
            self.get_logger().info(
                f"auto-saved map ({w}x{h} @ {info.resolution:.3f} m/cell) -> {base}.{{pgm,yaml}}")
            return base
        except Exception as e:        # noqa: BLE001 -- saving must never crash exploration
            self.get_logger().error(f"map auto-save failed: {e}")
            return None

    def _save_mesh(self, base):
        """Save the nvblox 3D reconstruction next to the 2D map: <base>.ply (mesh
        point cloud) + <base>.nvblx (serialized map, for relocalization). nvblox's
        save_ply / save_map services are in the APT ROS env (nvblox_msgs not
        importable in this conda node), so shell out: source slam_env.sh then
        `ros2 service call`. Best-effort -- logs and returns, never raises."""
        env = os.path.abspath(os.path.expanduser(self.nvblox_save_env))
        if not os.path.exists(env):
            self.get_logger().warn(
                f"3D-mesh save skipped: slam env script not found ({env}). "
                "Set the nvblox_save_env param to point at slam_env.sh.")
            return
        import subprocess
        for svc, ext in (("save_ply", ".ply"), ("save_map", ".nvblx")):
            target = base + ext
            cmd = (f'source "{env}" >/dev/null 2>&1 && '
                   f'ros2 service call /nvblox_node/{svc} nvblox_msgs/srv/FilePath '
                   f'"{{file_path: \'{target}\'}}"')
            try:
                r = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                                   text=True, timeout=30.0)
                if r.returncode == 0 and "success=True" in (r.stdout or ""):
                    self.get_logger().info(f"nvblox {svc} -> {target}")
                else:
                    self.get_logger().warn(
                        f"nvblox {svc} failed (rc={r.returncode}): "
                        f"{(r.stdout or r.stderr or '').strip()[:200]}")
            except Exception as e:        # noqa: BLE001 -- saving must never crash exploration
                self.get_logger().error(f"nvblox {svc} call errored: {e}")

    # --------------------------------------------------------------- utilities
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_markers(self, frontiers, chosen):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.global_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, f in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "frontiers"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = f.wx
            m.pose.position.y = f.wy
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            s = 0.15 + min(f.size, 200) * 0.001
            m.scale.x = m.scale.y = m.scale.z = s
            is_goal = chosen is not None and f is chosen
            m.color.r = 0.0 if is_goal else 1.0
            m.color.g = 1.0 if is_goal else 0.55
            m.color.b = 0.0
            m.color.a = 0.9
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
