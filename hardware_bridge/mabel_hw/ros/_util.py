"""_util.py — small shared plumbing for the ROS adapter nodes."""
from __future__ import annotations


def add_publisher_watchdog(node, topic: str, period_s: float = 2.0):
    """Log ERROR while more than one publisher sits on a command topic.

    This is the detection layer for the classic '/mabel_cmd two publishers →
    the base won't move' incident: DDS can't prevent a second commander in
    another process, but the owning node CAN see it and say so loudly.
    """
    def _check():
        try:
            n = node.count_publishers(topic)
        except Exception:
            return
        if n > 1:
            node.get_logger().error(
                f"{n} publishers on {topic} — exactly ONE commander is allowed; "
                "their commands are interleaving (idle zeros win). Find and "
                "stop the extra publisher.",
                throttle_duration_sec=10.0)

    return node.create_timer(period_s, _check)
