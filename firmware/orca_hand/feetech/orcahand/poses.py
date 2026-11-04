"""Named hand poses, expressed as per-joint percent of the calibrated range.

``percent`` here is 0 = raw_min (extended/open), 100 = raw_max (flexed/closed),
matching the legacy firmware convention. Keyed by bare joint name so the same
table works for either hand. Consumed by ``OrcaHand.set_named_position``.
"""

POSES = {
    "flat": {
        "index_pip": 10, "index_mcp": 10, "index_abd": 50,
        "middle_pip": 10, "middle_mcp": 10, "middle_abd": 50,
        "ring_pip": 10, "ring_mcp": 10, "ring_abd": 50,
        "pinky_pip": 10, "pinky_mcp": 10, "pinky_abd": 50,
        "thumb_ip": 10, "thumb_mp": 10, "thumb_abd": 50, "thumb_cmc": 10,
        "wrist": 50,
    },
    "fist": {
        "index_pip": 85, "index_mcp": 80, "index_abd": 50,
        "middle_pip": 85, "middle_mcp": 80, "middle_abd": 50,
        "ring_pip": 85, "ring_mcp": 80, "ring_abd": 50,
        "pinky_pip": 85, "pinky_mcp": 80, "pinky_abd": 50,
        "thumb_ip": 70, "thumb_mp": 70, "thumb_abd": 20, "thumb_cmc": 70,
        "wrist": 50,
    },
    "pinch": {
        "index_pip": 60, "index_mcp": 60, "index_abd": 50,
        "middle_pip": 10, "middle_mcp": 10, "middle_abd": 50,
        "ring_pip": 10, "ring_mcp": 10, "ring_abd": 50,
        "pinky_pip": 10, "pinky_mcp": 10, "pinky_abd": 50,
        "thumb_ip": 50, "thumb_mp": 55, "thumb_abd": 65, "thumb_cmc": 50,
        "wrist": 50,
    },
    "point": {
        "index_pip": 10, "index_mcp": 10, "index_abd": 50,
        "middle_pip": 85, "middle_mcp": 80, "middle_abd": 50,
        "ring_pip": 85, "ring_mcp": 80, "ring_abd": 50,
        "pinky_pip": 85, "pinky_mcp": 80, "pinky_abd": 50,
        "thumb_ip": 60, "thumb_mp": 60, "thumb_abd": 30, "thumb_cmc": 60,
        "wrist": 50,
    },
    "peace": {
        "index_pip": 10, "index_mcp": 10, "index_abd": 50,
        "middle_pip": 10, "middle_mcp": 10, "middle_abd": 50,
        "ring_pip": 85, "ring_mcp": 80, "ring_abd": 50,
        "pinky_pip": 85, "pinky_mcp": 80, "pinky_abd": 50,
        "thumb_ip": 60, "thumb_mp": 60, "thumb_abd": 30, "thumb_cmc": 60,
        "wrist": 50,
    },
    "ok": {
        "index_pip": 65, "index_mcp": 65, "index_abd": 50,
        "middle_pip": 10, "middle_mcp": 10, "middle_abd": 50,
        "ring_pip": 10, "ring_mcp": 10, "ring_abd": 50,
        "pinky_pip": 10, "pinky_mcp": 10, "pinky_abd": 50,
        "thumb_ip": 55, "thumb_mp": 60, "thumb_abd": 70, "thumb_cmc": 45,
        "wrist": 50,
    },
    "thumbs_up": {
        "index_pip": 80, "index_mcp": 80, "index_abd": 50,
        "middle_pip": 80, "middle_mcp": 80, "middle_abd": 50,
        "ring_pip": 80, "ring_mcp": 80, "ring_abd": 50,
        "pinky_pip": 80, "pinky_mcp": 80, "pinky_abd": 50,
        "thumb_ip": 10, "thumb_mp": 10, "thumb_abd": 30, "thumb_cmc": 5,
        "wrist": 50,
    },
    "rock": {
        "index_pip": 10, "index_mcp": 10, "index_abd": 50,
        "middle_pip": 80, "middle_mcp": 80, "middle_abd": 50,
        "ring_pip": 80, "ring_mcp": 80, "ring_abd": 50,
        "pinky_pip": 10, "pinky_mcp": 10, "pinky_abd": 50,
        "thumb_ip": 50, "thumb_mp": 50, "thumb_abd": 40, "thumb_cmc": 50,
        "wrist": 50,
    },
    "spider": {
        "index_pip": 50, "index_mcp": 30, "index_abd": 60,
        "middle_pip": 50, "middle_mcp": 30, "middle_abd": 50,
        "ring_pip": 50, "ring_mcp": 30, "ring_abd": 40,
        "pinky_pip": 50, "pinky_mcp": 30, "pinky_abd": 60,
        "thumb_ip": 30, "thumb_mp": 30, "thumb_abd": 60, "thumb_cmc": 20,
        "wrist": 50,
    },
}
