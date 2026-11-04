#!/usr/bin/env bash
# validate_wrist_streams.sh — confirm the bridge serves BOTH wrist cams to the
# network (MJPEG) AND to ROS, after the hardware bridge has loaded the two-sink
# fan-out (cameras/video_server.py). Run AFTER restarting the bridge.
set -uo pipefail
PORT="${CAM_HTTP_PORT:-8080}"          # bridge is the single camera server on :8080
HOST="127.0.0.1"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mabel 2>/dev/null || true
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "── 1. network MJPEG (bridge video server :$PORT) ───────────────────────"
curl -s --max-time 3 "http://$HOST:$PORT/" || echo "  (index unreachable — is the bridge up with the new code?)"
echo
for name in wrist_left wrist_right; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 "http://$HOST:$PORT/camera/$name/stream.mjpg")
  echo "  /camera/$name/stream.mjpg -> HTTP $code"
done

echo "── 2. decode one real frame from each network stream ───────────────────"
python3 - "$HOST" "$PORT" <<'PY'
import sys, urllib.request, numpy as np, cv2
host, port = sys.argv[1], sys.argv[2]
for name in ("wrist_left", "wrist_right"):
    try:
        data = urllib.request.urlopen(f"http://{host}:{port}/camera/{name}/stream.mjpg", timeout=5).read(400000)
        a = data.find(b"\xff\xd8"); b = data.find(b"\xff\xd9", a+2)
        im = cv2.imdecode(np.frombuffer(data[a:b+2], np.uint8), cv2.IMREAD_COLOR)
        ok = im is not None and im.std() > 3
        print(f"  {name}: frame={None if im is None else im.shape} std={0 if im is None else im.std():.1f} -> {'LIVE' if ok else 'no/blank frame'}")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
PY

echo "── 3. ROS topics (the second sink) ─────────────────────────────────────"
for t in /camera/wrist_left/image_raw /camera/wrist_right/image_raw; do
  hz=$(timeout 6 ros2 topic hz "$t" 2>/dev/null | grep -m1 average || echo "no data")
  echo "  $t : $hz"
done
echo "── done ────────────────────────────────────────────────────────────────"
