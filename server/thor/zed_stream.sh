#!/usr/bin/env bash
# zed_stream.sh — stream the ZED-M (stereo head cam) on thor into the EXISTING
# MABEL camera pipeline so the Vision Pro / iPhone / web clients see real pixels
# with no app change.
#
# The ZED-M outputs ONE hardware-stitched side-by-side stereo frame over UVC,
# which is exactly what real_cameras.StereoHeadCamera consumes: it splits the
# frame into left/right eyes and serves them on the same paths every client
# already pulls:
#     http://<thor>:8080/camera/head_left/stream.mjpg    (left eye)
#     http://<thor>:8080/camera/head_right/stream.mjpg   (right eye)
#     http://<thor>:8080/camera/main/stream.mjpg         (mono = left eye)
#     udp://<thor>:9091                                  (low-latency, headset)
#     http://<thor>:8080/                                (browser index w/ status)
#
# REQUIREMENT: the ZED-M must be on a USB-3.0 SuperSpeed link. ZED cameras expose
# their UVC video interface ONLY at SuperSpeed; on USB-2 the camera enumerates as
# HID-only and NO /dev/video* appears. The ZED Mini uses a USB-3.0 Micro-B port —
# use its original SS cable, not a plain micro-USB-2 cable.
#
# Usage:
#   bash thor/zed_stream.sh            # launch now (errors if no /dev/video*)
#   bash thor/zed_stream.sh --watch    # wait for the ZED to appear, then launch
#   bash thor/zed_stream.sh --selftest # grab one frame, save split eyes to /tmp
#   ZED_MODE=2560x720 ZED_FPS=30 bash thor/zed_stream.sh   # override capture mode
#
# Tunables (env):
#   ZED_MODE   capture WxH (stitched, both eyes). Default 2560x720 → 1280x720/eye
#              (720p60). Others: 1344x376 (WVGA100), 3840x1080 (1080p, /eye 1920),
#              4416x1242 (2.2K15). Per-eye = WxH with W halved.
#   ZED_OUT    streamed per-eye size. Default 'native' (full quality). e.g. 960x720.
#   ZED_FPS    capture fps (default 30; the ZED does 60 at 720p, 100 at WVGA).
#   ZED_QUALITY  JPEG quality 1-100 (default 85).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "$HERE/.." && pwd)"
CONDA_PY="${MABEL_PY:-$HOME/miniconda3/envs/base-module-env/bin/python}"

MODE="${ZED_MODE:-2560x720}"
OUT="${ZED_OUT:-native}"
FPS="${ZED_FPS:-60}"
QUALITY="${ZED_QUALITY:-85}"

[[ -x "$CONDA_PY" ]] || { echo "ERROR: python not found at $CONDA_PY (set MABEL_PY)"; exit 1; }

lan_ip() { ip -4 addr show 2>/dev/null | grep -oP 'inet \K10\.[0-9.]+|inet \K192\.168\.[0-9.]+' | head -1 | sed 's/inet //'; }

check_zed() {
  # Returns 0 if a ZED UVC video node is present.
  if ls /dev/video* >/dev/null 2>&1; then return 0; fi
  return 1
}

diagnose_no_video() {
  echo
  echo "  ✗ No /dev/video* found — the ZED-M is NOT streaming video."
  if lsusb 2>/dev/null | grep -qi 2b03; then
    local spd
    spd=$(lsusb -t 2>/dev/null | grep -iE "Driver=uvcvideo|2b03" | grep -oE "[0-9]+M" | head -1)
    echo "    The ZED is on USB but enumerated WITHOUT its video interface"
    echo "    (HID-only). It is on a USB-2 link, not SuperSpeed."
    echo
    echo "    FIX: connect the ZED-M with a USB-3.0 SuperSpeed cable (the ZED Mini"
    echo "         uses a USB-3.0 Micro-B port — its original cable, NOT a plain"
    echo "         micro-USB-2 cable) directly to a Thor USB-3 port."
    echo "    Verify it linked SuperSpeed:   lsusb -t   (look for the ZED at 5000M+)"
  else
    echo "    No Stereolabs (2b03:*) device on USB at all — check the connection."
  fi
  echo
}

launch() {
  local W H EYE
  W="${MODE%x*}"; H="${MODE#*x}"; EYE=$((W/2))
  local ip; ip="$(lan_ip)"; [[ -z "$ip" ]] && ip="<thor-ip>"
  echo "  ──────────────────────────────────────────────────────────────────"
  echo "    ZED-M  →  MABEL camera pipeline"
  echo "    capture ${W}x${H} stitched  →  ${EYE}x${H}/eye   out=${OUT}  ${FPS}fps  q${QUALITY}"
  echo
  echo "    browser   http://${ip}:8080/            (live status + click a stream)"
  echo "              http://${ip}:8080/camera/head_left/stream.mjpg"
  echo "    headset   udp://${ip}:9091  +  the /camera/* paths (no app change)"
  echo "    remote    via Tailscale IP instead of ${ip}"
  echo "  ──────────────────────────────────────────────────────────────────"
  echo
  # --no-rectify: the head_*_cam.yaml calibration is for the OLD 3D USB cam, not
  # the ZED. Raw split is geometrically honest until ZED calibration is added.
  # --no-wrist: only the ZED head cam is connected.
  exec "$CONDA_PY" "$SERVER_DIR/real_camera_server.py" \
    --head-mode "$MODE" --head-out "$OUT" --fps "$FPS" --jpeg-quality "$QUALITY" \
    --no-rectify --no-wrist "$@"
}

case "${1:-}" in
  --watch)
    echo "▸ waiting for the ZED-M video device to appear (Ctrl-C to stop)…"
    until check_zed; do sleep 1; done
    echo "▸ /dev/video* detected:"; ls -1 /dev/video* 2>/dev/null | sed 's/^/    /'
    launch
    ;;
  --selftest)
    if ! check_zed; then diagnose_no_video; exit 2; fi
    exec "$CONDA_PY" "$SERVER_DIR/real_camera_server.py" \
      --head-mode "$MODE" --no-rectify --no-wrist --selftest
    ;;
  *)
    if ! check_zed; then diagnose_no_video; exit 2; fi
    launch "$@"
    ;;
esac
