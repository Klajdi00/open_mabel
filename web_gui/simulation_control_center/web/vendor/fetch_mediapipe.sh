#!/usr/bin/env bash
# Re-vendor MediaPipe tasks-vision for the studio's webcam body-teleop so it has
# ZERO external CDN dependencies (jsdelivr / storage.googleapis.com are blocked on
# locked-down networks and over the relay). Writes into web/vendor/mediapipe/.
# bodyteleop.js loads /vendor/mediapipe/{vision_bundle.mjs,wasm/,*.task}.
set -euo pipefail
VER="0.10.14"
CDN="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/mediapipe"
mkdir -p "$DEST/wasm"

echo "▸ bundle"
curl -fsSL -o "$DEST/vision_bundle.mjs" "$CDN/vision_bundle.mjs"

echo "▸ wasm runtime"
for f in vision_wasm_internal.js vision_wasm_internal.wasm \
         vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -fsSL -o "$DEST/wasm/$f" "$CDN/wasm/$f"
done

echo "▸ models (must match the paths in bodyteleop.js)"
curl -fsSL -o "$DEST/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
curl -fsSL -o "$DEST/hand_landmarker.task" \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

echo "✔ vendored MediaPipe ${VER} → $DEST ($(du -sh "$DEST" | cut -f1))"
