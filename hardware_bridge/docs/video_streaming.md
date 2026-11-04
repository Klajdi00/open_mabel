# Video streaming

The hardware bridge is the only owner of the robot's cameras. Each camera runs in its
own capture worker process (`mabel_hw/cameras/camera_worker.py`), which publishes the
ROS image contract and pushes downscaled frames over a local socket to a single
video-relay process (`mabel_hw/cameras/video_relay.py`). The relay fans each stream out
to the network transports below. Every transport is demand-driven (nothing is encoded
for a stream nobody is watching), and MJPEG is always available as the fallback. Video
does not pass through `server/`.

```text
                        ┌─ ROS 2   /camera/<name>/*            (SLAM, RViz, recording)
camera ─ capture worker ┤
                        └─ video relay ─┬─ MJPEG / HTTP   :8080  /camera/<name>/stream.mjpg
                                        ├─ JPEG  / UDP    :9091  ("MV" framing)
                                        ├─ H.264 / UDP    :9092  ("MV" framing, NVENC)
                                        ├─ H.264 / WS     :8080  /camera/<name>/h264
                                        └─ dedicated ports :9620-9623 (head side-by-side, wrists, front)
```

Code: `mabel_hw/cameras/` (`manager.py`, `camera_worker.py`, `video_relay.py`,
`video_server.py`, `udp_sink.py`; drivers `realsense_d435.py`, `zed_mini.py`,
`wrist_d405.py`). Configuration: `CAMERAS` in `mabel_hw/config.py`.

## Stream names

`front` (D435i), `main` and `head_left` (ZED Mini left eye), `wrist_left`,
`wrist_right`. `main` falls back to the front camera while the head camera is absent and
switches back when it returns, without a client reconnect. The simulator serves the
same names, so clients work unchanged against either.

## Transports

| Transport | Port | Codec | Typical use |
|---|---|---|---|
| H.264 / UDP | 9092 | H.264 | lowest latency on LAN / VPN |
| H.264 / WebSocket | 8080, `/camera/<name>/h264` | H.264 | browsers (WebCodecs) and any TCP-only connection |
| JPEG / UDP | 9091 | JPEG | fallback without a codec dependency |
| MJPEG / HTTP | 8080, `/camera/<name>/stream.mjpg` | JPEG | universal fallback |
| Dedicated | 9620 head SBS, 9621 wrist left, 9622 wrist right, 9623 front | JPEG | per-camera ports used by `server/data_streams.py` |

UDP drops rather than queues, so a slow client never builds a backlog. H.264 at 720p
needs a few Mbit/s where MJPEG at the same resolution needs an order of magnitude more.
The simulator uses `h264_videotoolbox` (`server/sim_h264.py`) and emits the same
baseline 4:2:0 Annex-B format, so every client decodes one codec.

## Resolution and quality

The network frame size is independent of the capture and ROS resolution.

| Env (`MABEL_HW_*`) | Default | Meaning |
|---|---|---|
| `CAM_SERVE` | true | run the network video server |
| `MABEL_HTTP_PORT` (no prefix) | 8080 | MJPEG and H.264-over-WebSocket port |
| `CAM_NET_WIDTH` / `CAM_NET_QUALITY` | 640 / 65 | MJPEG and JPEG/UDP width and JPEG quality |
| `CAM_UDP` | true | enable the UDP sinks |
| `CAM_UDP_JPEG` / `CAM_UDP_H264` | true / true | per-transport enable |
| `CAM_UDP_JPEG_PORT` / `CAM_UDP_H264_PORT` | 9091 / 9092 | UDP ports |
| `CAM_H264_WIDTH` | 1280 | H.264 width; frames are never upscaled |
| `CAM_UDP_H264_BITRATE` | 4M | H.264 CBR bitrate per stream |
| `CAM_UDP_FPS` | 30 | encoder frame rate |
| `CAM_DEDICATED` | true | dedicated per-camera ports (`CAM_DEDI_*_PORT`) |
| `CAM_REALSENSE`, `CAM_ZED`, `CAM_WRIST` | true | per-camera enable |

## "MV" UDP framing (ports 9091 and 9092)

Defined in `server/udp_video.py`; the same framing carries a JPEG frame (9091) or one
H.264 access unit (9092).

```text
client -> {"sub": ["main", "wrist_left"]}          JSON, resent every 1 s as keepalive
server -> {"map": {"main": 0, "wrist_left": 1}}     once per subscription
server -> "MV" | ver u8 | stream u8 | seq u16 | chunk u16 | chunks u16 | ts_ms u32 | payload
          14-byte big-endian header; datagrams <= 1196 bytes (fits a 1280-byte MTU)
```

Reassemble by `(stream, seq)`; a newer `seq` discards an older partial frame.

## H.264 encoder

`mabel_hw/cameras/udp_sink.py` runs one `ffmpeg` `h264_nvenc` encoder per active stream
on the Jetson:

```text
-c:v h264_nvenc -pix_fmt yuv420p -profile:v baseline -preset p1 -tune ll -rc cbr
-b:v <bitrate> -g <fps> -forced-idr 1 -aud 1 -bsf:v dump_extra=freq=keyframe
```

- `-pix_fmt yuv420p -profile:v baseline` are required. Without them NVENC selects a
  4:4:4 profile that hardware decoders (VideoToolbox, WebCodecs) reject, which shows
  up as black video.
- An IDR frame with in-band SPS/PPS every second lets a late or lossy client recover
  within one second without signalling.
- One encoder feeds both the UDP and WebSocket consumers.

## Clients

- **Vision Pro** (`apps/visionos/MabelTeleop/Networking/`): `UDPVideoReceiver.swift`
  ("MV" transport), `H264Decoder.swift` (VideoToolbox), `WebSocketH264Receiver.swift`
  for TCP-only connections; JPEG/MJPEG remain the fallback.
- **Browser** (`web_gui/control_center/web/js/h264.js`): decodes H.264 over WebSocket
  with WebCodecs; the MJPEG image is the automatic fallback.

## Testing

```bash
python3 hardware_bridge/tools/video_transport_test.py   # all transports, synthetic frames (needs h264_nvenc)
python3 hardware_bridge/tools/video_latency_bench.py    # per-stream latency/bandwidth (on the robot)
bash    hardware_bridge/tools/validate_wrist_streams.sh # wrist streams reach a client
```

The relay logs measured capture-to-fan-out latency per stream (p50/p90) every 30 s
(search the bridge journal for "video latency"). Gateway-side H.264 measurements are in
`server/docs/server.pdf` (`server/tools/exp_h264_video.py`).
