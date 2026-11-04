# apps/ios — iPhone teleoperation app

A SwiftUI iPhone app for driving MABEL: a 3-D robot space with the live camera feeds, a base
and lift drive pad, per-joint control, an autonomy timeline editor, shared e-stop and torque
state, recording controls, and a home-screen widget / Live Activity. It is a client of the same
wire contract as the Vision Pro app (`server/wire_protocol.py`, WebSocket `:9090`), so every
command passes the `controller/` safety gate. It works the same against the real robot and the
MuJoCo twin, and falls back to an on-device demo when no robot is reachable.

## Build and run

Requires Xcode 26 with the iOS 26.5 SDK and a running gateway: `bash server/run.sh`
(simulation) or `bash server/real_run.sh` (real robot).

1. `open "apps/ios/MABEL Mobile App.xcodeproj"`
2. In *Signing & Capabilities* choose your own team for the app and widget targets, and change
   the bundle identifiers (`org.example.MABEL-Mobile-App-V2`, `….MabelWidget`, `…Tests`) and
   the app group (`group.org.example.MABEL-Mobile-App`) to your namespace.
3. Run the **MABEL Mobile App** scheme on an iPhone or the iOS Simulator.

## Connecting to a robot

The app races every known route with a TCP probe and uses the fastest that answers:
Wi-Fi LAN (Bonjour `_mabel-teleop._tcp`, or a host added by hand), Tailscale (the robot
host's `100.x.y.z` address, added by hand), and Bluetooth LE (`server/ble_bridge.py`;
control and telemetry only, no video). Add a robot under Settings → **Add robot**. If the robot
host runs with `MABEL_TOKEN` set, enter the same code under Settings → Real Robot Security
(stored in the Keychain). Example hosts live in `MABEL Mobile App/Services/NetworkConfig.swift`.

## Tests

```bash
cd apps/ios && xcodebuild test -project "MABEL Mobile App.xcodeproj" -scheme "MABEL Mobile App" \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'   # live-server tests skip without a gateway
```
