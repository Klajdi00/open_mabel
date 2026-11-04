//
//  DevicePoseTracker.swift
//  MABEL Mobile App
//
//  iPhone attitude → robot head active perception. Uses CoreMotion
//  device-motion (fused gyro + accelerometer, gravity-referenced; no camera,
//  no permission prompt).
//
//  Orientation-invariant by construction: we use ONLY the direction the phone
//  points — the back-camera axis (device −Z) expressed in the gravity frame —
//  and read its azimuth (yaw) and elevation (pitch). Because we never touch
//  the device's X/Y axes, holding the phone in landscape (rotated 90°) or
//  twisting/rolling it does NOT corrupt the mapping. Panning the phone left/
//  right → head yaw; tilting up/down → head pitch; roll is ignored.
//
//  The relative angles (since a recenter) are scaled to 85% (15% calmer than
//  the phone) and exponentially smoothed before becoming a 4×4 head transform.
//

import Foundation
import Combine
import CoreMotion

@MainActor
final class DevicePoseTracker: ObservableObject {
    @Published private(set) var active = false

    var motionScale = 0.85       // robot head moves 15% less than the phone
    var smoothing = 0.20         // exponential smoothing per sample (lower = calmer)
    /// Flip if "tilt up" looks down / "pan left" looks right on the robot.
    var yawSign = 1.0
    var pitchSign = 1.0      // tilt the phone up → the robot looks up (sign verified on-device)

    var onUpdate: (([Double]) -> Void)?

    private let motion = CMMotionManager()
    private var refYaw: Double?      // azimuth at recenter
    private var refPitch = 0.0
    private var smYaw = 0.0, smPitch = 0.0
    /// The smoothed angles currently applied — read at release so the ease-home
    /// glide starts exactly where tracking left the head.
    var current: (yaw: Double, pitch: Double) { (smYaw, smPitch) }

    var isAvailable: Bool { motion.isDeviceMotionAvailable }

    func start() {
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { active = motion.isDeviceMotionActive; return }
        motion.deviceMotionUpdateInterval = 1.0 / 30.0
        refYaw = nil; smYaw = 0; smPitch = 0
        motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] dm, _ in
            guard let self, let dm else { return }
            self.consume(dm.attitude.rotationMatrix)
        }
        active = true
    }

    func stop() { motion.stopDeviceMotionUpdates(); active = false }

    func recenter() { refYaw = nil; smYaw = 0; smPitch = 0 }

    // MARK: - Core

    private func consume(_ r: CMRotationMatrix) {
        // Device −Z (the rear camera / where the phone "looks") expressed in the
        // gravity-referenced world frame (Z = up). Row 3 of R is the device +Z
        // axis in world; negate for the forward (back-camera) direction.
        let fx = -r.m31, fy = -r.m32, fz = -r.m33
        let yaw = atan2(fy, fx)                 // azimuth in the horizontal plane
        let pitch = asin(min(1, max(-1, fz)))   // elevation above horizontal

        if refYaw == nil { refYaw = yaw; refPitch = pitch }
        var dYaw = yaw - refYaw!
        while dYaw >  .pi { dYaw -= 2 * .pi }
        while dYaw < -.pi { dYaw += 2 * .pi }
        let dPitch = pitch - refPitch

        // Reduce by 15% and smooth.
        let tYaw = dYaw * motionScale * yawSign
        let tPitch = dPitch * motionScale * pitchSign
        smYaw += (tYaw - smYaw) * smoothing
        smPitch += (tPitch - smPitch) * smoothing

        onUpdate?(Self.headMatrix(yaw: smYaw, pitch: smPitch))
    }

    /// Ry(yaw)·Rx(pitch) as a 4×4 row-major head transform (forward = −Z).
    static func headMatrix(yaw: Double, pitch: Double) -> [Double] {
        let cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch)
        return [ cy,  sy*sp,  sy*cp, 0,
                 0,   cp,    -sp,    0,
                -sy,  cy*sp,  cy*cp, 0,
                 0,   0,      0,     1]
    }
}
