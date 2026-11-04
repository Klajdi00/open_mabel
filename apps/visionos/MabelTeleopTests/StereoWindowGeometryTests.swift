//
//  StereoWindowGeometryTests.swift
//  MabelTeleopTests
//
//  Full immersion's head-locked stereo window (2026-10-16) must be TRUE SCALE
//  and ORTHOSTEREO. The Simulator renders one (mono) view and cannot be worn,
//  so these are the proofs: they ray-trace the display geometry the panel
//  builds and the sampling rule the StereoCameraMaterial applies, and check
//  every pixel reaches each eye at the angle its camera captured it from.
//

import XCTest
@testable import MabelTeleop

final class StereoWindowGeometryTests: XCTestCase {

    /// The MJCF head camera: fovy 83.3°, 4:3, window at 2 m, 63 mm IPD.
    private let g = StereoWindowGeometry(fovyDeg: 83.3, aspect: 4.0 / 3.0,
                                         distance: 2.0, ipd: 0.063)

    /// TRUE SCALE: the window subtends exactly the camera's field of view —
    /// horizontally and vertically, at every viewing distance.
    func test_windowSubtendsTheCameraFieldOfView() {
        for d: Float in [0.8, 1.5, 2.0, 5.0] {
            var gd = g; gd.distance = d
            let hfov = 2 * atan((gd.width / 2) / d)
            let vfov = 2 * atan((gd.height / 2) / d)
            XCTAssertEqual(hfov, gd.hfovRad, accuracy: 1e-5)
            XCTAssertEqual(vfov * 180 / .pi, 83.3, accuracy: 1e-3)
        }
        // 83.3° vertical on 4:3 is ≈99.7° horizontal (the MJCF's 112° diagonal module).
        XCTAssertEqual(g.hfovRad * 180 / .pi, 99.72, accuracy: 0.05)
        XCTAssertEqual(g.width, 4.744, accuracy: 0.002)
        XCTAssertEqual(g.height, 3.558, accuracy: 0.002)
    }

    // MARK: - Orthostereo by ray tracing

    /// Where on the window (x, metres) a texel of eye `side` (−1 left, +1
    /// right) is drawn, given the material's rule: LEFT samples u + s,
    /// RIGHT samples u − s  ⇒  texel u_img appears at panel u_p = u_img ∓ s.
    private func drawnX(uImg: Float, side: Float, shift: Float, _ gg: StereoWindowGeometry) -> Float {
        let up = uImg + side * shift          // left (side −1): u_p = u_img − s
        return (up - 0.5) * gg.width
    }

    /// The camera's angle for texel u_img (pinhole: linear in tanθ).
    private func cameraAngle(uImg: Float, _ gg: StereoWindowGeometry) -> Float {
        atan((uImg - 0.5) * 2 * tan(gg.hfovRad / 2))
    }

    /// The angle the eye at x = side·IPD/2 sees a point x on the window.
    private func eyeAngle(x: Float, side: Float, _ gg: StereoWindowGeometry) -> Float {
        atan((x - side * gg.ipd / 2) / gg.distance)
    }

    /// WITH the shift, every texel reaches each eye at exactly the angle the
    /// matching camera captured it from — at any window distance. That is the
    /// definition of an orthostereo display (parallel cameras, baseline = IPD).
    func test_shiftedWindow_isOrthostereoAtAnyDistance() {
        for d: Float in [0.8, 2.0, 5.0] {
            var gd = g; gd.distance = d
            for uImg in stride(from: Float(0.05), through: 0.95, by: 0.05) {
                for side: Float in [-1, 1] {
                    let x = drawnX(uImg: uImg, side: side, shift: gd.eyeShiftU, gd)
                    XCTAssertEqual(eyeAngle(x: x, side: side, gd), cameraAngle(uImg: uImg, gd),
                                   accuracy: 1e-5, "u=\(uImg) side=\(side) D=\(d)")
                }
            }
        }
    }

    /// WITHOUT it, the two eyes disagree by the predicted vergence error —
    /// the bug the shift exists to cancel (so this test proves the shift is
    /// not a no-op, and pins the number quoted in the docs: 1.80° at 2 m).
    func test_unshiftedWindow_hasTheVergenceError() {
        let x = drawnX(uImg: 0.5, side: -1, shift: 0, g)   // image centre, same spot for both eyes
        let left = eyeAngle(x: x, side: -1, g)
        let right = eyeAngle(x: x, side: +1, g)
        XCTAssertEqual(left - right, g.unshiftedVergenceError, accuracy: 1e-6)
        XCTAssertEqual(g.unshiftedVergenceError * 180 / .pi, 1.804, accuracy: 0.002)
        // …which pulls a point 1 m away in to 0.67 m.
        let trueV = 2 * atan(g.ipd / 2 / 1.0)
        let seen = g.ipd / 2 / tan((trueV + g.unshiftedVergenceError) / 2)
        XCTAssertEqual(seen, 0.666, accuracy: 0.003)
    }

    /// LEFT eye's image must move LEFT (toward that eye), i.e. the left
    /// texture is sampled at u + s with s > 0. A sign slip here would DOUBLE
    /// the error instead of cancelling it.
    func test_shiftSignCentresEachImageOnItsOwnEye() {
        XCTAssertGreaterThan(g.eyeShiftU, 0)
        XCTAssertEqual(drawnX(uImg: 0.5, side: -1, shift: g.eyeShiftU, g), -g.ipd / 2, accuracy: 1e-6)
        XCTAssertEqual(drawnX(uImg: 0.5, side: +1, shift: g.eyeShiftU, g), +g.ipd / 2, accuracy: 1e-6)
        XCTAssertEqual(g.eyeShiftU, 0.0315 / 4.744, accuracy: 1e-5)
    }

    /// WINDOW SIZE < 100 % (operator, 2026-10-16): the picture shrinks
    /// uniformly — tan of every angle scales by `scale` — and the per-eye shift
    /// still centres each eye's image on that eye (no vergence error added).
    func test_scaledWindow_shrinksUniformly_andStaysCentredPerEye() {
        var gs = g; gs.scale = 0.8
        XCTAssertEqual(gs.width / g.width, 0.8, accuracy: 1e-6)
        XCTAssertEqual(tan(gs.halfHeightAngle), 0.8 * tan(g.fovyRad / 2), accuracy: 1e-6)
        for side: Float in [-1, 1] {
            XCTAssertEqual(drawnX(uImg: 0.5, side: side, shift: gs.eyeShiftU, gs), side * gs.ipd / 2,
                           accuracy: 1e-6)
        }
        // Every texel reaches the eye at atan(0.8·tanθ) — a uniform magnification.
        for uImg in stride(from: Float(0.1), through: 0.9, by: 0.1) {
            let x = drawnX(uImg: uImg, side: -1, shift: gs.eyeShiftU, gs)
            XCTAssertEqual(tan(eyeAngle(x: x, side: -1, gs)), 0.8 * tan(cameraAngle(uImg: uImg, gs)),
                           accuracy: 1e-5)
        }
        XCTAssertEqual(StereoWindowGeometry.defaultScale, 0.8)
        XCTAssertTrue(StereoWindowGeometry.scaleRange.contains(1.0), "100 % = true scale must be reachable")
    }

    /// The head-locked overlays sit inside the video's angular extent and in
    /// FRONT of the window (else the window would hide them).
    func test_headLayoutIsInFrameAndInFront() {
        let halfH = g.hfovRad / 2, halfV = g.fovyRad / 2
        for p in [HeadLayout.wristPosition(side: -1), HeadLayout.wristPosition(side: 1),
                  HeadLayout.miniPosition] {
            XCTAssertLessThan(-p.z, g.distance)
            XCTAssertLessThan(abs(atan(p.x / -p.z)), halfH * 0.6)
            XCTAssertLessThan(abs(atan(p.y / -p.z)), halfV * 0.6)
        }
    }
}
