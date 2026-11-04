//
//  StageFramingTests.swift
//  MabelTeleopTests
//
//  JOINT-SPACE MODULE FRAMING — "is the thing I flew to actually visible?"
//
//  The operator report was "switching modules moves the camera so close the 3D
//  viewer cuts the robot off" (2026-07-28). The cause was that the zoom was a
//  hand-tuned constant — `1.37 / (radius · margin)`, clamped at 5.5 — that knew
//  nothing about the window it was drawing into, and nothing about the floating
//  chrome covering the top and bottom of that window. It could only be right at
//  one window size.
//
//  It is solved now, and this file pins the solve. The zoom is chosen so a
//  module of measured bounding-sphere radius R lands at `fillFraction` of the
//  CHROME-FREE box:
//
//      d_pt = 2 · R · homeScale · zoom / scenePerPoint   ≤   fill · usable
//
//  The RADII are not invented here: they are measured off the shipped rig
//  (`Resources/MabelRig/rig.json` + the PLYs, home pose) by the same subtree /
//  stop-set definitions `RobotSpace3DStage.moduleBox` uses. Re-measure and
//  update them when the MJCF → … → rig pipeline regenerates the assets; a
//  changed robot SHOULD fail this test.
//
//  What this proves: the arithmetic puts every module inside the viewport with
//  margin, and big enough to read, at every window size the app can be given.
//  What it cannot prove: that RealityKit then drew it there — that is what the
//  `[MABEL-FRAME]` measurement (and `debugModuleTour`) report from a live run.
//

import XCTest
import SwiftUI
import RealityKit
import simd
@testable import MabelTeleop

final class StageFramingTests: XCTestCase {

    /// Bounding-sphere radius about each module's framing anchor, in rig
    /// metres, measured from the shipped rig at the HOME pose.
    ///
    /// The live framing measures the CURRENT pose, so these are representative
    /// inputs, not the only ones — a raised lift and outstretched arms measured
    /// 0.538 m and 0.631 m on a live run. `testFitsAcrossThePoseEnvelope`
    /// sweeps the whole plausible range for that reason.
    static let measuredRadius: [(String, Float)] = [
        ("Left Arm",   0.5704),
        ("Right Arm",  0.5782),
        ("Left Hand",  0.2106),
        ("Right Hand", 0.2107),
        ("Neck",       0.1636),
        ("Lift",       0.3534),
        ("Torso",      0.4368),
    ]

    /// Radius envelope any module can reach as the robot moves: from the neck
    /// (the smallest module) to an arm at full extension with the lift up
    /// (which is most of the robot). The whole robot's own bounding sphere is
    /// ~0.6 m, so 0.9 is already beyond anything a single module can be.
    static let radiusEnvelope: [Float] = stride(from: Float(0.12), through: 0.90, by: 0.02).map { $0 }

    /// `RobotStageView.fitToView`: the robot (1.6 m of rig at scale 1) is sized
    /// to 66% of the view height, and the view height in scene metres is
    /// `size.height · scenePerPoint`.
    private func fitted(viewHeight: CGFloat, metresPerPoint: Float = 1.0 / 1360)
        -> (homeScale: Float, scenePerPoint: Float) {
        let h = Float(viewHeight) * metresPerPoint
        return (h * 0.66 / 1.6, metresPerPoint)
    }

    // MARK: - Every module fits, at every window size

    func testEveryModuleFitsTheUsableViewport() {
        // Window sizes the app realistically gets, from a small side panel to a
        // wide wall. Insets are the app's own chrome budget (`stageSafeInsets`).
        let sizes: [CGSize] = [CGSize(width: 900, height: 620),
                               CGSize(width: 1280, height: 800),
                               CGSize(width: 1600, height: 900),
                               CGSize(width: 2200, height: 1200),
                               CGSize(width: 700, height: 1100)]   // portrait-ish
        let insets = EdgeInsets(top: 52, leading: 0, bottom: 111, trailing: 0)

        for size in sizes {
            let f = fitted(viewHeight: size.height)
            let usable = RobotSpace3DStage.usableViewport(size, insets: insets)
            XCTAssertGreaterThan(usable, 1, "no usable viewport at \(size)")
            for (name, r) in Self.measuredRadius {
                let zoom = RobotSpace3DStage.fitZoom(radius: r, homeScale: f.homeScale,
                                                     scenePerPoint: f.scenePerPoint,
                                                     usable: usable)
                let d = RobotSpace3DStage.onScreenDiameter(radius: r, zoom: zoom,
                                                           homeScale: f.homeScale,
                                                           scenePerPoint: f.scenePerPoint)
                // FITS: the module's bounding sphere is inside the chrome-free
                // box. The sphere bounds every orbit angle, so this holds after
                // the operator rotates the view too.
                XCTAssertLessThanOrEqual(
                    d, usable + 0.5,
                    "\(name) is CLIPPED at \(Int(size.width))x\(Int(size.height)): "
                    + "\(Int(d)) pt across a \(Int(usable)) pt box (zoom \(zoom))")
                // READABLE: a framing that shows a speck is as useless as one
                // that crops. 28% of the short side is the floor.
                XCTAssertGreaterThan(
                    d, usable * 0.28,
                    "\(name) is TOO SMALL at \(Int(size.width))x\(Int(size.height)): "
                    + "\(Int(d)) pt across a \(Int(usable)) pt box (zoom \(zoom))")
            }
        }
    }

    /// The rig is posed by LIVE telemetry when a module is framed, so the
    /// radius is a moving number. Every value it can take must still fit.
    func testFitsAcrossThePoseEnvelope() {
        let sizes: [CGSize] = [CGSize(width: 900, height: 620),
                               CGSize(width: 1280, height: 908),
                               CGSize(width: 2200, height: 1200)]
        let insets = EdgeInsets(top: 96, leading: 0, bottom: 155, trailing: 0)
        for size in sizes {
            let f = fitted(viewHeight: size.height)
            let usable = RobotSpace3DStage.usableViewport(size, insets: insets)
            for r in Self.radiusEnvelope {
                let zoom = RobotSpace3DStage.fitZoom(radius: r, homeScale: f.homeScale,
                                                     scenePerPoint: f.scenePerPoint,
                                                     usable: usable)
                let d = RobotSpace3DStage.onScreenDiameter(radius: r, zoom: zoom,
                                                           homeScale: f.homeScale,
                                                           scenePerPoint: f.scenePerPoint)
                XCTAssertLessThanOrEqual(
                    d, usable + 0.5,
                    "a module of radius \(r) m is CLIPPED at "
                    + "\(Int(size.width))x\(Int(size.height)): \(Int(d)) pt in \(Int(usable)) pt")
                XCTAssertGreaterThan(
                    d, usable * 0.28,
                    "a module of radius \(r) m is TOO SMALL at "
                    + "\(Int(size.width))x\(Int(size.height)): \(Int(d)) pt in \(Int(usable)) pt")
            }
        }
    }

    /// The exact inversion: at the fill fraction, the on-screen diameter IS the
    /// requested fraction of the box — unless a clamp bound it.
    func testFitZoomInvertsExactly() {
        let f = fitted(viewHeight: 800)
        let usable: CGFloat = 640
        for (name, r) in Self.measuredRadius {
            let zoom = RobotSpace3DStage.fitZoom(radius: r, homeScale: f.homeScale,
                                                 scenePerPoint: f.scenePerPoint,
                                                 usable: usable)
            let clamped = zoom <= RobotSpace3DStage.minModuleZoom + 1e-4
                || zoom >= RobotSpace3DStage.maxModuleZoom - 1e-4
            guard !clamped else { continue }
            let d = RobotSpace3DStage.onScreenDiameter(radius: r, zoom: zoom,
                                                       homeScale: f.homeScale,
                                                       scenePerPoint: f.scenePerPoint)
            XCTAssertEqual(Float(d / usable), RobotSpace3DStage.fillFraction, accuracy: 1e-3,
                           "\(name) did not land at the fill fraction")
        }
    }

    /// A bigger window means a bigger module on screen, never a jump in the
    /// other direction — the old constant was window-blind, which is the bug.
    func testZoomTracksTheWindow() {
        let insets = EdgeInsets(top: 52, leading: 0, bottom: 111, trailing: 0)
        let r: Float = 0.2106                                   // a hand
        var lastDiameter: CGFloat = 0
        for h in stride(from: CGFloat(600), through: 1400, by: 200) {
            let size = CGSize(width: h * 1.6, height: h)
            let f = fitted(viewHeight: h)
            let usable = RobotSpace3DStage.usableViewport(size, insets: insets)
            let zoom = RobotSpace3DStage.fitZoom(radius: r, homeScale: f.homeScale,
                                                 scenePerPoint: f.scenePerPoint,
                                                 usable: usable)
            let d = RobotSpace3DStage.onScreenDiameter(radius: r, zoom: zoom,
                                                       homeScale: f.homeScale,
                                                       scenePerPoint: f.scenePerPoint)
            XCTAssertGreaterThanOrEqual(d, lastDiameter - 0.5,
                                        "a taller window made the hand SMALLER (h=\(h))")
            lastDiameter = d
        }
    }

    /// Chrome is not free: the same window with a three-row top bar must frame
    /// SMALLER than one with a single row, or the module ends up behind it.
    func testChromeShrinksTheFraming() {
        let size = CGSize(width: 1280, height: 800)
        let f = fitted(viewHeight: size.height)
        let thin = EdgeInsets(top: 52, leading: 0, bottom: 111, trailing: 0)
        let thick = EdgeInsets(top: 148, leading: 0, bottom: 111, trailing: 0)
        let r: Float = 0.3534                                   // the lift column
        func zoom(_ i: EdgeInsets) -> Float {
            RobotSpace3DStage.fitZoom(radius: r, homeScale: f.homeScale,
                                      scenePerPoint: f.scenePerPoint,
                                      usable: RobotSpace3DStage.usableViewport(size, insets: i))
        }
        XCTAssertLessThan(zoom(thick), zoom(thin),
                          "the framing ignored the taller top bar")
    }

    /// Degenerate inputs (before the first window fit) still produce a sane,
    /// clamped zoom rather than 0 or infinity.
    func testUnmeasuredStageFallsBackSafely() {
        for r in Self.measuredRadius.map(\.1) {
            let z = RobotSpace3DStage.fitZoom(radius: r, homeScale: 0,
                                              scenePerPoint: 0, usable: 0)
            XCTAssertGreaterThanOrEqual(z, RobotSpace3DStage.minModuleZoom)
            XCTAssertLessThanOrEqual(z, RobotSpace3DStage.maxModuleZoom)
        }
    }

    // MARK: - Orbit keeps the framing (the "snaps back to horizontal" bug)

    /// `RobotStage.orbit` used to overwrite the orientation with a level yaw
    /// about the rig ORIGIN, so the first drag inside a module framing threw
    /// away its elevation and its centring. Now it composes onto the adopted
    /// framing and keeps the anchor pinned.
    @MainActor
    func testOrbitPreservesTheAdoptedFraming() {
        let stage = RobotStage(scale: 0.4)
        // A module-style framing: rolled off the horizon, zoomed, off-centre.
        var t = Transform()
        t.scale = SIMD3(repeating: 0.9)
        t.rotation = RobotSpace3DStage.lookRotation(simd_normalize(SIMD3<Float>(0.4, 0.6, 1)))
        t.translation = SIMD3(0.05, -0.2, 0.1)
        let anchor = SIMD3<Float>(-0.17, 0.61, 0.17)          // the left arm's centre
        stage.rig.root.transform = t
        stage.adoptFraming(t, anchorLocal: anchor)

        let before = t.translation + t.rotation.act(anchor * t.scale.x)
        stage.orbit(dx: 0.4, dy: 0.15)
        let after = stage.rig.root.transform.translation
            + stage.rig.root.transform.rotation.act(anchor * stage.rig.root.transform.scale.x)
        XCTAssertEqual(simd_distance(before, after), 0, accuracy: 1e-5,
                       "orbiting moved the framed module off its anchor")
        // It really did rotate, and it did NOT collapse to a level yaw.
        let level = simd_quatf(angle: stage.yaw, axis: SIMD3(0, 1, 0))
        XCTAssertGreaterThan(simd_length(stage.rig.root.transform.rotation.vector - level.vector),
                             1e-3, "the framing's elevation was discarded")
        XCTAssertNotEqual(stage.orbitPitch, 0, accuracy: 1e-6)
    }

    /// Pinch-zoom pivots on the framed subject too — otherwise a zoom flings it
    /// out of the window (it scaled about the rig origin, which after a module
    /// framing is far off screen).
    @MainActor
    func testZoomPivotsOnTheFramedSubject() {
        let stage = RobotStage(scale: 0.4)
        var t = Transform()
        t.scale = SIMD3(repeating: 0.7)
        t.rotation = simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0))
        t.translation = SIMD3(0.02, -0.3, 0)
        let anchor = SIMD3<Float>(-0.57, 0.53, 0.20)          // the left hand
        stage.rig.root.transform = t
        stage.adoptFraming(t, anchorLocal: anchor)

        let before = t.translation + t.rotation.act(anchor * t.scale.x)
        stage.zoom(1.6)
        let now = stage.rig.root.transform
        let after = now.translation + now.rotation.act(anchor * now.scale.x)
        XCTAssertEqual(simd_distance(before, after), 0, accuracy: 1e-5,
                       "zooming moved the framed module off its anchor")
        XCTAssertGreaterThan(now.scale.x, t.scale.x)
    }

    /// With NO framing adopted (the volumetric window, the Body tab) the
    /// gestures must behave exactly as before: pure yaw, origin untouched.
    @MainActor
    func testUnframedOrbitIsUnchanged() {
        let stage = RobotStage(scale: 0.4)
        let p = SIMD3<Float>(0.1, -0.2, 0.3)
        stage.rig.root.position = p
        stage.orbit(dx: 0.3)
        XCTAssertEqual(simd_distance(stage.rig.root.position, p), 0, accuracy: 1e-6,
                       "an unframed orbit moved the rig")
        let expected = simd_quatf(angle: stage.yaw, axis: SIMD3(0, 1, 0))
        XCTAssertEqual(simd_length(stage.rig.root.orientation.vector - expected.vector), 0,
                       accuracy: 1e-5)
    }

    /// Elevation is clamped ASYMMETRICALLY: up stops short of the pole (where
    /// yaw and roll degenerate), down stops just under the horizon. Dropping
    /// below the ground plane shows the underside of the floor disc and an
    /// upside-down world, with no cue for how to get back.
    @MainActor
    func testOrbitPitchIsClamped() {
        let stage = RobotStage(scale: 0.4)
        stage.orbit(dx: 0, dy: 40)
        XCTAssertEqual(stage.orbitPitch, RobotStage.orbitPitchMax, accuracy: 1e-5)
        stage.orbitEnded()
        stage.orbit(dx: 0, dy: -40)
        XCTAssertEqual(stage.orbitPitch, RobotStage.orbitPitchMin, accuracy: 1e-5)
        XCTAssertLessThan(RobotStage.orbitPitchMin, 0, "a little below the horizon is useful")
        XCTAssertGreaterThan(RobotStage.orbitPitchMin, -0.4, "but never under the floor")
    }

    /// THE FLOOR STAYS LEVEL. Whatever framing is adopted and however the
    /// operator orbits it, the rig's up axis must project straight up on screen
    /// — otherwise the ground tips and the whole view reads as rotating
    /// (operator, 2026-07-28: "I want the camera view to lock the floor").
    @MainActor
    func testHorizonStaysLevelUnderEveryFraming() {
        // A module framing's rotation is arbitrary: elevated, off-axis, rolled.
        let dirs: [SIMD3<Float>] = [
            SIMD3(0, 0, 1), SIMD3(0.4, 0.6, 1), SIMD3(-0.8, 0.3, 0.5),
            SIMD3(0.2, -0.15, -1), SIMD3(1, 0.9, 0.1),
        ]
        for d in dirs {
            let stage = RobotStage(scale: 0.4)
            var t = Transform()
            t.scale = SIMD3(repeating: 0.8)
            t.rotation = RobotSpace3DStage.lookRotation(simd_normalize(d))
            t.translation = SIMD3(0.03, -0.25, 0.05)
            stage.rig.root.transform = t
            stage.adoptFraming(t, anchorLocal: SIMD3(0, 0.8, 0))

            for (dx, dy) in [(Float(0), Float(0)), (0.5, 0.0), (0.0, 0.4),
                             (-0.9, -0.3), (2.0, 0.9)] {
                stage.orbitEnded()
                stage.orbit(dx: dx, dy: dy)
                // The rig's up axis, on screen: its x-component IS the roll.
                let up = stage.rig.root.transform.rotation.act(SIMD3<Float>(0, 1, 0))
                XCTAssertEqual(up.x, 0, accuracy: 1e-4,
                               "horizon rolled by \(up.x) for dir \(d) drag (\(dx), \(dy))")
                XCTAssertGreaterThan(up.y, 0, "the world turned upside down")
            }
        }
    }

    /// A framing adopted with `levelled: false` (the head-image dive) keeps its
    /// exact rotation — it has to face a tilted image plane squarely or the
    /// crossfade swaps two different pictures.
    @MainActor
    func testUnlevelledFramingIsPreservedExactly() {
        let stage = RobotStage(scale: 0.4)
        var t = Transform()
        t.rotation = RobotSpace3DStage.lookRotation(simd_normalize(SIMD3<Float>(0.4, 0.6, 1)))
        t.scale = SIMD3(repeating: 0.9)
        stage.rig.root.transform = t
        stage.adoptFraming(t, anchorLocal: .zero, levelled: false)
        stage.orbit(dx: 0, dy: 0)
        XCTAssertEqual(simd_length(stage.rig.root.transform.rotation.vector - t.rotation.vector),
                       0, accuracy: 1e-5)
    }
}
