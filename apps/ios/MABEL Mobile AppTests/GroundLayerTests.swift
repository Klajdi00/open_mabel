//
//  GroundLayerTests.swift
//  MABEL Mobile AppTests
//
//  A LAYER CHIP MUST CHANGE WHAT IS DRAWN, AND NOTHING ELSE.
//
//  The bug these exist for (2026-07-29, reported as "the mesh button and the 2D
//  grid button don't work"): both buttons were wired correctly and both were
//  useless. `Mesh` drove the nvblox SLAM surface and `2D Grid` drove the
//  occupancy floorplan — and on a robot brought up without mapping, which is the
//  normal case on thor, BOTH streams are empty. So both chips lit up and changed
//  nothing on screen, while the only ground actually visible was a third chip
//  called `Ground`. Measured on the live robot the same day: `ws://…:9092` held
//  open for 12 s and delivered zero frames of every type.
//
//  The fix has two halves and this file guards the second:
//    · the empty layers now DIM and say why (a view concern, checked by eye);
//    · the STAGE layers — Robot, Floor, Grid — are locally generated, so they
//      always have something to toggle. Those are mechanical and testable.
//
//  The subtle one is `setRobotHidden`. The LiDAR ring, the TF triads and the
//  Cam 3D video planes all hang off nodes INSIDE the robot subtree, so the
//  obvious implementation (`robot.isHidden = true`) blanks the sensor picture
//  along with the model — the exact opposite of why an operator hides the robot.
//  `testHidingRobotKeepsSensorLayers` is the regression lock on that.
//

import XCTest
import SceneKit
import simd
@testable import MABEL_Mobile_App

final class GroundLayerTests: XCTestCase {

    // MARK: - Ground geometry

    func testSurfaceIsMuchSmallerThanTheGrid() {
        // Not a style preference — the measured failure. A ground plane the size
        // of the grid covers most of the frame from the viewer's ~3 m eye height
        // and turns the whole scene grey. The grid carries the far field.
        XCTAssertLessThan(GroundPlane.surfaceRadius, GroundPlane.radius / 3,
                          "the wash must stay a pad under the robot, not a room-sized filter")
    }

    func testSurfaceAndShadowBuild() {
        let s = GroundPlane.surface()
        XCTAssertFalse(s.childNodes.isEmpty, "ground surface built no geometry")
        // Depth writes would make the floor occlude the cloud, mesh and path.
        XCTAssertTrue(s.childNodes.allSatisfy {
            $0.geometry?.firstMaterial?.writesToDepthBuffer == false
        })
        // A nil texture (a CGContext this code got wrong) would leave an
        // invisible-but-present plane — a silent, and very confusing, failure.
        let sh = GroundPlane.contactShadow()
        XCTAssertNotNil(sh.geometry?.firstMaterial?.diffuse.contents,
                        "contact shadow has no texture")
        XCTAssertFalse(sh.geometry?.firstMaterial?.writesToDepthBuffer ?? true)
    }

    func testFineLatticeIsDenserAndStaysNear() {
        // The two lattices divide the work: fine texture near the robot, metric
        // reference everywhere. If the fine one were not actually finer, or spread
        // as far, we would be back to one cadence of hairlines in a void.
        XCTAssertLessThan(GroundPlane.fineStep, GroundPlane.step)
        let fine = GroundPlane.surface()
        var maxR: Float = 0
        var verts = 0
        for node in fine.childNodes {
            guard let src = node.geometry?.sources(for: .vertex).first else { continue }
            let vs = src.asVectors()
            verts += vs.count
            for v in vs { maxR = max(maxR, (v.x * v.x + v.z * v.z).squareRoot()) }
        }
        XCTAssertGreaterThan(verts, 0, "fine lattice built no geometry")
        XCTAssertLessThanOrEqual(maxR, GroundPlane.surfaceRadius + 0.01,
                                 "the fine texture escapes its near field")
    }

    func testGridIsClippedToADiscAndFadesOut() {
        let grid = GroundPlane.grid()
        XCTAssertFalse(grid.childNodes.isEmpty, "grid built no geometry")

        // Every vertex inside the disc: a square grid would poke past the wash and
        // announce its own corners.
        var maxR: Float = 0
        var alphas = Set<Int>()
        for node in grid.childNodes {
            guard let src = node.geometry?.sources(for: .vertex).first else { continue }
            for v in src.asVectors() {
                maxR = max(maxR, (v.x * v.x + v.z * v.z).squareRoot())
            }
            if let c = node.geometry?.firstMaterial?.diffuse.contents as? UIColor {
                var a: CGFloat = 0
                c.getWhite(nil, alpha: &a)
                alphas.insert(Int((a * 1000).rounded()))
            }
        }
        XCTAssertLessThanOrEqual(maxR, GroundPlane.radius + 0.01,
                                 "grid escapes its own disc")
        // Several distinct opacities = the radial fade is actually present. One
        // opacity would be the old flat 16 m square with a new coat of paint.
        XCTAssertGreaterThan(alphas.count, 4,
                             "grid has no radial fade (only \(alphas.count) opacity bands)")
    }

    func testStageSnapsToWholeCellsSoTheGridStaysWorldLocked() {
        // Following the robot CONTINUOUSLY would glue the lines to it and the
        // robot would look motionless. Snapping keeps the lines on world
        // positions while the pad stays underneath.
        let p = GroundPlane.snap(to: simd_float3(3.4, 5, -7.6))
        XCTAssertEqual(p.x, 3, accuracy: 1e-5)
        XCTAssertEqual(p.z, -8, accuracy: 1e-5)
        XCTAssertEqual(p.y, 0, accuracy: 1e-5, "the ground must stay on the floor")
        // And the offset is always a whole number of cells, so a line that was at
        // world x = k is still at world x = k after the snap.
        for v in stride(from: Float(-9.5), through: 9.5, by: 0.37) {
            let s = GroundPlane.snap(to: simd_float3(v, 0, v)).x
            XCTAssertEqual(s / GroundPlane.step, (s / GroundPlane.step).rounded(),
                           accuracy: 1e-4)
        }
    }

    // MARK: - The chips

    @MainActor
    func testFloorAndGridToggleIndependently() {
        let rig = LiveMapRig()

        rig.setFloorHidden(true)
        XCTAssertTrue(rig.floorSurface?.isHidden ?? false)
        XCTAssertTrue(rig.contactShadow?.isHidden ?? false,
                      "the shadow belongs to Floor — a pool with no floor under it "
                      + "is a smudge in mid-air")
        XCTAssertFalse(rig.floorGrid?.isHidden ?? true, "Floor must not take the Grid with it")

        rig.setFloorHidden(false)
        rig.setFloorGridHidden(true)
        XCTAssertTrue(rig.floorGrid?.isHidden ?? false)
        XCTAssertFalse(rig.floorSurface?.isHidden ?? true, "Grid must not take the Floor with it")
        XCTAssertFalse(rig.contactShadow?.isHidden ?? true)
    }

    @MainActor
    func testHidingRobotHidesTheModel() {
        let rig = LiveMapRig()
        let geometry = rig.robot.descendantsWithGeometry()
        XCTAssertGreaterThan(geometry.count, 10, "the rig loaded no meshes")

        rig.setRobotHidden(true)
        XCTAssertTrue(geometry.allSatisfy(\.isHidden), "some robot mesh stayed visible")
        rig.setRobotHidden(false)
        XCTAssertTrue(geometry.allSatisfy { !$0.isHidden })
    }

    @MainActor
    func testHidingRobotKeepsSensorLayers() {
        // THE regression lock. The LiDAR ring is parented to `robot` so it rides
        // the base pose; `robot.isHidden = true` would take it — and the TF triads,
        // and the Cam 3D video planes — down with the model. Hiding the robot is
        // how an operator SEES the sensor picture, so blanking it is exactly
        // backwards.
        let rig = LiveMapRig()
        rig.apply(lidar: (angleMin: -.pi, angleInc: 0.01, rangeMax: 8,
                          ranges: Array(repeating: 2.0, count: 128)), stride: 1)
        XCTAssertFalse(rig.lidarContainer.childNodes.isEmpty, "no LiDAR geometry to test with")

        rig.setRobotHidden(true)
        XCTAssertFalse(rig.lidarContainer.isHidden,
                       "hiding the robot must not hide the LiDAR ring")
        XCTAssertTrue(rig.lidarContainer.childNodes.allSatisfy { !$0.isHidden },
                      "hiding the robot must not hide the LiDAR points")
        // The POV camera lives under the head link and must keep rendering — the
        // 1st-person view is the one place you ALWAYS want the model out of the way.
        XCTAssertFalse(rig.povCamera.isHidden)
    }

    @MainActor
    func testStageFollowsTheRobot() {
        let rig = LiveMapRig()
        rig.robot.simdPosition = simd_float3(6.2, 0.078, -3.9)
        rig.syncStage()
        XCTAssertEqual(rig.stage.simdPosition.x, 6, accuracy: 1e-4)
        XCTAssertEqual(rig.stage.simdPosition.z, -4, accuracy: 1e-4)
    }
}

// MARK: - helpers

private extension SCNGeometrySource {
    /// Read a float3 vertex source back out. SceneKit gives no accessor for this.
    func asVectors() -> [SCNVector3] {
        guard semantic == .vertex, usesFloatComponents, componentsPerVector >= 3 else { return [] }
        var out: [SCNVector3] = []
        out.reserveCapacity(vectorCount)
        data.withUnsafeBytes { raw in
            for i in 0..<vectorCount {
                let o = dataOffset + i * dataStride
                out.append(SCNVector3(raw.loadUnaligned(fromByteOffset: o, as: Float.self),
                                      raw.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                                      raw.loadUnaligned(fromByteOffset: o + 8, as: Float.self)))
            }
        }
        return out
    }
}

private extension SCNNode {
    func descendantsWithGeometry() -> [SCNNode] {
        var out: [SCNNode] = []
        enumerateHierarchy { n, _ in
            // Skip the stage/shadow helpers parented in later — this is the MODEL.
            if n.geometry != nil, n.name != "contactShadow" { out.append(n) }
        }
        return out
    }
}
