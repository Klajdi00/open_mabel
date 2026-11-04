//
//  RigReloadRaceTests.swift
//  MabelTeleopTests
//
//  "When I connect to the Retarget Lab the 3D view shows two meshes overlapping" (2026-10-17).
//  The app boots on the rig saved last session (NOVA's) and the lab's hello says MABEL, so the
//  `.robotRigChanged` rebuild arrived while the first load was still parsing PLYs: nothing was
//  built yet, so nothing was removed, a second build started beside the first, and both
//  attached a full robot. Builds are now serialized; these pin "exactly one robot" for a swap
//  that lands mid-load, a swap after load, and a no-op swap.
//

import Testing
import RealityKit
@testable import MabelTeleop

@MainActor
@Suite("Rig rebuild never doubles the robot")
struct RigReloadRaceTests {

    /// Geometry hanging off `robot`, decorations (force/guide arrows) excluded.
    private func meshCount(_ rig: RobotRigEntity) -> Int {
        func walk(_ e: Entity) -> Int {
            if e.name.hasPrefix("decor:") { return 0 }
            return (e is ModelEntity ? 1 : 0) + e.children.reduce(0) { $0 + walk($1) }
        }
        return rig.robot.children.reduce(0) { $0 + walk($1) }
    }

    private func reference() async -> Int {
        let rig = RobotRigEntity()
        await rig.load()
        return meshCount(rig)
    }

    @Test func swapDuringFirstLoadBuildsOneRobot() async {
        let single = await reference()
        #expect(single > 10)
        let rig = RobotRigEntity()
        var sig = "nova_rig|v1|v1"
        rig.currentSignature = { sig }
        let first = Task { await rig.load() }
        // Let the load start and reach its PLY parse (~1 s for 14 MB; the race window).
        try? await Task.sleep(for: .milliseconds(150))
        #expect(rig.parts.isEmpty)          // still mid-load: the swap really lands inside it
        sig = "rig|v1|v1"                   // the lab's hello lands
        await rig.reloadIfRigChanged()
        await first.value
        #expect(meshCount(rig) == single)
        #expect(rig.parts.count == single)
    }

    @Test func swapAfterLoadReplacesTheRobot() async {
        let single = await reference()
        let rig = RobotRigEntity()
        var sig = "nova_rig|v1|v1"
        rig.currentSignature = { sig }
        await rig.load()
        sig = "rig|v1|v1"
        async let a: Void = rig.reloadIfRigChanged()     // two notifications back to back
        async let b: Void = rig.reloadIfRigChanged()
        _ = await (a, b)
        #expect(meshCount(rig) == single)
    }

    @Test func unchangedSignatureDoesNotRebuild() async {
        let rig = RobotRigEntity()
        rig.currentSignature = { "rig|v1|v1" }
        await rig.load()
        let before = rig.parts.map(ObjectIdentifier.init)
        await rig.reloadIfRigChanged()
        #expect(rig.parts.map(ObjectIdentifier.init) == before)
    }
}
