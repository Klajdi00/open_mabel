//
//  RobotVolumeView.swift
//  MabelTeleop
//
//  The VOLUMETRIC Robot space — a walk-around RealityKit volume: the live
//  articulated robot + its sensor world (SensorLayers: Cam 3D projections,
//  occupancy grid, map cloud + mesh, laser ring), with per-layer toggle chips
//  in the volume's bottom ornament. Opened from the right edge of the viewer.
//

import SwiftUI
import RealityKit

struct RobotVolumeView: View {
    // Plain reference (not @EnvironmentObject) — the volume must never
    // re-render at uiTick rate; all live data is pulled by timers/observers.
    let controller: RobotController
    @State private var model: RobotVolumeModel? = nil
    @State private var camOn = true
    @State private var gridOn = true
    @State private var cloudOn = true
    @State private var laserOn = true

    var body: some View {
        RealityView { content in
            let m = RobotVolumeModel(controller: controller)
            content.add(m.root)
            m.updateSub = content.subscribe(to: SceneEvents.Update.self) { [weak m] _ in
                m?.tick()
            }
            Task { @MainActor in
                model = m
                await m.loadRig()
            }
        } update: { _ in
            model?.setLayers(cam: camOn, grid: gridOn, cloud: cloudOn, laser: laserOn)
        }
        .ornament(attachmentAnchor: .scene(.bottom), contentAlignment: .top) {
            HStack(spacing: 6) {
                chip("Cam 3D", "camera.metering.matrix", $camOn)
                chip("2D Grid", "grid", $gridOn)
                chip("Cloud", "aqi.medium", $cloudOn)
                chip("Laser", "rays", $laserOn)
            }
            .padding(8)
            .glassBackgroundEffect(in: Capsule(style: .continuous))
            .padding(.top, 10)
        }
        .onDisappear { model?.teardown(); model = nil }
    }

    private func chip(_ title: String, _ icon: String, _ on: Binding<Bool>) -> some View {
        Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { on.wrappedValue.toggle() }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: icon).font(.system(size: 12, weight: .semibold))
                Text(title).font(.system(size: 12, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(on.wrappedValue ? .white : Theme.Palette.textSecondary)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background {
                if on.wrappedValue { Capsule().fill(Theme.accentGradient) }
                else { Capsule().fill(Theme.Palette.surface) }
            }
        }
        .buttonStyle(.plain)
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }
}

/// The volume's entity tree: an articulated rig + SensorLayers, floored and
/// scaled to the volume box. NOT ObservableObject on purpose.
@MainActor
final class RobotVolumeModel {

    let root = Entity()
    private let rig = RobotRigEntity()
    private var layers: SensorLayers?
    private weak var controller: RobotController?
    var updateSub: EventSubscription?
    private var lastPoseTick: UInt64 = .max

    init(controller: RobotController) {
        self.controller = controller
        // Content origin is the volume CENTRE; the 1.3 m box clips below −0.65.
        root.scale = SIMD3(repeating: 0.42)
        root.position = SIMD3(0, -0.58, 0)
        root.addChild(rig.root)
        // Immediate content: a soft pedestal under the robot's spot, visible the
        // instant the volume opens (the 14 MB rig parse takes a few seconds).
        var pedMat = PhysicallyBasedMaterial()
        pedMat.baseColor = .init(tint: UIColor(white: 0.22, alpha: 1))
        pedMat.roughness = 0.6
        let pedestal = ModelEntity(mesh: .generateCylinder(height: 0.02, radius: 0.55),
                                   materials: [pedMat])
        pedestal.position = SIMD3(0, -0.01, 0)
        root.addChild(pedestal)
        rig.pose(joints: controller.livePose.joints,
                 heading: controller.livePose.heading,
                 baseQuat: controller.livePose.baseQuat)
        let l = SensorLayers(controller: controller, rig: rig)
        l.start()
        layers = l
    }

    func loadRig() async {
        await rig.load()
    }

    func teardown() {
        updateSub?.cancel(); updateSub = nil
        layers?.stop(); layers = nil
    }

    func setLayers(cam: Bool, grid: Bool, cloud: Bool, laser: Bool) {
        layers?.setLayers(cam: cam, grid: grid, cloud: cloud, laser: laser, mesh: true, tf: false)
    }

    func tick() {
        guard let controller else { return }
        let pose = controller.livePose
        if pose.tick != lastPoseTick {
            lastPoseTick = pose.tick
            rig.pose(joints: pose.joints, heading: pose.heading, baseQuat: pose.baseQuat)
        }
    }
}
