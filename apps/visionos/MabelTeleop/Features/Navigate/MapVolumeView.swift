//
//  MapVolumeView.swift
//  MabelTeleop
//
//  The SLAM map in a volumetric window: walk around the room reconstruction
//  with the live MABEL miniature inside, pinch the floor to plan, and use
//  the same control set as the robot volume — Recenter · Life Size · Place
//  on Floor · Fit · Console. Room selection follows the Maps tab.
//

import SwiftUI

struct MapVolumeView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openWindow) private var openWindow
    @StateObject private var map = MapStage()
    @AppStorage("maps.selectedRoom") private var selectedRoomName = ""

    /// Volume is 2.2×2.4×2.2 m; its local origin is the centre.
    private let volumeFloorY: Float = -1.15
    private let volumeExtents = SIMD2<Float>(2.0, 2.1)

    var body: some View {
        MapStageView(map: map)
            .ornament(attachmentAnchor: .scene(.bottom), contentAlignment: .top) {
                HStack(spacing: 10) {
                    control("scope", "Recenter") {
                        map.recenter()
                    }
                    control("figure.stand", "Life Size") {
                        map.lifeSize(floorY: volumeFloorY)
                    }
                    control("arrow.down.to.line", "Place on Floor") {
                        map.placeOnFloor(y: volumeFloorY)
                    }
                    control("rectangle.compress.vertical", "Fit") {
                        map.fitTo(width: volumeExtents.x, height: volumeExtents.y)
                    }
                    Divider().frame(height: 22)
                    goButton
                    control("rectangle.on.rectangle", "Console") {
                        openWindow(id: WindowID.main.rawValue)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .glassBackgroundEffect()
                .padding(.top, 10)
            }
            .onAppear {
                map.controller = controller
                map.setRoom(currentRoom)
            }
            .onChange(of: selectedRoomName) { _, _ in
                map.setRoom(currentRoom)
            }
            .onDisappear { map.stopFollowing() }
    }

    private var currentRoom: SLAMRoom {
        SLAMRoom.library.first(where: { $0.name == selectedRoomName }) ?? SLAMRoom.library[0]
    }

    private var goButton: some View {
        Button {
            Haptics.rigid()
            map.startFollowing()
        } label: {
            Label(map.following ? "Driving…" : "Go",
                  systemImage: map.following ? "figure.walk.motion" : "play.fill")
                .font(.system(size: 13, weight: .semibold))
        }
        .buttonStyle(.borderedProminent)
        .tint(Theme.Palette.accent)
        .disabled(!map.hasGoal || map.following)
    }

    private func control(_ icon: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.system(size: 13, weight: .bold))
                Text(label).font(.system(size: 13, weight: .semibold))
            }
        }
        .buttonStyle(.bordered)
        .tint(Theme.Palette.accent)
    }
}
