//
//  NavigateView.swift
//  MabelTeleop
//
//  The map library — saved visual-SLAM reconstructions as TRUE 3D point
//  clouds with the live MABEL miniature inside. Look at any spot on the
//  floor and pinch: the A* global planner routes the base around the
//  furniture (same planner as the iOS app), draws the route, and Go drives
//  the real/sim base over the bridge with the same pure-pursuit follower.
//  Drag rotates the diorama, pinch-zoom scales it.
//

import SwiftUI
import RealityKit
import simd

struct NavigateView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openWindow) private var openWindow
    @StateObject private var map = MapStage()
    @State private var selected: SLAMRoom = SLAMRoom.library[0]
    @AppStorage("maps.selectedRoom") private var selectedRoomName = ""

    var body: some View {
        HStack(spacing: 12) {
            sidebar
                .frame(width: 232)
            inspector
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(.horizontal, 14)
        .onAppear {
            if let saved = SLAMRoom.library.first(where: { $0.name == selectedRoomName }) {
                selected = saved
            }
            map.controller = controller
            map.setRoom(selected)
        }
        .onDisappear { map.stopFollowing() }
    }

    // MARK: - Room list

    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 6) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Maps").font(.mabelTitle).foregroundStyle(Theme.Palette.textPrimary)
                    Text("\(SLAMRoom.library.count) reconstructions")
                        .font(.system(size: 10, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
                }
                .padding(.top, 4).padding(.bottom, 2)
                ForEach(SLAMRoom.library) { room in
                    roomCard(room)
                }
            }
            .padding(10)
        }
        .scrollIndicators(.hidden)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.lg, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Theme.Radius.lg, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        }
        .padding(.vertical, 52)
    }

    private func roomCard(_ room: SLAMRoom) -> some View {
        let active = room.id == selected.id
        return Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { selected = room }
            selectedRoomName = room.name          // map volume follows
            map.setRoom(room)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: room.icon)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(active ? .white : Theme.Palette.accent)
                    .frame(width: 30, height: 30)
                    .background(
                        active ? AnyShapeStyle(Theme.accentGradient)
                               : AnyShapeStyle(Theme.Palette.accent.opacity(0.14)),
                        in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                    )
                VStack(alignment: .leading, spacing: 0) {
                    Text(room.name).font(.mabel(12.5, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                    Text("\(Int(room.areaM2)) m² · \(room.pointCount / 1000)k")
                        .font(.system(size: 9.5, weight: .medium, design: .rounded)).foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(7)
            .background {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(active ? Theme.Palette.surfaceHi : Color.clear)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .strokeBorder(active ? Theme.Palette.accent.opacity(0.5) : Color.clear, lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
        }
        .buttonStyle(PressableStyle())
    }

    // MARK: - 3D inspector

    private var inspector: some View {
        ZStack(alignment: .top) {
            MapStageView(map: map)
            // Status + Go bar, plus a concise "demo map" notice when the real SLAM
            // isn't streaming — same eye level as the rest of the window chrome.
            VStack(spacing: 8) {
                topBar
                if !map.isLive { liveOfflinePill }
            }
            .padding(.top, 56)

            // Stage view controls — right-hand side, like the toggles.
            HStack { Spacer()
                VStack(spacing: 10) {
                    StageActionButton(icon: "scope", label: "Recenter") { map.recenter() }
                    StageActionButton(icon: "rectangle.compress.vertical", label: "Fit") { map.fitView() }
                    StageActionButton(icon: "video.fill", label: "Follow", active: map.tracking) {
                        map.setTracking(!map.tracking)
                    }
                    StageActionButton(icon: "cube.transparent", label: "Volume") {
                        openWindow(id: WindowID.mapVolume.rawValue)
                    }
                }
                .padding(.trailing, 12)
            }
            .frame(maxHeight: .infinity, alignment: .center)
        }
    }

    /// Shown over the synthetic room when no robot is streaming a live map — same
    /// glass pill chrome as the rest of the app.
    private var liveOfflinePill: some View {
        HintPill(systemImage: "antenna.radiowaves.left.and.right.slash",
                 title: "Live SLAM not connected",
                 subtitle: "Demo map · pinch the floor to plan A* & follow",
                 iconTint: Theme.Palette.warn)
    }

    private var topBar: some View {
        // Hugs its content and sits centred — not a full-width strip.
        HStack(spacing: 12) {
            statusLine
            inline("\(selected.pointCount / 1000)k pts", "circle.grid.3x3.fill")
            if map.hasGoal {
                inline(String(format: "%.1f m", map.pathLen),
                       "point.topleft.down.to.point.bottomright.curvepath.fill", tint: Theme.Palette.live)
            }
            Button {
                Haptics.rigid()
                map.startFollowing()
            } label: {
                Label(map.following ? "Driving…" : "Go",
                      systemImage: map.following ? "figure.walk.motion" : "play.fill")
                    .font(.system(size: 13, weight: .semibold)).foregroundStyle(.white)
                    .padding(.horizontal, 16).padding(.vertical, 7)
                    .background((map.hasGoal && !map.following) ? AnyShapeStyle(Theme.accentGradient)
                                : AnyShapeStyle(Color.white.opacity(0.18)), in: Capsule())
                    .contentShape(Capsule())
            }
            .buttonStyle(PressableStyle())
            .disabled(!map.hasGoal || map.following)
            .opacity((map.hasGoal && !map.following) ? 1 : 0.5)

        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .glassBackgroundEffect(in: Capsule())
        .frame(maxWidth: .infinity)        // centres the capsule in the view
    }

    private var statusLine: some View {
        let icon = map.following ? "figure.walk.motion" : (map.hasGoal ? "point.topleft.down.to.point.bottomright.curvepath.fill" : "hand.tap.fill")
        let text = map.following ? "Following path · pure pursuit"
            : (map.hasGoal ? "Route planned — press Go" : "Pinch the floor to set a goal")
        return HStack(spacing: 6) {
            Image(systemName: icon).font(.system(size: 12, weight: .bold))
            Text(text).font(.system(size: 12, weight: .bold)).lineLimit(1)
        }
        .foregroundStyle(map.following ? Theme.Palette.accent : (map.hasGoal ? Theme.Palette.live : Theme.Palette.textPrimary))
    }

    private func inline(_ text: String, _ icon: String, tint: Color = .white) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.system(size: 10, weight: .semibold)).foregroundStyle(tint.opacity(0.85))
            Text(text).font(.system(size: 12, weight: .semibold)).foregroundStyle(tint).lineLimit(1)
        }
    }
}
