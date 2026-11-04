//
//  SceneLibrary.swift
//  MabelTeleop (ported from MABEL Mobile App v2)
//
//  The sim scene library ("jump into a world like a game") + the scene-switch
//  loading overlay — iOS-native ports of the Simulation Control Center's scene
//  dropdown and switch card: category sections of thumbnail cards, then a
//  full-screen glass card with the new world's thumbnail, a milestone progress
//  bar and the physics/telemetry/cameras checklist.
//

import SwiftUI

// MARK: - Scene library sheet

struct SceneLibrarySheet: View {
    @ObservedObject var studio: SimStudio
    @Environment(\.dismiss) private var dismiss

    private var categories: [(String, [SimStudio.SceneInfo])] {
        let grouped = Dictionary(grouping: studio.scenes) { $0.category ?? "Worlds" }
        return grouped.keys.sorted().map { ($0, grouped[$0] ?? []) }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                // No scene backend (SCC not up): explain instead of a blank sheet.
                if studio.scenes.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "square.stack.3d.up.slash")
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(Theme.Palette.textTertiary)
                        Text("No scene library").font(.system(size: 15, weight: .semibold))
                        Text("Switching worlds needs the Simulation Control Center. Launch the sim with server/run.sh (not start_teleop.sh) and it appears here. Reset → Random still works from the bridge.")
                            .font(.system(size: 12)).multilineTextAlignment(.center)
                            .foregroundStyle(Theme.Palette.textSecondary)
                            .padding(.horizontal, 28)
                        Button("Check again") { studio.reprobe() }
                            .font(.system(size: 12, weight: .semibold)).padding(.top, 4)
                    }
                    .frame(maxWidth: .infinity).padding(.top, 60)
                }
                LazyVStack(alignment: .leading, spacing: 14, pinnedViews: []) {
                    ForEach(categories, id: \.0) { cat, list in
                        Text(cat.uppercased())
                            .font(.system(size: 10, weight: .heavy)).tracking(1.4)
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .padding(.horizontal, 16)
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10),
                                            GridItem(.flexible())], spacing: 10) {
                            ForEach(list) { scene in card(scene) }
                        }
                        .padding(.horizontal, 16)
                    }
                }
                .padding(.vertical, 12)
            }
            .background(Theme.Palette.canvas)
            .navigationTitle("Worlds")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func card(_ scene: SimStudio.SceneInfo) -> some View {
        let isCurrent = scene.name == studio.currentScene
        return Button {
            guard !isCurrent else { dismiss(); return }
            Haptics.rigid()
            dismiss()
            studio.switchScene(scene)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                ZStack(alignment: .topTrailing) {
                    SceneThumb(studio: studio, name: scene.name) {
                        Rectangle().fill(Theme.Palette.surface)
                            .overlay {
                                Image(systemName: "cube.transparent")
                                    .font(.system(size: 22, weight: .light))
                                    .foregroundStyle(Theme.Palette.textTertiary)
                            }
                    }
                    .frame(height: 96)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    if scene.dynamic == true {
                        Text("LIVE")
                            .font(.system(size: 8, weight: .heavy)).tracking(0.8)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 6).padding(.vertical, 3)
                            .background(Theme.Palette.live, in: Capsule())
                            .padding(6)
                    }
                }
                Text(scene.displayTitle)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .lineLimit(1)
                if let d = scene.description, !d.isEmpty {
                    Text(d)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(2, reservesSpace: true)
                } else {
                    Text(" ").font(.system(size: 10)).lineLimit(2, reservesSpace: true)
                }
            }
            .padding(8)
            .background {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Theme.Palette.surface)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(isCurrent ? Theme.Palette.accent : Theme.Palette.hairline,
                                  lineWidth: isCurrent ? 2 : 1)
            }
            .overlay(alignment: .topLeading) {
                if isCurrent {
                    Text("CURRENT")
                        .font(.system(size: 8, weight: .heavy)).tracking(0.8)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(Theme.Palette.accent, in: Capsule())
                        .padding(12)
                }
            }
        }
        .buttonStyle(PressableStyle())
    }
}

// MARK: - Scene-switch loading overlay

struct SceneLoadingOverlay: View {
    @ObservedObject var studio: SimStudio

    var body: some View {
        if let p = studio.switching {
            ZStack {
                // Blurred new-world backdrop under a dark scrim (SCC switch-back).
                SceneThumb(studio: studio, name: p.scene.name) { Color.clear }
                .ignoresSafeArea()
                .blur(radius: 28)
                .overlay { Color.black.opacity(0.72).ignoresSafeArea() }

                VStack(spacing: 14) {
                    SceneThumb(studio: studio, name: p.scene.name) {
                        Rectangle().fill(Theme.Palette.surface)
                    }
                    .frame(height: 150)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                    VStack(spacing: 3) {
                        Text(p.failed == nil ? "Loading world" : "Switch failed")
                            .font(.system(size: 11, weight: .heavy)).tracking(1.2)
                            .foregroundStyle(Theme.Palette.textTertiary)
                        Text(p.scene.displayTitle)
                            .font(.system(size: 20, weight: .bold))
                            .foregroundStyle(Theme.Palette.textPrimary)
                    }

                    if let err = p.failed {
                        Text(err)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(Theme.Palette.warn)
                            .multilineTextAlignment(.center)
                        Button("Dismiss") { studio.dismissSwitch() }
                            .buttonStyle(.borderedProminent)
                            .tint(Theme.Palette.accent)
                    } else {
                        // Determinate bar with a soft ease (never over-claims).
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Theme.Palette.surfaceHi)
                                Capsule()
                                    .fill(p.ready ? AnyShapeStyle(Theme.Palette.live)
                                                  : AnyShapeStyle(Theme.accentGradient))
                                    .frame(width: max(8, geo.size.width * p.pct))
                                    .animation(.easeOut(duration: 0.6), value: p.pct)
                            }
                        }
                        .frame(height: 7)
                        HStack {
                            Text(p.stage)
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(p.ready ? Theme.Palette.live : Theme.Palette.textSecondary)
                                .contentTransition(.opacity)
                            Spacer()
                            Text("\(Int(p.pct * 100))%")
                                .font(.mabelMonoSmall)
                                .foregroundStyle(Theme.Palette.textTertiary)
                        }

                        VStack(alignment: .leading, spacing: 7) {
                            checkRow("Physics engine", on: p.bridge)
                            checkRow("Telemetry link", on: p.telemetry)
                            checkRow("Cameras", on: p.camera)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 2)
                    }
                }
                .padding(18)
                .frame(maxWidth: 380)
                .background {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(.ultraThinMaterial)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                }
                .padding(24)
            }
            .transition(.opacity)
        }
    }

    private func checkRow(_ label: String, on: Bool) -> some View {
        HStack(spacing: 8) {
            if on {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.live)
            } else {
                ProgressView().controlSize(.mini).tint(Theme.Palette.textTertiary)
                    .frame(width: 13)
            }
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(on ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
        }
    }
}

/// A scene thumbnail that shows the PRELOADED image instantly when SimStudio has
/// it (the common case — the cache is warmed the moment the scene list arrives),
/// and only falls back to an on-demand AsyncImage for a straggler still in flight.
/// `placeholder` is what shows while neither is available.
private struct SceneThumb<Placeholder: View>: View {
    @ObservedObject var studio: SimStudio
    let name: String
    @ViewBuilder var placeholder: () -> Placeholder

    var body: some View {
        if let img = studio.thumbnails[name] {
            Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
        } else {
            AsyncImage(url: studio.thumbURL(name)) { phase in
                if let img = phase.image {
                    img.resizable().aspectRatio(contentMode: .fill)
                } else {
                    placeholder()
                }
            }
        }
    }
}
