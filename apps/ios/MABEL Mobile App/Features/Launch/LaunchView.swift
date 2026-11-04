//
//  LaunchView.swift
//  MABEL Mobile App
//
//  The welcome / loading screen: a friendly little MABEL tidying up — a broom
//  sweeping, a few sparkles vanishing as it cleans, a sleek visor scan, and
//  the wordmark fading in. Pure SwiftUI shapes, Apple-smooth easing.
//

import SwiftUI

struct LaunchView: View {
    @State private var sweep = false
    @State private var bob = false
    @State private var scan = false
    @State private var sparkle = false
    @State private var wordIn = false

    private let cream = Color(hex: 0xFCFAF5)
    private let ink = Color(hex: 0x1A1410)
    private let accent = Color(hex: 0xF26B3A)

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(hex: 0xFFA632), Color(hex: 0xF55B3C)],
                           startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            RadialGradient(colors: [Color(hex: 0xFFE7B0).opacity(0.5), .clear],
                           center: .init(x: 0.4, y: 0.32), startRadius: 4, endRadius: 360)
                .ignoresSafeArea()

            VStack(spacing: 34) {
                robot.offset(y: bob ? -6 : 0)
                wordmark
            }
        }
        .onAppear { start() }
    }

    // MARK: - Robot

    private var robot: some View {
        ZStack {
            // ground shadow
            Ellipse().fill(.black.opacity(0.14)).frame(width: 130, height: 18)
                .blur(radius: 4).offset(y: 78)

            // sparkles being swept up
            sparkles

            bodyView

            // sweeping arm + broom, hinged at the shoulder
            armBroom
                .rotationEffect(.degrees(sweep ? 14 : -26), anchor: .topLeading)
                .offset(x: 30, y: 2)
        }
        .frame(width: 210, height: 180)
    }

    private var bodyView: some View {
        VStack(spacing: 0) {
            // antenna
            Circle().fill(cream).frame(width: 12, height: 12)
                .overlay(Circle().stroke(accent, lineWidth: 3))
                .offset(y: 4)
            Rectangle().fill(cream).frame(width: 4, height: 12)
            // head/body
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(cream)
                .frame(width: 96, height: 92)
                .overlay {
                    // visor with a scanning light
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(ink)
                        .frame(width: 70, height: 44)
                        .overlay {
                            Capsule().fill(accent)
                                .frame(width: 20, height: 8)
                                .offset(x: scan ? 18 : -18)
                                .shadow(color: accent.opacity(0.8), radius: 6)
                        }
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .offset(y: -4)
                }
                .shadow(color: .black.opacity(0.18), radius: 14, y: 8)
        }
        .offset(x: -18)
    }

    private var armBroom: some View {
        VStack(spacing: 0) {
            // arm
            Capsule().fill(cream).frame(width: 10, height: 52)
            // broom handle
            Capsule().fill(Color(hex: 0xC98A52)).frame(width: 7, height: 46)
            // bristles
            UnevenRoundedRectangle(topLeadingRadius: 4, bottomLeadingRadius: 10,
                                   bottomTrailingRadius: 10, topTrailingRadius: 4, style: .continuous)
                .fill(Color(hex: 0xF2C14E))
                .frame(width: 34, height: 26)
                .overlay(alignment: .bottom) {
                    HStack(spacing: 2) {
                        ForEach(0..<6, id: \.self) { _ in
                            Capsule().fill(Color(hex: 0xD89B2E)).frame(width: 2, height: 8)
                        }
                    }.offset(y: 4)
                }
        }
    }

    private var sparkles: some View {
        ZStack {
            ForEach(Array(sparkleSpots.enumerated()), id: \.offset) { i, p in
                Image(systemName: "sparkle")
                    .font(.system(size: [10.0, 14, 9, 12][i % 4], weight: .bold))
                    .foregroundStyle(cream)
                    .opacity(sparkle ? 0 : 0.95)
                    .scaleEffect(sparkle ? 0.4 : 1)
                    .offset(x: p.x, y: p.y)
                    .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)
                        .delay(Double(i) * 0.18), value: sparkle)
            }
        }
    }
    private let sparkleSpots: [CGPoint] = [
        .init(x: 56, y: 46), .init(x: 80, y: 30), .init(x: 70, y: 60), .init(x: 92, y: 52)
    ]

    // MARK: - Wordmark

    private var wordmark: some View {
        VStack(spacing: 7) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("MABEL")
                    .font(.system(size: 34, weight: .heavy, design: .rounded))
                    .tracking(6)
                    .foregroundStyle(cream)
                // V2 chip — mirrors the app-icon badge (cream chip, orange mark).
                Text("V2")
                    .font(.system(size: 15, weight: .heavy, design: .rounded))
                    .foregroundStyle(Color(hex: 0xE2582A))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(cream, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            Text("TELEOPERATION CONSOLE")
                .font(.system(size: 10, weight: .semibold))
                .tracking(3)
                .foregroundStyle(cream.opacity(0.8))
            // pulsing dots
            HStack(spacing: 6) {
                ForEach(0..<3, id: \.self) { i in
                    Circle().fill(cream.opacity(0.9)).frame(width: 6, height: 6)
                        .scaleEffect(sparkle ? 1 : 0.5)
                        .animation(.easeInOut(duration: 0.6).repeatForever().delay(Double(i) * 0.18), value: sparkle)
                }
            }
            .padding(.top, 6)
        }
        .opacity(wordIn ? 1 : 0)
        .offset(y: wordIn ? 0 : 10)
    }

    // MARK: - Driver

    private func start() {
        withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) { sweep = true }
        withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) { bob = true }
        withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { scan = true }
        withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { sparkle = true }
        withAnimation(.spring(response: 0.7, dampingFraction: 0.8).delay(0.3)) { wordIn = true }
    }
}
