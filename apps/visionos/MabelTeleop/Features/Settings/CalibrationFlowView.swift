//
//  CalibrationFlowView.swift
//  MabelTeleop
//
//  The guided body-calibration experience: name your profile, hold three quick
//  poses, and review your measured proportions. Drives off the bridge's live
//  `calibrationStatus` telemetry (controller.calibration) and saves the result
//  on-device under the name you give. Designed to feel calm and Apple-clean —
//  one clear instruction at a time, generous space, gentle motion.
//

import SwiftUI

struct CalibrationFlowView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openImmersiveSpace) private var openImmersive

    /// Optional pre-filled name (re-calibrating an existing profile).
    var presetName: String = ""
    /// Overlay-presentation close hook (the Robot space presents this as a
    /// glass CARD, not a sheet — dismiss() is a no-op there).
    var onClose: (() -> Void)? = nil
    /// WHOLE-BODY flow (operator 2026-07-20): the space opens with mirroring
    /// DEFERRED, and this card owns the one engage point — capture (or Skip)
    /// → an unmissable 3·2·1 countdown → `engageWholeBody()`. The Settings
    /// sheet keeps the default (false): calibrating there never starts teleop.
    var engageOnFinish: Bool = false

    /// Close correctly for BOTH presentations. As a CARD overlay in the Robot
    /// window (the whole-body flow) `onClose` is set, and `@Environment(\.dismiss)`
    /// there dismisses the WINDOW — calling it closed the whole app the moment
    /// calibration finished ("the app just exits after calibration is done"). Only
    /// the Settings SHEET (onClose == nil) may use dismiss().
    private func close() {
        if let onClose { onClose() } else { dismiss() }
    }

    private enum Phase: Equatable {
        case intro, preparing, capturing, complete, failed(String)
        /// Post-capture (or post-skip) 3·2·1 before the robot mirrors.
        case countdown(Int)
    }
    @State private var phase: Phase = .intro
    @State private var name: String = ""
    /// The saved profile this card applied instead of measuring (countdown copy).
    @State private var usedSaved: CalibrationProfile? = nil
    /// Stall watchdog for the capture step (polled — see `CalibrationStall`).
    @State private var stall: CalibrationStall = .waiting

    /// The saved numbers "Use Saved Calibration" would apply right now.
    private var savedProfile: CalibrationProfile? {
        controller.preferredSavedProfile(forName: name)
    }

    var body: some View {
        // Compact card (2026-07-18): the SAME visual shell + scale as the
        // whole-body "are you clear" confirm card — 18 pt padding, 16 pt
        // header, 11-12 pt copy, content-hugging height. The old 480×660
        // hero layout read as a different app next to it.
        content
            .padding(18)
            .onAppear {
                if name.isEmpty { name = presetName }
                #if DEBUG
                // debugCountdown YES — jump straight to the 3·2·1 (the simulator
                // cannot complete a capture or tap Skip, so this is the only way
                // to exercise the countdown card + engage headlessly).
                if UserDefaults.standard.bool(forKey: "debugCountdown") {
                    phase = .countdown(3)
                }
                #endif
            }
            // Advance to the summary the moment the bridge reports the capture done.
            // Also accepted from .preparing: a fast capture can land before the
            // tracking-wait hop flips the phase, and onChange never re-fires for
            // a value that is already true — the card would stick forever.
            .onChange(of: controller.calibration?.done ?? false) { _, done in
                if done, phase == .capturing || phase == .preparing {
                    // Audible success (operator 2026-07-18): during the capture
                    // your eyes are on your extended arms / the robot, not the
                    // card — the chime says "captured" without looking.
                    Sounds.success()
                    Haptics.success()
                    withAnimation(Theme.Motion.smooth) {
                        // Whole-body flow: measured → count down → mirror.
                        phase = engageOnFinish ? .countdown(3) : .complete
                    }
                }
            }
            .animation(Theme.Motion.smooth, value: phase)
    }

    @ViewBuilder private var content: some View {
        switch phase {
        case .intro:            introStep
        case .preparing:        preparingStep
        case .capturing:        capturingStep
        case .complete:         completeStep
        case .failed(let why):  failedStep(why)
        case .countdown(let n): countdownStep(n)
        }
    }

    /// Open the immersive space if needed so ARKit hand tracking streams, then begin
    /// the capture — so calibration works straight from Settings, no Teleop tab first.
    ///
    /// Tracking failures are now SURFACED. This used to drop back to `.intro` with no
    /// explanation, so a denied Hand Tracking permission or a space that never opened
    /// read as "the button does nothing".
    private func beginCapture() async {
        if !controller.bodyTracking {
            withAnimation(Theme.Motion.smooth) { phase = .preparing }
            if !controller.immersiveActive {
                controller.fullImmersion = false
                // The space may already be open as the 1st-person cockpit's
                // stick host, which claims no ARKit session: PROMOTE it (a
                // second open on a live space fails, and `waitForTracking`
                // would then time out on a space that is right there).
                if controller.needsImmersiveOpen() {
                    _ = await openImmersive(id: WindowID.immersive.rawValue)
                }
            }
            if !(await controller.waitForTracking()) {
                withAnimation(Theme.Motion.smooth) {
                    phase = .failed(controller.trackingNotice
                        ?? "Hand tracking didn't start. Check Settings ▸ Privacy ▸ Hand Tracking for MABEL, then try again.")
                }
                return
            }
        }
        controller.startBodyCalibration(name: name)
        withAnimation(Theme.Motion.smooth) { phase = .capturing }
    }

    // MARK: - Intro: name + start

    private var introStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow("figure.arms.open", "Arm Extension Calibration")
            Text("One quick pose: extend both arms straight forward and MABEL scales its reach to yours. Skipping keeps the default reach; Use Saved re-applies your last measurement.")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 4) {
                Text("PROFILE NAME").font(.system(size: 9, weight: .heavy)).tracking(1.2)
                    .foregroundStyle(Theme.Palette.textTertiary)
                TextField("e.g. Operator", text: $name)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .textFieldStyle(.plain)
                    .submitLabel(.done)
            }
            .padding(12)
            .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }

            if !controller.canCalibrate { connectionHint }

            HStack(spacing: 10) {
                SecondaryButton(title: "Skip") { skipFlow() }
                PrimaryButton(title: "Begin", systemImage: "play.fill") {
                    Task { await beginCapture() }
                }
                .disabled(!controller.canCalibrate)
                .opacity(controller.canCalibrate ? 1 : 0.5)
            }
            useSavedButton
        }
    }

    // MARK: - Use saved calibration (operator 2026-10-12)

    /// "Use my saved numbers": skip the capture entirely. Shown whenever a
    /// profile has been saved on this headset, with the exact numbers + date it
    /// will apply. Same exit as Skip — in the whole-body flow it still runs the
    /// 3·2·1 countdown before mirroring.
    @ViewBuilder private var useSavedButton: some View {
        if let p = savedProfile {
            Button { useSaved(p) } label: {
                HStack(spacing: 10) {
                    Image(systemName: "person.crop.circle.badge.checkmark")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Theme.Palette.live)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Use Saved Calibration")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(Theme.Palette.textPrimary)
                        Text(p.savedLabel)
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
                .padding(.horizontal, 12).padding(.vertical, 9)
                .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Theme.Palette.live.opacity(0.35), lineWidth: 1) }
                .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .buttonStyle(.plain)
            .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 12, style: .continuous))
            .hoverEffect()
        }
    }

    private func useSaved(_ p: CalibrationProfile) {
        Haptics.select()
        controller.useSavedCalibration(p)
        usedSaved = p
        if engageOnFinish {
            withAnimation(Theme.Motion.smooth) { phase = .countdown(3) }
        } else {
            close()
        }
    }

    /// The way out of a capture the server never finishes (`CalibrationStall`).
    @ViewBuilder private var stallBanner: some View {
        if stall != .waiting {
            VStack(alignment: .leading, spacing: 8) {
                Label(stall == .serverSilent ? "The robot isn't answering calibration"
                                             : "Still not locked in",
                      systemImage: "clock.badge.exclamationmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(Theme.Palette.warn)
                Text(stall == .serverSilent
                     ? "No calibration status from the robot in \(Int(CalibrationStall.silentAfter)) s — this server may not support it. Use your saved numbers, or skip with the default reach."
                     : "The measurement hasn't settled. Keep holding, or use your saved numbers / skip.")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.75))
                    .fixedSize(horizontal: false, vertical: true)
                SecondaryButton(title: "Skip — default reach") {
                    controller.cancelBodyCalibration(); skipFlow()
                }
            }
            .transition(.opacity)
        }
    }

    /// Skip = keep the default reach. In the whole-body flow the COUNTDOWN
    /// still runs (skipping the measurement is not skipping the warning that
    /// the robot is about to mirror you); everywhere else it just closes.
    private func skipFlow() {
        if engageOnFinish {
            withAnimation(Theme.Motion.smooth) { phase = .countdown(3) }
        } else {
            close()
        }
    }

    /// Shown only when we're not connected — calibration runs on the robot bridge.
    /// (Hand tracking is started automatically when you tap Begin.)
    private var connectionHint: some View {
        HStack(spacing: 10) {
            Image(systemName: "antenna.radiowaves.left.and.right")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.Palette.warn)
            Text("Connect to MABEL first — pick your robot in the Robots list above.")
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(Theme.Space.md)
        .background(Theme.Palette.warn.opacity(0.10), in: RoundedRectangle(cornerRadius: Theme.Radius.md))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md).strokeBorder(Theme.Palette.warn.opacity(0.3), lineWidth: 1) }
    }

    /// Brief bridge between tapping Begin and the first pose while the immersive
    /// space opens and hand tracking comes up.
    private var preparingStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow("figure.arms.open", "Getting ready…")
            HStack(spacing: 10) {
                ProgressView().tint(Theme.Palette.accent)
                Text("Bring your hands up where you can see them so tracking can start.")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.85))
                    .fixedSize(horizontal: false, vertical: true)
            }
            useSavedButton
            SecondaryButton(title: "Cancel") {
                controller.cancelBodyCalibration(); close()
            }
        }
    }

    // MARK: - Capturing: the single arms-forward pose

    /// What the operator's body is doing RIGHT NOW. The local coach echoes the
    /// rule; when the bridge streams its OWN live measurement (additive
    /// `calibrationStatus.reachLive/hold`, 2026-07-19) that wins — the ring then
    /// can never fill ahead of the capture the server is actually timing (the
    /// old "ring full but never finishes" failure).
    private var live: CalibrationLive {
        var l = controller.calibrationLive
        if let cal = controller.calibration, cal.active {
            if let r = cal.reachLive, r > 0 { l.reach = r }
            if let h = cal.hold { l.hold = h }
        }
        return l
    }

    /// The one thing to fix, in priority order — never more than one instruction on
    /// screen. Mirrors exactly what the server needs to take the measurement.
    private var coach: (icon: String, title: String, detail: String, tint: Color) {
        if !live.hasHead {
            return ("exclamationmark.triangle.fill", "Head tracking lost",
                    "Look forward so the headset can locate you.", Theme.Palette.warn)
        }
        if live.hands == 0 {
            return ("hand.raised.slash.fill", "Bring your hands into view",
                    "The headset can't see your hands — hold them up in front of you.",
                    Theme.Palette.warn)
        }
        if live.hands == 1 {
            return ("hand.raised.fill", "Show both hands",
                    "Only one hand is tracked. Both arms are measured together.",
                    Theme.Palette.warn)
        }
        if live.steady {
            return ("dot.radiowaves.left.and.right", "Hold it right there",
                    "Measuring your reach…", Theme.Palette.live)
        }
        // The actual success criterion, in words: the arm-length number just
        // has to sit still. (Server rule: std ≤ 2 cm over ≥ 0.5 s, + 0.25 s.)
        return ("figure.arms.open", "Reach both arms straight out",
                "When the number stops moving for about a second, it locks in.",
                Theme.Palette.accent)
    }

    private var capturingStep: some View {
        let c = coach
        let ready = live.hands == 2 && live.hasHead
        return VStack(alignment: .leading, spacing: 12) {
            headerRow("figure.arms.open", "Extend Both Arms Forward")

            HStack(spacing: 14) {
                // LIVE progress ring: fills through the hold the server is
                // timing, so the operator SEES the capture happening.
                ZStack {
                    Circle()
                        .strokeBorder(Theme.Palette.surfaceHi, lineWidth: 6)
                        .frame(width: 108, height: 108)
                    Circle()
                        .trim(from: 0, to: live.hold)
                        .stroke(Theme.Palette.live,
                                style: StrokeStyle(lineWidth: 6, lineCap: .round))
                        .frame(width: 108, height: 108)
                        .rotationEffect(.degrees(-90))
                        .animation(.linear(duration: 0.06), value: live.hold)
                    VStack(spacing: 1) {
                        // The measured quantity, live — head→wrist, exactly
                        // what the server is averaging.
                        Text(ready ? String(format: "%.0f cm", live.reach * 100) : "—")
                            .font(.system(size: 20, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(Theme.Palette.textPrimary)
                            .contentTransition(.numericText())
                            .animation(Theme.Motion.snappy, value: Int(live.reach * 100))
                        Text("REACH")
                            .font(.system(size: 8, weight: .heavy)).tracking(1.2)
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    Label(c.title, systemImage: c.icon)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(c.tint)
                        .symbolEffect(.pulse, options: .repeating, isActive: live.steady)
                    Text(c.detail)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .fixedSize(horizontal: false, vertical: true)
                    // Which hands the headset actually has — the #1 reason a
                    // capture stalls.
                    HStack(spacing: 8) {
                        handChip("L", tracked: controller.leftHandTracked)
                        handChip("R", tracked: controller.rightHandTracked)
                    }
                }
                .animation(Theme.Motion.smooth, value: c.title)
            }

            stallBanner
            if stall != .waiting { useSavedButton }

            SecondaryButton(title: "Cancel", systemImage: "xmark") {
                controller.cancelBodyCalibration()
                close()
            }
        }
        .animation(Theme.Motion.smooth, value: stall)
        // Stall watchdog: re-evaluated twice a second while capturing. Ends
        // with the step (task auto-cancels when the phase moves on).
        .task {
            stall = .waiting
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                if Task.isCancelled { break }
                let v = controller.calibrationStall()
                if v != stall {
                    if v != .waiting { NSLog("[MABEL-CAL] stall verdict %@", "\(v)") }
                    stall = v
                }
            }
        }
    }

    private func handChip(_ label: String, tracked: Bool) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(tracked ? Theme.Palette.live : Theme.Palette.textTertiary)
                .frame(width: 6, height: 6)
            Text(tracked ? "\(label)" : "\(label) not seen")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(tracked ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
        }
        .padding(.horizontal, 9).frame(height: 24)
        .background(Theme.Palette.surface, in: Capsule())
        .overlay { Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        .animation(Theme.Motion.snappy, value: tracked)
    }

    // MARK: - Failed: tracking never started

    private func failedStep(_ why: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow("hand.raised.slash.fill", "Couldn't start hand tracking",
                      tint: Theme.Palette.warn)
            Text(why)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) {
                SecondaryButton(title: "Skip") { skipFlow() }
                PrimaryButton(title: "Try Again", systemImage: "arrow.clockwise") {
                    Task { await beginCapture() }
                }
            }
            useSavedButton
        }
    }

    // MARK: - Countdown → engage (whole-body flow only)

    /// Capture accepted (or skipped) — the robot mirrors only after an
    /// unmissable 3·2·1, so it never springs to the operator's pose the
    /// instant the measurement lands (operator 2026-07-20: "not right away").
    private func countdownStep(_ n: Int) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow("figure.arms.open", "Tracking Starts In…",
                      tint: Theme.Palette.live)
            HStack(spacing: 14) {
                Text("\(n)")
                    .font(.system(size: 54, weight: .heavy, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(Theme.Palette.live)
                    .contentTransition(.numericText(countsDown: true))
                    .frame(minWidth: 60)
                VStack(alignment: .leading, spacing: 6) {
                    if let p = usedSaved {
                        Text(String(format: "Saved calibration ×%.2f (%@) requested.", p.reach, p.name))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.85))
                    } else if let r = controller.calibration?.reach {
                        Text(String(format: "Reach scale ×%.2f applied.", r))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.85))
                    }
                    Text("Stand ready — the robot mirrors your head, arms and fingers the moment this hits zero.")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            SecondaryButton(title: "Cancel", systemImage: "xmark") { close() }
        }
        // One task for the WHOLE countdown: the phase's associated value changes
        // 3→2→1 but the case (and view identity) stays, so this fires once.
        .task { await runCountdown() }
    }

    private func runCountdown() async {
        for n in stride(from: 3, through: 1, by: -1) {
            withAnimation(Theme.Motion.snappy) { phase = .countdown(n) }
            NSLog("[MABEL-CAL] countdown %d", n)
            Sounds.tick()
            Haptics.soft()
            try? await Task.sleep(for: .seconds(1))
            // Cancel closed the card (task auto-cancels) or moved the phase on.
            if Task.isCancelled { return }
            guard case .countdown = phase else { return }
        }
        NSLog("[MABEL-CAL] countdown done — engaging whole-body tracking")
        controller.engageWholeBody()
        Haptics.success()
        close()
    }

    // MARK: - Complete: the measured summary

    private var completeStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow("checkmark.circle.fill", "Calibration Complete",
                      tint: Theme.Palette.live)
            Text("MABEL now matches your reach\(name.isEmpty ? "" : ", \(name)"). This profile is saved.")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
                .fixedSize(horizontal: false, vertical: true)

            if let m = controller.liveMetrics {
                let profile = CalibrationProfile(name: name.isEmpty ? m.user : name, metrics: m)
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(profile.detailRows, id: \.0) { row in
                        metricRow(row.0, row.1)
                    }
                }
                .padding(12)
                .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            }

            PrimaryButton(title: "Done", systemImage: "checkmark") { close() }
        }
    }

    // MARK: - Shared chrome

    /// Confirm-card-style header: icon + 16 pt title, close X on the trailing
    /// edge. One row replaces the old separate 44 pt close row + hero icon +
    /// large title (the popup now matches wholeBodyConfirmCard's scale). The X
    /// keeps a real gaze target and a matching hover shape (gotcha #10).
    private func headerRow(_ icon: String, _ title: String,
                           tint: Color = Theme.Palette.warn) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(tint)
            Text(title)
                .font(.system(size: 16, weight: .bold)).foregroundStyle(.white)
                .lineLimit(1).minimumScaleFactor(0.8)
            Spacer(minLength: 6)
            Button { close() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .frame(width: 30, height: 30)
                    .background(Theme.Palette.surfaceHi, in: Circle())
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .contentShape(.hoverEffect, Circle())
            .hoverEffect()
        }
    }

    private func metricRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.75))
            Spacer()
            Text(value).font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.Palette.textPrimary)
        }
    }
}

// MARK: - "Who's driving?" profile picker (shown when starting teleop)

/// Presented when the operator taps Start Teleop without an active profile: pick a
/// saved one, calibrate a new one, or skip. Keeps MABEL matched to whoever's driving.
struct ProfilePickerView: View {
    @EnvironmentObject private var controller: RobotController

    var onSelect: (CalibrationProfile) -> Void
    var onNew: () -> Void
    var onSkip: () -> Void

    var body: some View {
        VStack(spacing: Theme.Space.lg) {
            Spacer(minLength: 0)

            Image(systemName: "person.crop.circle.badge.checkmark")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(Theme.accentGradient)

            VStack(spacing: Theme.Space.xs) {
                Text("Who's driving?").font(.mabelLargeTitle).foregroundStyle(Theme.Palette.textPrimary)
                Text("Pick your calibration profile so MABEL reaches and grips exactly like you.")
                    .font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
                    .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
            }

            if !controller.calibrationProfiles.isEmpty {
                GlassCard {
                    VStack(spacing: Theme.Space.xs) {
                        ForEach(controller.calibrationProfiles) { p in
                            Button { onSelect(p) } label: {
                                HStack(spacing: 11) {
                                    Image(systemName: "figure.arms.open")
                                        .font(.system(size: 16, weight: .semibold))
                                        .foregroundStyle(Theme.Palette.accent).frame(width: 26)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text(p.name).font(.system(size: 15, weight: .semibold))
                                            .foregroundStyle(Theme.Palette.textPrimary)
                                        Text(p.summary).font(.system(size: 11, design: .monospaced))
                                            .foregroundStyle(Theme.Palette.textTertiary)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right").font(.system(size: 12, weight: .semibold))
                                        .foregroundStyle(Theme.Palette.textTertiary)
                                }
                                .padding(.vertical, 9).padding(.horizontal, 6)
                            }
                            .buttonStyle(.plain).hoverEffect()
                        }
                    }
                }
            }

            VStack(spacing: Theme.Space.sm) {
                PrimaryButton(title: "New Calibration", systemImage: "plus") { onNew() }
                SecondaryButton(title: controller.calibrationProfiles.isEmpty ? "Skip — use default" : "Skip for now") { onSkip() }
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: 540)
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.regularMaterial)
    }
}

// MARK: - Saved-profile detail

struct CalibrationProfileDetailView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.dismiss) private var dismiss

    let profile: CalibrationProfile
    /// Ask the parent to re-run calibration for this name.
    var onRecalibrate: (String) -> Void

    var body: some View {
        VStack(spacing: Theme.Space.lg) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(profile.name).font(.mabelViewTitle).foregroundStyle(Theme.Palette.textPrimary)
                    Text("Calibrated \(profile.createdAt.formatted(date: .abbreviated, time: .shortened))")
                        .font(.mabelCaption).foregroundStyle(Theme.Palette.textTertiary)
                }
                Spacer()
                if controller.isActiveProfile(profile) {
                    StatusPill(title: "Active", color: Theme.Palette.live)
                }
                Button { dismiss() } label: {
                    Image(systemName: "xmark").font(.system(size: 14, weight: .bold))
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .padding(10).background(Theme.Palette.surfaceHi, in: Circle())
                }
                .buttonStyle(.plain).hoverEffect()
            }

            GlassCard {
                VStack(alignment: .leading, spacing: Theme.Space.sm) {
                    Text("MEASUREMENTS").font(.mabelCaption)
                        .foregroundStyle(Theme.Palette.textTertiary).tracking(1.4)
                    ForEach(profile.detailRows, id: \.0) { row in
                        HStack {
                            Text(row.0).font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
                            Spacer()
                            Text(row.1).font(.mabelMono).foregroundStyle(Theme.Palette.textPrimary)
                        }
                    }
                }
            }

            VStack(spacing: Theme.Space.sm) {
                PrimaryButton(title: controller.isActiveProfile(profile) ? "Re‑apply to Robot" : "Apply to Robot",
                              systemImage: "checkmark.circle") {
                    controller.applyProfile(profile)
                    dismiss()
                }
                .disabled(!controller.connection.isLive)
                .opacity(controller.connection.isLive ? 1 : 0.5)

                SecondaryButton(title: "Re‑calibrate", systemImage: "arrow.clockwise") {
                    dismiss(); onRecalibrate(profile.name)
                }
                Button(role: .destructive) {
                    controller.deleteProfile(profile); dismiss()
                } label: {
                    Label("Delete Profile", systemImage: "trash")
                        .font(.mabelHeadline).foregroundStyle(Theme.Palette.danger)
                        .frame(maxWidth: .infinity).padding(.vertical, 12)
                }
                .buttonStyle(.plain).hoverEffect()
            }
            Spacer(minLength: 0)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: 540, maxHeight: .infinity, alignment: .top)
        .frame(maxWidth: .infinity)
        .background(.regularMaterial)
    }
}
