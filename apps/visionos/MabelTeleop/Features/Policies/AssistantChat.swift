//
//  AssistantChat.swift
//  MabelTeleop
//
//  Shared chat pieces for the Policies assistant, so the compact card and the
//  full-conversation pop-up render identically — visionOS console styling, with
//  a glass icon button that matches the Settings `secButton` (tinted fill +
//  hairline border) so every control reads as a button.
//

import SwiftUI
import Combine

/// Icon-only button in the Settings `secButton` language: a tinted circle with a
/// hairline border (or a filled accent circle when `filled`).
struct GlassIconButton: View {
    let systemImage: String
    var tint: Color = Theme.Palette.accent
    var filled: Bool = false
    var action: () -> Void

    var body: some View {
        Button { Haptics.tap(); action() } label: {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(filled ? .white : tint)
                .frame(width: 34, height: 34)
                .background(filled ? AnyShapeStyle(tint.gradient) : AnyShapeStyle(tint.opacity(0.14)), in: Circle())
                .overlay(Circle().strokeBorder(tint.opacity(filled ? 0 : 0.35), lineWidth: 1))
                .contentShape(Circle())
        }
        .buttonStyle(PressableStyle())
        .hoverEffect()
    }
}

/// One chat bubble. Robot replies sit left on frosted glass; the user's turns
/// sit right on the brand accent.
struct ChatBubble: View {
    let message: RobotHub.Message
    var lineLimit: Int? = nil

    var body: some View {
        let isRobot = message.role == .robot
        HStack {
            if !isRobot { Spacer(minLength: 40) }
            Text(message.text)
                .font(.system(size: 14))
                .foregroundStyle(isRobot ? Theme.Palette.textPrimary : .white)
                .lineLimit(lineLimit)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(isRobot ? AnyShapeStyle(Theme.Palette.surfaceHi) : AnyShapeStyle(Theme.accentGradient),
                            in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .frame(maxWidth: .infinity, alignment: isRobot ? .leading : .trailing)
            if isRobot { Spacer(minLength: 40) }
        }
    }
}

/// Shown while the assistant is composing an LLM reply.
struct TypingBubble: View {
    var body: some View {
        HStack {
            HStack(spacing: 5) {
                ProgressView().controlSize(.mini)
                Text("MABEL is thinking…").font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(Theme.Palette.surfaceHi, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            Spacer(minLength: 40)
        }
    }
}

/// The full conversation, auto-scrolled to the newest line. Fills its container.
struct ChatTranscript: View {
    @ObservedObject var hub: RobotHub

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(hub.messages) { ChatBubble(message: $0) }
                    if hub.thinking { TypingBubble() }
                    Color.clear.frame(height: 1).id("bottom")
                }
            }
            .onChange(of: hub.messages) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
            .onChange(of: hub.thinking) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
        }
    }
}

/// Things you can ask MABEL — a rotating set of tappable prompts so the operator
/// always has fresh ideas. Mixes physical commands with friendly home questions.
let assistantSuggestions: [(text: String, icon: String)] = [
    ("Relax your body", "figure.cooldown"),
    ("Go charge", "bolt.fill"),
    ("What's your battery?", "battery.75"),
    ("Turn on privacy mode", "eye.slash.fill"),
    ("Turn off the cameras", "video.slash.fill"),
    ("Check on the dog", "pawprint.fill"),
    ("Do the laundry", "washer.fill"),
    ("What's the weather?", "cloud.sun.fill"),
    ("Where are my keys?", "key.fill"),
    ("What did you do yesterday?", "clock.arrow.circlepath"),
    ("How many cokes are in the fridge?", "refrigerator.fill"),
    ("Tidy up the living room", "sparkles"),
]

/// A horizontally-scrolling strip of suggestion chips that rotates every few
/// seconds. Tapping one sends it to the assistant.
struct SuggestionBar: View {
    @ObservedObject var hub: RobotHub
    @State private var start = 0
    private let window = 4
    private let timer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()

    private var visible: [(text: String, icon: String)] {
        (0..<window).map { assistantSuggestions[(start + $0) % assistantSuggestions.count] }
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(visible, id: \.text) { s in
                    Button { hub.send(s.text); Haptics.tap() } label: {
                        HStack(spacing: 5) {
                            Image(systemName: s.icon).font(.system(size: 10, weight: .semibold))
                            Text(s.text).font(.system(size: 11, weight: .medium)).lineLimit(1)
                        }
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .padding(.horizontal, 10).padding(.vertical, 7)
                        .background(Theme.Palette.surfaceHi, in: Capsule())
                        .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
                    }
                    .buttonStyle(PressableStyle())
                    .hoverEffect()
                }
            }
            .padding(.vertical, 1)
        }
        .id(start)
        .onReceive(timer) { _ in
            withAnimation(.easeInOut(duration: 0.35)) { start = (start + window) % assistantSuggestions.count }
        }
    }
}

/// Text field + mic / send, shared by the card and the sheet.
struct ChatInputBar: View {
    @ObservedObject var hub: RobotHub

    var body: some View {
        HStack(spacing: 8) {
            TextField(hub.listening ? "Listening…" : "Message MABEL", text: $hub.draft)
                .textFieldStyle(.plain)
                .font(.system(size: 14)).foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, 14).padding(.vertical, 11)
                .background(Theme.Palette.surfaceHi, in: Capsule())
                .overlay(Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1))
                .onSubmit { hub.send() }
            if hub.draft.trimmingCharacters(in: .whitespaces).isEmpty && !hub.listening {
                Button { hub.toggleListening() } label: {
                    Image(systemName: hub.listening ? "mic.fill" : "mic.circle.fill")
                        .font(.system(size: 30))
                        .foregroundStyle(hub.listening ? Theme.Palette.danger : Theme.Palette.accent)
                }
                .buttonStyle(PressableStyle()).disabled(!hub.voiceAvailable).opacity(hub.voiceAvailable ? 1 : 0.4)
            } else {
                Button { hub.send() } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 30))
                        .foregroundStyle(Theme.Palette.accent)
                }.buttonStyle(PressableStyle())
            }
        }
    }
}

/// The big assistant pop-up — the whole conversation with the input pinned at the
/// bottom. Opened by tapping the compact card on the Policies page.
struct AssistantChatSheet: View {
    @ObservedObject var hub: RobotHub
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 14) {
            HStack {
                Text("Assistant").font(.mabelTitle).foregroundStyle(Theme.Palette.textPrimary)
                Spacer()
                GlassIconButton(systemImage: hub.voiceReply ? "speaker.wave.2.fill" : "speaker.slash.fill",
                                tint: hub.voiceReply ? Theme.Palette.accent : Theme.Palette.textTertiary,
                                filled: hub.voiceReply) { hub.voiceReply.toggle() }
                Button { dismiss() } label: {
                    Image(systemName: "xmark").font(.system(size: 13, weight: .bold))
                        .foregroundStyle(Theme.Palette.textTertiary).padding(10).background(Theme.Palette.surface, in: Circle())
                }.buttonStyle(.plain)
            }
            ChatTranscript(hub: hub)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            ChatInputBar(hub: hub)
        }
        .padding(28)
        .frame(width: 620, height: 720)
    }
}
