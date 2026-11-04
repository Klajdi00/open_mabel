//
//  AssistantChat.swift
//  MABEL Mobile App
//
//  Shared chat pieces for the Policies assistant, so the compact card on the
//  Policies page and the full-screen conversation sheet render identically:
//
//    • ChatBubble       — one message (robot leading / user trailing).
//    • TypingBubble     — "MABEL is thinking…" while the LLM composes.
//    • ChatTranscript   — scrolling, auto-pinned conversation.
//    • ChatInputBar     — glass text field + mic / send.
//    • AssistantChatSheet — the big pop-up opened by tapping the card.
//

import SwiftUI
import Combine

/// One chat bubble. Robot replies sit left on frosted glass; the user's turns
/// sit right on the brand accent.
struct ChatBubble: View {
    let message: RobotHub.Message
    var lineLimit: Int? = nil

    var body: some View {
        let isRobot = message.role == .robot
        HStack {
            if !isRobot { Spacer(minLength: 36) }
            Text(message.text)
                .font(.system(size: 14))
                .foregroundStyle(isRobot ? Theme.Palette.textPrimary : .white)
                .lineLimit(lineLimit)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(isRobot ? AnyShapeStyle(Theme.Palette.surfaceHi)
                                    : AnyShapeStyle(Theme.accentGradient),
                            in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .frame(maxWidth: .infinity, alignment: isRobot ? .leading : .trailing)
                .multilineTextAlignment(.leading)
            if isRobot { Spacer(minLength: 36) }
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
            Spacer(minLength: 36)
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
                .padding(.vertical, 2)
            }
            .onChange(of: hub.messages) { _, _ in
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: hub.thinking) { _, _ in
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }
}

/// Glass mic button — pulses while listening, dimmed when speech is unavailable.
struct MicButton: View {
    @ObservedObject var hub: RobotHub

    var body: some View {
        Button { hub.toggleListening() } label: {
            Image(systemName: hub.listening ? "mic.fill" : "mic.circle.fill")
                .font(.system(size: 30))
                .foregroundStyle(hub.listening ? Theme.Palette.danger : Theme.Palette.accent)
                .symbolEffect(.pulse, isActive: hub.listening)
        }
        .buttonStyle(PressableStyle())
        .disabled(!hub.voiceAvailable)
        .opacity(hub.voiceAvailable ? 1 : 0.4)
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
                }
            }
            .padding(.vertical, 1)
        }
        .transition(.opacity)
        .id(start)   // fade as the window rotates
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
            GlassField {
                TextField(hub.listening ? "Listening…" : "Message MABEL", text: $hub.draft)
                    .font(.system(size: 14))
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .onSubmit { hub.send() }
                    .frame(maxWidth: .infinity)
            }
            if hub.draft.trimmingCharacters(in: .whitespaces).isEmpty && !hub.listening {
                MicButton(hub: hub)
            } else {
                Button { hub.send() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                        .foregroundStyle(Theme.Palette.accent)
                }
                .buttonStyle(PressableStyle())
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
        NavigationStack {
            VStack(spacing: 12) {
                ChatTranscript(hub: hub)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                ChatInputBar(hub: hub)
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.sm)
            .padding(.bottom, Theme.Space.md)
            .background(Theme.Palette.canvas.ignoresSafeArea())
            .navigationTitle("Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    GlassIconButton(systemImage: hub.voiceReply ? "speaker.wave.2.fill" : "speaker.slash.fill",
                                    tint: hub.voiceReply ? Theme.Palette.accent : Theme.Palette.textSecondary,
                                    prominent: hub.voiceReply) { hub.voiceReply.toggle() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
