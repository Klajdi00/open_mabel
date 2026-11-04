//
//  RobotHub.swift
//  MABEL Mobile App
//
//  The brain behind the Policies tab's control surface: the robot's high-level
//  modes (dock/charge, go home, sleep, security patrol, speaker), and a
//  conversational assistant you can type to or talk to. Speech is transcribed
//  on-device (Speech framework) and the robot replies in the chat and aloud
//  (AVSpeechSynthesizer). High-level intents are matched to the same control
//  actions, so "go home" or "patrol the house" do the right thing.
//
//  The dedicated controls map to robot commands where one exists (e.g. Go Home /
//  Dock send a nav goal); the rest hold app-side mode state and give feedback —
//  the bridge wires these to the planner server-side.
//

import SwiftUI
import Combine
import Speech
import AVFoundation

@MainActor
final class RobotHub: NSObject, ObservableObject {

    // MARK: Chat
    struct Message: Identifiable, Equatable {
        let id = UUID()
        enum Role { case user, robot }
        let role: Role
        var text: String
    }

    @Published var messages: [Message] = [
        .init(role: .robot,
              text: "Hi, I'm MABEL. Tell me to go home, dock & charge, sleep, patrol the house, or run a task — by typing or holding the mic.")
    ]
    @Published var draft: String = ""
    @Published var listening = false
    @Published var voiceReply = true          // robot speaks its replies aloud
    @Published var thinking = false           // LLM is composing a reply

    /// True when a usable Anthropic key is configured (user key or owner bypass).
    var llmReady: Bool { SecureStore.hasUsableAnthropicKey }

    private static let systemPrompt = """
    You are MABEL, a friendly home robot assistant speaking through the user's phone. \
    You can chat, answer questions, and help the user plan and run home tasks. The user \
    can also command you with on-screen buttons to dock & charge, go home, sleep, enter \
    security / patrol mode, or run learned skills. Keep replies concise, warm, and \
    conversational — usually one to three sentences, suitable to be read aloud. If asked \
    to perform a physical action, acknowledge it briefly and naturally.
    """

    // MARK: High-level modes
    @Published var sleeping = false
    @Published var security = false           // patrol / monitor the home
    @Published var speaker = false            // remote voice → robot's speaker
    @Published var busyAction: String? = nil  // transient "Docking…" / "Going home…"

    weak var controller: RobotController?

    // MARK: Speech
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var recTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synth = AVSpeechSynthesizer()

    /// Voice dictation is only offered when the device actually supports it.
    var voiceAvailable: Bool { recognizer?.isAvailable ?? false }

    // MARK: - Chat

    func send(_ raw: String? = nil) {
        let text = (raw ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(.init(role: .user, text: text))
        draft = ""
        Haptics.soft()
        // Direct control commands act immediately with a deterministic reply.
        if let ack = controlIntent(text) { respond(ack); return }
        // Everything else goes to the LLM when configured, else a local fallback.
        if llmReady { replyViaLLM() } else { respond(localReply(text)) }
    }

    private func respond(_ text: String) {
        messages.append(.init(role: .robot, text: text))
        if voiceReply { speak(text) }
    }

    /// Tight matcher for the *physical* commands — these actuate the robot, so only
    /// explicit phrases match; casual mentions fall through to the LLM.
    private func controlIntent(_ text: String) -> String? {
        let t = text.lowercased()
        switch true {
        case t.contains("dock"), t.contains("go charge"), t.contains("charge yourself"):
            dock();            return "Returning to the dock to charge."
        case t.contains("go home"), t.contains("come home"), t.contains("come back"), t.contains("return home"):
            goHome();          return "On my way to the home position."
        case t.contains("wake up"), t == "wake":
            setSleep(false);   return "I'm awake and ready."
        case t.contains("go to sleep"), t.contains("sleep mode"):
            setSleep(true);    return "Going to sleep — wake me whenever you need me."
        case t.contains("patrol"), t.contains("security mode"), t.contains("monitor the house"):
            setSecurity(true); return "Security mode on. I'll patrol and keep an eye on the home."
        default:
            return nil
        }
    }

    /// Used only when no Anthropic key is configured — points the user at Settings.
    private func localReply(_ text: String) -> String {
        let t = text.lowercased()
        if t.contains("hello") || t.hasPrefix("hi") || t.contains("hey") {
            return "Hey! I can go home, dock & charge, sleep, or patrol. Add your Anthropic API key in Settings → Assistant and I can chat about anything."
        }
        return "I can go home, dock & charge, sleep, or patrol right now. To chat about anything else, add your Anthropic API key in Settings → Assistant."
    }

    /// Ask Claude for a reply, sending the whole conversation as context.
    private func replyViaLLM() {
        let client = ClaudeClient(apiKey: SecureStore.effectiveAnthropicKey,
                                  model: ClaudeClient.Model.current.rawValue)
        let turns = messages.map {
            ClaudeClient.Turn(role: $0.role == .user ? "user" : "assistant", text: $0.text)
        }
        withAnimation(Theme.Motion.gentle) { thinking = true }
        Task { [weak self] in
            do {
                let reply = try await client.send(system: Self.systemPrompt, turns: turns)
                await MainActor.run {
                    guard let self else { return }
                    self.thinking = false
                    self.respond(reply)
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.thinking = false
                    self.respond("I couldn't reach my language model — \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - High-level controls

    func goHome() {
        flash("Going home…")
        controller?.sendNavGoal(x: 0, y: 0, yaw: 0)
        if sleeping { setSleep(false) }
    }

    func dock() {
        flash("Docking…")
        // Dock is the charging pose at the map origin for now; the server maps the
        // dock fiducial in a real deployment.
        controller?.sendNavGoal(x: 0, y: 0, yaw: 0)
        if sleeping { setSleep(false) }
    }

    func setSleep(_ on: Bool) {
        withAnimation(Theme.Motion.snappy) {
            sleeping = on
            if on { security = false; speaker = false }
        }
        Haptics.tap()
    }
    func toggleSleep() { setSleep(!sleeping) }

    func setSecurity(_ on: Bool) {
        withAnimation(Theme.Motion.snappy) {
            security = on
            if on { sleeping = false }
        }
        Haptics.tap()
    }
    func toggleSecurity() { setSecurity(!security) }

    func setSpeaker(_ on: Bool) {
        withAnimation(Theme.Motion.snappy) {
            speaker = on
            if on { sleeping = false }
        }
        Haptics.tap()
    }
    func toggleSpeaker() { setSpeaker(!speaker) }

    /// Show a transient "doing X" banner that clears itself.
    private func flash(_ label: String) {
        busyAction = label
        Haptics.rigid()
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(3))
            await MainActor.run { if self?.busyAction == label { self?.busyAction = nil } }
        }
    }

    // MARK: - Voice dictation

    func toggleListening() { listening ? stopListening() : startListening() }

    func startListening() {
        SFSpeechRecognizer.requestAuthorization { [weak self] auth in
            Task { @MainActor in
                guard let self else { return }
                guard auth == .authorized else {
                    self.messages.append(.init(role: .robot,
                        text: "I need Speech Recognition permission to hear you — enable it in Settings."))
                    return
                }
                self.beginAudio()
            }
        }
    }

    private func beginAudio() {
        guard let recognizer, recognizer.isAvailable else { return }
        // Tidy any prior session.
        recTask?.cancel(); recTask = nil

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            messages.append(.init(role: .robot, text: "I couldn't open the microphone just now."))
            return
        }

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        request = req

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }

        audioEngine.prepare()
        do { try audioEngine.start() } catch {
            messages.append(.init(role: .robot, text: "I couldn't start listening."))
            return
        }

        listening = true
        Haptics.tap()
        recTask = recognizer.recognitionTask(with: req) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result { self.draft = result.bestTranscription.formattedString }
                if error != nil || (result?.isFinal ?? false) { self.finishAudio() }
            }
        }
    }

    func stopListening() {
        finishAudio()
        // Send whatever was transcribed.
        if !draft.trimmingCharacters(in: .whitespaces).isEmpty { send() }
    }

    private func finishAudio() {
        guard listening else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        recTask?.cancel()
        request = nil; recTask = nil
        listening = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    // MARK: - Text-to-speech (robot's voice)

    private func speak(_ text: String) {
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = 0.5
        u.pitchMultiplier = 1.0
        synth.speak(u)
    }
}
