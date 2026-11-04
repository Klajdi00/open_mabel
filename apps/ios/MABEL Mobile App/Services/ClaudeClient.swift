//
//  ClaudeClient.swift
//  MABEL Mobile App
//
//  Minimal Anthropic Messages API client. There is no official Anthropic Swift
//  SDK, so this talks raw HTTPS to api.anthropic.com/v1/messages via URLSession.
//  It backs the in-app robot assistant with a real Claude model. The API key
//  lives in the Keychain (SecureStore.anthropicKey) — for a personal robot app
//  the key is on-device; a production deployment would proxy through a server so
//  the key never ships to clients.
//

import Foundation

struct ClaudeClient {
    /// One conversation turn. `role` is "user" or "assistant" (the wire format).
    struct Turn { let role: String; let text: String }

    /// Models offered in Settings. Defaults to the latest, most capable Claude.
    enum Model: String, CaseIterable, Identifiable {
        case opus48   = "claude-opus-4-8"
        case sonnet46 = "claude-sonnet-4-6"
        case haiku45  = "claude-haiku-4-5"
        var id: String { rawValue }
        /// Just the model, for tight spots (the settings row): a collapsed picker
        /// has one line, and "Opus 4.8 · most capable" wrapped onto two and ran
        /// under the chevron.
        var name: String {
            switch self {
            case .opus48:   return "Opus 4.8"
            case .sonnet46: return "Sonnet 4.6"
            case .haiku45:  return "Haiku 4.5"
            }
        }
        /// What choosing it buys you — shown in the MENU, where there is room.
        var blurb: String {
            switch self {
            case .opus48:   return "most capable"
            case .sonnet46: return "balanced"
            case .haiku45:  return "fastest"
            }
        }
        var label: String { "\(name) · \(blurb)" }
        static let storageKey = "llm.model"
        static var current: Model {
            Model(rawValue: UserDefaults.standard.string(forKey: storageKey) ?? "") ?? .opus48
        }
    }

    var apiKey: String
    var model: String

    enum ClientError: LocalizedError {
        case noKey
        case http(Int, String)
        case badResponse
        var errorDescription: String? {
            switch self {
            case .noKey:                return "No Anthropic API key set."
            case .http(let code, let m):
                // Surface the API's own message when it's short and useful.
                let trimmed = m.count > 160 ? String(m.prefix(160)) + "…" : m
                return "API error \(code): \(trimmed)"
            case .badResponse:          return "Unexpected response from the model."
            }
        }
    }

    /// Send the conversation + a system prompt and return Claude's text reply.
    func send(system: String, turns: [Turn]) async throws -> String {
        guard !apiKey.isEmpty else { throw ClientError.noKey }

        var req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "content-type")
        req.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        req.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        req.timeoutInterval = 30

        // The API requires the first message to be a user turn — drop any leading
        // assistant turns (e.g. the assistant's greeting) and keep it bounded.
        var trimmed = turns
        while let first = trimmed.first, first.role != "user" { trimmed.removeFirst() }
        let recent = Array(trimmed.suffix(20))

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 600,
            "system": system,
            "messages": recent.map { ["role": $0.role, "content": $0.text] },
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw ClientError.badResponse }
        guard http.statusCode == 200 else {
            // Pull the API's error message out of the JSON envelope when present.
            var detail = String(data: data, encoding: .utf8) ?? ""
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let err = obj["error"] as? [String: Any], let msg = err["message"] as? String {
                detail = msg
            }
            throw ClientError.http(http.statusCode, detail)
        }

        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = obj["content"] as? [[String: Any]] else { throw ClientError.badResponse }
        let text = content
            .compactMap { ($0["type"] as? String) == "text" ? $0["text"] as? String : nil }
            .joined()
        return text.isEmpty ? "(no reply)" : text
    }
}
