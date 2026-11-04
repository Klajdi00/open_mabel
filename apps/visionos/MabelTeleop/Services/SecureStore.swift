//
//  SecureStore.swift
//  MabelTeleop
//
//  Keychain-backed secrets that must NEVER live in the app binary or the repo —
//  notably the REAL-ROBOT ACCESS CODE. The sim is open; the physical robot is
//  gated by a secret the operator types once. We keep it in the Keychain (not
//  UserDefaults) so it is encrypted at rest and never serialized into a backup of
//  plain prefs. The bridge on thor enforces the same code on every transport, so
//  this is the client half of "not anyone can drive the real robot".
//  Ported 1:1 from the MABEL iOS app so both clients share the secret store.
//

import Foundation
import Security

enum SecureStore {
    /// Keychain account for the physical robot's access code.
    private static let realCodeAccount = "mabel.realRobotAccessCode"
    /// Keychain account for the operator's Anthropic API key (powers the assistant).
    private static let anthropicKeyAccount = "mabel.anthropicAPIKey"
    private static let service = "com.mabel.teleop.secrets"

    /// The operator-entered access code for the real robot (nil if never set).
    static var realRobotCode: String? {
        get { read(realCodeAccount) }
        set { write(realCodeAccount, newValue) }
    }

    /// True once the operator has stored a real-robot code on this device.
    static var hasRealRobotCode: Bool {
        guard let c = realRobotCode else { return false }
        return !c.isEmpty
    }

    /// The value the operator typed into the assistant key field. Stored only in
    /// this device's Keychain. By default this is the user's OWN Anthropic key so
    /// downloads never spend the developer's tokens — unless it's the owner bypass
    /// code, which swaps in the developer's default key (see `effectiveAnthropicKey`).
    static var anthropicKey: String? {
        get { read(anthropicKeyAccount) }
        set { write(anthropicKeyAccount, newValue) }
    }

    /// True once any value is stored in the assistant key field.
    static var hasAnthropicKey: Bool {
        guard let k = anthropicKey else { return false }
        return !k.isEmpty
    }

    /// The owner bypass code: typing this as the "key" activates the developer's
    /// default Claude key on this device, so the owner doesn't paste a personal key.
    /// NOT shipped in the repo — injected at build time via the `OWNER_BYPASS_CODE`
    /// Info.plist value (set it from an untracked xcconfig / build setting) or the
    /// same-named scheme env var. Empty (unconfigured) disables the bypass.
    static var ownerBypassCode: String {
        if let env = ProcessInfo.processInfo.environment["OWNER_BYPASS_CODE"], !env.isEmpty { return env }
        return (Bundle.main.object(forInfoDictionaryKey: "OWNER_BYPASS_CODE") as? String) ?? ""
    }

    /// Whether the stored value is the owner bypass code (developer mode). An empty
    /// (unconfigured) bypass code never matches — fail closed.
    static var ownerModeActive: Bool {
        let code = ownerBypassCode
        return !code.isEmpty && (anthropicKey ?? "") == code
    }

    /// The developer's default key — NOT shipped in the repo. Injected at build time
    /// via the `ANTHROPIC_DEFAULT_KEY` Info.plist value (set it from an untracked
    /// xcconfig / build setting), or the same-named scheme env var for Xcode runs.
    static var developerDefaultKey: String {
        if let env = ProcessInfo.processInfo.environment["ANTHROPIC_DEFAULT_KEY"], !env.isEmpty { return env }
        return (Bundle.main.object(forInfoDictionaryKey: "ANTHROPIC_DEFAULT_KEY") as? String) ?? ""
    }

    /// The key actually used for requests: the owner bypass swaps in the developer
    /// default; everyone else uses the key they entered.
    static var effectiveAnthropicKey: String {
        let stored = anthropicKey ?? ""
        return stored == ownerBypassCode ? developerDefaultKey : stored
    }

    /// True when there's a usable key (a real user key, or owner mode with a
    /// configured developer default).
    static var hasUsableAnthropicKey: Bool { !effectiveAnthropicKey.isEmpty }

    // MARK: - Keychain primitives

    private static func write(_ account: String, _ value: String?) {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(base as CFDictionary)            // replace any existing
        guard let value, !value.isEmpty,
              let data = value.data(using: .utf8) else { return }   // nil/empty == clear
        var add = base
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(add as CFDictionary, nil)
    }

    private static func read(_ account: String) -> String? {
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
