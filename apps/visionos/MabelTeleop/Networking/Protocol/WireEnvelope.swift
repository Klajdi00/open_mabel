//
//  WireEnvelope.swift
//  MabelTeleop
//
//  The headset and robot share one WebSocket but exchange several message
//  types (outbound telemetry, inbound robot state, pings, errors). Wrapping
//  each payload in a typed envelope keeps the schema extensible without
//  versioning the whole socket.
//

import Foundation

public enum WireMessageType: String, Codable, Sendable {
    case teleopFrame = "teleop_frame"     // headset → robot
    case robotState  = "robot_state"      // robot   → headset
    case ping        = "ping"             // either direction
    case pong        = "pong"
    case error       = "error"
    case hello       = "hello"            // handshake on connect
    case reset       = "reset"            // headset → sim: reset to home
    // Mute/unmute the ROBOT's mic: {"micEnabled": Bool}. Echoed in robot_state.audio.
    // CONTROL ONLY — voice PCM rides the dedicated duplex :9610 plane (AUDIO 0x07
    // records), never this socket. See server/docs/data_streams.md.
    case audioConfig = "audio_config"     // headset → robot
}

/// A minimal envelope. `payload` is decoded on demand by the consumer
/// based on `type` — keeps decoding zero-allocation when we don't care
/// about a given message.
public struct WireEnvelope: Codable, Sendable {
    public var type: WireMessageType
    public var payload: Data

    public init(type: WireMessageType, payload: Data) {
        self.type = type
        self.payload = payload
    }

    enum CodingKeys: String, CodingKey {
        case type, payload
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try c.decode(WireMessageType.self, forKey: .type)
        // Payload travels as a nested JSON object; re-encode to Data so
        // callers can use JSONDecoder on their specific type.
        let raw = try c.decode(AnyCodable.self, forKey: .payload)
        self.payload = try JSONEncoder().encode(raw)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        let decoded = try JSONDecoder().decode(AnyCodable.self, from: payload)
        try c.encode(decoded, forKey: .payload)
    }
}

// MARK: - Helpers

public enum WireCodec {
    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = []
        return e
    }()
    private static let decoder = JSONDecoder()

    public static func encode<T: Encodable>(_ value: T, as type: WireMessageType) throws -> Data {
        let payload = try encoder.encode(value)
        let envelope = WireEnvelope(type: type, payload: payload)
        return try encoder.encode(envelope)
    }

    public static func decodeEnvelope(_ data: Data) throws -> WireEnvelope {
        try decoder.decode(WireEnvelope.self, from: data)
    }

    public static func decodePayload<T: Decodable>(_ type: T.Type, from envelope: WireEnvelope) throws -> T {
        try decoder.decode(T.self, from: envelope.payload)
    }
}

// MARK: - AnyCodable (minimal, internal-only)

/// Lets the envelope round-trip payloads whose concrete type is
/// decided by the message `type` field. Intentionally internal.
struct AnyCodable: Codable {
    let value: Any

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil()                     { value = NSNull() }
        else if let b = try? c.decode(Bool.self)    { value = b }
        else if let i = try? c.decode(Int.self)     { value = i }
        else if let d = try? c.decode(Double.self)  { value = d }
        else if let s = try? c.decode(String.self)  { value = s }
        else if let a = try? c.decode([AnyCodable].self) { value = a.map(\.value) }
        else if let o = try? c.decode([String: AnyCodable].self) {
            value = o.mapValues(\.value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: c, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case is NSNull:            try c.encodeNil()
        case let b as Bool:        try c.encode(b)
        case let i as Int:         try c.encode(i)
        case let d as Double:      try c.encode(d)
        case let s as String:      try c.encode(s)
        case let a as [Any]:       try c.encode(a.map(AnyCodable.init(value:)))
        case let o as [String:Any]:try c.encode(o.mapValues(AnyCodable.init(value:)))
        default:
            throw EncodingError.invalidValue(value,
                .init(codingPath: c.codingPath, debugDescription: "Unsupported type"))
        }
    }

    init(value: Any) { self.value = value }
}
