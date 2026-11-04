import Foundation

/// Sticky operator stop intent. Confirmation refers to controller flags only.
struct RobotStopState: Equatable {
    private(set) var requested = false
    private(set) var confirmed = false
    private(set) var acknowledged = false
    private(set) var controllerStopped = false
    private(set) var requestedAt: TimeInterval?
    private(set) var lastSentAt: TimeInterval?
    private(set) var sendError: String?
    static let retryInterval: TimeInterval = 0.75

    mutating func request(now: TimeInterval) {
        requested = true; confirmed = false; acknowledged = false; controllerStopped = false; requestedAt = now
        lastSentAt = nil; sendError = nil
    }
    mutating func resume() {
        requested = false; confirmed = false; acknowledged = false; controllerStopped = false; requestedAt = nil
        lastSentAt = nil; sendError = nil
    }
    mutating func observe(_ state: RobotState) {
        confirmed = state.controllerStopConfirmed
        controllerStopped = state.controllerStopAcknowledged
        if controllerStopped { acknowledged = true; sendError = nil }
    }
    mutating func disconnected() {
        confirmed = false; controllerStopped = false; lastSentAt = nil
        if requested && !acknowledged { sendError = "Connection lost" }
    }
    mutating func failed(_ message: String) { confirmed = false; sendError = message }
    mutating func sent(now: TimeInterval) { lastSentAt = now }
    func shouldRetry(now: TimeInterval) -> Bool {
        requested && !acknowledged && now - (lastSentAt ?? -.infinity) >= Self.retryInterval
    }
    func label(now: TimeInterval) -> String {
        if confirmed { return "Controller stopped" }
        if requested && acknowledged {
            return controllerStopped ? "Controller stopped · motor status unconfirmed" : "Controller stop state changed"
        }
        if requested {
            return sendError != nil || now - (requestedAt ?? now) >= 1.5 ? "Stop unconfirmed" : "Stop requested"
        }
        return "Controller state unconfirmed"
    }
}
