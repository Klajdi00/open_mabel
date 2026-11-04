import XCTest
@testable import MABEL_Mobile_App

@MainActor
final class RobotSafetyTests: XCTestCase {
    private func stopCommands(_ link: FakeLink) -> [Bool] {
        link.sent.compactMap { command in
            if case .estop(let active) = command { return active }
            return nil
        }
    }

    private func stoppedState(unconfirmed: [String]? = []) -> RobotState {
        var state = RobotState()
        state.estop = true
        state.torqueState = ["any": false]
        state.moduleTorque = ["left_arm": false, "right_arm": false, "neck": false, "lift": true]
        state.stopUnconfirmed = unconfirmed
        return state
    }

    func testOfflineTorqueOnDoesNotSendOrClaimSuccess() {
        let link = FakeLink()
        let controller = RobotController(link: link)
        controller.releaseEStop()
        XCTAssertTrue(stopCommands(link).isEmpty)
        XCTAssertTrue(controller.estopEngaged)
        XCTAssertFalse(controller.torqueOnRequested)
        XCTAssertTrue(controller.safetyNotice?.contains("No robot connection") == true)
    }

    func testTorqueOnWaitsForControllerEcho() {
        let link = FakeLink()
        let controller = RobotController(link: link)
        link.connect()
        controller.releaseEStop()
        XCTAssertEqual(stopCommands(link), [false])
        XCTAssertTrue(controller.estopEngaged)
        XCTAssertTrue(controller.torqueOnRequested)
        link.onState?(stoppedState())
        XCTAssertTrue(controller.estopEngaged)
        XCTAssertTrue(controller.torqueOnRequested)
        var resumed = RobotState()
        resumed.estop = false
        link.onState?(resumed)
        XCTAssertFalse(controller.estopEngaged)
        XCTAssertFalse(controller.torqueOnRequested)
        controller.disconnect()
    }

    func testInterruptedTorqueOnIsNeverReplayedOrUnlockedByOldEcho() {
        let link = FakeLink()
        let controller = RobotController(link: link)
        link.connect()
        controller.releaseEStop()
        link.dropUnexpectedly()
        XCTAssertFalse(controller.torqueOnRequested)
        XCTAssertTrue(controller.estopEngaged)
        link.connect()
        var resumed = RobotState()
        resumed.estop = false
        link.onState?(resumed)
        XCTAssertEqual(stopCommands(link), [false], "a reconnect must never enable torque")
        XCTAssertTrue(controller.estopEngaged, "an interrupted request requires another explicit tap")
        controller.disconnect()
    }

    func testStopIsAlwaysAvailableEvenOffline() {
        let link = FakeLink()
        let controller = RobotController(link: link)
        controller.engageEStop()
        XCTAssertEqual(stopCommands(link), [true])
        XCTAssertTrue(controller.estopEngaged)
        XCTAssertTrue(controller.stopState.requested)
        XCTAssertEqual(controller.stopStatusLabel, "Stop unconfirmed")
        controller.disconnect()
    }

    func testHostAckEndsRetriesDespiteAmbiguousPhysicalMotorStatus() {
        var stop = RobotStopState()
        stop.request(now: 0)
        stop.sent(now: 0)
        XCTAssertFalse(stop.shouldRetry(now: 0.5))
        XCTAssertTrue(stop.shouldRetry(now: 1))
        stop.observe(stoppedState(unconfirmed: ["neck_yaw"]))
        XCTAssertTrue(stop.acknowledged)
        XCTAssertFalse(stop.confirmed)
        XCTAssertFalse(stop.shouldRetry(now: 100))
        XCTAssertEqual(stop.label(now: 100), "Controller stopped · motor status unconfirmed")
        stop.disconnected()
        XCTAssertFalse(stop.shouldRetry(now: 200))
        var releasedByOtherClient = RobotState()
        releasedByOtherClient.estop = false
        releasedByOtherClient.torqueState = ["any": true]
        stop.observe(releasedByOtherClient)
        XCTAssertFalse(stop.shouldRetry(now: 300), "an acknowledged old stop cannot undo a later explicit resume")
        stop.request(now: 400)
        XCTAssertTrue(stop.shouldRetry(now: 400), "a fresh operator stop starts a new request")
    }

    func testAcknowledgedStopIsNotResentOnReconnectOrRemoteResume() {
        let link = FakeLink()
        let controller = RobotController(link: link)
        link.connect()
        controller.engageEStop()
        link.onState?(stoppedState(unconfirmed: ["neck_yaw"]))
        XCTAssertTrue(controller.stopState.acknowledged)
        link.dropUnexpectedly()
        link.connect()
        var resumed = RobotState()
        resumed.estop = false
        resumed.torqueState = ["any": true]
        link.onState?(resumed)
        XCTAssertEqual(stopCommands(link), [true])
        controller.engageEStop()
        XCTAssertEqual(stopCommands(link), [true, true])
        controller.disconnect()
    }

    func testNestedTorqueSafetyDecodeAndLiftExclusion() {
        var state = RobotState()
        WebSocketLink.applySafety([
            "estop": true,
            "stopUnconfirmed": ["neck_yaw"],
            "torque": ["any": true, "modules": [
                "left_arm": ["torque": false], "right_arm": ["torque": false],
                "neck": ["torque": false], "lift": ["torque": true]
            ]]
        ], to: &state)
        XCTAssertEqual(state.moduleTorque["left_arm"], false)
        XCTAssertEqual(state.stopUnconfirmed, ["neck_yaw"])
        XCTAssertTrue(state.controllerStopAcknowledged)
        XCTAssertFalse(state.controllerStopConfirmed)
        state.moduleTorque["neck"] = true
        XCTAssertFalse(state.controllerStopAcknowledged)
        state.moduleTorque["neck"] = false
        state.stopUnconfirmed = []
        XCTAssertTrue(state.controllerStopConfirmed)
    }

    func testMissingSafetyTelemetryCannotAcknowledgeStop() {
        var state = RobotState()
        XCTAssertFalse(state.controllerStopAcknowledged)
        state.estop = true
        XCTAssertFalse(state.controllerStopAcknowledged)
        state.torqueState = ["arms": false]
        XCTAssertTrue(state.controllerStopAcknowledged, "legacy Bluetooth reports the arms flag")
        state.estop = false
        XCTAssertFalse(state.controllerStopAcknowledged)
    }

    func testDisconnectedWebSocketReportsSafetySendFailure() {
        let link = WebSocketLink(host: "127.0.0.1", wsPort: 1)
        var message: String?
        link.onSafetyError = { message = $0 }
        link.send(.estop(false))
        XCTAssertEqual(message, "No robot connection")
    }

    func testObsoleteSocketCallbacksCannotReconnectExplicitlyDisconnectedLink() async {
        let link = WebSocketLink(host: "127.0.0.1", wsPort: 1)
        let oldSocket = URLSession.shared.webSocketTask(with: URL(string: "ws://127.0.0.1:1")!)
        link.urlSession(.shared, webSocketTask: oldSocket, didOpenWithProtocol: nil)
        await flushMainQueue()
        XCTAssertEqual(link.status, .disconnected)
        link.urlSession(.shared, webSocketTask: oldSocket, didCloseWith: .goingAway, reason: nil)
        link.urlSession(.shared, task: oldSocket, didCompleteWithError: URLError(.cancelled))
        await flushMainQueue()
        XCTAssertEqual(link.status, .disconnected)
    }

    func testOldHandshakeCannotMarkReplacementSocketConnected() async {
        let link = WebSocketLink(host: "127.0.0.1", wsPort: 1)
        let oldSocket = URLSession.shared.webSocketTask(with: URL(string: "ws://127.0.0.1:1")!)
        var connectedCallbacks = 0
        link.onStatus = { if $0 == .connected { connectedCallbacks += 1 } }
        link.connect() // creates a different current task; port 1 has no robot or server
        link.urlSession(.shared, webSocketTask: oldSocket, didOpenWithProtocol: nil)
        await flushMainQueue()
        XCTAssertEqual(connectedCallbacks, 0, "only the current socket may finish its handshake")
        link.disconnect()
    }

    private func flushMainQueue() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.main.async { continuation.resume() }
        }
    }
}
