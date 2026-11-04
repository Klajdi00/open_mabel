//
//  DemoLink.swift
//  MABEL Mobile App
//
//  An on-device physics-lite simulator that makes the whole app feel
//  alive without a robot attached. It integrates operator commands into
//  a plausible RobotState at 30 Hz, drifts battery/thermals slowly, and
//  smoothly servos joints toward their targets.
//

import Foundation
import QuartzCore

final class DemoLink: RobotLink {
    private(set) var status: ConnectionStatus = .disconnected {
        didSet { onStatus?(status) }
    }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((_ state: String, _ key: String?, _ holder: String) -> Void)?
    var onMapList: ((_ maps: [MapInfo], _ loaded: String?) -> Void)?
    /// Demo Mode has no clip library — the offline simulator has no disk to save
    /// one to. The picker shows its empty state rather than pretending.
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    /// The demo has no ROS map server, so the procedural rooms ARE the saved maps.
    private var demoLoadedMap: String? = nil
    private func emitMapList() {
        let maps = SLAMRoom.library.map { MapInfo(name: $0.name, hasMesh: true, hasReloc: false) }
        onMapList?(maps, demoLoadedMap)
    }

    private var state = RobotState()
    private var displayLink: CADisplayLink?
    private var lastTick: CFTimeInterval = 0

    // Command intents the loop integrates.
    private var cmdForward = 0.0, cmdStrafe = 0.0, cmdYaw = 0.0
    private var cmdLiftRate = 0.0
    private var jointTargets = RobotModel.homePose
    private var estopped = false

    // MARK: - RobotLink

    func connect() {
        status = .connecting
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
            guard let self else { return }
            self.status = .demo
            self.onServerInfo?("on-device-sim")   // the demo is a simulation
            self.startLoop()
        }
    }

    func disconnect() {
        displayLink?.invalidate()
        displayLink = nil
        status = .disconnected
    }

    func send(_ command: RobotCommand) {
        switch command {
        case let .twist(f, s, y):
            cmdForward = f; cmdStrafe = s; cmdYaw = y
        case let .jointTarget(id, value):
            jointTargets[id] = value
        case let .liftFraction(frac):
            jointTargets["lift_lower"] = frac * 0.3175
            jointTargets["lift_upper"] = frac * 0.3175
        case let .liftRate(r):
            cmdLiftRate = r
        case .goHome:
            jointTargets = RobotModel.homePose
        case .resetSlam:
            break   // the on-device map demo restarts via LiveMapStore.restartMapping()
        case .resetSim, .resetSimRandom:
            // Reset to the initial start environment: home pose + zeroed pose.
            jointTargets = RobotModel.homePose
            state.joints = RobotModel.homePose
            state.heading = 0; state.linearSpeed = 0; state.angularSpeed = 0
            cmdForward = 0; cmdStrafe = 0; cmdYaw = 0; cmdLiftRate = 0
        case let .estop(on):
            estopped = on
            state.estop = on
            state.torqueState["any"] = !on
            if on { cmdForward = 0; cmdStrafe = 0; cmdYaw = 0 }
        case let .headPose(m):
            // Drive the neck from the head's forward (−Z) so the demo twin
            // also looks where the phone points.
            guard m.count == 16 else { break }
            jointTargets["neck_yaw"]   = atan2(-m[2], -m[10]) * 0.6
            jointTargets["neck_pitch"] = asin(max(-1, min(1, m[6]))) * 0.6
        case .listMaps:
            emitMapList()
        case let .loadMap(name):
            let live = name.isEmpty || name == "live" || name == "none"
            demoLoadedMap = live ? nil : name
            emitMapList()
        case .controlSpec, .dispatchPolicy, .externalForce, .record, .selectTask,
             .headTracking, .setSpeedCaps, .setRobotName,
             // The two operator switches are control-LAW changes evaluated
             // server-side; the on-device demo replays joint targets and has no
             // gravity model or retargeter to switch, so it accepts and drops
             // them rather than pretending the toggle did something.
             .gravityComp, .wholeBodyRetarget,
             .requestControl, .releaseControl, .navGoal, .dataCollect, .registerPush,
             .setMicEnabled,
             // Animation demos need a server to write clips to; the on-device
             // simulator has no disk-backed library, so these are accepted and
             // dropped rather than faked.
             .demoRecord, .demoReplay, .demoDelete:
            break   // demo applies joint targets locally; WBC + Nav2 are server-side
                    // and there is no robot mic to mute in the on-device demo
        case .demoList:
            // Answer honestly with an empty library so the picker shows its empty
            // state instead of spinning on a reply that will never come.
            onDemoClips?([])
        }
    }

    // MARK: - Simulation loop

    private func startLoop() {
        lastTick = CACurrentMediaTime()
        let link = CADisplayLink(target: self, selector: #selector(step))
        link.preferredFrameRateRange = .init(minimum: 24, maximum: 30, preferred: 30)
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @objc private func step() {
        let now = CACurrentMediaTime()
        let dt = min(max(now - lastTick, 1.0 / 120), 1.0 / 15)
        lastTick = now

        integrateBase(dt: dt)
        integrateLift(dt: dt)
        servoJoints(dt: dt)
        driftHealth(dt: dt)

        // Synthetic planar LiDAR at ~5 Hz — same sparse-attach contract as the
        // bridge (lidar rides a state only when a FRESH sweep exists), so the
        // tile and its latching path exercise identically offline.
        if now - lastScanTime >= 0.2 {
            lastScanTime = now
            state.lidar = demoScan()
        } else {
            state.lidar = nil
        }

        // Synthetic joint dynamics at ~10 Hz — same sparse contract as the
        // bridge's jointVelocities/jointTorques (and temps, so the telemetry
        // bars' temperature grading exercises offline). Vel is the true
        // derivative of the servoed pose; torque a plausible viscous+gravity
        // proxy; temps track drive load like cpuTemp.
        if now - lastDynTime >= 0.1 {
            let dynDt = max(now - lastDynTime, 1e-3)
            lastDynTime = now
            var vel: [String: Double] = [:], tor: [String: Double] = [:], temp: [String: Double] = [:]
            let load = abs(state.linearSpeed) + abs(state.angularSpeed)
            for spec in RobotModel.joints {
                let q = state.joints[spec.id] ?? 0
                let dq = (q - (prevJoints[spec.id] ?? q)) / dynDt
                vel[spec.id] = (dq * 1e4).rounded() / 1e4
                tor[spec.id] = ((2.2 * dq + 0.7 * sin(q)) * 1e3).rounded() / 1e3
                let base = 33.0 + 9.0 * abs(sin(q)) + load * 6
                let held = demoTemps[spec.id] ?? base
                demoTemps[spec.id] = held + (base - held) * 0.1
                temp[spec.id] = (demoTemps[spec.id]! * 10).rounded() / 10
                prevJoints[spec.id] = q
            }
            state.jointVelocities = vel
            state.jointTorques = tor
            state.jointTemps = temp
        } else {
            state.jointVelocities = nil
            state.jointTorques = nil
            state.jointTemps = nil
        }

        state.timestamp = now
        onState?(state)
    }

    private var lastScanTime: TimeInterval = 0
    private var lastDynTime: TimeInterval = 0
    private var prevJoints: [String: Double] = [:]
    private var demoTemps: [String: Double] = [:]

    /// Ray-cast the demo room (an axis-aligned rectangle + two round obstacles)
    /// from the robot's pose so the sweep rotates with `state.heading` — enough
    /// realism to eyeball "LiDAR alive and tracking" without MuJoCo.
    private func demoScan() -> RobotState.LidarScan {
        let n = 360
        var scan = RobotState.LidarScan()
        scan.angleMin = -Double.pi
        scan.angleInc = 2 * Double.pi / Double(n)
        scan.rangeMax = 12
        scan.stamp = state.timestamp
        let px = 0.6, py = -0.4                        // robot offset in the room
        let xMin = -3.5, xMax = 3.5, yMin = -2.5, yMax = 2.5
        let obstacles: [(x: Double, y: Double, r: Double)] = [(1.8, 0.8, 0.3), (-1.2, -1.0, 0.25)]
        scan.ranges = (0..<n).map { i in
            let a = scan.angleMin + Double(i) * scan.angleInc + state.heading
            let dx = cos(a), dy = sin(a)
            var t = Double.greatestFiniteMagnitude
            if dx > 1e-9 { t = min(t, (xMax - px) / dx) }
            if dx < -1e-9 { t = min(t, (xMin - px) / dx) }
            if dy > 1e-9 { t = min(t, (yMax - py) / dy) }
            if dy < -1e-9 { t = min(t, (yMin - py) / dy) }
            for o in obstacles {
                let ox = o.x - px, oy = o.y - py
                let b = ox * dx + oy * dy
                guard b > 0 else { continue }
                let d2 = ox * ox + oy * oy - b * b
                let r2 = o.r * o.r
                if d2 < r2 { t = min(t, b - (r2 - d2).squareRoot()) }
            }
            return min(scan.rangeMax, t + Double.random(in: -0.01...0.01))
        }
        return scan
    }

    private func integrateBase(dt: Double) {
        // First-order lag toward commanded velocities for a natural feel.
        let target = estopped ? 0 : sqrt(cmdForward * cmdForward + cmdStrafe * cmdStrafe)
        state.linearSpeed += (target - state.linearSpeed) * min(1, dt * 4)
        state.angularSpeed += ((estopped ? 0 : cmdYaw) - state.angularSpeed) * min(1, dt * 4)
        state.heading += state.angularSpeed * dt
    }

    private func integrateLift(dt: Double) {
        guard !estopped, abs(cmdLiftRate) > 1e-4 else { return }
        // Jog both lift stages by the commanded rate (sim LIFT_SPEED scale).
        let delta = cmdLiftRate * MotionModel.liftSpeed * dt
        for id in ["lift_lower", "lift_upper"] {
            let v = (jointTargets[id] ?? 0) + delta
            jointTargets[id] = min(0.3175, max(0, v))
        }
    }

    private func servoJoints(dt: Double) {
        let rate = min(1, dt * 3.2)   // critically-ish damped approach
        for spec in RobotModel.joints {
            let target = jointTargets[spec.id] ?? 0
            let current = state.joints[spec.id] ?? 0
            state.joints[spec.id] = current + (target - current) * rate
        }
    }

    private func driftHealth(dt: Double) {
        let load = abs(state.linearSpeed) + abs(state.angularSpeed)
        // Battery drains with effort; thermals follow load.
        state.batteryLevel = max(0, state.batteryLevel - dt * (0.0006 + load * 0.0009))
        state.batteryVoltage = 42 + state.batteryLevel * 8.4
        let targetTemp = 44 + load * 12
        state.cpuTemp += (targetTemp - state.cpuTemp) * min(1, dt * 0.5)
        state.latencyMs = 14 + Double.random(in: 0...8)
        state.wifiRSSI = -48 - Int.random(in: 0...12)
    }
}
