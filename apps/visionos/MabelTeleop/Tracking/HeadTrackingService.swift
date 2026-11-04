//
//  HeadTrackingService.swift
//  MabelTeleop
//
//  Polls WorldTrackingProvider for the device pose. visionOS doesn't
//  push head-pose updates as a stream (unlike hands), so the consumer
//  samples whenever it's ready to transmit a frame.
//
//  Restart safety: ARKit providers are one-shot. ProviderRef is a
//  lock-protected box that lets currentPose() remain nonisolated (safe
//  to call from Task.detached) while allowing the provider to be swapped
//  on each start() without recreating the whole service.
//

import Foundation
import ARKit
import QuartzCore
import simd

// MARK: - Thread-safe provider box

/// Wraps the mutable WorldTrackingProvider reference behind an NSLock so
/// that nonisolated callers can safely access it from any thread.
private final class ProviderRef: @unchecked Sendable {
    private let lock = NSLock()
    private var _provider: WorldTrackingProvider = WorldTrackingProvider()

    func swap(to p: WorldTrackingProvider) { lock.withLock { _provider = p } }
    func get() -> WorldTrackingProvider    { lock.withLock { _provider }    }
}

// MARK: - HeadTrackingService

@MainActor
public final class HeadTrackingService {

    private let providerRef = ProviderRef()
    private var session: ARKitSession = ARKitSession()
    private var isRunning = false

    public init() {}

    public static var isSupported: Bool {
        WorldTrackingProvider.isSupported
    }

    public func requestAuthorization() async -> Bool {
        let result = await session.requestAuthorization(for: [.worldSensing])
        return result[.worldSensing] == .allowed
    }

    public func start() async throws {
        guard !isRunning else { return }
        // Always create fresh instances — ARKit providers cannot be restarted.
        session = ARKitSession()
        let p   = WorldTrackingProvider()
        providerRef.swap(to: p)
        try await session.run([p])
        isRunning = true
    }

    public func stop() {
        session.stop()
        isRunning = false
    }

    /// Samples the current head pose. Returns `nil` if tracking is not yet
    /// ready or the provider has no anchor available.
    /// nonisolated — safe to call from Task.detached and any background context.
    public nonisolated func currentPose(at timestamp: TimeInterval = CACurrentMediaTime()) -> HeadPose? {
        let provider = providerRef.get()
        guard let anchor = provider.queryDeviceAnchor(atTimestamp: timestamp) else {
            return nil
        }
        return HeadPose(
            transform: Transform4x4(anchor.originFromAnchorTransform),
            trackingState: anchor.isTracked ? .normal : .limited
        )
    }
}
