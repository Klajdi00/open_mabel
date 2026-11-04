//
//  VideoStreamAspect.swift
//  MABEL Mobile App
//
//  Latest MEASURED frame size per wire video stream ("main", "front", …),
//  written by every MJPEG consumer as frames decode. Read wherever render
//  geometry must match what is ACTUALLY on the wire — the Cam 3D projection
//  planes and the 1st-person handover FOV. The bundled rig JSON's w×h is the
//  exporter's canvas size, NOT the stream size (sim streams the head at
//  960×720 4:3 while the spec says 480×270 16:9), so anything sized from the
//  JSON alone renders the video squeezed.
//

import Foundation
import CoreGraphics

enum VideoStreamAspect {
    private static let lock = NSLock()
    private static var sizes: [String: CGSize] = [:]

    static func set(_ stream: String, _ size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        lock.lock(); sizes[stream] = size; lock.unlock()
    }

    static func aspect(_ stream: String) -> Double? {
        lock.lock(); defer { lock.unlock() }
        guard let s = sizes[stream] else { return nil }
        return Double(s.width / s.height)
    }
}
