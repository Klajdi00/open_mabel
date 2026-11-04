//
//  H264Decoder.swift
//  MabelTeleop
//
//  Hardware H.264 decode (VideoToolbox) for the UDP :9092 video
//  path — the visionOS twin of the iOS app's Services/H264Decoder.swift. Input is
//  ONE Annex-B access unit per call (what UDPVideoReceiver reassembles from the
//  bridge's :9092 'MV' stream).
//  Output is a CGImage, so it drops into the SAME render path as the MJPEG fallback
//  (SwiftUI Image tiles AND the RealityKit StereoWindowPanel texture upload).
//
//  Pipeline per access unit:
//    Annex-B  → split NAL units → cache SPS(7)/PPS(8), collect VCL(1,5)
//    SPS+PPS  → CMVideoFormatDescription (rebuilt whenever the params change)
//    VCL      → AVCC (4-byte length-prefixed) → CMSampleBuffer
//             → VTDecompressionSession (RealTime) → CVPixelBuffer → CGImage
//
//  The bridge emits SPS/PPS in-band at every IDR (~1 s), so a late
//  or lossy joiner recovers within one GOP with
//  no out-of-band signalling.
//
//  Runs on the receiver's serial decode queue. H.264 is inter-frame, so — unlike the
//  JPEG path — access units are fed IN ORDER and never dropped-while-busy (dropping a
//  P-frame corrupts the stream until the next IDR).
//

import Foundation
import VideoToolbox
import CoreMedia
import CoreVideo
import CoreGraphics
import OSLog

final class H264Decoder {

    private var session: VTDecompressionSession?
    private var formatDesc: CMVideoFormatDescription?
    private var sps: Data?
    private var pps: Data?
    private let onImage: @Sendable (CGImage) -> Void
    private let log = Logger(subsystem: "com.mabel.teleop", category: "H264")

    init(onImage: @escaping @Sendable (CGImage) -> Void) { self.onImage = onImage }

    /// True after the transport LOST an access unit: every P-frame until the next IDR
    /// references a picture we never decoded, so feeding them paints smeared garbage
    /// (Wi-Fi drops one 1.2 kB chunk of a frame and the whole frame is gone). Skip to the
    /// next IDR instead — a ≤ one-GOP freeze on the last good picture, never corruption.
    /// Starts true: a late joiner's first P-frames are equally unreferenced.
    private(set) var needKeyframe = true
    /// Tell the decoder an access unit was lost upstream (sequence gap).
    func noteLoss() { needKeyframe = true }
    deinit { invalidateSession() }

    private func invalidateSession() {
        if let s = session { VTDecompressionSessionInvalidate(s); session = nil }
    }

    /// Decode one Annex-B access unit.
    func decode(accessUnit au: Data) {
        let nalus = Self.splitNALUs(au)
        var vcl: [Data] = []
        var paramsChanged = false
        for n in nalus {
            guard let first = n.first else { continue }
            switch first & 0x1F {
            case 7: if n != sps { sps = n; paramsChanged = true }     // SPS
            case 8: if n != pps { pps = n; paramsChanged = true }     // PPS
            case 1, 5: vcl.append(n)                                  // non-IDR / IDR slice
            default: break                                           // AUD(9), SEI(6), … drop
            }
        }
        if paramsChanged, let sps, let pps { rebuildFormat(sps: sps, pps: pps) }
        guard !vcl.isEmpty, let fmt = formatDesc else { return }
        if needKeyframe {
            guard vcl.contains(where: { ($0.first ?? 0) & 0x1F == 5 }) else { return }
            needKeyframe = false
        }
        if session == nil { createSession(format: fmt) }
        guard let session else { return }

        var avcc = Data()
        for n in vcl {
            var len = UInt32(n.count).bigEndian
            withUnsafeBytes(of: &len) { avcc.append(contentsOf: $0) }
            avcc.append(n)
        }
        guard let sample = Self.makeSampleBuffer(avcc, format: fmt) else { return }

        let flags: VTDecodeFrameFlags = [._EnableAsynchronousDecompression]
        #if DEBUG
        let tIn = Date().timeIntervalSince1970
        #endif
        VTDecompressionSessionDecodeFrame(
            session, sampleBuffer: sample, flags: flags, infoFlagsOut: nil
        ) { [weak self] status, _, imageBuffer, _, _ in
            guard let self, status == noErr, let imageBuffer else { return }
            #if DEBUG
            Self.logStamp(imageBuffer, tIn: tIn)
            #endif
            var cg: CGImage?
            VTCreateCGImageFromCVPixelBuffer(imageBuffer, options: nil, imageOut: &cg)
            if let cg { self.onImage(cg) }
        }
    }

    #if DEBUG
    /// Latency probe (defaults `debugVideoStamp YES`): reads the camera service's USB-arrival stamp
    /// (NOVA nova_head_camera/video.py: white + black reference cell, then 24 bits MSB first, 16 px
    /// cells along the bottom-left) and logs it with the AU-complete and decoded wall times, for
    /// NOVA/experiments/teleop_latency/vp_sim_latency.py.
    private static let stampOn = UserDefaults.standard.bool(forKey: "debugVideoStamp")
    private static func logStamp(_ pb: CVPixelBuffer, tIn: TimeInterval) {
        guard stampOn else { return }
        let tDec = Date().timeIntervalSince1970
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), row = CVPixelBufferGetBytesPerRow(pb)
        guard w >= 26 * 16, h >= 16 else { return }
        let px = base.assumingMemoryBound(to: UInt8.self)
        func lum(_ k: Int) -> Double {
            let x = k * 16 + 8, y = h - 8
            let p = px + y * row + x * 4                                  // BGRA
            return (Double(p[0]) + Double(p[1]) + Double(p[2])) / 3
        }
        let white = lum(0), black = lum(1)
        guard white - black > 60 else { return }
        var v: UInt32 = 0
        for k in 0..<24 { v = v << 1 | ((lum(k + 2) - black) / (white - black) > 0.5 ? 1 : 0) }
        NSLog("[MABEL-VLAT] stamp=%u in=%.4f dec=%.4f w=%d", v, tIn * 1000, tDec * 1000, w)
    }
    #endif

    // MARK: - Format description + session

    private func rebuildFormat(sps: Data, pps: Data) {
        invalidateSession()                              // params changed → fresh session
        formatDesc = nil
        sps.withUnsafeBytes { (sp: UnsafeRawBufferPointer) in
            pps.withUnsafeBytes { (pp: UnsafeRawBufferPointer) in
                guard let spsBase = sp.bindMemory(to: UInt8.self).baseAddress,
                      let ppsBase = pp.bindMemory(to: UInt8.self).baseAddress else { return }
                let ptrs: [UnsafePointer<UInt8>] = [spsBase, ppsBase]
                let sizes: [Int] = [sps.count, pps.count]
                var fmt: CMVideoFormatDescription?
                let st = ptrs.withUnsafeBufferPointer { pb in
                    sizes.withUnsafeBufferPointer { sb in
                        CMVideoFormatDescriptionCreateFromH264ParameterSets(
                            allocator: kCFAllocatorDefault,
                            parameterSetCount: 2,
                            parameterSetPointers: pb.baseAddress!,
                            parameterSetSizes: sb.baseAddress!,
                            nalUnitHeaderLength: 4,
                            formatDescriptionOut: &fmt)
                    }
                }
                if st == noErr { self.formatDesc = fmt }
                else { self.log.error("CMVideoFormatDescription failed: \(st)") }
            }
        }
    }

    private func createSession(format: CMVideoFormatDescription) {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        var s: VTDecompressionSession?
        let st = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: format,
            decoderSpecification: nil,
            imageBufferAttributes: attrs as CFDictionary,
            outputCallback: nil,
            decompressionSessionOut: &s)
        if st == noErr, let s {
            VTSessionSetProperty(s, key: kVTDecompressionPropertyKey_RealTime,
                                 value: kCFBooleanTrue)
            session = s
        } else {
            log.error("VTDecompressionSessionCreate failed: \(st)")
        }
    }

    // MARK: - Annex-B helpers

    /// Split an Annex-B buffer into NAL unit payloads (start codes stripped).
    static func splitNALUs(_ d: Data) -> [Data] {
        let b = [UInt8](d)
        let n = b.count
        var out: [Data] = []
        func startLen(_ at: Int) -> Int {
            if at + 3 < n, b[at] == 0, b[at+1] == 0, b[at+2] == 0, b[at+3] == 1 { return 4 }
            if at + 2 < n, b[at] == 0, b[at+1] == 0, b[at+2] == 1 { return 3 }
            return 0
        }
        var i = 0
        while i < n && startLen(i) == 0 { i += 1 }        // skip to first start code
        while i < n {
            let sl = startLen(i); if sl == 0 { break }
            let start = i + sl
            var j = start
            while j < n && startLen(j) == 0 { j += 1 }
            if start < j { out.append(Data(b[start..<j])) }
            i = j
        }
        return out
    }

    /// Wrap AVCC (length-prefixed) frame bytes in a CMSampleBuffer (CMBlockBuffer owns
    /// its own memory — no manual free, no dangling pointer).
    static func makeSampleBuffer(_ avcc: Data,
                                 format: CMVideoFormatDescription) -> CMSampleBuffer? {
        var block: CMBlockBuffer?
        var st = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: avcc.count,
            blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
            dataLength: avcc.count, flags: kCMBlockBufferAssureMemoryNowFlag,
            blockBufferOut: &block)
        guard st == kCMBlockBufferNoErr, let block else { return nil }
        st = avcc.withUnsafeBytes { raw in
            CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: block,
                                          offsetIntoDestination: 0, dataLength: avcc.count)
        }
        guard st == kCMBlockBufferNoErr else { return nil }
        var sample: CMSampleBuffer?
        var sizes = [avcc.count]
        st = CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: format,
            sampleCount: 1, sampleTimingEntryCount: 0, sampleTimingArray: nil,
            sampleSizeEntryCount: 1, sampleSizeArray: &sizes, sampleBufferOut: &sample)
        return st == noErr ? sample : nil
    }
}
