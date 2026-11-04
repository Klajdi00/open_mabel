//
//  LoopingVideo.swift
//  MABEL Mobile App
//
//  A silent, auto-looping video tile used to preview a policy's task
//  ("how the complete task looks"). Backed by AVPlayerLooper for seamless
//  repeats. Falls back gracefully if the clip is missing.
//

import SwiftUI
import AVFoundation

struct LoopingVideo: UIViewRepresentable {
    let resource: String           // bundle resource name (no extension)
    var fileExtension = "mp4"

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        if let url = Bundle.main.url(forResource: resource, withExtension: fileExtension) {
            view.play(url: url)
        }
        return view
    }

    func updateUIView(_ uiView: PlayerView, context: Context) {
        if uiView.currentResource != resource,
           let url = Bundle.main.url(forResource: resource, withExtension: fileExtension) {
            uiView.play(url: url)
        }
    }

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        private var looper: AVPlayerLooper?
        private let queuePlayer = AVQueuePlayer()
        private(set) var currentResource: String?

        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }

        func play(url: URL) {
            currentResource = url.deletingPathExtension().lastPathComponent
            let item = AVPlayerItem(url: url)
            looper = AVPlayerLooper(player: queuePlayer, templateItem: item)
            queuePlayer.isMuted = true
            playerLayer.player = queuePlayer
            playerLayer.videoGravity = .resizeAspectFill
            queuePlayer.play()
        }
    }
}
