//
//  MediaPipeVideoPreview.swift
//  pose
//
//  原生 MediaPipe 影片預覽（AVPlayer），不經 QuickPoseSimulatedCameraView。
//

import AVFoundation
import SwiftUI

struct MediaPipeVideoPreview: UIViewRepresentable {
    let player: AVPlayer?

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = player
        return view
    }

    func updateUIView(_ uiView: PlayerView, context: Context) {
        if uiView.playerLayer.player !== player {
            uiView.playerLayer.player = player
        }
    }

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}
