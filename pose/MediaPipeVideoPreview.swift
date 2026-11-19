//
//  MediaPipeVideoPreview.swift
//  pose
//
//  原生 MediaPipe 影片預覽（AVPlayer）。回報 playerLayer.videoRect
//  供骨架 overlay 精確對位（含 aspectFill 裁切）。
//

import AVFoundation
import SwiftUI

struct MediaPipeVideoPreview: UIViewRepresentable {
    let player: AVPlayer?
    @Binding var videoRect: CGRect

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = player
        view.onVideoRectChange = { rect in
            DispatchQueue.main.async {
                if self.videoRect != rect {
                    self.videoRect = rect
                }
            }
        }
        return view
    }

    func updateUIView(_ uiView: PlayerView, context: Context) {
        if uiView.playerLayer.player !== player {
            uiView.playerLayer.player = player
        }
        uiView.onVideoRectChange = { rect in
            DispatchQueue.main.async {
                if self.videoRect != rect {
                    self.videoRect = rect
                }
            }
        }
        uiView.reportVideoRect()
    }

    final class PlayerView: UIView {
        var onVideoRectChange: ((CGRect) -> Void)?

        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }

        override func layoutSubviews() {
            super.layoutSubviews()
            reportVideoRect()
        }

        func reportVideoRect() {
            // videoRect：實際繪出影片的區域（已套用 preferredTransform + gravity）。
            onVideoRectChange?(playerLayer.videoRect)
        }
    }
}
