//
//  MediaPipeCameraPreview.swift
//  pose
//
//  原生 AVCapture 預覽層，供 MediaPipe Pose Landmarker 使用（不經 QuickPoseCameraView）。
//

import AVFoundation
import SwiftUI

struct MediaPipeCameraPreview: UIViewRepresentable {
    let engine: MediaPipeNativePoseEngine

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = engine.captureSessionForPreview
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = engine.captureSessionForPreview
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
