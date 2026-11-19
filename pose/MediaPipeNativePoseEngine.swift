//
//  MediaPipeNativePoseEngine.swift
//  pose
//
//  真正的 Google MediaPipe Tasks Vision Pose Landmarker（非 QuickPose 內建）。
//  僅在「MediaPipe」引擎啟用：AVCapture → PoseLandmarker → PoseNode。
//

import AVFoundation
import Combine
import Foundation
import SwiftUI

#if canImport(MediaPipeTasksVision)
import MediaPipeTasksVision
#endif

final class MediaPipeNativePoseEngine: NSObject, ObservableObject {
    @Published var previewNodes: [PoseNode] = []
    @Published var fpsText = "FPS: —"
    @Published var statusHint = ""
    @Published var adviceLines: [String] = ["MediaPipe 原生骨架待機中"]
    @Published var stepHUD = String(format: "L:%3d R:%3d 總:%3d ·—   —bpm", 0, 0, 0)
    @Published var recentSteps: [StepEvent] = []
    @Published var dbNodeCount = 0
    @Published var isRunning = false
    @Published var isEngineStarting = false
    @Published var lastError: String?

    var detectionActive = false
    var activityMode: GaitActivityMode = .walking
    var analysisPipeline: PoseAnalysisPipeline?
    var bodyGaitProfile: BodyGaitProfile?
    var prefersMediaPipeStore = true
    var usesBackViewPipeline = false
    var usesSideViewPipeline = false

    let backViewPipeline = BackViewPosePipeline()
    let sideViewPreprocessor = SideViewPreprocessor()
    let jointConfidenceFilter = JointConfidenceFilter()

    var onStreamEnqueue: (([PoseNode], Double) -> Void)?
    var onStepEvents: (([StepEvent]) -> Void)?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "pose.mediapipe.capture")
    private var frameCount = 0
    private var fpsWindowStart = Date()
    private var useFrontCamera = false

#if canImport(MediaPipeTasksVision)
    /// 僅在 capture queue 存取。
    nonisolated(unsafe) private var poseLandmarker: PoseLandmarker?
#endif

    /// 給 SwiftUI 預覽層綁定用。
    var captureSessionForPreview: AVCaptureSession { session }

    @MainActor
    func syncCaptureView(_ mode: PoseCaptureViewMode) {
        usesBackViewPipeline = mode == .back
        usesSideViewPipeline = mode == .side
        switch mode {
        case .back:
            jointConfidenceFilter.threshold = 0.25
            jointConfidenceFilter.coreThreshold = 0.7
            sideViewPreprocessor.reset()
        case .side:
            jointConfidenceFilter.threshold = 0.25
            jointConfidenceFilter.coreThreshold = 0.5
            backViewPipeline.reset()
        case .front:
            jointConfidenceFilter.threshold = 0.5
            jointConfidenceFilter.coreThreshold = 0.5
            backViewPipeline.reset()
            sideViewPreprocessor.reset()
        }
    }

    @MainActor
    func clearAnalysisBuffers() {
        jointConfidenceFilter.reset()
        backViewPipeline.reset()
        sideViewPreprocessor.reset()
        recentSteps = []
        stepHUD = String(format: "L:%3d R:%3d 總:%3d ·—   —bpm", 0, 0, 0)
        dbNodeCount = 0
    }

    @MainActor
    func flushStabilizedNodes() -> [PoseNode]? {
        jointConfidenceFilter.flush()
    }

    @MainActor
    func start(useFrontCamera: Bool) {
        self.useFrontCamera = useFrontCamera
        lastError = nil
        isEngineStarting = true
        statusHint = "正在啟動 MediaPipe…"
#if canImport(MediaPipeTasksVision)
        sessionQueue.async { [weak self] in
            self?.configureSessionIfNeeded()
            self?.ensureLandmarker()
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
                self?.isEngineStarting = false
                self?.statusHint = "MediaPipe 原生偵測中"
                self?.adviceLines = ["MediaPipe Pose Landmarker 已啟動。按「開始」才會寫入資料庫。"]
            }
        }
#else
        isEngineStarting = false
        lastError = "尚未連結 MediaPipeTasksVision。請在 Xcode 加入 SPM：https://github.com/google-ai-edge/mediapipe → MediaPipeTasksVision，並加入 Models/pose_landmarker_full.task"
        statusHint = lastError ?? ""
        adviceLines = [statusHint]
#endif
    }

    @MainActor
    func stop() {
        detectionActive = false
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.statusHint = "MediaPipe 已停止"
            }
        }
    }

    func switchCamera(useFrontCamera: Bool) {
        guard self.useFrontCamera != useFrontCamera else { return }
        self.useFrontCamera = useFrontCamera
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            for input in self.session.inputs {
                self.session.removeInput(input)
            }
            self.addCameraInput(front: useFrontCamera)
            self.session.commitConfiguration()
        }
    }

#if canImport(MediaPipeTasksVision)
    private func ensureLandmarker() {
        guard poseLandmarker == nil else { return }
        guard let modelPath = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else {
            Task { @MainActor in
                self.lastError = "找不到 pose_landmarker_full.task，請確認已加入 App Bundle"
                self.statusHint = self.lastError ?? ""
            }
            return
        }
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.poseLandmarkerLiveStreamDelegate = self
        do {
            poseLandmarker = try PoseLandmarker(options: options)
        } catch {
            Task { @MainActor in
                self.lastError = "MediaPipe 初始化失敗：\(error.localizedDescription)"
                self.statusHint = self.lastError ?? ""
            }
        }
    }
#endif

    private func configureSessionIfNeeded() {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        addCameraInput(front: useFrontCamera)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = useFrontCamera
            }
        }
        session.commitConfiguration()
    }

    private func addCameraInput(front: Bool) {
        let position: AVCaptureDevice.Position = front ? .front : .back
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            Task { @MainActor in
                self.lastError = "無法開啟\(front ? "前" : "後")鏡頭"
            }
            return
        }
        session.addInput(input)
    }
}

#if canImport(MediaPipeTasksVision)
extension MediaPipeNativePoseEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        guard let image = try? MPImage(pixelBuffer: pixelBuffer) else { return }
        // 在 capture queue 直接送入 Landmarker；結果由 delegate 回呼。
        try? poseLandmarker?.detectAsync(image: image, timestampInMilliseconds: ts)
    }
}

extension MediaPipeNativePoseEngine: PoseLandmarkerLiveStreamDelegate {
    nonisolated func poseLandmarker(
        _ poseLandmarker: PoseLandmarker,
        didFinishDetection result: PoseLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error {
            Task { @MainActor in
                self.statusHint = "MediaPipe：\(error.localizedDescription)"
            }
            return
        }
        guard let result, let pose = result.landmarks.first else {
            Task { @MainActor in
                self.previewNodes = []
                if self.detectionActive {
                    self.adviceLines = ["畫面中尚未偵測到人物，請站到鏡頭前。"]
                }
            }
            return
        }

        let samples: [MediaPipeLandmarkMapper.LandmarkSample] = pose.map { lm in
            MediaPipeLandmarkMapper.LandmarkSample(
                x: Double(lm.x),
                y: Double(lm.y),
                z: Double(lm.z),
                visibility: Self.probability(lm.visibility),
                presence: Self.probability(lm.presence ?? lm.visibility)
            )
        }

        Task { @MainActor in
            self.handleNativeLandmarks(samples)
        }
    }
}
#endif

#if !canImport(MediaPipeTasksVision)
extension MediaPipeNativePoseEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {}
}
#endif

extension MediaPipeNativePoseEngine {
#if canImport(MediaPipeTasksVision)
    /// MediaPipe `NormalizedLandmark.visibility`／`presence` 為可選 NSNumber。
    fileprivate static func probability(_ value: NSNumber?) -> Double {
        guard let value else { return 0 }
        return Double(truncating: value)
    }
#endif

    @MainActor
    fileprivate func handleNativeLandmarks(_ samples: [MediaPipeLandmarkMapper.LandmarkSample]) {
        frameCount += 1
        let elapsed = Date().timeIntervalSince(fpsWindowStart)
        if elapsed >= 1 {
            let fps = Double(frameCount) / elapsed
            fpsText = String(format: "FPS: %.0f", fps)
            frameCount = 0
            fpsWindowStart = Date()
        }

        let rawNodes = MediaPipeLandmarkMapper.nodes(from: samples)
        let stabilized = jointConfidenceFilter.push(rawNodes)
        let preview = stabilized ?? rawNodes
        if preview.contains(where: MediaPipeSkeletonGraph.isVisible) {
            previewNodes = preview
        }

        guard detectionActive else {
            statusHint = "MediaPipe 骨架預覽中，按「開始」才會寫入資料庫"
            return
        }

        let minKeep = (usesBackViewPipeline || usesSideViewPipeline) ? 0.25 : 0.5
        let nodes = stabilized ?? rawNodes.filter { $0.confidenceScore >= minKeep }

        var finalLines: [String] = []
        var finalIssues = Set<PoseIssueCode>()

        if usesBackViewPipeline {
            let posture = PoseAdvice.evaluate(nodes: nodes)
            finalLines.append(contentsOf: posture.lines)
            finalIssues.formUnion(posture.issues)
            let backFeatures = backViewPipeline.process(nodes: nodes)
            if backFeatures.coreReliable || !backFeatures.adviceLines.isEmpty {
                finalLines.append(contentsOf: backFeatures.adviceLines)
            }
            finalIssues.formUnion(backFeatures.debouncedIssues)
            finalIssues.subtract(PoseIssueCode.sagittalPlaneCodes)
        } else {
            let posture = PoseAdvice.evaluate(nodes: nodes)
            let snapshot = MediaPipeLandmarkMapper.walkingSnapshot(from: nodes)
            let form = activityMode == .running
                ? RunningFormAdvisor.evaluate(snapshot)
                : WalkingFormAdvisor.evaluate(snapshot)
            finalLines.append(contentsOf: posture.lines)
            finalLines.append(contentsOf: form.lines)
            finalIssues.formUnion(posture.issues)
            finalIssues.formUnion(form.issues)
            if usesSideViewPipeline {
                let sideFeatures = sideViewPreprocessor.process(nodes: nodes)
                if sideFeatures.coreReliable || !sideFeatures.adviceLines.isEmpty {
                    finalLines.append(contentsOf: sideFeatures.adviceLines)
                }
                finalIssues.formUnion(sideFeatures.debouncedIssues)
            }
        }

        let store: PoseDatabase = prefersMediaPipeStore ? .mediaPipe : .shared
        store.recordFrame(nodes: nodes)
        let ts = Date().timeIntervalSince1970
        dbNodeCount += nodes.count

        let formSnapshot = MediaPipeLandmarkMapper.walkingSnapshot(from: nodes)
        if let pipeline = analysisPipeline {
            let result = pipeline.process(
                keypointFrame: MediaPipeLandmarkMapper.keypointFrame(from: nodes),
                lines: finalLines,
                issues: finalIssues,
                formSnapshot: usesBackViewPipeline ? nil : formSnapshot
            )
            adviceLines = result.lines
            stepHUD = result.hudSummary
            recentSteps = result.recentEvents
            onStepEvents?(result.emittedSteps)
        } else {
            adviceLines = finalLines.isEmpty ? ["偵測中…"] : finalLines
        }

        onStreamEnqueue?(nodes, ts)
        statusHint = ""
    }
}
