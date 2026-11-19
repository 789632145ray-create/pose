//
//  MediaPipeNativePoseEngine.swift
//  pose
//
//  真正的 Google MediaPipe Tasks Vision Pose Landmarker（不經 QuickPose）。
//  相機：liveStream；影片：VIDEO 模式逐幀 detect。兩者都寫入 mediapipe.realm。
//

import AVFoundation
import Combine
import Foundation
import QuartzCore
import SwiftUI
import UIKit

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
    /// 影片模式供 SwiftUI 預覽層綁定。
    @Published var videoPlayer: AVPlayer?
    /// MediaPipe 推理影格的直立尺寸；骨架 overlay 依此做 aspectFill 對位。
    @Published var frameContentSize: CGSize = .zero

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
    /// 影片播完時回呼（主執行緒），方便自動產生摘要。
    var onVideoPlaybackEnded: (() -> Void)?

    private enum RuntimeKind {
        case idle
        case camera
        case video
    }

    private var runtimeKind: RuntimeKind = .idle
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "pose.mediapipe.capture")
    private var frameCount = 0
    private var fpsWindowStart = Date()
    private var useFrontCamera = false

    private var videoOutputItem: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?
    private var displayLinkProxy: DisplayLinkProxy?
    private var lastVideoTimestampMs = -1
    private var endObserver: NSObjectProtocol?
    /// 影片軌 preferredTransform 對應的影像方向（與 AVPlayerLayer 顯示一致）。
    private var videoFrameOrientation: UIImage.Orientation = .up
    private var videoUprightSize: CGSize = .zero

#if canImport(MediaPipeTasksVision)
    /// 相機 liveStream；僅在 capture queue 存取。
    nonisolated(unsafe) private var liveLandmarker: PoseLandmarker?
    /// 影片 VIDEO 模式；主執行緒／display link 存取。
    private var videoLandmarker: PoseLandmarker?
#endif

    /// 給 SwiftUI 相機預覽層綁定用。
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

    // MARK: - Camera

    @MainActor
    func start(useFrontCamera: Bool) {
        tearDownVideoRuntime()
        self.useFrontCamera = useFrontCamera
        lastError = nil
        isEngineStarting = true
        statusHint = "正在啟動 MediaPipe…"
#if canImport(MediaPipeTasksVision)
        runtimeKind = .camera
        sessionQueue.async { [weak self] in
            self?.configureSessionIfNeeded()
            self?.ensureLiveLandmarker()
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
        stopDisplayLink()
        if runtimeKind == .video {
            videoPlayer?.pause()
        }
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async {
                self?.isRunning = self?.runtimeKind == .video && self?.videoPlayer != nil
                if self?.runtimeKind == .camera {
                    self?.runtimeKind = .idle
                    self?.statusHint = "MediaPipe 已停止"
                }
            }
        }
    }

    /// 離開畫面時徹底釋放相機與影片 runtime。
    @MainActor
    func shutdown() {
        detectionActive = false
        stopDisplayLink()
        tearDownVideoRuntime()
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
#if canImport(MediaPipeTasksVision)
            self?.liveLandmarker = nil
#endif
        }
        runtimeKind = .idle
        isRunning = false
        statusHint = "MediaPipe 已停止"
    }

    func switchCamera(useFrontCamera: Bool) {
        guard runtimeKind == .camera else { return }
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

    // MARK: - Video（原生 VIDEO 模式，不經 QuickPose）

    @MainActor
    func prepareVideo(url: URL) {
        stopCameraSessionOnly()
        tearDownVideoRuntime()
        lastError = nil
        lastVideoTimestampMs = -1
        previewNodes = []
        frameContentSize = .zero

#if canImport(MediaPipeTasksVision)
        guard ensureVideoLandmarker() else { return }

        let asset = AVURLAsset(url: url)
        if let track = asset.tracks(withMediaType: .video).first {
            let transform = track.preferredTransform
            let natural = track.naturalSize
            videoFrameOrientation = Self.imageOrientation(from: transform)
            videoUprightSize = Self.uprightSize(natural: natural, orientation: videoFrameOrientation)
            frameContentSize = videoUprightSize
        } else {
            videoFrameOrientation = .up
            videoUprightSize = .zero
        }

        let item = AVPlayerItem(asset: asset)
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        item.add(output)
        videoOutputItem = output

        let player = AVPlayer(playerItem: item)
        player.actionAtItemEnd = .pause
        videoPlayer = player
        runtimeKind = .video
        isRunning = true
        isEngineStarting = false
        statusHint = "MediaPipe 影片已載入（原生 Landmarker）"
        adviceLines = ["按「開始」以 Google MediaPipe 分析影片；不會經過 QuickPose。"]

        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleVideoEnded()
            }
        }
#else
        lastError = "尚未連結 MediaPipeTasksVision，無法以原生 MediaPipe 分析影片。"
        statusHint = lastError ?? ""
        adviceLines = [statusHint]
#endif
    }

    /// 開始／停止影片寫入：播放並抽幀送入 Landmarker。
    @MainActor
    func setVideoDetectionActive(_ active: Bool) {
        guard runtimeKind == .video else { return }
        detectionActive = active
        if active {
            lastVideoTimestampMs = -1
            jointConfidenceFilter.reset()
            if let player = videoPlayer {
                player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
                player.play()
            }
            startDisplayLink()
            statusHint = "MediaPipe 影片分析中"
            adviceLines = ["原生 MediaPipe 正在分析影片…"]
        } else {
            stopDisplayLink()
            videoPlayer?.pause()
        }
    }

    @MainActor
    private func handleVideoEnded() {
        stopDisplayLink()
        detectionActive = false
        statusHint = "影片播放結束"
        onVideoPlaybackEnded?()
    }

    @MainActor
    private func tearDownVideoRuntime() {
        stopDisplayLink()
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
            self.endObserver = nil
        }
        videoPlayer?.pause()
        videoPlayer = nil
        videoOutputItem = nil
        lastVideoTimestampMs = -1
        videoFrameOrientation = .up
        videoUprightSize = .zero
        if runtimeKind == .video {
            frameContentSize = .zero
        }
#if canImport(MediaPipeTasksVision)
        videoLandmarker = nil
#endif
        if runtimeKind == .video {
            runtimeKind = .idle
        }
    }

    @MainActor
    private func stopCameraSessionOnly() {
        detectionActive = false
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
#if canImport(MediaPipeTasksVision)
        sessionQueue.async { [weak self] in
            self?.liveLandmarker = nil
        }
#endif
        if runtimeKind == .camera {
            runtimeKind = .idle
        }
    }

    @MainActor
    private func startDisplayLink() {
        stopDisplayLink()
        let proxy = DisplayLinkProxy(engine: self)
        displayLinkProxy = proxy
        let link = CADisplayLink(target: proxy, selector: #selector(DisplayLinkProxy.tick))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @MainActor
    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
        displayLinkProxy = nil
    }

    @MainActor
    fileprivate func pullVideoFrameIfNeeded() {
        guard runtimeKind == .video, detectionActive else { return }
        guard let player = videoPlayer, let output = videoOutputItem else { return }
        let time = player.currentTime()
        guard time.isNumeric, time.seconds >= 0 else { return }
        let itemTime = time
        guard output.hasNewPixelBuffer(forItemTime: itemTime) else { return }
        var displayTime = CMTime.zero
        guard let pixelBuffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &displayTime) else {
            return
        }
        let ts = max(0, Int(CMTimeGetSeconds(itemTime) * 1000))
        guard ts > lastVideoTimestampMs else { return }
        lastVideoTimestampMs = ts

        if videoUprightSize == .zero {
            let raw = CGSize(
                width: CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                height: CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            )
            videoUprightSize = Self.uprightSize(natural: raw, orientation: videoFrameOrientation)
        }
        if frameContentSize != videoUprightSize, videoUprightSize.width > 0 {
            frameContentSize = videoUprightSize
        }

#if canImport(MediaPipeTasksVision)
        guard let landmarker = videoLandmarker else { return }
        // 必須帶上與 AVPlayerLayer 相同的方向，否則骨架會壓扁／偏移。
        guard let image = try? MPImage(pixelBuffer: pixelBuffer, orientation: videoFrameOrientation) else {
            return
        }
        do {
            let result = try landmarker.detect(videoFrame: image, timestampInMilliseconds: ts)
            applyLandmarkerResult(result, noPersonHint: "影片中尚未偵測到人物。")
        } catch {
            statusHint = "MediaPipe 影片：\(error.localizedDescription)"
        }
#endif
    }

    /// AVAssetTrack.preferredTransform → UIImage.Orientation（對齊播放器顯示）。
    nonisolated private static func imageOrientation(from transform: CGAffineTransform) -> UIImage.Orientation {
        let angle = atan2(transform.b, transform.a)
        let deg = angle * 180 / .pi
        switch Int(deg.rounded()) {
        case 90, -270: return .right
        case -90, 270: return .left
        case 180, -180: return .down
        default: return .up
        }
    }

    nonisolated private static func uprightSize(natural: CGSize, orientation: UIImage.Orientation) -> CGSize {
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            return CGSize(width: natural.height, height: natural.width)
        default:
            return natural
        }
    }

#if canImport(MediaPipeTasksVision)
    private func ensureLiveLandmarker() {
        guard liveLandmarker == nil else { return }
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
            liveLandmarker = try PoseLandmarker(options: options)
        } catch {
            Task { @MainActor in
                self.lastError = "MediaPipe 初始化失敗：\(error.localizedDescription)"
                self.statusHint = self.lastError ?? ""
            }
        }
    }

    @MainActor
    @discardableResult
    private func ensureVideoLandmarker() -> Bool {
        if videoLandmarker != nil { return true }
        guard let modelPath = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else {
            lastError = "找不到 pose_landmarker_full.task，請確認已加入 App Bundle"
            statusHint = lastError ?? ""
            adviceLines = [statusHint]
            return false
        }
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .video
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        do {
            videoLandmarker = try PoseLandmarker(options: options)
            return true
        } catch {
            lastError = "MediaPipe 影片初始化失敗：\(error.localizedDescription)"
            statusHint = lastError ?? ""
            adviceLines = [statusHint]
            return false
        }
    }

    @MainActor
    private func applyLandmarkerResult(_ result: PoseLandmarkerResult?, noPersonHint: String) {
        guard let result, let pose = result.landmarks.first else {
            previewNodes = []
            if detectionActive {
                adviceLines = [noPersonHint]
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
        handleNativeLandmarks(samples)
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

// MARK: - DisplayLink bridge

private final class DisplayLinkProxy: NSObject {
    weak var engine: MediaPipeNativePoseEngine?
    init(engine: MediaPipeNativePoseEngine) { self.engine = engine }
    @objc func tick() {
        Task { @MainActor in
            self.engine?.pullVideoFrameIfNeeded()
        }
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
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        guard let image = try? MPImage(pixelBuffer: pixelBuffer) else { return }
        try? liveLandmarker?.detectAsync(image: image, timestampInMilliseconds: ts)
        Task { @MainActor in
            let size = CGSize(width: width, height: height)
            if self.frameContentSize != size {
                self.frameContentSize = size
            }
        }
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
        Task { @MainActor in
            self.applyLandmarkerResult(result, noPersonHint: "畫面中尚未偵測到人物，請站到鏡頭前。")
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
    /// 相容 MediaPipe `NSNumber?` 與未來若改為 `Float?` 的 visibility／presence。
    fileprivate static func probability(_ value: Any?) -> Double {
        switch value {
        case let number as NSNumber:
            return Double(truncating: number)
        case let float as Float:
            return Double(float)
        case let double as Double:
            return double
        default:
            return 0
        }
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
