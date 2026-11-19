//
//  PoseDetectionView.swift
//  pose
//
//  完整偵測管線（MediaPipe BlazePose Full 規則建議，或自訓模型 /predict）。
//  相機權限改為「使用者按鈕後才請求」；授權前不掛載 QuickPoseCameraView。
//

import AVFoundation
import Combine
import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers
import QuickPoseCore
import QuickPoseSwiftUI

private enum PauseAdvice {
    static func lines(for activity: GaitActivityMode) -> [String] {
        [activity.coreTitle + "："] + activity.corePrincipleLines + [
            "偵測已暫停。建議你：",
            "做 2～3 次深長呼吸，放鬆肩膀與下顎。",
            "若剛才覺得站不穩，可輕輕活動踝、膝與髖，再按「開始」繼續。"
        ]
    }
}

/// QuickPose 引擎 + 影格處理（class 持有狀態，避免 onFrame 閉包捕獲 struct 導致永遠讀到舊的 detectionActive）。
@MainActor
final class QuickPoseEngine: ObservableObject {
    let pose: QuickPose

    var detectionActive = false
    private(set) var loopActive = false
    /// MediaPipe Full（`.good`）或自訓模型共用完整管線時由此指定模型。
    var assessmentEngine: PoseAssessmentEngine = .trainedModel
    /// 自訓＋MediaPipe 模型或 MediaPipe 引擎時寫入 mediapipe.realm。
    var prefersMediaPipeStore = false
    /// 僅「拍攝視角＝背面」時啟用背面特徵與分級信心門檻。
    var usesBackViewPipeline = false
    /// 僅「拍攝視角＝側面」時啟用側面前處理／特徵管線。
    var usesSideViewPipeline = false
    var activityMode: GaitActivityMode = .walking
    private var restartTask: Task<Void, Never>?

    weak var analysisPipeline: PoseAnalysisPipeline?
    var bodyGaitProfile: BodyGaitProfile?
    var onStreamEnqueue: (([PoseNode], TimeInterval) -> Void)?
    var onStepEvents: (([StepEvent]) -> Void)?
    /// 遮擋關節過濾；背面／側面用較寬手臂門檻，正面用一般 0.5。
    private let jointConfidenceFilter = JointConfidenceFilter(threshold: 0.5, coreThreshold: 0.5)
    /// 背面視角特徵管線（正規化、生物力學指標、防抖提醒）。
    let backViewPipeline = BackViewPosePipeline()
    /// 側面視角前處理器（訓練／推論共用同一套邏輯）。
    let sideViewPreprocessor = SideViewPreprocessor()

    func syncConfidenceFilterForCaptureView(_ mode: PoseCaptureViewMode) {
        usesBackViewPipeline = mode == .back
        usesSideViewPipeline = mode == .side
        switch mode {
        case .back:
            jointConfidenceFilter.threshold = 0.25
            jointConfidenceFilter.coreThreshold = 0.7
            sideViewPreprocessor.reset()
        case .side:
            // 側面以近側篩選為主；門檻略寬以保留近側肢體連續性。
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

    @Published var fpsText = "FPS: —"
    @Published var adviceLines: [String] = ["正在檢查相機權限…"]
    @Published var overlayImage: UIImage?
    @Published var previewNodes: [PoseNode] = []
    @Published var isEngineStarting = false
    @Published var statusHint = ""
    @Published var stepHUD = "步數 L:0 R:0 總:0"
    @Published var dbNodeCount = 0
    @Published var recentSteps: [StepEvent] = []
    @Published private(set) var frameCallbackCount = 0

    init() {
        let raw = (Bundle.main.object(forInfoDictionaryKey: "QuickPoseSDKKey") as? String) ?? ""
        pose = QuickPose(sdkKey: raw.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func stopLoop() {
        restartTask?.cancel()
        restartTask = nil
        if loopActive {
            pose.stop()
            loopActive = false
        }
    }

    /// 骨架列上的 QuickPose／MediaPipe 都沿用 Full（`.good`）工作階段，啟動後不重新載入輕量模型。
    func startLoopIfNeeded() {
        guard !loopActive else { return }
        loopActive = true

        let features: [QuickPose.Feature] = [.overlay(.wholeBody)]
        // 固定 Full：切換引擎不 reload 模型，骨架線才不會只在第一個引擎出現。
        let modelConfig = QuickPose.ModelConfig(
            detailedFaceTracking: false,
            detailedHandTracking: false,
            modelComplexity: .good
        )

        pose.start(features: features, modelConfig: modelConfig, onStart: { [weak self] in
            Task { @MainActor in
                self?.statusHint = ""
                self?.isEngineStarting = false
            }
        }) { [weak self] status, image, _, _, landmarks in
            Task { @MainActor in
                self?.processFrame(status: status, image: image, landmarks: landmarks)
            }
        }
    }

    /// 切換來源 / 倒數結束後：stop 再 start，等同原生模式的「重新 start」。
    func restartLoop() {
        scheduleRestart(hint: "正在重新啟動骨架…")
    }

    /// 換模型複雜度必須先 stop，稍等再 start，否則 overlay 線條不會再畫。
    func scheduleRestart(hint: String = "正在切換偵測引擎…") {
        restartTask?.cancel()
        if loopActive {
            pose.stop()
            loopActive = false
        }
        overlayImage = nil
        isEngineStarting = true
        statusHint = hint
        restartTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard let self, !Task.isCancelled else { return }
            self.startLoopIfNeeded()
        }
    }

    func processFrame(status: QuickPose.Status, image: UIImage?, landmarks: QuickPose.Landmarks?) {
        frameCallbackCount += 1

        switch status {
        case .sdkValidationError:
            detectionActive = false
            isEngineStarting = false
            overlayImage = image
            adviceLines = [
                "SDK 金鑰驗證失敗。",
                "請確認 QuickPoseSDKKey 正確，且 Bundle ID 已在 dev.quickpose.ai 設定。"
            ]
            statusHint = "SDK 驗證失敗"
            return
        default:
            break
        }

        // 骨架線要持續更新；暫停或剛切換引擎時也要畫，否則只會停在第一個模型。
        if let image {
            overlayImage = image
        }

        let rawNodes = landmarks.map { PoseNodeExtractor.extractAll(from: $0) }
        // 每幀只 push 一次：低信心關節延遲一幀插值；首幀尚無定稿時先用原始點預覽。
        let stabilized = rawNodes.flatMap { jointConfidenceFilter.push($0) }
        let preview = stabilized ?? rawNodes
        if let preview, preview.contains(where: MediaPipeSkeletonGraph.isVisible) {
            previewNodes = preview
        }

        switch status {
        case .success(let info):
            let fps = "FPS: \(info.fps)"
            fpsText = fps
            isEngineStarting = false
            guard detectionActive else {
                statusHint = "骨架預覽中，按「開始」才會寫入資料庫"
                return
            }
            if let landmarks, let pipeline = analysisPipeline {
                let posture = PoseAdvice.evaluate(from: landmarks)
                let gait = PoseGaitAdvisor.gaitAdvice(
                    from: landmarks,
                    profile: bodyGaitProfile,
                    activity: activityMode
                )
                var mergedIssues = posture.issues
                mergedIssues.formUnion(gait.issues)
                var mergedLines = posture.lines
                if !gait.lines.isEmpty {
                    mergedLines.append(contentsOf: gait.lines)
                }
                // 寫入／串流使用定稿幀（手臂等低信心關節已插值或丟棄）。
                let minKeep = (usesBackViewPipeline || usesSideViewPipeline) ? 0.25 : 0.5
                let nodes = stabilized ?? (rawNodes ?? []).filter { $0.confidenceScore >= minKeep }
                var finalLines = mergedLines
                var finalIssues = mergedIssues
                // 背面／側面特徵只在對應拍攝視角啟用。
                if usesBackViewPipeline {
                    let backFeatures = backViewPipeline.process(nodes: nodes)
                    if backFeatures.coreReliable || !backFeatures.adviceLines.isEmpty {
                        finalLines.append(contentsOf: backFeatures.adviceLines)
                    }
                    finalIssues.formUnion(backFeatures.debouncedIssues)
                } else if usesSideViewPipeline {
                    let sideFeatures = sideViewPreprocessor.process(nodes: nodes)
                    if sideFeatures.coreReliable || !sideFeatures.adviceLines.isEmpty {
                        finalLines.append(contentsOf: sideFeatures.adviceLines)
                    }
                    finalIssues.formUnion(sideFeatures.debouncedIssues)
                }
                let advice = PoseFrameAdvice(lines: finalLines, issues: finalIssues)
                let store: PoseDatabase = prefersMediaPipeStore ? .mediaPipe : .shared
                store.recordFrame(nodes: nodes)
                let ts = Date().timeIntervalSince1970
                let savedCount = nodes.count
                let result = pipeline.process(
                    landmarks: landmarks,
                    lines: advice.lines,
                    issues: advice.issues
                )
                if assessmentEngine.usesQualityPredict {
                    onStreamEnqueue?(nodes, ts)
                }
                adviceLines = result.lines
                stepHUD = result.hudSummary
                recentSteps = result.recentEvents
                statusHint = ""
                dbNodeCount += savedCount
                onStepEvents?(result.emittedSteps)
            } else {
                adviceLines = ["偵測中，尚未取得關節資料。"]
            }

        case .noPersonFound:
            isEngineStarting = false
            fpsText = "FPS: —"
            if detectionActive {
                adviceLines = ["畫面中尚未偵測到人物，請站到鏡頭前。"]
                analysisPipeline?.reset()
                jointConfidenceFilter.reset()
                backViewPipeline.reset()
                sideViewPreprocessor.reset()
            }

        case .sdkValidationError:
            break

        @unknown default:
            isEngineStarting = false
            if detectionActive {
                adviceLines = ["偵測引擎回報未知狀態，請按「暫停」後再「開始」重試。"]
            }
        }
    }

    func resetSessionUI() {
        overlayImage = nil
        previewNodes = []
        isEngineStarting = false
        dbNodeCount = 0
        recentSteps = []
        stepHUD = "步數 L:0 R:0 總:0"
        clearAnalysisBuffers()
    }

    /// 清空骨架緩衝／步數相關暫存，保留相機預覽畫面。
    func clearAnalysisBuffers() {
        jointConfidenceFilter.reset()
        backViewPipeline.reset()
        sideViewPreprocessor.reset()
        recentSteps = []
        stepHUD = "步數 L:0 R:0 總:0"
        dbNodeCount = 0
    }

    /// 暫停前吐出緩衝中最後一幀（已做信心插值）。
    func flushStabilizedNodes() -> [PoseNode]? {
        jointConfidenceFilter.flush()
    }
}

private struct PickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dst = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString + "-" + received.file.lastPathComponent)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: received.file, to: dst)
            return PickedMovie(url: dst)
        }
    }
}

enum DetectionSource: Hashable {
    case liveCamera
    case video(URL)
}

private struct PoseFrameAdvice {
    var lines: [String]
    var issues: Set<PoseIssueCode>
}

private enum PoseAdvice {
    static func evaluate(from landmarks: QuickPose.Landmarks) -> PoseFrameAdvice {
        let ls = landmarks.landmark(forBody: .shoulder(side: .left))
        let rs = landmarks.landmark(forBody: .shoulder(side: .right))
        let lh = landmarks.landmark(forBody: .hip(side: .left))
        let rh = landmarks.landmark(forBody: .hip(side: .right))
        let nose = landmarks.landmark(forBody: .nose)

        func ok(_ p: QuickPose.Point3d) -> Bool {
            p.visibility > 0.4 && p.presence > 0.4
        }

        guard ok(ls), ok(rs), ok(lh), ok(rh), ok(nose) else {
            return PoseFrameAdvice(
                lines: ["關節可見度不足，請後退一點、提高光線，或讓全身入鏡。"],
                issues: [.lowVisibility]
            )
        }

        var out: [String] = []
        var issues: Set<PoseIssueCode> = []
        let shoulderTilt = abs(ls.y - rs.y)
        let hipTilt = abs(lh.y - rh.y)
        let shoulderMidX = (ls.x + rs.x) / 2
        let hipMidX = (lh.x + rh.x) / 2
        let trunkOffset = abs(shoulderMidX - hipMidX)
        let headOffset = abs(nose.x - shoulderMidX)

        if shoulderTilt > 0.05 {
            out.append("肩膀有明顯高低差，試著放鬆並讓雙肩保持水平。")
            issues.insert(.shoulderTilt)
        } else {
            out.append("肩膀對齊良好。")
        }

        if hipTilt > 0.05 {
            out.append("骨盆有些傾斜，重心可再平均分配到雙腳。")
            issues.insert(.hipTilt)
        } else {
            out.append("骨盆穩定度不錯。")
        }

        if trunkOffset > 0.06 {
            out.append("上半身有側傾，建議收核心，讓肩膀回到髖部正上方。")
            issues.insert(.trunkSideBend)
        } else {
            out.append("軀幹中線穩定。")
        }

        if headOffset > 0.08 {
            out.append("頭部偏離身體中線，請把視線與頭部回正。")
            issues.insert(.headOffMidline)
        } else {
            out.append("頭部位置良好。")
        }

        return PoseFrameAdvice(lines: out, issues: issues)
    }
}

private enum CameraGate: Equatable {
    case checking
    case needUserTap
    case denied
    case authorized
}

struct PoseDetectionView: View {
    @EnvironmentObject private var auth: AuthManager
    @EnvironmentObject private var modeStore: DetectionModeStore

    private static var sdkKey: String {
        let raw = (Bundle.main.object(forInfoDictionaryKey: "QuickPoseSDKKey") as? String) ?? ""
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static var sdkKeyIsPlaceholder: Bool {
        let k = sdkKey.lowercased()
        return k.isEmpty || k.contains("your_sdk") || k == "your_sdk_key_here"
    }

    private static var hasCameraUsageString: Bool {
        let s = Bundle.main.object(forInfoDictionaryKey: "NSCameraUsageDescription") as? String
        return (s?.isEmpty == false)
    }

    /// 必須用 @StateObject 持有；@State 對 class 不穩定，相機 delegate 可能綁到舊實例而 onFrame 永不觸發。
    @StateObject private var quickPoseEngine = QuickPoseEngine()

    @State private var cameraGate: CameraGate = .checking
    /// 步態分析管線（影格處理由 quickPoseEngine 讀取此參考）。
    @State private var analysisPipeline = PoseAnalysisPipeline()

    @State private var isPaused = true
    @State private var countdownSeconds: Int?
    @State private var countdownTask: Task<Void, Never>?
    /// 影片模式仍用短倒數；相機改走 LiveTimedSession 狀態機。
    private static let videoCountdownSeconds = 2
    /// 實時相機：idle → preparing → analyzing → completed。
    @State private var liveSessionState: LiveSessionState = .idle
    @State private var liveSessionTask: Task<Void, Never>?
    @State private var sessionAudio = SessionAudioCues()
    /// 實時相機前後鏡頭（跑步機側面／背面建議用後鏡頭）。
    @AppStorage("poseUseFrontCamera") private var useFrontCamera = false
    @State private var engineAttached = false
    @State private var stepFlashSide: PoseFootSide?
    @State private var stepFlashOpacity: Double = 0

    @State private var detectionSource: DetectionSource = .liveCamera
    @State private var pickedItem: PhotosPickerItem?
    @State private var showPhotosPicker = false
    @State private var showFileImporter = false
    @State private var isLoadingVideo = false
    @State private var showSummary = false
    @State private var summaryLines: [String] = []
    @State private var videoLoadError: String?
    @State private var pendingVideo: PendingVideo?
    @State private var showHistory = false
    @State private var showMemberProfile = false
    @State private var showAppMenu = false
    /// 底部控制區可收起，讓骨架畫面不被擋住。
    @State private var isBottomPanelHidden = false
    @State private var lastSavedID: UUID?
    /// 本次偵測 session 是否已自動寫入歷史（避免重複儲存）。
    @State private var historySavedForSession = false
    @StateObject private var summaryStore = SummaryStore()

    /// 目前應開啟的本機節點庫（MediaPipe 引擎或自訓＋MediaPipe 模型 → mediapipe.realm）。
    private var nodeDatabase: PoseDatabase {
        usesMediaPipeDataPlane ? .mediaPipe : .shared
    }
    /// 開始 session 時鎖定的庫，避免中途切引擎寫錯檔。
    @State private var sessionIsMediaPipe = false
    private var sessionDatabase: PoseDatabase {
        sessionIsMediaPipe ? .mediaPipe : .shared
    }
    /// 目前進行中的資料庫 session id（nil 代表沒有進行中的 session）。
    @State private var dbSessionID: String?
    /// 最近一次已結束、仍可供「品質辨識與摘要」讀取的 session（暫停後 dbSessionID 會清空）。
    @State private var lastAnalyzableSessionID: String?
    @State private var lastAnalyzableIsMediaPipe = false
    /// 暫停前從串流截下的影格；Realm 讀取失敗時仍可供品質辨識。
    @State private var cachedSessionFrames: [PoseStreamClient.WireFrame] = []
    @State private var showDatabase = false

    /// 每幀即時串流上傳 + 即時預測。
    private let stream = PoseStreamClient()
    /// 累積多次即時 /predict，收斂成單一 final_label + final_confidence。
    private let predictionBuffer = LivePredictionBuffer()
    @State private var livePrediction: LivePrediction?
    @State private var streamLoopTask: Task<Void, Never>?
    /// 影片整段模型辨識結果（按「品質辨識」後由 /predict 回傳）。
    @State private var videoQualityResult: LivePrediction?
    @State private var isAnalyzingVideoQuality = false

    private struct PendingVideo: Equatable {
        let url: URL
        let displayName: String
    }

    private var currentSourceLabel: String {
        switch detectionSource {
        case .liveCamera: return "相機（即時）"
        case .video(let url): return "影片：\(url.lastPathComponent)"
        }
    }

    private var assessmentEngine: PoseAssessmentEngine {
        modeStore.engine
    }

    /// 自訓模型可再選骨架／品質模型族系：MediaPipe → mediapipe.realm + /mediapipe/predict。
    private var usesMediaPipeTrainedPipeline: Bool {
        assessmentEngine == .trainedModel && modeStore.skeletonModel == .mediaPipe
    }

    /// 寫入與預測要走 MediaPipe 管線（獨立引擎，或自訓＋MediaPipe 模型）。
    private var usesMediaPipeDataPlane: Bool {
        assessmentEngine == .mediaPipe || usesMediaPipeTrainedPipeline
    }

    private var activePredictPath: String {
        usesMediaPipeDataPlane ? "/mediapipe/predict" : assessmentEngine.predictPath
    }

    private var activePredictEngineName: String {
        usesMediaPipeDataPlane ? "mediapipe" : assessmentEngine.predictEngineName
    }

    /// 相機／影片來源切換時強制 remount，確保 onAppear → start 生命週期與原生模式一致。
    private var cameraContentID: String {
        switch detectionSource {
        case .liveCamera:
            return "live-camera-\(useFrontCamera ? "front" : "back")"
        case .video(let url):
            return "video-\(url.absoluteString)"
        }
    }

    /// 前鏡頭預覽是鏡像，節點 x 要翻轉才會對上身體；後鏡頭不翻。
    private var skeletonFlipsHorizontally: Bool {
        switch detectionSource {
        case .liveCamera:
            return useFrontCamera && !ProcessInfo.processInfo.isiOSAppOnMac
        case .video:
            return false
        }
    }

    private let chromeBackground = Color(red: 0.12, green: 0.12, blue: 0.14)

    var body: some View {
        contentWithSheets
    }

    private var contentWithSheets: some View {
        contentWithOverlays
            .sheet(isPresented: $showSummary) {
                VideoSummarySheet(
                    lines: summaryLines,
                    qualityResult: videoQualityResult,
                    isAnalyzingQuality: isAnalyzingVideoQuality,
                    showsModelQuality: assessmentEngine.usesQualityPredict,
                    autoSaved: historySavedForSession,
                    onShowHistory: {
                        showSummary = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                            showHistory = true
                        }
                    },
                    onClose: {
                        showSummary = false
                        if liveSessionState == .completed {
                            liveSessionState = .idle
                            applyPausedIdleState()
                        }
                    }
                )
            }
            .sheet(isPresented: $showHistory) {
                SummaryHistorySheet(store: summaryStore) {
                    showHistory = false
                }
            }
            .sheet(isPresented: $showDatabase) {
                PoseDatabaseSheet(database: nodeDatabase, stream: stream) {
                    showDatabase = false
                }
            }
            .sheet(isPresented: $showMemberProfile) {
                MemberProfileSheet(auth: auth) {
                    showMemberProfile = false
                }
            }
            .sheet(isPresented: $showAppMenu) {
                appMenuSheetContent
            }
    }

    private var appMenuSheetContent: some View {
        AppMenuSheet(
            onHistory: {
                showAppMenu = false
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    showHistory = true
                }
            },
            onMemberProfile: {
                showAppMenu = false
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    showMemberProfile = true
                }
            },
            onSignOut: {
                showAppMenu = false
                auth.signOut()
            }
        )
        .environmentObject(auth)
        .environmentObject(modeStore)
        .presentationDetents([.medium, .large])
    }

    private var contentWithOverlays: some View {
        contentWithLifecycle
            .overlay {
                if let pending = pendingVideo {
                    pendingConfirmOverlay(pending)
                }
            }
            .overlay {
                // 影片短倒數（舊流程）。
                if let sec = countdownSeconds, !liveSessionState.isBusy {
                    countdownOverlay(seconds: sec)
                }
            }
            .overlay {
                if case .preparing(let sec) = liveSessionState {
                    livePreparingOverlay(seconds: sec)
                }
            }
            .overlay(alignment: .top) {
                if case .analyzing(let sec) = liveSessionState {
                    liveAnalyzingBanner(remaining: sec)
                }
            }
    }

    private var contentWithLifecycle: some View {
        rootZStack
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
            .onAppear {
                refreshCameraGate()
                syncAssessmentEngine()
                syncActivityMode()
                syncCaptureViewMode()
                wireQuickPoseEngineCallbacks()
                syncBodyGaitProfile()
            }
            .onChange(of: modeStore.engine) { _, newEngine in
                handleAssessmentEngineChange(newEngine)
            }
            .onChange(of: modeStore.skeletonModel) { _, _ in
                handleSkeletonModelChange()
            }
            .onChange(of: modeStore.activity) { _, _ in
                syncActivityMode()
                if isPaused { applyPausedIdleState() }
            }
            .onChange(of: modeStore.captureView) { _, _ in
                syncCaptureViewMode()
            }
            .onChange(of: auth.userProfile) { _, _ in
                syncBodyGaitProfile()
            }
            .task(id: cameraGate) {
                guard cameraGate == .authorized, case .liveCamera = detectionSource else { return }
                bootstrapAuthorizedSession()
            }
            .task(id: detectionSource) {
                await handleDetectionSourceChange()
            }
            .onChange(of: pickedItem) { _, newItem in
                Task { await handlePickedItem(newItem) }
            }
            .photosPicker(isPresented: $showPhotosPicker, selection: $pickedItem, matching: .videos)
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .avi],
                allowsMultipleSelection: false
            ) { result in
                handleImportedFile(result)
            }
            .alert("影片載入失敗", isPresented: Binding(
                get: { videoLoadError != nil },
                set: { if !$0 { videoLoadError = nil } }
            )) {
                Button("好") { videoLoadError = nil }
            } message: {
                Text(videoLoadError ?? "")
            }
            .onDisappear {
                cancelCountdown()
                cancelLiveSessionTimer()
                sessionAudio.stopAll()
                liveSessionState = .idle
                quickPoseEngine.detectionActive = false
                quickPoseEngine.stopLoop()
                finishDatabaseSession()
                stopStreaming()
                engineAttached = false
                quickPoseEngine.resetSessionUI()
                analysisPipeline.reset()
            }
    }

    @ViewBuilder
    private var rootZStack: some View {
        ZStack {
            chromeBackground
            activeDetectionContent
        }
    }

    @ViewBuilder
    private var activeDetectionContent: some View {
        switch detectionSource {
        case .video:
            cameraAndPoseContent
        case .liveCamera:
            switch cameraGate {
            case .checking:
                VStack(spacing: 16) {
                    ProgressView()
                        .tint(.white)
                    Text("正在檢查相機設定…")
                        .foregroundStyle(.white)
                }

            case .needUserTap:
                cameraPermissionPanel

            case .denied:
                deniedPanel

            case .authorized:
                cameraAndPoseContent
            }
        }
    }

    // MARK: - 相機權限（手動點擊）

    private var cameraPermissionPanel: some View {
        VStack(spacing: 20) {
            Text("Pose偵測")
                .font(.title2.weight(.bold))
                .foregroundStyle(.orange)

#if targetEnvironment(simulator)
            Text("iOS 模擬器沒有相機，請用實機安裝此 App 才能測試相機授權與偵測。")
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.95))
                .padding(.horizontal)
#else
            Text("需要相機才能做姿勢偵測。請點下方按鈕，系統才會顯示「允許使用相機」。")
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.95))
                .padding(.horizontal)

            if !Self.hasCameraUsageString {
                Text("設定錯誤：App 內找不到 NSCameraUsageDescription，系統不會顯示授權視窗。請檢查 Xcode Target 的 Info。")
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Button {
                requestCameraAfterUserTap()
            } label: {
                Text("允許使用相機")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.accentColor)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .padding(.horizontal, 32)
            .disabled(!Self.hasCameraUsageString)

            Text("若曾拒絕過，請到「設定 → 隱私權與安全性 → 相機」開啟「Pose偵測」，或點下面開啟設定。")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button("開啟設定") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .foregroundStyle(.cyan)
#endif
            videoImportButtons(primaryLabel: "或改為上傳影片偵測")
        }
    }

    @MainActor
    private var stepEventList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(quickPoseEngine.recentSteps.suffix(5).reversed()) { event in
                let interval = event.intervalFromPrevious.map { String(format: "+%.2fs", $0) } ?? "—"
                let bpm = event.cadenceBPM.map { String(format: "%.0f bpm", $0) } ?? "—"
                let color: Color = event.side == .left ? .cyan : .pink
                Text("第 \(event.index) 步  \(event.side.localizedLabel)腳   \(interval)   \(bpm)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(color)
            }
        }
    }

    /// 底部固定高度區塊；內容可垂直滑動，避免選項變多後資料庫／選影片被裁切。
    @MainActor
    private func bottomPanel(height panelHeight: CGFloat, safeBottom: CGFloat) -> some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 8) {
                Text("向下滑動可看到資料庫與選影片")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.55))
                    .frame(maxWidth: .infinity, alignment: .center)

                DetectionEnginePickerRow()
                // 僅「自訓模型」顯示：選 QuickPose 或 MediaPipe 的自訓品質模型／資料庫。
                if assessmentEngine == .trainedModel {
                    SkeletonModelPickerRow()
                }
                GaitActivityPickerRow()
                CaptureViewPickerRow()
                bottomControls

                Button {
                    showDatabase = true
                } label: {
                    Label("\(nodeDatabase.kind.buttonTitle)（本次 \(quickPoseEngine.dbNodeCount) 筆）", systemImage: "cylinder.split.1x2.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.bordered)
                .tint(.teal)

                bottomAdviceSection
                bottomStepsSection
            }
            .padding(12)
            .padding(.bottom, max(safeBottom, 12))
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: panelHeight + safeBottom, alignment: .top)
        .background(.black.opacity(0.65))
        .animation(nil, value: quickPoseEngine.adviceLines.count)
    }

    @MainActor
    private var bottomAdviceSection: some View {
        // 外層已可滑動，這裡不再嵌套 ScrollView，避免手勢衝突。
        VStack(alignment: .leading, spacing: 4) {
            Text(assessmentEngine.adviceSectionTitle)
                .font(.headline)
                .foregroundStyle(.white)
            ForEach(Array(quickPoseEngine.adviceLines.enumerated()), id: \.offset) { _, line in
                Text("• \(line)")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.95))
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @MainActor
    private var bottomStepsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider().background(.white.opacity(0.2))
            Text("每一步")
                .font(.headline)
                .foregroundStyle(.white)
            if quickPoseEngine.recentSteps.isEmpty {
                Text("尚無步數紀錄")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.45))
            } else {
                stepEventList
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @MainActor
    @ViewBuilder
    private var bottomControls: some View {
        switch detectionSource {
        case .liveCamera:
            HStack(spacing: 10) {
                Button {
                    startLiveTimedSessionFromUser()
                } label: {
                    Label("開始", systemImage: "play.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(liveSessionState.isBusy || Self.sdkKeyIsPlaceholder || countdownSeconds != nil)

                Button {
                    cancelLiveTimedSessionFromUser()
                } label: {
                    Label("取消", systemImage: "xmark.circle.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(!liveSessionState.isBusy || Self.sdkKeyIsPlaceholder)
            }
            Button {
                guard !liveSessionState.isBusy else { return }
                useFrontCamera.toggle()
            } label: {
                Label(
                    useFrontCamera ? "目前：前鏡頭（點擊切換後鏡頭）" : "目前：後鏡頭（點擊切換前鏡頭）",
                    systemImage: "camera.rotate"
                )
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.bordered)
            .tint(.cyan)
            .disabled(liveSessionState.isBusy || ProcessInfo.processInfo.isiOSAppOnMac)
            videoImportButtons(primaryLabel: "上傳影片偵測")

        case .video:
            HStack(spacing: 10) {
                Button {
                    resumeDetectionFromUser()
                } label: {
                    Label("開始", systemImage: "play.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(!isPaused || Self.sdkKeyIsPlaceholder || countdownSeconds != nil)

                Button {
                    pauseDetectionFromUser()
                } label: {
                    Label("暫停", systemImage: "pause.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(isPaused || !engineAttached || Self.sdkKeyIsPlaceholder || countdownSeconds != nil)
            }
            HStack(spacing: 10) {
                Button {
                    analyzeVideoQualityAndShowSummary()
                } label: {
                    HStack {
                        if isAnalyzingVideoQuality { ProgressView().tint(.white) }
                        Label(
                            assessmentEngine.usesQualityPredict ? "品質辨識與摘要" : "分析摘要",
                            systemImage: assessmentEngine.usesQualityPredict ? "brain.head.profile" : "text.alignleft"
                        )
                            .font(.subheadline.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
                .disabled(isAnalyzingVideoQuality || Self.sdkKeyIsPlaceholder)

                Button {
                    switchBackToCamera()
                } label: {
                    Label("回到相機", systemImage: "camera.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.gray)
            }
            videoImportButtons(primaryLabel: "重新選擇影片")
        }
    }

    @MainActor
    private var appMenuButton: some View {
        Button {
            showAppMenu = true
        } label: {
            Image(systemName: "line.3.horizontal.circle.fill")
                .font(.title2)
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("選單")
    }

    @MainActor
    @ViewBuilder
    private func videoImportButtons(primaryLabel: String) -> some View {
        VStack(spacing: 8) {
            Text("影片在手機本機分析，不需要 Railway")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.7))
                .frame(maxWidth: .infinity, alignment: .leading)
            HStack(spacing: 10) {
                Button {
                    showPhotosPicker = true
                } label: {
                    HStack(spacing: 8) {
                        if isLoadingVideo {
                            ProgressView().tint(.white)
                        }
                        Label(primaryLabel, systemImage: "film.fill")
                            .font(.subheadline.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .disabled(isLoadingVideo)

                Button {
                    showFileImporter = true
                } label: {
                    Label("檔案", systemImage: "folder.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.bordered)
                .tint(.purple)
                .disabled(isLoadingVideo)
            }
        }
    }

    @MainActor
    private func handle(emitted: [StepEvent]) {
        guard let last = emitted.last else { return }
        stepFlashSide = last.side
        withAnimation(.easeOut(duration: 0.05)) {
            stepFlashOpacity = 0.85
        }
        withAnimation(.easeOut(duration: 0.35).delay(0.05)) {
            stepFlashOpacity = 0
        }
    }

    @MainActor
    private func resetStepUIState() {
        quickPoseEngine.stepHUD = "步數 L:0 R:0 總:0"
        quickPoseEngine.recentSteps = []
        stepFlashSide = nil
        stepFlashOpacity = 0
    }

    @MainActor
    private func wireQuickPoseEngineCallbacks() {
        syncAssessmentEngine()
        syncActivityMode()
        syncCaptureViewMode()
        quickPoseEngine.analysisPipeline = analysisPipeline
        syncBodyGaitProfile()
        quickPoseEngine.onStreamEnqueue = { [stream] nodes, ts in
            stream.enqueue(nodes: nodes, timestamp: ts)
        }
        quickPoseEngine.onStepEvents = { emitted in
            handle(emitted: emitted)
        }
    }

    @MainActor
    private func syncAssessmentEngine() {
        quickPoseEngine.assessmentEngine = assessmentEngine
        quickPoseEngine.prefersMediaPipeStore = usesMediaPipeDataPlane
    }

    @MainActor
    private func syncActivityMode() {
        let activity = modeStore.activity
        quickPoseEngine.activityMode = activity
        analysisPipeline.activityMode = activity
    }

    @MainActor
    private func syncCaptureViewMode() {
        quickPoseEngine.syncConfidenceFilterForCaptureView(modeStore.captureView)
    }

    @MainActor
    private func handleAssessmentEngineChange(_ newEngine: PoseAssessmentEngine) {
        let previous = quickPoseEngine.assessmentEngine
        syncAssessmentEngine()
        guard previous != newEngine else { return }
        livePrediction = nil
        videoQualityResult = nil
        finishDatabaseSession()
        stopStreaming()
        if engineAttached, quickPoseEngine.detectionActive {
            sessionIsMediaPipe = usesMediaPipeDataPlane
            dbSessionID = nodeDatabase.beginSession(sourceLabel: currentSourceLabel)
            quickPoseEngine.dbNodeCount = 0
            startStreaming()
        }
        if PoseAssessmentEngine.requiresPoseRuntimeRestart(from: previous, to: newEngine) {
            quickPoseEngine.scheduleRestart(hint: "正在切換\(newEngine.title)骨架…")
        } else {
            quickPoseEngine.statusHint = "已切換\(newEngine.title)，骨架持續顯示"
        }
    }

    /// 自訓模型下切換 QuickPose／MediaPipe 品質模型：換 Realm 與 /predict 路徑，不刪舊資料。
    @MainActor
    private func handleSkeletonModelChange() {
        guard assessmentEngine == .trainedModel else { return }
        syncAssessmentEngine()
        livePrediction = nil
        videoQualityResult = nil
        finishDatabaseSession()
        stopStreaming()
        if engineAttached, quickPoseEngine.detectionActive {
            sessionIsMediaPipe = usesMediaPipeDataPlane
            dbSessionID = nodeDatabase.beginSession(sourceLabel: currentSourceLabel)
            quickPoseEngine.dbNodeCount = 0
            startStreaming()
        }
        quickPoseEngine.statusHint = "已切換\(modeStore.skeletonModel.title)，按「開始」寫入對應資料庫"
    }

    @MainActor
    private func syncBodyGaitProfile() {
        let profile = BodyGaitProfile(userProfile: auth.userProfile)
        analysisPipeline.bodyProfile = profile
        quickPoseEngine.bodyGaitProfile = profile
    }

    @MainActor
    private func handlePickedItem(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        isLoadingVideo = true
        defer {
            isLoadingVideo = false
            pickedItem = nil
        }
        do {
            if let movie = try await item.loadTransferable(type: PickedMovie.self) {
                presentPendingMovie(url: movie.url, displayName: movie.url.lastPathComponent)
                return
            }
            if let data = try await item.loadTransferable(type: Data.self) {
                let name = item.itemIdentifier ?? "video.mov"
                let dst = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString + "-" + URL(fileURLWithPath: name).lastPathComponent)
                try data.write(to: dst)
                presentPendingMovie(url: dst, displayName: dst.lastPathComponent)
                return
            }
            videoLoadError = "找不到影片內容。請改用「檔案」選擇，或確認影片已下載到本機。"
        } catch {
            videoLoadError = "影片載入錯誤：\(error.localizedDescription)。可改點「檔案」從檔案 App 匯入。"
        }
    }

    @MainActor
    private func handleImportedFile(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            videoLoadError = "無法開啟檔案：\(error.localizedDescription)"
        case .success(let urls):
            guard let url = urls.first else { return }
            let accessed = url.startAccessingSecurityScopedResource()
            defer {
                if accessed { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let dst = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString + "-" + url.lastPathComponent)
                try? FileManager.default.removeItem(at: dst)
                try FileManager.default.copyItem(at: url, to: dst)
                presentPendingMovie(url: dst, displayName: url.lastPathComponent)
            } catch {
                videoLoadError = "影片複製失敗：\(error.localizedDescription)"
            }
        }
    }

    @MainActor
    private func presentPendingMovie(url: URL, displayName: String) {
        if pendingVideo != nil {
            cancelPendingVideo()
        }
        if case .video = detectionSource {
            stopDetectionActivity(fullReset: false)
        }
        pendingVideo = PendingVideo(url: url, displayName: displayName)
    }

    /// 停止資料處理與 session；切換來源或離開偵測時使用（會 stop QuickPose）。
    @MainActor
    private func stopDetectionActivity(fullReset: Bool) {
        cancelCountdown()
        quickPoseEngine.detectionActive = false
        let snap = stream.snapshotSessionFrames()
        if !snap.isEmpty {
            cachedSessionFrames = snap
        }
        finishDatabaseSession()
        stopStreaming()
        engineAttached = false
        quickPoseEngine.resetSessionUI()
        if fullReset {
            analysisPipeline.reset()
            resetStepUIState()
            livePrediction = nil
            videoQualityResult = nil
        }
    }

    /// 使用者暫停：只停資料寫入，QuickPose 引擎保持運轉（官方 demo 僅 onDisappear 才 stop）。
    @MainActor
    private func pauseDetectionProcessing(fullReset: Bool, preserveAnalysis: Bool = false) {
        cancelCountdown()
        quickPoseEngine.detectionActive = false
        // 吐出信心過濾緩衝的最後一幀再截串流影格。
        if let flushed = quickPoseEngine.flushStabilizedNodes(), !flushed.isEmpty {
            let ts = Date().timeIntervalSince1970
            sessionDatabase.recordFrame(nodes: flushed)
            if assessmentEngine.usesQualityPredict {
                stream.enqueue(nodes: flushed, timestamp: ts)
            }
        }
        // 必須在 stopStreaming／reset 前截下影格，否則品質辨識會變成「尚無節點資料」。
        let snap = stream.snapshotSessionFrames()
        if !snap.isEmpty {
            cachedSessionFrames = snap
        }
        finishDatabaseSession()
        stopStreaming()
        quickPoseEngine.isEngineStarting = false
        if fullReset {
            if !preserveAnalysis {
                analysisPipeline.reset()
                resetStepUIState()
                livePrediction = nil
                videoQualityResult = nil
            }
            quickPoseEngine.dbNodeCount = 0
        }
    }

    @MainActor
    private func confirmPendingVideo() {
        guard let pending = pendingVideo else { return }
        pendingVideo = nil
        lastSavedID = nil
        historySavedForSession = false
        stopDetectionActivity(fullReset: true)
        quickPoseEngine.fpsText = "FPS: —"
        quickPoseEngine.adviceLines = ["正在載入影片…"]
        detectionSource = .video(pending.url)
        isPaused = true
        beginVideoDetectionCountdown()
    }

    @MainActor
    private func cancelPendingVideo() {
        if let url = pendingVideo?.url {
            try? FileManager.default.removeItem(at: url)
        }
        pendingVideo = nil
    }

    @MainActor
    private func buildSummaryLinesForHistory(includeQuality: Bool = true) -> [String] {
        var lines = analysisPipeline.videoSummary()
        if modeStore.isBackView {
            lines.append(contentsOf: quickPoseEngine.backViewPipeline.sessionSummaryLines())
        } else if modeStore.isSideView {
            lines.append(contentsOf: quickPoseEngine.sideViewPreprocessor.sessionSummaryLines())
        }
        guard includeQuality, assessmentEngine.usesQualityPredict else { return lines }
        if let pred = videoQualityResult ?? livePrediction,
           pred.note == nil,
           let label = pred.label {
            let pct = Int(pred.confidencePercent.rounded())
            let qualityLine = "模型辨識：姿勢「\(label == "good" ? "好" : "壞")」（信心 \(pct)%）"
            lines.removeAll { $0.hasPrefix("模型辨識：") }
            lines.insert(qualityLine, at: 0)
        }
        return lines
    }

    /// 偵測／分析完成後自動寫入歷史紀錄（同一 session 只存一次）。
    @MainActor
    private func autoSaveSummaryToHistory() {
        guard !historySavedForSession else { return }
        let lines = summaryLines.isEmpty ? buildSummaryLinesForHistory() : summaryLines
        guard !lines.isEmpty else { return }
        let summary = SavedSummary(
            id: UUID(),
            date: Date(),
            sourceLabel: currentSourceLabel,
            totalSteps: analysisPipeline.totalSteps,
            leftSteps: analysisPipeline.leftSteps,
            rightSteps: analysisPipeline.rightSteps,
            avgCadenceBPM: analysisPipeline.avgCadenceBPM,
            lines: lines
        )
        summaryStore.add(summary)
        lastSavedID = summary.id
        historySavedForSession = true
    }

    @MainActor
    private func saveCurrentSummary() {
        autoSaveSummaryToHistory()
    }

    /// 結束目前資料庫 session，把最終步態統計一併寫入。
    @MainActor
    private func finishDatabaseSession() {
        guard let id = dbSessionID else { return }
        // 節點已落盤；保留 id 供暫停後「品質辨識與摘要」讀取。
        lastAnalyzableSessionID = id
        lastAnalyzableIsMediaPipe = sessionIsMediaPipe
        sessionDatabase.endSession(
            totalSteps: analysisPipeline.totalSteps,
            leftSteps: analysisPipeline.leftSteps,
            rightSteps: analysisPipeline.rightSteps,
            avgCadenceBPM: analysisPipeline.avgCadenceBPM
        )
        dbSessionID = nil
    }

    private var analyzableDatabase: PoseDatabase {
        if dbSessionID != nil {
            return sessionDatabase
        }
        return lastAnalyzableIsMediaPipe ? .mediaPipe : .shared
    }

    /// 開始一段即時串流（開後端 session + 啟動每 0.5 秒批次上傳與每秒預測的迴圈）。
    /// 相機與影片共用 `/predict`；若 `/sessions` 開不起來，仍退回本機視窗預測（與影片模式一致），不再整段中止成「未連線」。
    @MainActor
    private func startStreaming() {
        livePrediction = nil
        predictionBuffer.reset()
        guard assessmentEngine.usesQualityPredict else { return }
        let source = currentSourceLabel
        let preferUpload = assessmentEngine.usesTrainedModelPredict
        stream.predictPath = activePredictPath
        stream.predictEngine = activePredictEngineName
        streamLoopTask?.cancel()
        streamLoopTask = Task {
            var uploadLiveFrames = preferUpload
            if preferUpload {
                let ok = await stream.begin(mode: .predict, sourceLabel: source)
                if !ok {
                    // 憑證／伺服器仍可能可打 /predict（影片模式就是這條路）。
                    stream.preparePredictWindow()
                    uploadLiveFrames = false
                    await MainActor.run {
                        livePrediction = LivePrediction(
                            label: nil,
                            probabilityGood: 0,
                            note: "雲端 session 未開，改用本機預測"
                        )
                    }
                }
            } else {
                stream.preparePredictWindow()
            }
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if Task.isCancelled { break }
                if uploadLiveFrames {
                    await stream.flush()
                }
                tick += 1
                if tick % 2 == 0 { // 約每秒預測一次
                    if let pred = await stream.predictIfReady() {
                        await MainActor.run {
                            // 單窗預測先寫入緩衝，再以加權投票收斂，避免結果跳動。
                            if let aggregated = predictionBuffer.append(pred) {
                                let stable = aggregated.asLivePrediction()
                                livePrediction = stable
                                if case .video = detectionSource {
                                    videoQualityResult = stable
                                }
                            } else {
                                livePrediction = pred
                            }
                        }
                    }
                }
            }
        }
    }

    /// 結束串流：停止迴圈並回報統計。
    /// - Parameter preserveDisplayPrediction: 完成採樣後保留 HUD／摘要用的預測結果，不要被清掉。
    @MainActor
    private func stopStreaming(preserveDisplayPrediction: Bool = false) {
        streamLoopTask?.cancel()
        streamLoopTask = nil
        let total = analysisPipeline.totalSteps
        let left = analysisPipeline.leftSteps
        let right = analysisPipeline.rightSteps
        let bpm = analysisPipeline.avgCadenceBPM
        Task { await stream.end(totalSteps: total, leftSteps: left, rightSteps: right, avgCadenceBPM: bpm) }
        if !preserveDisplayPrediction {
            livePrediction = nil
            predictionBuffer.reset()
        }
    }

    @MainActor
    private func pendingConfirmOverlay(_ pending: PendingVideo) -> some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 16) {
                Image(systemName: "film.stack")
                    .font(.system(size: 44))
                    .foregroundStyle(.purple)
                Text("確認分析這支影片")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.white)
                Text(pending.displayName)
                    .font(.callout.monospaced())
                    .foregroundStyle(.white.opacity(0.85))
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Text("確認後倒數 \(Self.videoCountdownSeconds) 秒開始偵測，可隨時切回相機。")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                HStack(spacing: 12) {
                    Button(role: .cancel) {
                        cancelPendingVideo()
                    } label: {
                        Text("取消")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.bordered)
                    .tint(.gray)

                    Button {
                        confirmPendingVideo()
                    } label: {
                        Label("開始分析", systemImage: "play.fill")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 24)
            .frame(maxWidth: 360)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
            .padding(28)
        }
        .transition(.opacity)
    }

    @MainActor
    private func switchBackToCamera() {
        stopDetectionActivity(fullReset: true)
        isPaused = true
        quickPoseEngine.fpsText = "FPS: —（已暫停）"
        cancelLiveSessionTimer()
        sessionAudio.stopAll()
        liveSessionState = .idle
        quickPoseEngine.adviceLines = [
            "偵測已暫停。請按「開始」：先準備 \(LiveTimedSessionConfig.prepareSeconds) 秒，再採樣 \(LiveTimedSessionConfig.analyzeSeconds) 秒。"
        ]
        detectionSource = .liveCamera
        pickedItem = nil
    }

    @MainActor
    private func handleDetectionSourceChange() async {
        guard !Self.sdkKeyIsPlaceholder else { return }
        guard countdownSeconds == nil, !liveSessionState.isBusy, !quickPoseEngine.isEngineStarting else { return }
        if isPaused {
            if !engineAttached { applyPausedIdleState() }
            return
        }
        guard engineAttached else { return }
        if case .video = detectionSource {
            try? await Task.sleep(nanoseconds: 300_000_000)
        } else {
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        attachQuickPoseEngine(resuming: true)
    }

    @MainActor
    private func analyzeVideoQualityAndShowSummary() {
        // 僅在仍有評估幀時覆寫摘要；暫停後 pipeline 已 reset，不可用「尚未取得…」蓋掉既有摘要。
        if analysisPipeline.hasEvaluatedFrames || summaryLines.isEmpty {
            summaryLines = buildSummaryLinesForHistory(includeQuality: false)
        }

        guard assessmentEngine.usesQualityPredict else {
            videoQualityResult = nil
            isAnalyzingVideoQuality = false
            showSummary = true
            autoSaveSummaryToHistory()
            return
        }

        let fallbackQuality = videoQualityResult ?? livePrediction
        var wireFrames = cachedSessionFrames
        if wireFrames.isEmpty {
            wireFrames = stream.snapshotSessionFrames()
            if !wireFrames.isEmpty { cachedSessionFrames = wireFrames }
        }
        let sessionID = dbSessionID ?? lastAnalyzableSessionID
        let database = analyzableDatabase

        stream.predictPath = activePredictPath
        stream.predictEngine = activePredictEngineName
        videoQualityResult = nil
        isAnalyzingVideoQuality = true
        showSummary = true

        Task {
            if wireFrames.isEmpty, let sessionID {
                try? await Task.sleep(nanoseconds: 150_000_000)
                let records = JointConfidenceFilter.interpolateSession(
                    database.allFrames(sessionID: sessionID)
                )
                let result = await stream.predict(frames: records)
                await MainActor.run {
                    applyVideoQualityResult(result, fallback: fallbackQuality)
                }
                return
            }
            let result: LivePrediction?
            if wireFrames.isEmpty {
                result = fallbackQuality
                    ?? LivePrediction(label: nil, probabilityGood: 0, note: "尚無節點資料")
            } else {
                result = await stream.predict(frames: wireFrames)
            }
            await MainActor.run {
                applyVideoQualityResult(result, fallback: fallbackQuality)
            }
        }
    }

    @MainActor
    private func applyVideoQualityResult(_ result: LivePrediction?, fallback: LivePrediction?) {
        let resolved: LivePrediction?
        if let result, result.note == nil {
            resolved = result
        } else if let fallback, fallback.note == nil {
            resolved = fallback
        } else {
            resolved = result ?? fallback
        }
        videoQualityResult = resolved
        isAnalyzingVideoQuality = false
        // 摘要卡片與條列必須同一筆結果；覆寫先前即時預測寫入的「模型辨識」行。
        summaryLines.removeAll { $0.hasPrefix("模型辨識：") }
        if let resolved, resolved.note == nil, let label = resolved.label {
            let pct = Int(resolved.confidencePercent.rounded())
            let qualityLine = "模型辨識：姿勢「\(label == "good" ? "好" : "壞")」（信心 \(pct)%）"
            summaryLines.insert(qualityLine, at: 0)
        }
        // 若歷史已先存過舊信心度，更新同一筆摘要。
        if historySavedForSession, let id = lastSavedID {
            summaryStore.updateLines(id: id, lines: summaryLines.isEmpty ? buildSummaryLinesForHistory() : summaryLines)
        } else {
            autoSaveSummaryToHistory()
        }
    }

    /// 影片暫停後用已截下的影格做品質辨識，並打開摘要。
    @MainActor
    private func runQualityAnalysisAfterCapture(fallback: LivePrediction?) {
        guard assessmentEngine.usesQualityPredict else {
            showSummary = true
            autoSaveSummaryToHistory()
            return
        }
        stream.predictPath = activePredictPath
        stream.predictEngine = activePredictEngineName
        videoQualityResult = nil
        isAnalyzingVideoQuality = true
        showSummary = true

        let wireFrames = cachedSessionFrames
        let sessionID = lastAnalyzableSessionID
        let database = analyzableDatabase

        Task {
            var result: LivePrediction?
            if !wireFrames.isEmpty {
                result = await stream.predict(frames: wireFrames)
            } else if let sessionID {
                try? await Task.sleep(nanoseconds: 150_000_000)
                let records = JointConfidenceFilter.interpolateSession(
                    database.allFrames(sessionID: sessionID)
                )
                result = await stream.predict(frames: records)
            } else {
                result = fallback
                    ?? LivePrediction(label: nil, probabilityGood: 0, note: "尚無節點資料")
            }
            await MainActor.run {
                applyVideoQualityResult(result, fallback: fallback)
                quickPoseEngine.adviceLines = ["偵測已暫停，摘要已存入歷史紀錄。"] + PauseAdvice.lines(for: modeStore.activity)
            }
        }
    }

    private var deniedPanel: some View {
        VStack(spacing: 20) {
            Text("相機已被關閉")
                .font(.title2.weight(.bold))
                .foregroundStyle(.white)
            Text("請到「設定 → 隱私權與安全性 → 相機」，開啟「Pose偵測」。")
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.9))
                .padding(.horizontal)
            Button("開啟設定") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
            videoImportButtons(primaryLabel: "或改為上傳影片偵測")
        }
    }

    private func refreshCameraGate() {
#if targetEnvironment(simulator)
        quickPoseEngine.adviceLines = ["iOS 模擬器沒有相機，請用實機執行本 App。"]
        quickPoseEngine.statusHint = "請使用實機"
        cameraGate = .needUserTap
        return
#endif

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraGate = .authorized
        case .notDetermined:
            cameraGate = .needUserTap
            quickPoseEngine.adviceLines = ["請點「允許使用相機」，再於系統對話框選「允許」。"]
        case .denied, .restricted:
            cameraGate = .denied
            quickPoseEngine.adviceLines = ["相機權限已被關閉，請到設定中開啟。"]
        @unknown default:
            cameraGate = .needUserTap
        }
    }

    private func requestCameraAfterUserTap() {
        quickPoseEngine.adviceLines = ["正在等待系統回應…"]
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    cameraGate = .authorized
                } else {
                    cameraGate = .denied
                    quickPoseEngine.adviceLines = ["已拒絕相機。請到設定中開啟，或再試一次。"]
                }
            }
        }
    }

    // MARK: - 偵測畫面（授權後）

    private var cameraAndPoseContent: some View {
        GeometryReader { geometry in
            let w = max(geometry.size.width, 1)
            let h = max(geometry.size.height, 1)
            let bottomPanelHeight = cameraBottomPanelHeight(for: h)
            let safeBottom = geometry.safeAreaInsets.bottom

            cameraPreviewStack(width: w, height: h)
                .overlay(alignment: .top) {
                    stepFlashOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .top) {
                    poseHUDOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .bottom) {
                    bottomChrome(height: bottomPanelHeight, safeBottom: safeBottom)
                }
                .overlay(alignment: .bottomTrailing) {
                    quickPoseVersionLabel
                        .padding(.bottom, isBottomPanelHidden ? 36 : 0)
                }
        }
    }

    /// 底部選單；「隱藏」膠囊置中貼在選單頂邊（你標的白線位置），版面內不裁切。
    @MainActor
    private func bottomChrome(height panelHeight: CGFloat, safeBottom: CGFloat) -> some View {
        VStack(spacing: 8) {
            bottomPanelToggle
                .frame(maxWidth: .infinity, alignment: .center)

            if !isBottomPanelHidden {
                bottomPanel(height: panelHeight, safeBottom: safeBottom)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, isBottomPanelHidden ? max(safeBottom, 12) : 0)
        .animation(.easeInOut(duration: 0.25), value: isBottomPanelHidden)
    }

    @MainActor
    private var bottomPanelToggle: some View {
        Button {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            withAnimation(.easeInOut(duration: 0.25)) {
                isBottomPanelHidden.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isBottomPanelHidden ? "chevron.up" : "chevron.down")
                    .font(.caption.weight(.bold))
                Text(isBottomPanelHidden ? "顯示控制" : "隱藏")
                    .font(.caption.weight(.bold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.black.opacity(0.78), in: Capsule())
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.28), lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isBottomPanelHidden ? "顯示下方偵測模型選單" : "隱藏下方偵測模型選單")
        .accessibilityHint("收起或展開偵測引擎與骨架模型選單")
    }

    private func cameraBottomPanelHeight(for height: CGFloat) -> CGFloat {
        let ratio: CGFloat = {
            switch detectionSource {
            case .video: return 0.52
            case .liveCamera: return 0.48
            }
        }()
        // 選項變多後靠 ScrollView 捲動；高度給足可見區，其餘滑下去找資料庫／選影片。
        return min(max(height * ratio, 360), 520)
    }

    @ViewBuilder
    private func cameraPreviewStack(width w: CGFloat, height h: CGFloat) -> some View {
        ZStack(alignment: .top) {
            cameraSourceView(width: w, height: h)

            QuickPoseOverlayView(overlayImage: $quickPoseEngine.overlayImage, contentMode: .fill)
                .frame(width: w, height: h)
                .opacity(quickPoseEngine.overlayImage == nil ? 0 : 1)
                .allowsHitTesting(false)

            if assessmentEngine == .mediaPipe {
                MediaPipeSkeletonOverlay(
                    nodes: quickPoseEngine.previewNodes,
                    flipHorizontally: skeletonFlipsHorizontally
                )
                .frame(width: w, height: h)
                .allowsHitTesting(false)
            }
        }
        .id(cameraContentID)
        .frame(width: w, height: h)
        .onAppear {
            wireQuickPoseEngineCallbacks()
            quickPoseEngine.restartLoop()
        }
    }

    @ViewBuilder
    private func cameraSourceView(width w: CGFloat, height h: CGFloat) -> some View {
        switch detectionSource {
        case .video(let url):
            QuickPoseSimulatedCameraView(useFrontCamera: false, delegate: quickPoseEngine.pose, video: url)
                .frame(width: w, height: h)
        case .liveCamera:
            if ProcessInfo.processInfo.isiOSAppOnMac,
               let url = Bundle.main.url(forResource: "happy-dance", withExtension: "mov") {
                QuickPoseSimulatedCameraView(useFrontCamera: false, delegate: quickPoseEngine.pose, video: url)
                    .frame(width: w, height: h)
            } else {
                QuickPoseCameraView(
                    useFrontCamera: useFrontCamera,
                    delegate: quickPoseEngine.pose,
                    videoGravity: .resizeAspectFill
                )
                .frame(width: w, height: h)
                .clipped()
            }
        }
    }

    @ViewBuilder
    private func stepFlashOverlay(safeTop: CGFloat) -> some View {
        if let side = stepFlashSide {
            Rectangle()
                .fill(side == .left ? Color.cyan : Color.pink)
                .frame(height: 6)
                .opacity(stepFlashOpacity)
                .padding(.top, safeTop)
                .allowsHitTesting(false)
        }
    }

    private func poseHUDOverlay(safeTop: CGFloat) -> some View {
        HStack(alignment: .top, spacing: 10) {
            poseStatusBadge
            Spacer(minLength: 8)
            if case .liveCamera = detectionSource, !ProcessInfo.processInfo.isiOSAppOnMac {
                Button {
                    guard !liveSessionState.isBusy else { return }
                    useFrontCamera.toggle()
                } label: {
                    Image(systemName: "camera.rotate.fill")
                        .font(.body.weight(.bold))
                        .foregroundStyle(.white)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(liveSessionState.isBusy)
                .accessibilityLabel(useFrontCamera ? "切換為後鏡頭" : "切換為前鏡頭")
            }
            // 右上角改為選單（影片仍可從下方「上傳影片偵測」選）。
            appMenuButton
        }
        .padding(.horizontal, 12)
        // GeometryReader 在 ignoresSafeArea 下 safeTop 可能是 0；右上角又是控制中心熱區。
        .padding(.top, max(safeTop, 54) + 6)
        .zIndex(2)
    }

    private var poseStatusBadge: some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 2) {
                Text(
                    assessmentEngine == .trainedModel
                        ? "\(assessmentEngine.hudBadgeTitle) · \(modeStore.skeletonModel.shortTitle)"
                        : assessmentEngine.hudBadgeTitle
                )
                    .font(.caption.weight(.bold))
                    .foregroundStyle(usesMediaPipeDataPlane ? .mint : .orange)
                Text(modeStore.activity.coreTitle)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(modeStore.activity == .running ? .orange : .cyan)
                if modeStore.isBackView {
                    Text("拍攝視角 · 背面")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.mint)
                } else if modeStore.isSideView {
                    Text("拍攝視角 · 側面")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.orange)
                }
                if case .liveCamera = detectionSource, !ProcessInfo.processInfo.isiOSAppOnMac {
                    Text(useFrontCamera ? "鏡頭 · 前" : "鏡頭 · 後")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.cyan)
                }
            }
            Text(quickPoseEngine.fpsText)
                .font(.system(size: 16, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
            if !quickPoseEngine.statusHint.isEmpty {
                Text(quickPoseEngine.statusHint)
                    .font(.caption)
                    .foregroundStyle(.yellow)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(quickPoseEngine.stepHUD)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.mint)
                .fixedSize(horizontal: false, vertical: true)
            Text("資料庫節點：\(quickPoseEngine.dbNodeCount)")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.teal)
            if usesMediaPipeDataPlane {
                Text("寫入 \(MediaPipeRealm.fileName)")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.mint)
            }
            if quickPoseEngine.loopActive, quickPoseEngine.frameCallbackCount == 0 {
                Text("等待第一幀…")
                    .font(.caption2)
                    .foregroundStyle(.yellow)
            } else if quickPoseEngine.frameCallbackCount > 0 {
                Text("引擎回呼 \(quickPoseEngine.frameCallbackCount) 次")
                    .font(.caption2)
                    .foregroundStyle(.green.opacity(0.85))
            }
            qualityBadgeIfNeeded
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var qualityBadgeIfNeeded: some View {
        if case .video = detectionSource {
            Text(assessmentEngine == .mediaPipe ? "MediaPipe 影片偵測中" : "影片偵測中")
                .font(.caption2.weight(.bold))
                .foregroundStyle(.purple)
            if assessmentEngine.usesQualityPredict, let result = videoQualityResult ?? livePrediction {
                videoQualityBadge(result, compact: true)
            }
        } else if assessmentEngine.usesQualityPredict, let pred = livePrediction {
            videoQualityBadge(pred, compact: true)
        }
    }

    private var quickPoseVersionLabel: some View {
        VStack(alignment: .trailing, spacing: 2) {
            if assessmentEngine == .mediaPipe {
                Text("MediaPipe \(quickPoseEngine.pose.modelWeight())")
                    .font(.caption2)
                    .foregroundStyle(.mint.opacity(0.9))
            }
            Text("QuickPose v\(quickPoseEngine.pose.quickPoseVersion())")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(8)
    }

    // MARK: - QuickPose

    @MainActor
    private func bootstrapAuthorizedSession() {
        quickPoseEngine.statusHint = ""

        if Self.sdkKeyIsPlaceholder {
            quickPoseEngine.adviceLines = [
                "相機已授權，但尚未設定有效的 QuickPoseSDKKey。",
                "請在 Info.plist 將 QuickPoseSDKKey 改成你在 https://dev.quickpose.ai 申請的金鑰，",
                "並確認 Bundle ID（此專案為 my-first-app.pose）已在後台綁定。"
            ]
            quickPoseEngine.statusHint = "請設定 SDK Key"
            engineAttached = false
            return
        }

        applyPausedIdleState()
    }

    @MainActor
    private func applyPausedIdleState() {
        guard isPaused, !engineAttached, !quickPoseEngine.isEngineStarting else { return }
        guard countdownSeconds == nil, !liveSessionState.isBusy else { return }
        quickPoseEngine.fpsText = "FPS: —（已暫停）"
        switch detectionSource {
        case .liveCamera:
            quickPoseEngine.adviceLines = [
                "\(assessmentEngine.hudBadgeTitle) 待機中。按「開始」後先準備 \(LiveTimedSessionConfig.prepareSeconds) 秒（不上傳數據），再自動採樣 \(LiveTimedSessionConfig.analyzeSeconds) 秒。"
            ] + modeStore.activity.corePrincipleLines
        case .video:
            quickPoseEngine.adviceLines = [
                "影片已載入（\(assessmentEngine.hudBadgeTitle)）。請按「開始」，倒數 \(Self.videoCountdownSeconds) 秒後開始偵測。"
            ] + modeStore.activity.corePrincipleLines
        }
    }

    @MainActor
    private func resumeDetectionFromUser() {
        guard !Self.sdkKeyIsPlaceholder else { return }
        guard countdownSeconds == nil, !liveSessionState.isBusy else { return }
        // 僅影片模式走短倒數；相機改用定時狀態機。
        guard case .video = detectionSource else {
            startLiveTimedSessionFromUser()
            return
        }
        beginVideoDetectionCountdown()
    }

    /// 影片短倒數結束後 attach QuickPose。
    @MainActor
    private func beginVideoDetectionCountdown() {
        cancelCountdown()
        quickPoseEngine.adviceLines = ["準備開始偵測…"]
        isPaused = true
        let seconds = Self.videoCountdownSeconds

        countdownTask = Task {
            for sec in stride(from: seconds, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                await MainActor.run { countdownSeconds = sec }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }
            try? await Task.sleep(nanoseconds: 300_000_000)
            await MainActor.run {
                countdownSeconds = nil
                countdownTask = nil
                isPaused = false
                attachQuickPoseEngine(resuming: engineAttached)
            }
        }
    }

    @MainActor
    private func cancelCountdown() {
        countdownTask?.cancel()
        countdownTask = nil
        countdownSeconds = nil
    }

    @MainActor
    private func cancelLiveSessionTimer() {
        liveSessionTask?.cancel()
        liveSessionTask = nil
    }

    /// 相機：清空緩衝 → 準備倒數 → 定時採樣 → 自動完成並顯示結果。
    @MainActor
    private func startLiveTimedSessionFromUser() {
        guard !Self.sdkKeyIsPlaceholder else { return }
        guard case .liveCamera = detectionSource else { return }
        guard !liveSessionState.isBusy, countdownSeconds == nil else { return }

        cancelCountdown()
        cancelLiveSessionTimer()
        sessionAudio.stopAll()

        // 進入 preparing 前徹底清空上一次暫存，避免污染。
        clearLiveSessionBuffersForFreshStart()

        isPaused = true
        engineAttached = false
        quickPoseEngine.detectionActive = false
        quickPoseEngine.startLoopIfNeeded()
        quickPoseEngine.fpsText = "FPS: 準備中…"
        quickPoseEngine.adviceLines = [
            "請上跑步機並調整至跑步速度。準備 \(LiveTimedSessionConfig.prepareSeconds) 秒後自動開始採樣。"
        ]
        quickPoseEngine.statusHint = "準備中，不收集姿勢數據"

        let prepareTotal = LiveTimedSessionConfig.prepareSeconds
        liveSessionState = .preparing(remaining: prepareTotal)
        sessionAudio.announcePreparing(seconds: prepareTotal)

        liveSessionTask = Task { @MainActor in
            for sec in stride(from: prepareTotal, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                liveSessionState = .preparing(remaining: sec)
                if sec <= 3 {
                    sessionAudio.playTickBeep()
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }
            enterLiveAnalyzingPhase()
        }
    }

    /// 正式採樣：僅此階段將骨架餵給特徵／模型。
    @MainActor
    private func enterLiveAnalyzingPhase() {
        // 再清一次，確保準備階段完全沒有殘留寫入。
        analysisPipeline.reset()
        resetStepUIState()
        quickPoseEngine.clearAnalysisBuffers()
        predictionBuffer.reset()
        livePrediction = nil
        videoQualityResult = nil
        cachedSessionFrames = []
        historySavedForSession = false
        lastSavedID = nil

        isPaused = false
        quickPoseEngine.detectionActive = true
        engineAttached = true
        quickPoseEngine.isEngineStarting = false
        // 準備階段已在預覽，勿 restart 造成黑屏空窗；只確保 loop 仍在跑。
        quickPoseEngine.startLoopIfNeeded()
        quickPoseEngine.fpsText = "FPS: 分析中…"
        quickPoseEngine.statusHint = "分析中，請維持穩定跑姿"
        quickPoseEngine.adviceLines = ["開始分析：請維持穩定跑姿 \(LiveTimedSessionConfig.analyzeSeconds) 秒。"]

        finishDatabaseSession()
        stopStreaming(preserveDisplayPrediction: true)
        sessionIsMediaPipe = usesMediaPipeDataPlane
        dbSessionID = nodeDatabase.beginSession(sourceLabel: currentSourceLabel)
        startStreaming()

        let analyzeTotal = LiveTimedSessionConfig.analyzeSeconds
        liveSessionState = .analyzing(remaining: analyzeTotal)
        sessionAudio.announceAnalyzing()

        liveSessionTask = Task { @MainActor in
            for sec in stride(from: analyzeTotal, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                liveSessionState = .analyzing(remaining: sec)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }
            completeLiveTimedSession()
        }
    }

    /// 採樣時間到：立刻鎖定數據，聚合結果並顯示摘要。
    @MainActor
    private func completeLiveTimedSession() {
        cancelLiveSessionTimer()

        // 瞬間鎖定：之後的減速／下機畫面不得再寫入。
        quickPoseEngine.detectionActive = false

        if let flushed = quickPoseEngine.flushStabilizedNodes(), !flushed.isEmpty {
            let ts = Date().timeIntervalSince1970
            sessionDatabase.recordFrame(nodes: flushed)
            if assessmentEngine.usesQualityPredict {
                stream.enqueue(nodes: flushed, timestamp: ts)
            }
        }

        let snap = stream.snapshotSessionFrames()
        if !snap.isEmpty {
            cachedSessionFrames = snap
        }

        let lockedLive = predictionBuffer.currentAggregate()?.asLivePrediction() ?? livePrediction

        summaryLines = buildSummaryLinesForHistory(includeQuality: false)

        finishDatabaseSession()
        stopStreaming(preserveDisplayPrediction: true)

        if let locked = lockedLive {
            livePrediction = locked
            videoQualityResult = locked
        }

        isPaused = true
        engineAttached = false
        liveSessionState = .completed
        quickPoseEngine.fpsText = "FPS: —（採樣完成）"
        quickPoseEngine.statusHint = ""
        quickPoseEngine.adviceLines = ["分析完成，可以減速了。正在產生報告…"]

        sessionAudio.announceCompleted()

        runQualityAnalysisAfterCapture(fallback: lockedLive)
    }

    /// 準備／分析中取消：清空暫存並回 idle，不產生報告。
    @MainActor
    private func cancelLiveTimedSessionFromUser() {
        guard liveSessionState.isBusy else { return }
        cancelLiveSessionTimer()
        sessionAudio.stopAll()
        cancelCountdown()

        quickPoseEngine.detectionActive = false
        finishDatabaseSession()
        stopStreaming()
        analysisPipeline.reset()
        resetStepUIState()
        quickPoseEngine.clearAnalysisBuffers()
        livePrediction = nil
        videoQualityResult = nil
        summaryLines = []
        cachedSessionFrames = []
        historySavedForSession = false
        lastSavedID = nil

        isPaused = true
        engineAttached = false
        liveSessionState = .idle
        quickPoseEngine.fpsText = "FPS: —（已取消）"
        quickPoseEngine.statusHint = ""
        quickPoseEngine.adviceLines = [
            "已取消並清空暫存數據。按「開始」可重新準備 \(LiveTimedSessionConfig.prepareSeconds) 秒後採樣。"
        ] + modeStore.activity.corePrincipleLines
    }

    /// 開始新 session 前清空骨架緩衝、步數與模型預測陣列。
    @MainActor
    private func clearLiveSessionBuffersForFreshStart() {
        finishDatabaseSession()
        stopStreaming()
        analysisPipeline.reset()
        resetStepUIState()
        quickPoseEngine.clearAnalysisBuffers()
        predictionBuffer.reset()
        livePrediction = nil
        videoQualityResult = nil
        summaryLines = []
        cachedSessionFrames = []
        historySavedForSession = false
        lastSavedID = nil
        quickPoseEngine.dbNodeCount = 0
    }

    @MainActor
    private func pauseDetectionFromUser() {
        cancelCountdown()
        // 相機定時流程用「取消」；此處僅服務影片模式暫停。
        guard case .video = detectionSource else {
            if liveSessionState.isBusy {
                cancelLiveTimedSessionFromUser()
            }
            return
        }

        // 影片模式：暫停即自動產生並顯示分析摘要（先截資料再 teardown）。
        if engineAttached {
            // 先不要寫入即時預測信心度；等整段品質辨識完成後再統一寫入，避免卡片與條列不一致。
            if analysisPipeline.hasEvaluatedFrames {
                summaryLines = buildSummaryLinesForHistory(includeQuality: false)
            }
            let fallbackQuality = videoQualityResult ?? livePrediction
            pauseDetectionProcessing(fullReset: true, preserveAnalysis: true)
            isPaused = true
            quickPoseEngine.fpsText = "FPS: —（已暫停）"
            quickPoseEngine.adviceLines = ["偵測已暫停，正在產生分析摘要…"]
            quickPoseEngine.statusHint = ""
            runQualityAnalysisAfterCapture(fallback: fallbackQuality)
            return
        }

        pauseDetectionProcessing(fullReset: true)
        isPaused = true
        quickPoseEngine.fpsText = "FPS: —（已暫停）"
        quickPoseEngine.adviceLines = PauseAdvice.lines(for: modeStore.activity)
        quickPoseEngine.statusHint = ""
    }

    @MainActor
    private func attachQuickPoseEngine(resuming: Bool) {
        if Self.sdkKeyIsPlaceholder { return }

        wireQuickPoseEngineCallbacks()
        syncAssessmentEngine()

        isPaused = false
        quickPoseEngine.detectionActive = true
        engineAttached = true
        quickPoseEngine.isEngineStarting = true
        quickPoseEngine.fpsText = "FPS: 偵測中…"

        finishDatabaseSession()
        stopStreaming()
        analysisPipeline.reset()
        resetStepUIState()

        historySavedForSession = false
        lastSavedID = nil
        quickPoseEngine.dbNodeCount = 0
        cachedSessionFrames = []
        sessionIsMediaPipe = usesMediaPipeDataPlane
        dbSessionID = nodeDatabase.beginSession(sourceLabel: currentSourceLabel)
        startStreaming()

        quickPoseEngine.adviceLines = [resuming ? "正在恢復偵測…" : "正在啟動偵測…"]
        quickPoseEngine.statusHint = ""

        quickPoseEngine.restartLoop()
    }

    @MainActor
    @ViewBuilder
    private func videoQualityBadge(_ result: LivePrediction, compact: Bool) -> some View {
        if let note = result.note {
            Text(compact ? "品質：\(note)" : note)
                .font(compact ? .caption.weight(.semibold) : .body)
                .foregroundStyle(.yellow)
        } else if let label = result.label {
            let isGood = label == "good"
            let pct = Int(result.confidencePercent.rounded())
            if compact {
                Text("品質：\(isGood ? "好" : "壞") \(pct)%")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(isGood ? .green : .red)
            } else {
                HStack(spacing: 8) {
                    Image(systemName: isGood ? "hand.thumbsup.fill" : "hand.thumbsdown.fill")
                    Text(isGood ? "好" : "壞")
                        .font(.title.weight(.bold))
                    Text("\(pct)%")
                        .font(.title3.weight(.semibold))
                }
                .foregroundStyle(isGood ? .green : .red)
            }
        }
    }

    @MainActor
    private func countdownOverlay(seconds: Int) -> some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 12) {
                Text("\(seconds)")
                    .font(.system(size: 88, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText())
                Text("秒後開始偵測")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.9))
            }
        }
        .allowsHitTesting(true)
    }

    /// 準備階段：遠距離可視的特大倒數（固定 LiveTimedSessionConfig.prepareSeconds，不是舊的 5 秒）。
    @MainActor
    private func livePreparingOverlay(seconds: Int) -> some View {
        ZStack {
            Color.black.opacity(0.62).ignoresSafeArea()
            VStack(spacing: 18) {
                Text("準備倒數 \(LiveTimedSessionConfig.prepareSeconds) 秒")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.yellow)
                Text("\(seconds)")
                    .font(.system(size: 120, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText())
                    .monospacedDigit()
                Text("請上跑步機並調整至跑步速度")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                Text("此階段不收集姿勢數據 · 倒數結束後才採樣 \(LiveTimedSessionConfig.analyzeSeconds) 秒")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.75))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
                Button {
                    cancelLiveTimedSessionFromUser()
                } label: {
                    Label("取消 / 重設", systemImage: "xmark.circle.fill")
                        .font(.headline.weight(.bold))
                        .frame(maxWidth: 280)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .padding(.top, 12)
            }
        }
        .allowsHitTesting(true)
        .accessibilityElement(children: .contain)
    }

    /// 分析階段：頂部高對比進度＋剩餘秒數（骨架仍由底層繪製）。
    @MainActor
    private func liveAnalyzingBanner(remaining: Int) -> some View {
        let total = Double(LiveTimedSessionConfig.analyzeSeconds)
        let progress = total > 0 ? Double(LiveTimedSessionConfig.analyzeSeconds - remaining) / total : 0
        return VStack(spacing: 0) {
            HStack(spacing: 12) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 14, height: 14)
                        .shadow(color: .red.opacity(0.8), radius: 4)
                    Text("分析中")
                        .font(.title3.weight(.black))
                        .foregroundStyle(.white)
                }
                Spacer(minLength: 8)
                Text("剩餘 \(remaining) 秒")
                    .font(.system(size: 28, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                Button {
                    cancelLiveTimedSessionFromUser()
                } label: {
                    Text("取消")
                        .font(.subheadline.weight(.bold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.9), in: Capsule())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.top, 56)
            .padding(.bottom, 10)

            ProgressView(value: min(max(progress, 0), 1))
                .progressViewStyle(.linear)
                .tint(.red)
                .scaleEffect(x: 1, y: 2, anchor: .center)
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
        }
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0.78), Color.black.opacity(0.35), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .allowsHitTesting(true)
    }
}

// MARK: - 摘要列圖示（達成／需改善／中性資訊）

private enum SummaryLineTone {
    case positive
    case warning
    case neutral

    var systemImage: String {
        switch self {
        case .positive: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .neutral: return "info.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .positive: return .green
        case .warning: return .orange
        case .neutral: return .secondary
        }
    }

    /// 依文案判斷：只有明確「達標／符合」才打勾；需改善／請注意用警告；步數、核心說明等用 info。
    static func tone(for line: String) -> SummaryLineTone {
        let text = line.trimmingCharacters(in: .whitespacesAndNewlines)

        // 先判達成，避免「沒有明顯需要改善」被「改善」誤判成警告。
        let positiveHints = [
            "整體姿勢穩定",
            "沒有明顯需要改善",
            "符合正確",
            "核心指標穩定",
            "背面核心指標穩定",
            "側面核心指標穩定",
            "穩定，符合",
            "bpm 符合",
            "合理的安全步態",
            "合理的自我保護",
            "bpm 活潑",
            "姿勢「好」",
        ]
        if positiveHints.contains(where: { text.contains($0) }) {
            return .positive
        }

        let warningHints = [
            "需注意", "偏大", "不足", "持續異常", "異常：",
            "請", "偏低", "略高", "差距偏大",
            "可針對上述", "改善程度",
            "尚未取得", "無法", "失敗",
            "前傾過大", "過度跨步", "拖步", "左右晃",
            "姿勢「壞」",
        ]
        if warningHints.contains(where: { text.contains($0) }) {
            return .warning
        }
        // 「約 N% 時段…」多半是問題佔比，原則說明不會用這個句型。
        if text.contains("約"), text.contains("%"), text.contains("時段") {
            return .warning
        }
        return .neutral
    }
}

private struct SummaryAdviceRow: View {
    let line: String

    var body: some View {
        let tone = SummaryLineTone.tone(for: line)
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: tone.systemImage)
                .foregroundStyle(tone.color)
            Text(line)
                .font(.body)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityLabel("\(tone == .warning ? "需注意" : tone == .positive ? "達標" : "資訊")：\(line)")
    }
}

// MARK: - 影片分析摘要 Sheet

private struct VideoSummarySheet: View {
    let lines: [String]
    let qualityResult: LivePrediction?
    let isAnalyzingQuality: Bool
    var showsModelQuality: Bool = true
    let autoSaved: Bool
    let onShowHistory: () -> Void
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("影片分析摘要")
                        .font(.title2.weight(.bold))

                    if autoSaved {
                        Label("已自動存入歷史紀錄", systemImage: "checkmark.seal.fill")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.green)
                            .padding(.vertical, 4)
                    }

                    if showsModelQuality {
                        qualityResultCard
                    }

                    ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                        SummaryAdviceRow(line: line)
                    }
                    Text("註：暫停影片時會自動產生摘要並寫入歷史紀錄。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.top, 6)

                    Button {
                        onShowHistory()
                    } label: {
                        Label("查看歷史紀錄", systemImage: "clock.arrow.circlepath")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
                    .padding(.top, 6)
                }
                .padding(20)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成", action: onClose)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    @ViewBuilder
    private var qualityResultCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("姿勢品質（模型辨識）")
                .font(.headline)
            if isAnalyzingQuality {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("正在用訓練模型分析影片…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 8)
            } else if let result = qualityResult {
                if let note = result.note {
                    Label(note, systemImage: "exclamationmark.triangle.fill")
                        .font(.subheadline)
                        .foregroundStyle(.orange)
                } else if let label = result.label {
                    let isGood = label == "good"
                    HStack(spacing: 16) {
                        Image(systemName: isGood ? "checkmark.seal.fill" : "xmark.seal.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(isGood ? .green : .red)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(isGood ? "姿勢：好" : "姿勢：壞")
                                .font(.title.weight(.bold))
                                .foregroundStyle(isGood ? .green : .red)
                            Text("模型信心度：\(Int(result.confidencePercent.rounded()))%")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("尚無辨識結果")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - 歷史紀錄 Sheet

private struct SummaryHistorySheet: View {
    @ObservedObject var store: SummaryStore
    let onClose: () -> Void
    @State private var showClearConfirm = false

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "zh_Hant_TW")
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        NavigationStack {
            Group {
                if store.items.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "tray")
                            .font(.system(size: 40))
                            .foregroundStyle(.secondary)
                        Text("還沒有任何歷史紀錄")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Text("完成影片品質辨識，或相機偵測後按「暫停」，摘要會自動存入這裡。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(store.items) { item in
                            NavigationLink {
                                summaryDetail(item)
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(Self.dateFormatter.string(from: item.date))
                                        .font(.subheadline.weight(.semibold))
                                    Text(item.sourceLabel)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    HStack(spacing: 12) {
                                        Label("\(item.totalSteps) 步", systemImage: "figure.walk")
                                        if let bpm = item.avgCadenceBPM {
                                            Label(String(format: "%.0f bpm", bpm), systemImage: "metronome")
                                        }
                                        Label("L \(item.leftSteps) / R \(item.rightSteps)", systemImage: "arrow.left.arrow.right")
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .onDelete { offsets in
                            store.remove(at: offsets)
                        }
                    }
                }
            }
            .navigationTitle("歷史紀錄")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !store.items.isEmpty {
                        Button(role: .destructive) {
                            showClearConfirm = true
                        } label: {
                            Text("清空")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成", action: onClose)
                }
            }
            .alert("確認清空歷史紀錄？", isPresented: $showClearConfirm) {
                Button("取消", role: .cancel) {}
                Button("清空", role: .destructive) {
                    store.clear()
                }
            } message: {
                Text("將刪除全部歷史摘要，此操作無法復原。")
            }
        }
    }

    private func summaryDetail(_ item: SavedSummary) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(Self.dateFormatter.string(from: item.date))
                    .font(.headline)
                Text(item.sourceLabel)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Divider()
                ForEach(Array(item.lines.enumerated()), id: \.offset) { _, line in
                    SummaryAdviceRow(line: line)
                }
            }
            .padding(20)
        }
        .navigationTitle("摘要詳情")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - 姿勢節點資料庫 Sheet

private struct PoseDatabaseSheet: View {
    let database: PoseDatabase
    let stream: PoseStreamClient
    let onClose: () -> Void

    @State private var records: [PoseSessionRecord] = []
    @State private var totalNodes: Int = 0
    @State private var exportURL: URL?
    @State private var exportError: String?
    @State private var showClearConfirm = false

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "zh_Hant_TW")
        f.dateStyle = .short
        f.timeStyle = .medium
        return f
    }()

    var body: some View {
        NavigationStack {
            Group {
                if records.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "cylinder.split.1x2")
                            .font(.system(size: 40))
                            .foregroundStyle(.secondary)
                        Text(database.kind == .mediaPipe ? "MediaPipe 資料庫尚無節點" : "資料庫尚無任何節點")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Text(database.kind.emptyHint)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                        Text(database.fileURL.lastPathComponent)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        Section {
                            LabeledContent("偵測次數", value: "\(records.count)")
                            LabeledContent("節點總筆數", value: "\(totalNodes)")
                            LabeledContent("本機檔案", value: database.fileURL.lastPathComponent)
                        } header: {
                            Text("總覽")
                        }

                        Section("各次偵測") {
                            ForEach(records) { rec in
                                NavigationLink {
                                    PoseSessionDetailView(database: database, stream: stream, record: rec) {
                                        reload()
                                    }
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Text(Self.dateFormatter.string(from: rec.startedAt))
                                                .font(.subheadline.weight(.semibold))
                                            Spacer()
                                            if let label = rec.trainingLabel {
                                                Text(label == "good" ? "好" : "壞")
                                                    .font(.caption2.weight(.bold))
                                                    .padding(.horizontal, 8)
                                                    .padding(.vertical, 2)
                                                    .background(label == "good" ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                                                    .foregroundStyle(label == "good" ? .green : .red)
                                                    .clipShape(Capsule())
                                            }
                                        }
                                        Text(rec.sourceLabel)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                        HStack(spacing: 12) {
                                            Label("\(rec.frameCount) 幀", systemImage: "film")
                                            Label("\(rec.nodeCount) 節點", systemImage: "point.3.connected.trianglepath.dotted")
                                            Label("\(rec.totalSteps) 步", systemImage: "figure.walk")
                                        }
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle(database.kind.sheetTitle)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !records.isEmpty {
                        Button(role: .destructive) {
                            showClearConfirm = true
                        } label: {
                            Text("清空")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 12) {
                        if records.contains(where: { $0.trainingLabel == "good" || $0.trainingLabel == "bad" }) {
                            Button("匯出 JSON") {
                                do {
                                    exportError = nil
                                    exportURL = try database.exportLabeledTrainingJSON()
                                } catch {
                                    exportError = error.localizedDescription
                                }
                            }
                        }
                        Button("完成", action: onClose)
                    }
                }
            }
            .alert("確認清空資料庫？", isPresented: $showClearConfirm) {
                Button("取消", role: .cancel) {}
                Button("清空", role: .destructive) {
                    database.clearAll()
                    reload()
                }
            } message: {
                Text("將刪除全部姿勢節點紀錄，此操作無法復原。")
            }
            .sheet(isPresented: Binding(
                get: { exportURL != nil },
                set: { if !$0 { exportURL = nil } }
            )) {
                if let exportURL {
                    NavigationStack {
                        ShareLink(item: exportURL, preview: SharePreview(exportURL.lastPathComponent)) {
                            Label("分享到 Mac / 檔案", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .padding()
                        .navigationTitle("匯出訓練 JSON")
                        .navigationBarTitleDisplayMode(.inline)
                        .toolbar {
                            ToolbarItem(placement: .topBarTrailing) {
                                Button("關閉") { self.exportURL = nil }
                            }
                        }
                    }
                    .presentationDetents([.medium])
                }
            }
            .alert("匯出失敗", isPresented: Binding(
                get: { exportError != nil },
                set: { if !$0 { exportError = nil } }
            )) {
                Button("好", role: .cancel) {}
            } message: {
                Text(exportError ?? "")
            }
            .onAppear { reload() }
        }
    }

    private func reload() {
        records = database.sessions()
        totalNodes = database.totalNodeCount()
    }
}

// MARK: - 單次偵測詳情（本機暫存資料瀏覽）

private struct PoseSessionDetailView: View {
    let database: PoseDatabase
    let stream: PoseStreamClient
    let record: PoseSessionRecord
    let onUpdated: () -> Void

    @State private var isUploading = false
    @State private var statusMessage: String?
    @State private var isUploadError = false
    @State private var trainingLabel: String?

    var body: some View {
        let nodes = database.firstFrameNodes(sessionID: record.id)
        let canLabel = record.frameCount > 0 || record.nodeCount > 0 || !nodes.isEmpty
        return List {
            Section {
                LabeledContent("開始時間", value: PoseSessionDetailView.dateFormatter.string(from: record.startedAt))
                LabeledContent("來源", value: record.sourceLabel)
                LabeledContent("影格數", value: "\(record.frameCount)")
                LabeledContent("節點筆數", value: "\(record.nodeCount)")
                LabeledContent("步數", value: "總 \(record.totalSteps)（左 \(record.leftSteps) / 右 \(record.rightSteps)）")
                if let bpm = record.avgCadenceBPM {
                    LabeledContent("平均步頻", value: String(format: "%.0f bpm", bpm))
                }
                if let label = trainingLabel ?? record.trainingLabel {
                    LabeledContent("訓練標籤", value: label == "good" ? "好" : "壞")
                }
            } header: {
                Text("此次偵測")
            } footer: {
                Text(database.kind == .mediaPipe
                     ? "MediaPipe 節點存在本機 mediapipe.realm。標好／壞後可上傳雲端，或在資料庫按「匯出 JSON」，到 Mac 跑 python train.py --engine mediapipe --from-json 檔案.json。"
                     : "在此標記好 / 壞並上傳至雲端，或匯出 JSON 給 train.py --from-json。")
            }

            if canLabel {
                Section("訓練標記（上傳至雲端）") {
                    Button {
                        uploadLabel("good")
                    } label: {
                        HStack {
                            if isUploading { ProgressView() }
                            Label("標為「好」並上傳", systemImage: "hand.thumbsup.fill")
                        }
                    }
                    .disabled(isUploading)

                    Button {
                        uploadLabel("bad")
                    } label: {
                        HStack {
                            if isUploading { ProgressView() }
                            Label("標為「壞」並上傳", systemImage: "hand.thumbsdown.fill")
                        }
                    }
                    .disabled(isUploading)
                }
            } else {
                Section {
                    Text("尚無節點可標記。請先開始偵測，暫停後再打開資料庫。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            if let statusMessage {
                Section {
                    Text(statusMessage)
                        .font(.footnote)
                        .foregroundStyle(isUploadError ? .red : .green)
                }
            }

            Section("第一幀節點樣本（共 \(nodes.count) 個）") {
                ForEach(Array(nodes.enumerated()), id: \.offset) { _, n in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(n.joint)
                            .font(.subheadline.weight(.semibold))
                        Text(String(format: "x %.3f  y %.3f  z %.3f", n.x, n.y, n.z))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Text(String(format: "可見度 %.2f · 存在度 %.2f", n.visibility, n.presence))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("偵測詳情")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { trainingLabel = record.trainingLabel }
    }

    private func uploadLabel(_ label: String) {
        statusMessage = nil
        let frames = JointConfidenceFilter.interpolateSession(
            database.allFrames(sessionID: record.id)
        )
        guard !frames.isEmpty else {
            isUploadError = true
            statusMessage = "此筆尚無節點。請回到偵測畫面按「暫停」後再上傳。"
            return
        }
        isUploading = true
        Task {
            let result = await stream.uploadLabeledSession(
                label: label,
                sourceLabel: record.sourceLabel,
                totalSteps: record.totalSteps,
                leftSteps: record.leftSteps,
                rightSteps: record.rightSteps,
                avgCadenceBPM: record.avgCadenceBPM,
                frames: frames,
                path: database.kind.uploadPath
            )
            await MainActor.run {
                isUploading = false
                isUploadError = !result.success
                trainingLabel = label
                database.markTrainingUpload(sessionID: record.id, label: label, uploadedToCloud: result.success)
                onUpdated()
                if result.success {
                    statusMessage = result.message
                } else {
                    statusMessage = "本機已標記，但沒傳到雲端：\(result.message)。可回資料庫按「匯出 JSON」，到 Mac 用 train.py --from-json 訓練。"
                }
            }
        }
    }

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "zh_Hant_TW")
        f.dateStyle = .short
        f.timeStyle = .medium
        return f
    }()
}
