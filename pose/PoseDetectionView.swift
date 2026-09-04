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
    static let lines: [String] = [
        "偵測已暫停。建議你：",
        "做 2～3 次深長呼吸，放鬆肩膀與下顎。",
        "若剛才覺得站不穩，可輕輕活動踝、膝與髖，再按「開始」繼續。"
    ]
}

/// QuickPose 引擎 + 影格處理（class 持有狀態，避免 onFrame 閉包捕獲 struct 導致永遠讀到舊的 detectionActive）。
@MainActor
final class QuickPoseEngine: ObservableObject {
    let pose: QuickPose

    var detectionActive = false
    private(set) var loopActive = false
    /// MediaPipe Full（`.good`）或自訓模型共用完整管線時由此指定模型。
    var assessmentEngine: PoseAssessmentEngine = .trainedModel

    weak var analysisPipeline: PoseAnalysisPipeline?
    var bodyGaitProfile: BodyGaitProfile?
    var onStreamEnqueue: (([PoseNode], TimeInterval) -> Void)?
    var onStepEvents: (([StepEvent]) -> Void)?

    @Published var fpsText = "FPS: —"
    @Published var adviceLines: [String] = ["正在檢查相機權限…"]
    @Published var overlayImage: UIImage?
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
        if loopActive {
            pose.stop()
            loopActive = false
        }
    }

    /// MediaPipe 使用 BlazePose Full（`.good`）；自訓模型同樣用 Full 骨架再送後端。
    func startLoopIfNeeded() {
        guard !loopActive else { return }
        loopActive = true

        let features: [QuickPose.Feature]
        let modelConfig: QuickPose.ModelConfig
        switch assessmentEngine {
        case .mediaPipe:
            features = [.overlay(.wholeBody), .showPoints()]
            modelConfig = QuickPose.ModelConfig(
                detailedFaceTracking: false,
                detailedHandTracking: false,
                modelComplexity: .good
            )
        case .trainedModel:
            features = [.overlay(.wholeBody)]
            modelConfig = QuickPose.ModelConfig(
                detailedFaceTracking: false,
                detailedHandTracking: false,
                modelComplexity: .good
            )
        case .quickPose:
            features = [.overlay(.wholeBody)]
            modelConfig = QuickPose.ModelConfig()
        }

        pose.start(features: features, modelConfig: modelConfig) { [weak self] status, image, _, _, landmarks in
            Task { @MainActor in
                self?.processFrame(status: status, image: image, landmarks: landmarks)
            }
        }
    }

    /// 切換來源 / 倒數結束後：stop 再 start，等同原生模式的「重新 start」。
    func restartLoop() {
        stopLoop()
        startLoopIfNeeded()
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

        guard detectionActive else { return }

        let img = image
        switch status {
        case .success(let info):
            let fps = "FPS: \(info.fps)"
            if let landmarks, let pipeline = analysisPipeline {
                let posture = PoseAdvice.evaluate(from: landmarks)
                let gait = PoseGaitAdvisor.gaitAdvice(from: landmarks, profile: bodyGaitProfile)
                var mergedIssues = posture.issues
                mergedIssues.formUnion(gait.issues)
                var mergedLines = posture.lines
                if !gait.lines.isEmpty {
                    mergedLines.append(contentsOf: gait.lines)
                }
                let advice = PoseFrameAdvice(lines: mergedLines, issues: mergedIssues)
                let nodes = PoseNodeExtractor.extractAll(from: landmarks)
                PoseDatabase.shared.recordFrame(nodes: nodes)
                let ts = Date().timeIntervalSince1970
                let savedCount = nodes.count
                let result = pipeline.process(
                    landmarks: landmarks,
                    lines: advice.lines,
                    issues: advice.issues
                )
                isEngineStarting = false
                if assessmentEngine.usesTrainedModelPredict {
                    onStreamEnqueue?(nodes, ts)
                }
                overlayImage = img
                fpsText = fps
                adviceLines = result.lines
                stepHUD = result.hudSummary
                recentSteps = result.recentEvents
                statusHint = ""
                dbNodeCount += savedCount
                onStepEvents?(result.emittedSteps)
            } else {
                isEngineStarting = false
                overlayImage = img
                fpsText = fps
                adviceLines = ["偵測中，尚未取得關節資料。"]
            }

        case .noPersonFound:
            isEngineStarting = false
            overlayImage = img
            fpsText = "FPS: —"
            adviceLines = ["畫面中尚未偵測到人物，請站到鏡頭前。"]
            analysisPipeline?.reset()

        case .sdkValidationError:
            break

        @unknown default:
            isEngineStarting = false
            overlayImage = img
            adviceLines = ["偵測引擎回報未知狀態，請按「暫停」後再「開始」重試。"]
        }
    }

    func resetSessionUI() {
        overlayImage = nil
        isEngineStarting = false
        dbNodeCount = 0
        recentSteps = []
        stepHUD = "步數 L:0 R:0 總:0"
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
    private static let cameraCountdownSeconds = 5
    private static let videoCountdownSeconds = 2
    @State private var engineAttached = false
    @State private var stepFlashSide: PoseFootSide?
    @State private var stepFlashOpacity: Double = 0

    @State private var detectionSource: DetectionSource = .liveCamera
    @State private var pickedItem: PhotosPickerItem?
    @State private var isLoadingVideo = false
    @State private var showSummary = false
    @State private var summaryLines: [String] = []
    @State private var videoLoadError: String?
    @State private var pendingVideo: PendingVideo?
    @State private var showHistory = false
    @State private var showMemberProfile = false
    @State private var showAppMenu = false
    @State private var lastSavedID: UUID?
    /// 本次偵測 session 是否已自動寫入歷史（避免重複儲存）。
    @State private var historySavedForSession = false
    @StateObject private var summaryStore = SummaryStore()

    /// 姿勢節點資料庫；偵測期間每一幀的全部節點都寫入此處。
    private let database = PoseDatabase.shared
    /// 目前進行中的資料庫 session id（nil 代表沒有進行中的 session）。
    @State private var dbSessionID: String?
    @State private var showDatabase = false

    /// 每幀即時串流上傳 + 即時預測。
    private let stream = PoseStreamClient()
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

    /// 相機／影片來源切換時強制 remount，確保 onAppear → start 生命週期與原生模式一致。
    private var cameraContentID: String {
        switch detectionSource {
        case .liveCamera: return "live-camera"
        case .video(let url): return "video-\(url.absoluteString)"
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
                    showsModelQuality: assessmentEngine.usesTrainedModelPredict,
                    autoSaved: historySavedForSession,
                    onShowHistory: {
                        showSummary = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                            showHistory = true
                        }
                    },
                    onClose: { showSummary = false }
                )
            }
            .sheet(isPresented: $showHistory) {
                SummaryHistorySheet(store: summaryStore) {
                    showHistory = false
                }
            }
            .sheet(isPresented: $showDatabase) {
                PoseDatabaseSheet(database: database, stream: stream) {
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
                if let sec = countdownSeconds {
                    countdownOverlay(seconds: sec)
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
                wireQuickPoseEngineCallbacks()
                syncBodyGaitProfile()
            }
            .onChange(of: modeStore.engine) { _, newEngine in
                handleAssessmentEngineChange(newEngine)
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
            uploadVideoButton(label: "或改為上傳影片偵測")
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

    /// 底部固定高度區塊，避免偵測中文字變多時整片介面上下跳動。
    @MainActor
    private func bottomPanel(height panelHeight: CGFloat, safeBottom: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            DetectionEnginePickerRow()
            bottomControls

            Button {
                showDatabase = true
            } label: {
                Label("姿勢節點資料庫（本次 \(quickPoseEngine.dbNodeCount) 筆）", systemImage: "cylinder.split.1x2.fill")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.bordered)
            .tint(.teal)

            Spacer(minLength: 0)

            bottomAdviceSection
            bottomStepsSection
        }
        .padding(12)
        .padding(.bottom, safeBottom)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: panelHeight + safeBottom, alignment: .top)
        .background(.black.opacity(0.65))
        .animation(nil, value: quickPoseEngine.adviceLines.count)
    }

    @MainActor
    private var bottomAdviceSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(assessmentEngine.adviceSectionTitle)
                .font(.headline)
                .foregroundStyle(.white)
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 4) {
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
        }
        .frame(height: 88, alignment: .top)
    }

    @MainActor
    private var bottomStepsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider().background(.white.opacity(0.2))
            Text("每一步")
                .font(.headline)
                .foregroundStyle(.white)
            ScrollView(.vertical, showsIndicators: false) {
                Group {
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
        }
        .frame(height: 100, alignment: .top)
    }

    @MainActor
    @ViewBuilder
    private var bottomControls: some View {
        switch detectionSource {
        case .liveCamera:
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
            uploadVideoButton(label: "上傳影片偵測")
                .frame(maxWidth: .infinity)

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
                            assessmentEngine.usesTrainedModelPredict ? "品質辨識與摘要" : "分析摘要",
                            systemImage: assessmentEngine.usesTrainedModelPredict ? "brain.head.profile" : "text.alignleft"
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
            uploadVideoButton(label: "重新選擇影片")
                .frame(maxWidth: .infinity)
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
                .padding(8)
                .background(.black.opacity(0.45), in: Circle())
        }
        .buttonStyle(.plain)
    }

    @MainActor
    @ViewBuilder
    private func uploadVideoButton(label: String) -> some View {
        PhotosPicker(selection: $pickedItem, matching: .videos, photoLibrary: .shared()) {
            HStack(spacing: 8) {
                if isLoadingVideo {
                    ProgressView().tint(.white)
                }
                Label(label, systemImage: "film.fill")
                    .font(.subheadline.weight(.semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
        .tint(.purple)
        .disabled(isLoadingVideo || Self.sdkKeyIsPlaceholder)
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
    }

    @MainActor
    private func handleAssessmentEngineChange(_ newEngine: PoseAssessmentEngine) {
        guard newEngine.usesFullDetectionPipeline else { return }
        let previous = quickPoseEngine.assessmentEngine
        syncAssessmentEngine()
        guard previous != newEngine else { return }
        livePrediction = nil
        videoQualityResult = nil
        if quickPoseEngine.loopActive {
            quickPoseEngine.restartLoop()
        }
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
                await MainActor.run {
                    if pendingVideo != nil {
                        cancelPendingVideo()
                    }
                    if case .video = detectionSource {
                        stopDetectionActivity(fullReset: false)
                    }
                    pendingVideo = PendingVideo(url: movie.url, displayName: movie.url.lastPathComponent)
                }
            } else {
                videoLoadError = "找不到影片內容，請重新選擇。"
            }
        } catch {
            videoLoadError = "影片載入錯誤：\(error.localizedDescription)"
        }
    }

    /// 停止資料處理與 session；切換來源或離開偵測時使用（會 stop QuickPose）。
    @MainActor
    private func stopDetectionActivity(fullReset: Bool) {
        cancelCountdown()
        quickPoseEngine.detectionActive = false
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
    private func pauseDetectionProcessing(fullReset: Bool) {
        cancelCountdown()
        quickPoseEngine.detectionActive = false
        finishDatabaseSession()
        stopStreaming()
        quickPoseEngine.isEngineStarting = false
        quickPoseEngine.overlayImage = nil
        if fullReset {
            analysisPipeline.reset()
            resetStepUIState()
            livePrediction = nil
            videoQualityResult = nil
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
        beginDetectionCountdown(isVideo: true)
    }

    @MainActor
    private func cancelPendingVideo() {
        if let url = pendingVideo?.url {
            try? FileManager.default.removeItem(at: url)
        }
        pendingVideo = nil
    }

    @MainActor
    private func buildSummaryLinesForHistory() -> [String] {
        var lines = analysisPipeline.videoSummary()
        if assessmentEngine.usesTrainedModelPredict,
           let pred = videoQualityResult ?? livePrediction,
           pred.note == nil,
           let label = pred.label {
            let pct = Int(pred.confidencePercent)
            let qualityLine = "模型辨識：姿勢「\(label == "good" ? "好" : "壞")」（信心 \(pct)%）"
            if !lines.contains(where: { $0.hasPrefix("模型辨識：") }) {
                lines.insert(qualityLine, at: 0)
            }
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
        guard dbSessionID != nil else { return }
        database.endSession(
            totalSteps: analysisPipeline.totalSteps,
            leftSteps: analysisPipeline.leftSteps,
            rightSteps: analysisPipeline.rightSteps,
            avgCadenceBPM: analysisPipeline.avgCadenceBPM
        )
        dbSessionID = nil
    }

    /// 開始一段即時串流（開後端 session + 啟動每 0.5 秒批次上傳與每秒預測的迴圈）。
    @MainActor
    private func startStreaming() {
        livePrediction = nil
        guard assessmentEngine.usesTrainedModelPredict else { return }
        let source = currentSourceLabel
        streamLoopTask?.cancel()
        streamLoopTask = Task {
            let ok = await stream.begin(mode: .predict, sourceLabel: source)
            guard ok else {
                await MainActor.run { livePrediction = LivePrediction(label: nil, probabilityGood: 0, note: "未連線/未登入") }
                return
            }
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if Task.isCancelled { break }
                await stream.flush()
                tick += 1
                if tick % 2 == 0 { // 約每秒預測一次
                    if let pred = await stream.predictIfReady() {
                        await MainActor.run {
                            livePrediction = pred
                            if case .video = detectionSource, pred.note == nil {
                                videoQualityResult = pred
                            }
                        }
                    }
                }
            }
        }
    }

    /// 結束串流：停止迴圈並回報統計。
    @MainActor
    private func stopStreaming() {
        streamLoopTask?.cancel()
        streamLoopTask = nil
        let total = analysisPipeline.totalSteps
        let left = analysisPipeline.leftSteps
        let right = analysisPipeline.rightSteps
        let bpm = analysisPipeline.avgCadenceBPM
        Task { await stream.end(totalSteps: total, leftSteps: left, rightSteps: right, avgCadenceBPM: bpm) }
        livePrediction = nil
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
        quickPoseEngine.adviceLines = ["偵測已暫停。請按「開始」，倒數 5 秒後開始偵測。"]
        detectionSource = .liveCamera
        pickedItem = nil
    }

    @MainActor
    private func handleDetectionSourceChange() async {
        guard !Self.sdkKeyIsPlaceholder else { return }
        guard countdownSeconds == nil, !quickPoseEngine.isEngineStarting else { return }
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
        summaryLines = analysisPipeline.videoSummary()
        videoQualityResult = nil
        showSummary = true

        guard assessmentEngine.usesTrainedModelPredict else {
            isAnalyzingVideoQuality = false
            autoSaveSummaryToHistory()
            return
        }

        isAnalyzingVideoQuality = true

        let sessionID = dbSessionID
        Task {
            let frames = sessionID.map { database.allFrames(sessionID: $0) } ?? []
            let result = await stream.predict(frames: frames)
            await MainActor.run {
                videoQualityResult = result
                isAnalyzingVideoQuality = false
                if let result, result.note == nil, let label = result.label {
                    let pct = Int(result.confidencePercent)
                    let qualityLine = "模型辨識：姿勢「\(label == "good" ? "好" : "壞")」（信心 \(pct)%）"
                    if !summaryLines.contains(where: { $0.hasPrefix("模型辨識：") }) {
                        summaryLines.insert(qualityLine, at: 0)
                    }
                }
                autoSaveSummaryToHistory()
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
            uploadVideoButton(label: "或改為上傳影片偵測")
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

            cameraPreviewStack(width: w, height: h)
                .overlay(alignment: .top) {
                    stepFlashOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .topLeading) {
                    poseHUDOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .bottom) {
                    bottomPanel(height: bottomPanelHeight, safeBottom: geometry.safeAreaInsets.bottom)
                }
                .overlay(alignment: .bottomTrailing) {
                    quickPoseVersionLabel
                }
        }
    }

    private func cameraBottomPanelHeight(for height: CGFloat) -> CGFloat {
        let ratio: CGFloat = {
            switch detectionSource {
            case .video: return 0.46
            case .liveCamera: return 0.40
            }
        }()
        return min(max(height * ratio, 300), 420)
    }

    @ViewBuilder
    private func cameraPreviewStack(width w: CGFloat, height h: CGFloat) -> some View {
        ZStack(alignment: .top) {
            cameraSourceView(width: w, height: h)

            QuickPoseOverlayView(overlayImage: $quickPoseEngine.overlayImage, contentMode: .fill)
                .frame(width: w, height: h)
                .opacity(quickPoseEngine.overlayImage == nil ? 0 : 1)
                .allowsHitTesting(false)
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
                QuickPoseCameraView(useFrontCamera: true, delegate: quickPoseEngine.pose, videoGravity: .resizeAspectFill)
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
        VStack(alignment: .leading, spacing: 10) {
            poseStatusBadge
            appMenuButton
        }
        .padding(.leading, 12)
        .padding(.top, safeTop + 8)
        .zIndex(2)
    }

    private var poseStatusBadge: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(assessmentEngine.hudBadgeTitle)
                .font(.caption.weight(.bold))
                .foregroundStyle(assessmentEngine == .mediaPipe ? .mint : .orange)
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
            if assessmentEngine.usesTrainedModelPredict, let result = videoQualityResult ?? livePrediction {
                videoQualityBadge(result, compact: true)
            }
        } else if assessmentEngine.usesTrainedModelPredict, let pred = livePrediction {
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
        guard countdownSeconds == nil else { return }
        quickPoseEngine.fpsText = "FPS: —（已暫停）"
        switch detectionSource {
        case .liveCamera:
            quickPoseEngine.adviceLines = ["\(assessmentEngine.hudBadgeTitle) 已暫停。請按「開始」，倒數 5 秒後開始偵測。"]
        case .video:
            quickPoseEngine.adviceLines = ["影片已載入（\(assessmentEngine.hudBadgeTitle)）。請按「開始」，倒數 \(Self.videoCountdownSeconds) 秒後開始偵測。"]
        }
    }

    @MainActor
    private func resumeDetectionFromUser() {
        guard !Self.sdkKeyIsPlaceholder else { return }
        guard countdownSeconds == nil else { return }
        let isVideo: Bool
        if case .video = detectionSource { isVideo = true } else { isVideo = false }
        beginDetectionCountdown(isVideo: isVideo)
    }

    /// 倒數結束後 attach QuickPose；影片 2 秒、相機 5 秒。
    @MainActor
    private func beginDetectionCountdown(isVideo: Bool) {
        cancelCountdown()
        quickPoseEngine.adviceLines = ["準備開始偵測…"]
        isPaused = true
        let seconds = isVideo ? Self.videoCountdownSeconds : Self.cameraCountdownSeconds

        countdownTask = Task {
            for sec in stride(from: seconds, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                await MainActor.run { countdownSeconds = sec }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }
            let warmupNs: UInt64 = isVideo ? 300_000_000 : 200_000_000
            try? await Task.sleep(nanoseconds: warmupNs)
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
    private func pauseDetectionFromUser() {
        cancelCountdown()
        if engineAttached {
            summaryLines = buildSummaryLinesForHistory()
            autoSaveSummaryToHistory()
        }
        pauseDetectionProcessing(fullReset: true)
        isPaused = true
        quickPoseEngine.fpsText = "FPS: —（已暫停）"
        quickPoseEngine.adviceLines = PauseAdvice.lines
        if historySavedForSession {
            quickPoseEngine.adviceLines = ["偵測已暫停，摘要已存入歷史紀錄。"] + PauseAdvice.lines
        }
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
        dbSessionID = database.beginSession(sourceLabel: currentSourceLabel)
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
            let pct = Int(result.confidencePercent)
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
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.indigo)
                            Text(line)
                                .font(.body)
                                .multilineTextAlignment(.leading)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Text("註：分析完成後會自動寫入歷史紀錄；可在影片繼續播放後再開啟，數值會更完整。")
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
                            Text(String(format: "模型信心度：%.0f%%", result.confidencePercent))
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
                            store.clear()
                        } label: {
                            Text("清空")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成", action: onClose)
                }
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
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.indigo)
                        Text(line)
                            .font(.body)
                            .fixedSize(horizontal: false, vertical: true)
                    }
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
                        Text("資料庫尚無任何節點")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Text("開始相機或影片偵測後，節點會存入資料庫；完成後可在此標「好 / 壞」上傳訓練。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        Section {
                            LabeledContent("偵測次數", value: "\(records.count)")
                            LabeledContent("節點總筆數", value: "\(totalNodes)")
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
            .navigationTitle("節點資料庫")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !records.isEmpty {
                        Button(role: .destructive) {
                            database.clearAll()
                            reload()
                        } label: {
                            Text("清空")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成", action: onClose)
                }
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
        let canLabel = record.frameCount > 0 && record.endedAt != nil
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
                Text("在此標記好 / 壞並上傳至雲端，即可用 train.py 訓練模型。")
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
                    Text("偵測進行中或尚無節點，請暫停後再標記。")
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
        isUploading = true
        let frames = database.allFrames(sessionID: record.id)
        Task {
            let result = await stream.uploadLabeledSession(
                label: label,
                sourceLabel: record.sourceLabel,
                totalSteps: record.totalSteps,
                leftSteps: record.leftSteps,
                rightSteps: record.rightSteps,
                avgCadenceBPM: record.avgCadenceBPM,
                frames: frames
            )
            await MainActor.run {
                isUploading = false
                isUploadError = !result.success
                statusMessage = result.message
                if result.success {
                    trainingLabel = label
                    database.markTrainingUpload(sessionID: record.id, label: label)
                    onUpdated()
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
