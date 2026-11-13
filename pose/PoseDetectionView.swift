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
    var activityMode: GaitActivityMode = .walking
    private var restartTask: Task<Void, Never>?

    weak var analysisPipeline: PoseAnalysisPipeline?
    var bodyGaitProfile: BodyGaitProfile?
    var onStreamEnqueue: (([PoseNode], TimeInterval) -> Void)?
    var onStepEvents: (([StepEvent]) -> Void)?

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

    /// MediaPipe 使用 BlazePose Full（`.good`）；自訓模型同樣用 Full 骨架再送後端。
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
        if let landmarks {
            let nodes = PoseNodeExtractor.extractAll(from: landmarks)
            if nodes.contains(where: MediaPipeSkeletonGraph.isVisible) {
                previewNodes = nodes
            }
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
                let advice = PoseFrameAdvice(lines: mergedLines, issues: mergedIssues)
                let nodes = PoseNodeExtractor.extractAll(from: landmarks)
                PoseDatabase.store(for: assessmentEngine).recordFrame(nodes: nodes)
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
    @State private var showCoachingDetail = false
    @State private var lastSavedID: UUID?
    /// 本次偵測 session 是否已自動寫入歷史（避免重複儲存）。
    @State private var historySavedForSession = false
    @StateObject private var summaryStore = SummaryStore()

    /// 目前引擎對應的本機節點庫（MediaPipe 為獨立 mediapipe.realm）。
    private var nodeDatabase: PoseDatabase {
        PoseDatabase.store(for: assessmentEngine)
    }
    /// 開始 session 時鎖定的庫，避免中途切引擎寫錯檔。
    @State private var sessionIsMediaPipe = false
    private var sessionDatabase: PoseDatabase {
        sessionIsMediaPipe ? .mediaPipe : .shared
    }
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

    /// 前鏡頭預覽是鏡像，節點 x 要翻轉才會對上身體。
    private var skeletonFlipsHorizontally: Bool {
        switch detectionSource {
        case .liveCamera:
            return !ProcessInfo.processInfo.isiOSAppOnMac
        case .video:
            return false
        }
    }

    private let chromeBackground = PoseChrome.canvas

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
                    onClose: { showSummary = false }
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
            .sheet(isPresented: $showCoachingDetail) {
                coachingDetailSheet
            }
    }

    private var coachingDetailSheet: some View {
        NavigationStack {
            List {
                Section(assessmentEngine.adviceSectionTitle) {
                    ForEach(Array(quickPoseEngine.adviceLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.body)
                    }
                }
                Section("最近步態") {
                    if quickPoseEngine.recentSteps.isEmpty {
                        Text("尚無步數紀錄")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(quickPoseEngine.recentSteps.suffix(12).reversed()) { event in
                            let interval = event.intervalFromPrevious.map { String(format: "+%.2fs", $0) } ?? "—"
                            let bpm = event.cadenceBPM.map { String(format: "%.0f bpm", $0) } ?? "—"
                            Text("第 \(event.index) 步 · \(event.side.localizedLabel)腳 · \(interval) · \(bpm)")
                                .font(.subheadline.monospacedDigit())
                        }
                    }
                }
            }
            .navigationTitle("即時建議")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { showCoachingDetail = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private var appMenuSheetContent: some View {
        AppMenuSheet(
            developerInfo: currentDeveloperInfo,
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

    private var currentDeveloperInfo: PoseDeveloperInfo {
        PoseDeveloperInfo(
            engineTitle: assessmentEngine.hudBadgeTitle,
            activityTitle: modeStore.activity.title,
            realmFileName: assessmentEngine == .mediaPipe ? MediaPipeRealm.fileName : "pose.realm",
            dbNodeCount: quickPoseEngine.dbNodeCount,
            frameCallbackCount: quickPoseEngine.frameCallbackCount,
            fpsText: quickPoseEngine.fpsText,
            quickPoseVersion: quickPoseEngine.pose.quickPoseVersion(),
            mediaPipeModelWeight: assessmentEngine == .mediaPipe ? quickPoseEngine.pose.modelWeight() : nil,
            localVideoNote: "影片在手機本機分析，不需要 Railway。"
        )
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
                syncActivityMode()
                wireQuickPoseEngineCallbacks()
                syncBodyGaitProfile()
            }
            .onChange(of: modeStore.engine) { _, newEngine in
                handleAssessmentEngineChange(newEngine)
            }
            .onChange(of: modeStore.activity) { _, _ in
                syncActivityMode()
                if isPaused { applyPausedIdleState() }
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
                .foregroundStyle(PoseChrome.primaryText)

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
                    .background(PoseChrome.accent)
                    .foregroundStyle(.black)
                    .clipShape(RoundedRectangle(cornerRadius: PoseChrome.cardRadius, style: .continuous))
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
            .foregroundStyle(PoseChrome.accent)
#endif
            videoImportButtons(primaryLabel: "或改為上傳影片偵測")
        }
    }

    @MainActor
    private func videoImportButtons(primaryLabel: String) -> some View {
        VStack(spacing: 8) {
            Button {
                showPhotosPicker = true
            } label: {
                Label(primaryLabel, systemImage: "film.fill")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .tint(PoseChrome.accent)
            .foregroundStyle(.black)
            .disabled(isLoadingVideo)

            Button {
                showFileImporter = true
            } label: {
                Label("從檔案選擇", systemImage: "folder.fill")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.bordered)
            .tint(.white)
            .disabled(isLoadingVideo)
        }
        .padding(.horizontal, 32)
    }

    /// 貼齊安全區的精簡控制面板：引擎／活動切換 + 核心操作列。
    @MainActor
    private func bottomPanel(height panelHeight: CGFloat, safeBottom: CGFloat) -> some View {
        VStack(spacing: PoseChrome.spacingSM) {
            if case .video = detectionSource {
                analyzeSummaryCapsule
            }

            PoseGlassCard {
                VStack(spacing: PoseChrome.spacingSM) {
                    DetectionEnginePickerRow()
                    GaitActivityPickerRow()
                    compactTransportRow
                    coachingTicker
                }
            }
        }
        .padding(.horizontal, PoseChrome.spacingSM)
        .padding(.bottom, max(safeBottom, 8))
        .frame(maxWidth: .infinity, alignment: .bottom)
        .frame(height: panelHeight + safeBottom, alignment: .bottom)
    }

    @MainActor
    private var analyzeSummaryCapsule: some View {
        Button {
            PoseChrome.impact(.light)
            analyzeVideoQualityAndShowSummary()
        } label: {
            HStack(spacing: 8) {
                if isAnalyzingVideoQuality {
                    ProgressView()
                        .tint(PoseChrome.primaryText)
                        .scaleEffect(0.85)
                } else {
                    Image(systemName: assessmentEngine.usesQualityPredict ? "sparkles" : "text.alignleft")
                        .font(.caption.weight(.bold))
                }
                Text(assessmentEngine.usesQualityPredict ? "品質辨識與摘要" : "分析摘要")
                    .font(.caption.weight(.bold))
                Spacer(minLength: 0)
                Image(systemName: "chevron.up")
                    .font(.caption2.weight(.bold))
                    .opacity(0.7)
            }
            .foregroundStyle(PoseChrome.primaryText)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background {
                Capsule(style: .continuous)
                    .fill(.ultraThinMaterial)
                    .background(Capsule(style: .continuous).fill(PoseChrome.glassFill))
            }
        }
        .buttonStyle(.plain)
        .disabled(isAnalyzingVideoQuality || Self.sdkKeyIsPlaceholder)
        .opacity(isAnalyzingVideoQuality || Self.sdkKeyIsPlaceholder ? 0.55 : 1)
    }

    @MainActor
    private var compactTransportRow: some View {
        HStack(alignment: .bottom, spacing: 10) {
            PoseCircleIconButton(
                systemImage: "film.fill",
                label: videoPickerLabel,
                isDisabled: isLoadingVideo
            ) {
                showPhotosPicker = true
            }

            PoseCircleIconButton(
                systemImage: "folder.fill",
                label: "檔案",
                isDisabled: isLoadingVideo
            ) {
                showFileImporter = true
            }

            Spacer(minLength: 4)

            playPausePrimaryButton

            Spacer(minLength: 4)

            if case .video = detectionSource {
                PoseCircleIconButton(systemImage: "camera.fill", label: "相機") {
                    switchBackToCamera()
                }
            } else {
                PoseCircleIconButton(systemImage: "camera.fill", label: "相機", isDisabled: true) {}
            }

            PoseCircleIconButton(systemImage: "cylinder.split.1x2.fill", label: "資料庫") {
                showDatabase = true
            }
        }
    }

    private var videoPickerLabel: String {
        if case .video = detectionSource { return "重選" }
        return "影片"
    }

    @MainActor
    private var playPausePrimaryButton: some View {
        let canResume = isPaused && !Self.sdkKeyIsPlaceholder && countdownSeconds == nil
        let canPause = !isPaused && engineAttached && !Self.sdkKeyIsPlaceholder && countdownSeconds == nil
        return PoseCircleIconButton(
            systemImage: isPaused ? "play.fill" : "pause.fill",
            label: isPaused ? "開始" : "暫停",
            isProminent: true,
            isDisabled: isPaused ? !canResume : !canPause
        ) {
            withAnimation(PoseChrome.controlSpring) {
                if isPaused {
                    resumeDetectionFromUser()
                } else {
                    pauseDetectionFromUser()
                }
            }
        }
    }

    @MainActor
    private var coachingTicker: some View {
        Button {
            PoseChrome.impact(.light)
            showCoachingDetail = true
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "lightbulb.fill")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(PoseChrome.accent)
                Text(quickPoseEngine.adviceLines.first ?? "偵測建議會顯示在這裡")
                    .font(.caption)
                    .foregroundStyle(PoseChrome.primaryText.opacity(0.9))
                    .lineLimit(1)
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(PoseChrome.label)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(PoseChrome.secondaryFill, in: RoundedRectangle(cornerRadius: PoseChrome.capsuleRadius, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    @MainActor
    private var appMenuButton: some View {
        Button {
            PoseChrome.impact(.light)
            showAppMenu = true
        } label: {
            Image(systemName: "line.3.horizontal")
                .font(.body.weight(.bold))
                .foregroundStyle(PoseChrome.primaryText)
                .frame(width: 40, height: 40)
                .background(PoseChrome.secondaryFill, in: Circle())
                .overlay(Circle().strokeBorder(Color.white.opacity(0.12), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("選單")
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
    private func syncActivityMode() {
        let activity = modeStore.activity
        quickPoseEngine.activityMode = activity
        analysisPipeline.activityMode = activity
    }

    @MainActor
    private func handleAssessmentEngineChange(_ newEngine: PoseAssessmentEngine) {
        let previous = quickPoseEngine.assessmentEngine
        syncAssessmentEngine()
        guard previous != newEngine else { return }
        livePrediction = nil
        videoQualityResult = nil
        finishDatabaseSession()
        if engineAttached, quickPoseEngine.detectionActive {
            sessionIsMediaPipe = newEngine == .mediaPipe
            dbSessionID = nodeDatabase.beginSession(sourceLabel: currentSourceLabel)
            quickPoseEngine.dbNodeCount = 0
        }
        if PoseAssessmentEngine.requiresPoseRuntimeRestart(from: previous, to: newEngine) {
            quickPoseEngine.scheduleRestart(hint: "正在切換\(newEngine.title)骨架…")
        } else {
            quickPoseEngine.statusHint = "已切換\(newEngine.title)，骨架持續顯示"
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
        if assessmentEngine.usesQualityPredict,
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
        sessionDatabase.endSession(
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
        guard assessmentEngine.usesQualityPredict else { return }
        let source = currentSourceLabel
        let uploadLiveFrames = assessmentEngine.usesTrainedModelPredict
        stream.predictPath = assessmentEngine.predictPath
        stream.predictEngine = assessmentEngine.predictEngineName
        streamLoopTask?.cancel()
        streamLoopTask = Task {
            if uploadLiveFrames {
                let ok = await stream.begin(mode: .predict, sourceLabel: source)
                guard ok else {
                    await MainActor.run { livePrediction = LivePrediction(label: nil, probabilityGood: 0, note: "未連線/未登入") }
                    return
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
                    .foregroundStyle(PoseChrome.accent)
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
                    .tint(.white)

                    Button {
                        confirmPendingVideo()
                    } label: {
                        Label("開始分析", systemImage: "play.fill")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(PoseChrome.accent)
                    .foregroundStyle(.black)
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 24)
            .frame(maxWidth: 360)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: PoseChrome.cardRadius, style: .continuous))
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

        guard assessmentEngine.usesQualityPredict else {
            isAnalyzingVideoQuality = false
            autoSaveSummaryToHistory()
            return
        }

        isAnalyzingVideoQuality = true
        stream.predictPath = assessmentEngine.predictPath
        stream.predictEngine = assessmentEngine.predictEngineName

        let sessionID = dbSessionID
        Task {
            let frames = sessionID.map { sessionDatabase.allFrames(sessionID: $0) } ?? []
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
            .tint(PoseChrome.accent)
            .foregroundStyle(.black)
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
            let bottomPanelHeight = cameraBottomPanelHeight(for: detectionSource)

            cameraPreviewStack(width: w, height: h)
                .overlay(alignment: .top) {
                    stepFlashOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .top) {
                    poseHUDOverlay(safeTop: geometry.safeAreaInsets.top)
                }
                .overlay(alignment: .bottom) {
                    bottomPanel(height: bottomPanelHeight, safeBottom: geometry.safeAreaInsets.bottom)
                }
        }
    }

    private func cameraBottomPanelHeight(for source: DetectionSource) -> CGFloat {
        // 控制列精簡後僅保留安全區上方一小段，釋放跑者畫面。
        switch source {
        case .video: return 188
        case .liveCamera: return 168
        }
    }

    private var hasVisibleSkeletonNodes: Bool {
        quickPoseEngine.previewNodes.contains(where: MediaPipeSkeletonGraph.isVisible)
    }

    @ViewBuilder
    private func cameraPreviewStack(width w: CGFloat, height h: CGFloat) -> some View {
        ZStack(alignment: .top) {
            cameraSourceView(width: w, height: h)

            // 有自繪骨架時隱藏 SDK 青色粗線，改用專業動作捕捉樣式。
            QuickPoseOverlayView(overlayImage: $quickPoseEngine.overlayImage, contentMode: .fill)
                .frame(width: w, height: h)
                .opacity(quickPoseEngine.overlayImage == nil || hasVisibleSkeletonNodes ? 0 : 1)
                .allowsHitTesting(false)

            MediaPipeSkeletonOverlay(
                nodes: quickPoseEngine.previewNodes,
                flipHorizontally: skeletonFlipsHorizontally
            )
            .frame(width: w, height: h)
            .opacity(hasVisibleSkeletonNodes ? 1 : 0)
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
        if stepFlashSide != nil {
            Rectangle()
                .fill(PoseChrome.accent)
                .frame(height: 3)
                .opacity(stepFlashOpacity)
                .padding(.top, safeTop)
                .allowsHitTesting(false)
        }
    }

    private func poseHUDOverlay(safeTop: CGFloat) -> some View {
        VStack(alignment: .trailing, spacing: PoseChrome.spacingXS) {
            HStack(alignment: .top, spacing: PoseChrome.spacingSM) {
                poseStatsCard
                appMenuButton
            }
            fpsMicroBadge
        }
        .padding(.horizontal, PoseChrome.spacingSM)
        // GeometryReader 在 ignoresSafeArea 下 safeTop 可能是 0；避開控制中心熱區。
        .padding(.top, max(safeTop, 54) + 6)
        .zIndex(2)
    }

    private var poseStatsCard: some View {
        PoseGlassCard {
            VStack(alignment: .leading, spacing: PoseChrome.spacingSM) {
                HStack {
                    Text(modeStore.activity.title.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .tracking(0.8)
                        .foregroundStyle(PoseChrome.label)
                    Spacer(minLength: 0)
                    if !quickPoseEngine.statusHint.isEmpty {
                        Text(quickPoseEngine.statusHint)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(PoseChrome.label)
                            .lineLimit(1)
                    }
                }

                HStack(alignment: .top, spacing: PoseChrome.spacingSM) {
                    PoseStatCell(
                        title: "CADENCE",
                        value: cadenceDisplayValue,
                        subtitle: "bpm"
                    )
                    PoseStatCell(
                        title: "STEPS",
                        value: "\(analysisPipeline.totalSteps)",
                        subtitle: "L \(analysisPipeline.leftSteps) / R \(analysisPipeline.rightSteps)"
                    )
                    PoseStatCell(
                        title: "QUALITY",
                        value: qualityDisplayValue,
                        badge: qualityBadgeLabel
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var cadenceDisplayValue: String {
        if let bpm = analysisPipeline.avgCadenceBPM {
            return String(format: "%.0f", bpm)
        }
        return "—"
    }

    private var activeQualityPrediction: LivePrediction? {
        if assessmentEngine.usesQualityPredict {
            return videoQualityResult ?? livePrediction
        }
        return nil
    }

    private var qualityDisplayValue: String {
        guard let pred = activeQualityPrediction, pred.note == nil, pred.label != nil else {
            return "—"
        }
        return "\(Int(pred.confidencePercent))%"
    }

    private var qualityBadgeLabel: String? {
        guard let pred = activeQualityPrediction, pred.note == nil, pred.label != nil else {
            return nil
        }
        return pred.qualityTitle
    }

    private var fpsMicroBadge: some View {
        Text(compactFPSLabel)
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .monospacedDigit()
            .foregroundStyle(PoseChrome.primaryText.opacity(0.8))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(PoseChrome.glassFill, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.1), lineWidth: 1))
    }

    private var compactFPSLabel: String {
        let raw = quickPoseEngine.fpsText
        if let range = raw.range(of: #"\d+"#, options: .regularExpression) {
            return "FPS \(raw[range])"
        }
        return "FPS —"
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
            quickPoseEngine.adviceLines = [
                "\(assessmentEngine.hudBadgeTitle) 已暫停。請按「開始」，倒數 5 秒後開始偵測。"
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
        quickPoseEngine.adviceLines = PauseAdvice.lines(for: modeStore.activity)
        if historySavedForSession {
            quickPoseEngine.adviceLines = ["偵測已暫停，摘要已存入歷史紀錄。"] + PauseAdvice.lines(for: modeStore.activity)
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
        sessionIsMediaPipe = assessmentEngine == .mediaPipe
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
                .foregroundStyle(PoseChrome.label)
        } else if let label = result.label {
            let isGood = label == "good"
            let pct = Int(result.confidencePercent)
            if compact {
                Text("品質：\(isGood ? "好" : "壞") \(pct)%")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(isGood ? PoseChrome.accent : .red)
            } else {
                HStack(spacing: 8) {
                    Image(systemName: isGood ? "hand.thumbsup.fill" : "hand.thumbsdown.fill")
                    Text(isGood ? "好" : "壞")
                        .font(.title.weight(.bold))
                    Text("\(pct)%")
                        .font(.title3.weight(.semibold))
                }
                .foregroundStyle(isGood ? PoseChrome.accent : .red)
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
                            .foregroundStyle(PoseChrome.accent)
                            .padding(.vertical, 4)
                    }

                    if showsModelQuality {
                        qualityResultCard
                    }

                    ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(PoseChrome.accent)
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
                    .tint(PoseChrome.accent)
                    .foregroundStyle(.black)
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
                        .foregroundStyle(PoseChrome.label)
                } else if let label = result.label {
                    let isGood = label == "good"
                    HStack(spacing: 16) {
                        Image(systemName: isGood ? "checkmark.seal.fill" : "xmark.seal.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(isGood ? PoseChrome.accent : .red)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(isGood ? "姿勢：好" : "姿勢：壞")
                                .font(.title.weight(.bold))
                                .foregroundStyle(isGood ? PoseChrome.accent : .red)
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
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: PoseChrome.cardRadius, style: .continuous))
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
                                .foregroundStyle(PoseChrome.accent)
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
    @State private var exportURL: URL?
    @State private var exportError: String?

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
                            database.clearAll()
                            reload()
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
        let frames = database.allFrames(sessionID: record.id)
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
