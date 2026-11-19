//
//  PoseDetectionShellView.swift
//  pose
//
//  登入後三種引擎都進入 PoseDetectionView 完整管線。
//

import Combine
import SwiftUI

/// 拍攝視角：背面／側面特徵管線各自只在對應模式啟用。
enum PoseCaptureViewMode: String, CaseIterable, Identifiable {
    case front
    case side
    case back

    var id: String { rawValue }

    var title: String {
        switch self {
        case .front: return "正面"
        case .side: return "側面"
        case .back: return "背面"
        }
    }

    var systemImage: String {
        switch self {
        case .front: return "person.fill"
        case .side: return "arrow.left.and.right"
        case .back: return "person.fill.turn.right"
        }
    }
}

@MainActor
final class DetectionModeStore: ObservableObject {
    @AppStorage("poseDetectionEngine") private var engineRaw = PoseAssessmentEngine.trainedModel.rawValue
    @AppStorage("poseGaitActivity") private var activityRaw = GaitActivityMode.walking.rawValue
    @AppStorage("poseCaptureViewMode") private var captureViewRaw = PoseCaptureViewMode.front.rawValue

    var engine: PoseAssessmentEngine {
        get { PoseAssessmentEngine.resolved(fromStored: engineRaw) }
        set {
            engineRaw = newValue.rawValue
            newValue.save()
            objectWillChange.send()
        }
    }

    var activity: GaitActivityMode {
        get { GaitActivityMode.resolved(fromStored: activityRaw) }
        set {
            activityRaw = newValue.rawValue
            objectWillChange.send()
        }
    }

    var captureView: PoseCaptureViewMode {
        get { PoseCaptureViewMode(rawValue: captureViewRaw) ?? .front }
        set {
            captureViewRaw = newValue.rawValue
            objectWillChange.send()
        }
    }

    var isBackView: Bool { captureView == .back }
    var isSideView: Bool { captureView == .side }
}

struct PoseDetectionShellView: View {
    @EnvironmentObject private var auth: AuthManager
    @StateObject private var modeStore = DetectionModeStore()
    @AppStorage("didShowEngineChooser") private var didShowEngineChooser = false
    @State private var showEngineChooser = false
    @State private var showUserGuide = false

    var body: some View {
        Group {
            PoseDetectionView()
        }
        .environmentObject(modeStore)
        .sheet(isPresented: $showUserGuide, onDismiss: handleUserGuideDismissed) {
            UserGuideSheet {
                showUserGuide = false
            }
        }
        .sheet(isPresented: $showEngineChooser) {
            EngineChooserSheet(modeStore: modeStore) {
                didShowEngineChooser = true
                showEngineChooser = false
            }
        }
        .onAppear {
            presentFirstRunSheetsIfNeeded()
        }
        .onChange(of: auth.username) { _, _ in
            presentFirstRunSheetsIfNeeded()
        }
    }

    /// 新註冊會員先看使用說明；看完（或無需顯示）再跳出引擎選擇。
    private func presentFirstRunSheetsIfNeeded() {
        guard !showUserGuide, !showEngineChooser else { return }
        if auth.shouldPresentUserGuide {
            showUserGuide = true
            return
        }
        if !didShowEngineChooser {
            showEngineChooser = true
        }
    }

    private func handleUserGuideDismissed() {
        auth.markUserGuideSeen()
        if !didShowEngineChooser {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                showEngineChooser = true
            }
        }
    }
}

private struct EngineChooserSheet: View {
    @ObservedObject var modeStore: DetectionModeStore
    let onDone: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(PoseAssessmentEngine.allCases) { engine in
                        Button {
                            modeStore.engine = engine
                            onDone()
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: engine.systemImage)
                                    .font(.title3)
                                    .foregroundStyle(engine == .mediaPipe ? Color.mint : Color.accentColor)
                                    .frame(width: 32)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(engine.title)
                                        .font(.headline)
                                        .foregroundStyle(.primary)
                                    Text(engine.detail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if modeStore.engine == engine {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                        }
                    }
                } header: {
                    Text("請選擇偵測引擎")
                } footer: {
                    Text("QuickPose／自訓模型寫入 pose.realm；MediaPipe 引擎以 Google Pose Landmarker 原生抓關節（相機與影片），寫入 mediapipe.realm，不再經 QuickPose。舊的「自訓＋MediaPipe 骨架」選項已移除，請清空 mediapipe 資料庫後重新採樣再訓練。")
                }
            }
            .navigationTitle("偵測引擎")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("稍後") { onDone() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

// MARK: - 共用：偵測引擎切換（底部／選單）

struct DetectionEnginePickerRow: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("偵測引擎")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.75))
            HStack(spacing: 6) {
                ForEach(PoseAssessmentEngine.allCases) { engine in
                    let selected = modeStore.engine == engine
                    Button {
                        modeStore.engine = engine
                    } label: {
                        VStack(spacing: 4) {
                            Image(systemName: engine.systemImage)
                                .font(.body.weight(.semibold))
                            Text(engine.title)
                                .font(.caption.weight(.bold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.75)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.bordered)
                    .tint(selected ? engineTint(engine) : .gray)
                    .accessibilityLabel(engine.title)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                }
            }
        }
    }

    private func engineTint(_ engine: PoseAssessmentEngine) -> Color {
        switch engine {
        case .quickPose: return .cyan
        case .mediaPipe: return .mint
        case .trainedModel: return .indigo
        }
    }
}

struct GaitActivityPickerRow: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("活動")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.75))
            HStack(spacing: 6) {
                ForEach(GaitActivityMode.allCases) { mode in
                    let selected = modeStore.activity == mode
                    Button {
                        modeStore.activity = mode
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: mode.systemImage)
                                .font(.caption.weight(.semibold))
                            Text(mode.title)
                                .font(.caption.weight(.bold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.bordered)
                    .tint(selected ? (mode == .running ? .orange : .cyan) : .gray)
                    .accessibilityLabel(mode.title)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                }
            }
        }
    }
}

struct CaptureViewPickerRow: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("拍攝視角")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.75))
            HStack(spacing: 6) {
                ForEach(PoseCaptureViewMode.allCases) { mode in
                    let selected = modeStore.captureView == mode
                    Button {
                        modeStore.captureView = mode
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: mode.systemImage)
                                .font(.caption.weight(.semibold))
                            Text(mode.title)
                                .font(.caption.weight(.bold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.bordered)
                    .tint(selected ? captureTint(mode) : .gray)
                    .accessibilityLabel(mode.title)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                }
            }
            if modeStore.isBackView {
                Text("背面模式：啟用背面特徵、手臂低信心門檻與耳－肩頭部推估。")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.65))
            } else if modeStore.isSideView {
                Text("側面模式：朝向翻轉正規化、只採近側肢體，並計算前傾／膝角／跨步過度等特徵。")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.65))
            }
        }
    }

    private func captureTint(_ mode: PoseCaptureViewMode) -> Color {
        switch mode {
        case .back: return .mint
        case .side: return .orange
        case .front: return .cyan
        }
    }
}

struct DetectionEngineMenuItems: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        Menu("偵測引擎") {
            ForEach(PoseAssessmentEngine.allCases) { engine in
                Button {
                    modeStore.engine = engine
                } label: {
                    if modeStore.engine == engine {
                        Label(engine.title, systemImage: "checkmark")
                    } else {
                        Text(engine.title)
                    }
                }
            }
        }
    }
}
