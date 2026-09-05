//
//  PoseDetectionShellView.swift
//  pose
//
//  登入後三種引擎都進入 PoseDetectionView 完整管線。
//

import Combine
import SwiftUI

@MainActor
final class DetectionModeStore: ObservableObject {
    @AppStorage("poseDetectionEngine") private var engineRaw = PoseAssessmentEngine.trainedModel.rawValue

    var engine: PoseAssessmentEngine {
        get { PoseAssessmentEngine.resolved(fromStored: engineRaw) }
        set {
            engineRaw = newValue.rawValue
            newValue.save()
            objectWillChange.send()
        }
    }
}

struct PoseDetectionShellView: View {
    @EnvironmentObject private var auth: AuthManager
    @StateObject private var modeStore = DetectionModeStore()
    @AppStorage("didShowEngineChooser") private var didShowEngineChooser = false
    @State private var showEngineChooser = false

    var body: some View {
        Group {
            PoseDetectionView()
        }
        .environmentObject(modeStore)
        .sheet(isPresented: $showEngineChooser) {
            EngineChooserSheet(modeStore: modeStore) {
                didShowEngineChooser = true
                showEngineChooser = false
            }
        }
        .onAppear {
            if !didShowEngineChooser {
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
                    Text("三種引擎都可相機／上傳影片，並把節點標好／壞上傳。MediaPipe 存 mediapipe.realm，QuickPose 與自訓模型存 pose.realm。")
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
