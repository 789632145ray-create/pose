//
//  PoseDetectionShellView.swift
//  pose
//
//  登入後依使用者選擇顯示 QuickPose SDK 或訓練模型完整管線。
//

import Combine
import SwiftUI

enum PoseDetectionEngine: String, CaseIterable, Identifiable {
    case quickPoseSDK = "quickpose"
    case trainedModel = "trained"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quickPoseSDK: return "QuickPose SDK"
        case .trainedModel: return "訓練模型"
        }
    }

    var detail: String {
        switch self {
        case .quickPoseSDK:
            return "官方骨架偵測，即時 FPS 與 overlay"
        case .trainedModel:
            return "步態分析、節點資料庫、雲端品質模型"
        }
    }
}

@MainActor
final class DetectionModeStore: ObservableObject {
    @AppStorage("poseDetectionEngine") private var engineRaw = PoseDetectionEngine.trainedModel.rawValue

    var engine: PoseDetectionEngine {
        get { PoseDetectionEngine(rawValue: engineRaw) ?? .trainedModel }
        set { engineRaw = newValue.rawValue }
    }
}

struct PoseDetectionShellView: View {
    @EnvironmentObject private var auth: AuthManager
    @StateObject private var modeStore = DetectionModeStore()

    var body: some View {
        Group {
            switch modeStore.engine {
            case .trainedModel:
                PoseDetectionView()
            case .quickPoseSDK:
                QuickPoseBasicDetectionView()
            }
        }
        .environmentObject(modeStore)
    }
}

// MARK: - 共用：偵測引擎切換（底部／選單）

struct DetectionEnginePickerRow: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        Menu {
            ForEach(PoseDetectionEngine.allCases) { engine in
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
        } label: {
            HStack(spacing: 8) {
                Image(systemName: modeStore.engine == .quickPoseSDK ? "figure.walk" : "brain.head.profile")
                Text("偵測引擎：\(modeStore.engine.title)")
                    .font(.subheadline.weight(.semibold))
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption2.weight(.semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .padding(.horizontal, 4)
        }
        .buttonStyle(.bordered)
        .tint(.cyan)
    }
}

struct DetectionEngineMenuItems: View {
    @EnvironmentObject private var modeStore: DetectionModeStore

    var body: some View {
        Menu("偵測引擎") {
            ForEach(PoseDetectionEngine.allCases) { engine in
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
