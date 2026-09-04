//
//  PoseDetectionShellView.swift
//  pose
//
//  登入後依使用者選擇顯示 QuickPose 原生 overlay、MediaPipe 完整管線，或自訓模型。
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

    var body: some View {
        Group {
            switch modeStore.engine {
            case .trainedModel, .mediaPipe:
                PoseDetectionView()
            case .quickPose:
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
        } label: {
            HStack(spacing: 8) {
                Image(systemName: modeStore.engine.systemImage)
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
