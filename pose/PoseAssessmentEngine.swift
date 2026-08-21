//
//  PoseAssessmentEngine.swift
//  pose
//
//  偵測系統版本：
//  - QuickPose：輕量骨架 + 規則建議
//  - MediaPipe：完整 BlazePose 骨架 + 規則建議
//  - 自訓模型：後端 RandomForest 好／壞品質辨識
//

import Foundation

enum PoseAssessmentEngine: String, CaseIterable, Identifiable, Codable {
    case quickPose = "quickpose"
    case mediaPipe = "mediapipe"
    case trainedModel = "trained_model"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quickPose: return "QuickPose"
        case .mediaPipe: return "MediaPipe"
        case .trainedModel: return "自訓模型"
        }
    }

    var subtitle: String {
        switch self {
        case .quickPose:
            return "輕量骨架 + 規則建議（肩膀、骨盆、頭部）"
        case .mediaPipe:
            return "MediaPipe Full 骨架 + 規則建議（較精準）"
        case .trainedModel:
            return "後端 RandomForest 好／壞品質辨識"
        }
    }

    var adviceSectionTitle: String {
        switch self {
        case .quickPose, .mediaPipe: return "姿勢建議"
        case .trainedModel: return "模型辨識"
        }
    }

    /// 使用規則式姿勢建議（非後端 ML）。
    var usesRuleBasedAdvice: Bool {
        self != .trainedModel
    }

    /// 需要串流節點到後端並呼叫 /predict。
    var usesTrainedModelPredict: Bool {
        self == .trainedModel
    }

    var hudBadgeTitle: String {
        switch self {
        case .quickPose: return "QuickPose 規則"
        case .mediaPipe: return "MediaPipe Full"
        case .trainedModel: return "自訓模型"
        }
    }

    private static let storageKey = "PoseAssessmentEngine"

    static func loadSaved() -> PoseAssessmentEngine {
        guard let raw = UserDefaults.standard.string(forKey: storageKey),
              let value = PoseAssessmentEngine(rawValue: raw) else {
            return .quickPose
        }
        return value
    }

    func save() {
        UserDefaults.standard.set(rawValue, forKey: Self.storageKey)
    }
}
