//
//  PoseAssessmentEngine.swift
//  pose
//
//  偵測系統版本（三種都走完整管線：相機／影片、節點庫、標好壞）：
//  - QuickPose：輕量骨架 + 規則建議
//  - MediaPipe：BlazePose Full 骨架 + 規則建議／步態分析
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

    var detail: String {
        switch self {
        case .quickPose:
            return "完整偵測畫面：相機／影片、步態、節點庫、標好／壞"
        case .mediaPipe:
            return "MediaPipe BlazePose Full 骨架、步態分析與規則建議"
        case .trainedModel:
            return "步態分析、節點資料庫、雲端品質模型"
        }
    }

    var systemImage: String {
        switch self {
        case .quickPose: return "figure.walk"
        case .mediaPipe: return "person.fill.viewfinder"
        case .trainedModel: return "brain.head.profile"
        }
    }

    var adviceSectionTitle: String {
        switch self {
        case .quickPose, .mediaPipe: return "姿勢建議"
        case .trainedModel: return "模型辨識"
        }
    }

    /// 三種引擎都走完整偵測管線（相機／影片、步態、節點資料庫、標好壞）。
    var usesFullDetectionPipeline: Bool { true }

    /// 使用規則式姿勢建議（非後端 ML）。
    var usesRuleBasedAdvice: Bool {
        self != .trainedModel
    }

    /// 需要串流節點到後端並呼叫 /predict。
    var usesTrainedModelPredict: Bool {
        self == .trainedModel
    }

    /// 三種引擎共用同一個已啟動的 QuickPose Full 工作階段畫骨架線。
    var usesFullPoseModel: Bool { true }

    /// 切換引擎只改分析／資料庫／是否呼叫 /predict。stop/start 會讓 overlay 線條只在第一個引擎出現。
    static func requiresPoseRuntimeRestart(from _: PoseAssessmentEngine, to _: PoseAssessmentEngine) -> Bool {
        false
    }

    var hudBadgeTitle: String {
        switch self {
        case .quickPose: return "QuickPose 規則"
        case .mediaPipe: return "MediaPipe Full"
        case .trainedModel: return "自訓模型"
        }
    }

    private static let storageKey = "PoseAssessmentEngine"

    /// 相容舊版 AppStorage：`trained`（自訓模型）、`quickpose`。
    static func resolved(fromStored raw: String) -> PoseAssessmentEngine {
        if let value = PoseAssessmentEngine(rawValue: raw) {
            return value
        }
        if raw == "trained" { return .trainedModel }
        return .trainedModel
    }

    static func loadSaved() -> PoseAssessmentEngine {
        if let raw = UserDefaults.standard.string(forKey: storageKey) {
            return resolved(fromStored: raw)
        }
        if let legacy = UserDefaults.standard.string(forKey: "poseDetectionEngine") {
            return resolved(fromStored: legacy)
        }
        return .quickPose
    }

    func save() {
        UserDefaults.standard.set(rawValue, forKey: Self.storageKey)
    }
}
