//
//  PoseAssessmentEngine.swift
//  pose
//
//  偵測系統版本（三種都走完整管線：相機／影片、節點庫、標好壞）：
//  - QuickPose：輕量骨架 + 規則建議
//  - MediaPipe：Google Pose Landmarker 原生抓關節（相機／影片）→ mediapipe.realm
//  - 自訓模型：可選兩套雲端品質模型（QuickPose 訓練／MediaPipe 訓練）做好／壞辨識
//

import Foundation

/// 自訓模型底下可選的雲端品質模型（對應兩份 joblib）。
enum TrainedQualityModel: String, CaseIterable, Identifiable, Codable {
    case quickPose = "pose"
    case mediaPipe = "mediapipe"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quickPose: return "QuickPose 模型"
        case .mediaPipe: return "MediaPipe 模型"
        }
    }

    var shortTitle: String {
        switch self {
        case .quickPose: return "QuickPose"
        case .mediaPipe: return "MediaPipe"
        }
    }

    var detail: String {
        switch self {
        case .quickPose:
            return "pose_quality_model.joblib · QuickPose 骨架 · /predict"
        case .mediaPipe:
            return "mediapipe_quality_model.joblib · 原生骨架 · /mediapipe/predict"
        }
    }

    var systemImage: String {
        switch self {
        case .quickPose: return "figure.walk"
        case .mediaPipe: return "person.fill.viewfinder"
        }
    }

    var predictPath: String {
        self == .mediaPipe ? "/mediapipe/predict" : "/predict"
    }

    var predictEngineName: String {
        rawValue
    }

    /// MediaPipe 模型必須用原生 Landmarker 特徵，才能對上訓練資料。
    var usesNativeMediaPipe: Bool {
        self == .mediaPipe
    }

    private static let storageKey = "TrainedQualityModel"

    static func resolved(fromStored raw: String) -> TrainedQualityModel {
        TrainedQualityModel(rawValue: raw) ?? .quickPose
    }

    static func loadSaved() -> TrainedQualityModel {
        if let raw = UserDefaults.standard.string(forKey: storageKey) {
            return resolved(fromStored: raw)
        }
        return .quickPose
    }

    func save() {
        UserDefaults.standard.set(rawValue, forKey: Self.storageKey)
    }
}

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
            return "QuickPose SDK 骨架 + 規則建議"
        case .mediaPipe:
            return "Google MediaPipe 原生抓關節（相機／影片）"
        case .trainedModel:
            return "可選 QuickPose／MediaPipe 兩套自訓品質模型"
        }
    }

    var detail: String {
        switch self {
        case .quickPose:
            return "完整偵測畫面：相機／影片、步態、節點庫、標好／壞"
        case .mediaPipe:
            return "原生 Pose Landmarker 蒐集訓練資料；寫入 mediapipe.realm，不經 QuickPose"
        case .trainedModel:
            return "選下方辨識模型：QuickPose 或 MediaPipe 自訓 joblib"
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

    /// 需要串流節點到後端並呼叫 /predict（自訓模型）。
    var usesTrainedModelPredict: Bool {
        self == .trainedModel
    }

    /// 用雲端模型做品質辨識：自訓模型走選中的 joblib，MediaPipe 引擎走 mediapipe 模型。
    var usesQualityPredict: Bool {
        self == .trainedModel || self == .mediaPipe
    }

    /// 未指定自訓子模型時的預設路徑（MediaPipe 引擎／QuickPose）。
    var predictPath: String {
        self == .mediaPipe ? "/mediapipe/predict" : "/predict"
    }

    var predictEngineName: String {
        self == .mediaPipe ? "mediapipe" : "pose"
    }

    func predictPath(qualityModel: TrainedQualityModel) -> String {
        if self == .trainedModel { return qualityModel.predictPath }
        return predictPath
    }

    func predictEngineName(qualityModel: TrainedQualityModel) -> String {
        if self == .trainedModel { return qualityModel.predictEngineName }
        return predictEngineName
    }

    /// 是否應走原生 MediaPipe Landmarker（含自訓選 MediaPipe 模型）。
    func usesNativeMediaPipe(qualityModel: TrainedQualityModel) -> Bool {
        self == .mediaPipe || (self == .trainedModel && qualityModel.usesNativeMediaPipe)
    }

    /// QuickPose／自訓（QuickPose 模型）共用 QuickPose Full；MediaPipe 走原生 Landmarker。
    var usesFullPoseModel: Bool { true }

    /// 進出原生 MediaPipe runtime 時需重啟相機／影片管線。
    static func requiresPoseRuntimeRestart(
        from: PoseAssessmentEngine,
        to: PoseAssessmentEngine,
        fromQuality: TrainedQualityModel = .quickPose,
        toQuality: TrainedQualityModel = .quickPose
    ) -> Bool {
        from.usesNativeMediaPipe(qualityModel: fromQuality)
            != to.usesNativeMediaPipe(qualityModel: toQuality)
    }

    var hudBadgeTitle: String {
        hudBadgeTitle(qualityModel: .quickPose)
    }

    func hudBadgeTitle(qualityModel: TrainedQualityModel) -> String {
        switch self {
        case .quickPose: return "QuickPose 規則"
        case .mediaPipe: return "MediaPipe Full"
        case .trainedModel: return "自訓 · \(qualityModel.shortTitle)"
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
