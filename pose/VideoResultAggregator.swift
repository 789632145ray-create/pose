//
//  VideoResultAggregator.swift
//  pose
//
//  影片級「好／壞」後處理聚合：與 backend/video_result_aggregator.py 同一套規則。
//  1) 過濾低信心雜訊（minConf；全滅則退回全部）
//  2) 信心加權投票決定 final_label
//  3) Top-K（前 20%，1~5）平均或 Peak 作為 final_confidence
//  4) 輸出供 App 直接渲染的結構
//

import Foundation

enum VideoScoreMode: String, CaseIterable, Identifiable {
    case topK = "topk"
    case peak = "peak"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .topK: return "Top-K 平均"
        case .peak: return "絕對最高"
        }
    }
}

struct FramePredictionSample: Equatable {
    /// "good" 或 "bad"
    let label: String
    /// 該標籤的信心 0~1
    let confidence: Double

    init?(label: String?, probabilityGood: Double) {
        guard let raw = label?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              raw == "good" || raw == "bad" else { return nil }
        let pGood = max(0, min(1, probabilityGood))
        self.label = raw
        self.confidence = raw == "good" ? pGood : (1 - pGood)
    }

    init(label: String, confidence: Double) {
        self.label = label.lowercased()
        self.confidence = max(0, min(1, confidence))
    }
}

struct VideoAggregateResult: Equatable {
    let finalLabel: String
    let finalConfidence: Double
    let dominanceRatio: Double
    let validSamplesCount: Int

    var confidencePercent: Double { finalConfidence * 100 }

    /// 轉成既有 LivePrediction，讓 HUD／摘要不必改版面。
    func asLivePrediction() -> LivePrediction {
        let pGood = finalLabel == "good" ? finalConfidence : (1 - finalConfidence)
        return LivePrediction(
            label: finalLabel,
            probabilityGood: pGood,
            note: nil,
            finalLabel: finalLabel,
            finalConfidence: finalConfidence,
            dominanceRatio: dominanceRatio,
            validSamplesCount: validSamplesCount
        )
    }

    var jsonObject: [String: Any] {
        [
            "final_label": finalLabel,
            "final_confidence": finalConfidence,
            "dominance_ratio": dominanceRatio,
            "valid_samples_count": validSamplesCount,
        ]
    }
}

struct VideoResultAggregator {
    var minConf: Double = 0.5
    var scoreMode: VideoScoreMode = .topK

    func aggregate(_ predictions: [FramePredictionSample]) -> VideoAggregateResult {
        guard !predictions.isEmpty else {
            return VideoAggregateResult(
                finalLabel: "bad",
                finalConfidence: 0,
                dominanceRatio: 0,
                validSamplesCount: 0
            )
        }

        let filtered = predictions.filter { $0.confidence >= minConf }
        let used = filtered.isEmpty ? predictions : filtered

        let sumGood = used.filter { $0.label == "good" }.map(\.confidence).reduce(0, +)
        let sumBad = used.filter { $0.label == "bad" }.map(\.confidence).reduce(0, +)
        let total = sumGood + sumBad

        let finalLabel: String
        let dominance: Double
        if total <= 0 {
            let goodCount = used.filter { $0.label == "good" }.count
            finalLabel = goodCount >= used.count - goodCount ? "good" : "bad"
            dominance = 0
        } else if sumGood >= sumBad {
            finalLabel = "good"
            dominance = sumGood / total
        } else {
            finalLabel = "bad"
            dominance = sumBad / total
        }

        let winnerConfs = used.filter { $0.label == finalLabel }.map(\.confidence).sorted(by: >)
        let finalConfidence: Double
        if winnerConfs.isEmpty {
            finalConfidence = 0
        } else if scoreMode == .peak {
            finalConfidence = winnerConfs[0]
        } else {
            let k = Self.topKCount(winnerConfs.count)
            let slice = Array(winnerConfs.prefix(k))
            finalConfidence = slice.reduce(0, +) / Double(slice.count)
        }

        return VideoAggregateResult(
            finalLabel: finalLabel,
            finalConfidence: max(0, min(1, finalConfidence)),
            dominanceRatio: max(0, min(1, dominance)),
            validSamplesCount: used.count
        )
    }

    /// 前 20%，至少 1、最多 5。
    static func topKCount(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let raw = Int((Double(n) * 0.20).rounded())
        return max(1, min(5, max(1, raw)))
    }
}

/// 累積多次 /predict 結果，輸出單一穩定結論。
@MainActor
final class LivePredictionBuffer {
    private(set) var samples: [FramePredictionSample] = []
    var aggregator = VideoResultAggregator()
    /// 避免緩衝無限增長（約每秒一筆時可蓋住數分鐘）。
    var maxSamples: Int = 240

    func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    @discardableResult
    func append(_ prediction: LivePrediction) -> VideoAggregateResult? {
        guard prediction.note == nil,
              let sample = FramePredictionSample(label: prediction.label, probabilityGood: prediction.probabilityGood) else {
            return nil
        }
        samples.append(sample)
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
        return aggregator.aggregate(samples)
    }

    func currentAggregate() -> VideoAggregateResult? {
        guard !samples.isEmpty else { return nil }
        return aggregator.aggregate(samples)
    }
}
