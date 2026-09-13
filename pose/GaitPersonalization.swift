//
//  GaitPersonalization.swift
//  pose
//
//  依使用者 BMI、身高、年齡調整步態評估閾值與建議文案。
//  - 高 BMI：小步幅、高步頻、膝屈曲緩衝；低垂直振幅可視為保護性表現。
//  - 高身高：嚴格檢查過度跨步。
//  - 青壯年：可期待較大髖伸展與推蹬；中高齡：重平衡、防拖步、步幅縮短屬正常。
//

import Foundation
import QuickPoseCore

// MARK: - 年齡分層

enum AgeGaitGroup: Equatable {
    /// 45 歲以下，關節活動度與肌力通常較佳。
    case youngAdult
    /// 45 歲以上，活動度退化，評估重心轉為平衡與安全。
    case middleAgedPlus

    init(age: Int) {
        self = age >= 45 ? .middleAgedPlus : .youngAdult
    }

    var label: String {
        switch self {
        case .youngAdult: return "青壯年"
        case .middleAgedPlus: return "中高齡"
        }
    }
}

// MARK: - 個體化步態檔案

struct BodyGaitProfile: Equatable {
    enum Trait: String, CaseIterable {
        case highBMI
        case tall
    }

    let age: Int
    let ageGroup: AgeGaitGroup
    let heightCm: Double
    let weightKg: Double
    let bmi: Double
    let traits: Set<Trait>

    init?(userProfile: UserProfile?) {
        guard let profile = userProfile else { return nil }
        age = profile.age
        ageGroup = AgeGaitGroup(age: profile.age)
        heightCm = profile.heightCm
        weightKg = profile.weightKg
        guard let bmi = profile.bmi else { return nil }
        self.bmi = bmi

        var traits = Set<Trait>()
        if bmi >= 27 { traits.insert(.highBMI) }
        if profile.heightCm >= 175 { traits.insert(.tall) }
        self.traits = traits
    }

    var isHighBMI: Bool { traits.contains(.highBMI) }
    var isTall: Bool { traits.contains(.tall) }
    var isYoungAdult: Bool { ageGroup == .youngAdult }
    var isMiddleAgedPlus: Bool { ageGroup == .middleAgedPlus }
    var hasPersonalization: Bool { true }

    /// 建議最低步頻（bpm）。
    var recommendedMinCadence: Double {
        if isMiddleAgedPlus { return 78 }
        if isHighBMI { return 100 }
        if isTall { return 92 }
        return 85
    }

    /// 腳踝 x 超前骨盆中線的容忍上限（越小越嚴格）；中高齡允許較短步幅。
    var maxFootAheadOffset: Double {
        if isMiddleAgedPlus { return 0.058 }
        if isTall { return 0.042 }
        if isHighBMI { return 0.050 }
        return 0.065
    }

    /// 觸地時膝屈曲比例下限。
    var minKneeFlexionRatio: Double {
        if isHighBMI { return 0.38 }
        if isMiddleAgedPlus { return 0.24 }
        return 0.28
    }

    /// 腳踝相對骨盆的最低抬離量（y 差值）；低於此視為可能拖步。
    var minAnkleClearance: Double {
        if isMiddleAgedPlus { return 0.018 }
        return 0.012
    }

    /// 青壯年推蹬期：膝蓋應略在髖後方（x 差）的最小值。
    var minHipExtensionOffset: Double {
        if isYoungAdult { return 0.015 }
        return 0
    }

    /// 垂直振幅：高 BMI 或中高齡偏低可視為合理。
    var lowVerticalOscillationPraiseThreshold: Double {
        if isHighBMI || isMiddleAgedPlus { return 0.028 }
        return 0.018
    }

    var profileSummaryLine: String? {
        var parts: [String] = ["\(ageGroup.label)（\(age) 歲）"]
        if isHighBMI {
            parts.append(String(format: "BMI %.1f（偏重）", bmi))
        }
        if isTall {
            parts.append(String(format: "身高 %.0f cm（偏高）", heightCm))
        }
        return "個人化評估：" + parts.joined(separator: "、")
    }
}

// MARK: - 即時步態建議

enum PoseGaitAdvisor {
    /// 走路核心（對側擺臂、腳跟→腳尖）＋ 依個體化檔案追加的步態提示。
    static func gaitAdvice(from landmarks: QuickPose.Landmarks, profile: BodyGaitProfile?) -> (lines: [String], issues: Set<PoseIssueCode>) {
        let form = WalkingFormAdvisor.evaluate(from: landmarks)
        var lines = form.lines
        var issues = form.issues
        guard let profile else { return (lines, issues) }

        func ok(_ p: QuickPose.Point3d) -> Bool {
            p.visibility > 0.35 && p.presence > 0.35
        }

        let lh = landmarks.landmark(forBody: .hip(side: .left))
        let rh = landmarks.landmark(forBody: .hip(side: .right))
        guard ok(lh), ok(rh) else { return (lines, issues) }

        let hipMidX = (lh.x + rh.x) / 2
        let pelvisY = (lh.y + rh.y) / 2
        let formIssues = issues

        // 中高齡：軀幹直立與平衡（肩→髖中線）
        if profile.isMiddleAgedPlus {
            let ls = landmarks.landmark(forBody: .shoulder(side: .left))
            let rs = landmarks.landmark(forBody: .shoulder(side: .right))
            let nose = landmarks.landmark(forBody: .nose)
            if ok(ls), ok(rs) {
                let shoulderMidX = (ls.x + rs.x) / 2
                let hipMidX2 = hipMidX
                let trunkLean = abs(shoulderMidX - hipMidX2)
                if trunkLean > 0.07 {
                    lines.append("軀幹前後或左右偏移偏大，中高齡請優先保持直立與重心穩定，避免跌倒。")
                    issues.insert(.trunkInstability)
                } else if ok(nose), abs(nose.y - ls.y) > 0.12 {
                    lines.append("頭部或上背略前傾，試著輕收下巴、視線平視前方。")
                    issues.insert(.forwardStoop)
                }
            }
        }

        for side: QuickPose.Side in [.left, .right] {
            let label = side == .left ? "左" : "右"
            let ankle = landmarks.landmark(forBody: .ankle(side: side))
            let knee = landmarks.landmark(forBody: .knee(side: side))
            let hip = landmarks.landmark(forBody: .hip(side: side))
            let heel = landmarks.landmark(forBody: .heel(side: side))
            guard ok(ankle), ok(knee), ok(hip) else { continue }

            let footAhead = ankle.x - hipMidX
            if profile.isTall, !profile.isMiddleAgedPlus, footAhead > profile.maxFootAheadOffset {
                lines.append("\(label)腳著地點偏身體前方，長腿易過度跨步，試著縮小步幅、落在重心下方。")
                issues.insert(.overStriding)
            } else if profile.isHighBMI, footAhead > profile.maxFootAheadOffset {
                lines.append("\(label)腳步稍大，高 BMI 建議縮小步幅並提高步頻以減少衝擊。")
                issues.insert(.overStriding)
            }

            let span = ankle.y - hip.y
            if span > 0.02 {
                let flexRatio = (knee.y - hip.y) / span
                if profile.isHighBMI, flexRatio < profile.minKneeFlexionRatio {
                    lines.append("\(label)膝蓋觸地時屈曲不足，請刻意微彎膝蓋做緩衝。")
                    issues.insert(.insufficientKneeFlexion)
                }
            }

            // 腳踝抬離量（y 越小＝越高）； clearance = pelvisY - ankle.y
            let clearance = pelvisY - ankle.y
            if profile.isMiddleAgedPlus, clearance < profile.minAnkleClearance, ok(heel) {
                lines.append("\(label)腳尖可能未充分離地（拖步），請刻意抬腳尖、小步慢行。")
                issues.insert(.footDrag)
            }

            // 青壯年：推蹬期髖伸展（膝在髖後方）
            if profile.isYoungAdult, span > 0.025 {
                let hipExtension = hip.x - knee.x
                if hipExtension < profile.minHipExtensionOffset {
                    lines.append("\(label)推蹬時髖伸展偏小，可試著在後腳離地前將膝蓋略往髖後帶。")
                    issues.insert(.limitedHipExtension)
                }
            }
        }

        let onlyCoreOrPraise = issues == formIssues && (
            lines == WalkingFormAdvisor.corePrincipleLines
            || lines.contains(where: { $0.contains("對側擺臂") || $0.contains("腳跟先著地") })
        )
        if onlyCoreOrPraise {
            if profile.isMiddleAgedPlus {
                lines.append("步態提示：步幅略小、擺臂減少在中高齡屬正常；重點是軀幹直立、重心穩定、避免拖步。")
            } else if profile.isHighBMI {
                lines.append("步態提示：偏重體型建議「小步幅、高步頻」，觸地時保持膝蓋微彎。")
            } else if profile.isTall {
                lines.append("步態提示：高身高者請控制跨步，腳掌儘量落在骨盆正下方。")
            } else if profile.isYoungAdult {
                lines.append("步態提示：青壯年可在腳尖蹬地時帶出髖伸展，維持軀幹穩定即可。")
            }
        }

        return (lines, issues)
    }
}

// MARK: - 累積指標（供整段摘要）

struct GaitSessionMetrics {
    var pelvisYValues: [Double] = []
    var overStrideFrames: Int = 0
    var lowKneeFlexFrames: Int = 0
    var footDragFrames: Int = 0
    var limitedHipExtensionFrames: Int = 0
    var trunkInstabilityFrames: Int = 0
    var evaluatedFrames: Int = 0
    var walkingFormFrames: Int = 0
    var raisedArmFrames: Int = 0
    var excessiveArmFrames: Int = 0
    var ipsilateralArmFrames: Int = 0
    var forefootStrikeFrames: Int = 0
    var incompleteToeOffFrames: Int = 0
    var goodArmSwingFrames: Int = 0
    var goodFootRollFrames: Int = 0

    mutating func ingest(landmarks: QuickPose.Landmarks, profile: BodyGaitProfile?) {
        WalkingFormAdvisor.accumulate(metrics: &self, snapshot: WalkingFormAdvisor.snapshot(from: landmarks))
        guard let profile else { return }

        func ok(_ p: QuickPose.Point3d) -> Bool {
            p.visibility > 0.35 && p.presence > 0.35
        }

        let lh = landmarks.landmark(forBody: .hip(side: .left))
        let rh = landmarks.landmark(forBody: .hip(side: .right))
        guard ok(lh), ok(rh) else { return }

        evaluatedFrames += 1
        let pelvisY = (lh.y + rh.y) / 2
        pelvisYValues.append(pelvisY)
        let hipMidX = (lh.x + rh.x) / 2

        if profile.isMiddleAgedPlus {
            let ls = landmarks.landmark(forBody: .shoulder(side: .left))
            let rs = landmarks.landmark(forBody: .shoulder(side: .right))
            if ok(ls), ok(rs) {
                let trunkLean = abs((ls.x + rs.x) / 2 - hipMidX)
                if trunkLean > 0.07 { trunkInstabilityFrames += 1 }
            }
        }

        for side: QuickPose.Side in [.left, .right] {
            let ankle = landmarks.landmark(forBody: .ankle(side: side))
            let knee = landmarks.landmark(forBody: .knee(side: side))
            let hip = landmarks.landmark(forBody: .hip(side: side))
            guard ok(ankle), ok(knee), ok(hip) else { continue }

            if ankle.x - hipMidX > profile.maxFootAheadOffset {
                overStrideFrames += 1
            }

            let span = ankle.y - hip.y
            if span > 0.02 {
                let flexRatio = (knee.y - hip.y) / span
                if profile.isHighBMI, flexRatio < profile.minKneeFlexionRatio {
                    lowKneeFlexFrames += 1
                }
            }

            let clearance = pelvisY - ankle.y
            if profile.isMiddleAgedPlus, clearance < profile.minAnkleClearance {
                footDragFrames += 1
            }

            if profile.isYoungAdult, span > 0.025, (hip.x - knee.x) < profile.minHipExtensionOffset {
                limitedHipExtensionFrames += 1
            }
        }
    }

    var verticalOscillation: Double? {
        guard pelvisYValues.count >= 10 else { return nil }
        guard let minY = pelvisYValues.min(), let maxY = pelvisYValues.max() else { return nil }
        return maxY - minY
    }
}

enum GaitPersonalizationSummary {
    static func lines(
        profile: BodyGaitProfile?,
        avgCadence: Double?,
        totalSteps: Int,
        metrics: GaitSessionMetrics
    ) -> [String] {
        guard let profile, profile.hasPersonalization else { return [] }

        var out: [String] = []
        if let summary = profile.profileSummaryLine {
            out.append(summary)
        }

        if let cadence = avgCadence, totalSteps >= 4 {
            if profile.isMiddleAgedPlus {
                if cadence >= profile.recommendedMinCadence {
                    out.append(String(format: "步頻 %.0f bpm 穩定，符合中高齡安全步態（≥%.0f）；步幅略小屬正常。", cadence, profile.recommendedMinCadence))
                } else if cadence < 65 {
                    out.append(String(format: "步頻 %.0f bpm 偏低，若伴隨拖步請提高警覺；可試小步、稍快節奏。", cadence))
                }
            } else if profile.isHighBMI {
                if cadence >= profile.recommendedMinCadence {
                    out.append(String(format: "步頻 %.0f bpm 符合偏重體型建議（≥%.0f），有助縮小步幅、降低衝擊。", cadence, profile.recommendedMinCadence))
                } else {
                    out.append(String(format: "步頻 %.0f bpm 偏低，高 BMI 建議提高至 ≥%.0f bpm，配合較小步幅。", cadence, profile.recommendedMinCadence))
                }
            } else if profile.isTall, cadence < profile.recommendedMinCadence {
                out.append(String(format: "步頻 %.0f bpm 可再提高，長腿者較高步頻有助避免過度跨步。", cadence))
            } else if profile.isYoungAdult, cadence >= profile.recommendedMinCadence {
                out.append(String(format: "步頻 %.0f bpm 活潑，青壯年可在此節奏下維持推蹬與髖伸展。", cadence))
            }
        }

        if let vo = metrics.verticalOscillation {
            if vo <= profile.lowVerticalOscillationPraiseThreshold, profile.isHighBMI || profile.isMiddleAgedPlus {
                let note = profile.isMiddleAgedPlus
                    ? "對中高齡而言是常見且合理的安全步態（低彈跳、貼地感）。"
                    : "對高 BMI 而言是常見且合理的自我保護步態（貼地滑移感）。"
                out.append("垂直振幅偏低（約 \(String(format: "%.2f", vo))），\(note)")
            } else if profile.isHighBMI {
                out.append("垂直振幅略高，偏重體型可再降低彈跳、縮小步幅以減輕關節負荷。")
            }
        }

        let frames = max(metrics.evaluatedFrames, 1)
        let overPct = Double(metrics.overStrideFrames) / Double(frames) * 100
        if overPct >= 12, (profile.isTall || profile.isHighBMI) && !profile.isMiddleAgedPlus {
            out.append(String(format: "約 %.0f%% 時段腳掌落在重心前方，建議縮步並落在骨盆正下方。", overPct))
        }

        let kneePct = Double(metrics.lowKneeFlexFrames) / Double(frames) * 100
        if kneePct >= 12, profile.isHighBMI {
            out.append(String(format: "約 %.0f%% 時段觸地膝屈曲不足，請加強觸地瞬間的微彎緩衝。", kneePct))
        }

        let dragPct = Double(metrics.footDragFrames) / Double(frames) * 100
        if dragPct >= 10, profile.isMiddleAgedPlus {
            out.append(String(format: "約 %.0f%% 時段疑似拖步（腳尖未充分離地），請提高抬腳意識以防絆倒。", dragPct))
        }

        let trunkPct = Double(metrics.trunkInstabilityFrames) / Double(frames) * 100
        if trunkPct >= 10, profile.isMiddleAgedPlus {
            out.append(String(format: "約 %.0f%% 時段軀幹偏移，請優先練習直立與重心轉移平穩。", trunkPct))
        } else if profile.isMiddleAgedPlus, dragPct < 8, trunkPct < 8 {
            out.append("整體步態穩定：即使步幅較小、擺臂減少，仍符合中高齡安全姿勢標準。")
        }

        let hipPct = Double(metrics.limitedHipExtensionFrames) / Double(frames) * 100
        if hipPct >= 12, profile.isYoungAdult {
            out.append(String(format: "約 %.0f%% 時段髖伸展偏小，可加强後腳推蹬與踝背屈。", hipPct))
        }

        return out
    }
}
