//
//  WalkingFormAdvisor.swift
//  pose
//
//  走路正確姿勢核心：
//  1) 手臂自然下垂，隨著對側腳步前後擺動，幅度不宜過大。
//  2) 腳跟先著地，力量順勢平穩地過渡到腳掌，最後由腳尖蹬地推動身體前進。
//

import Foundation
import QuickPoseCore

struct WalkingJointSample: Equatable {
    var x: Double
    var y: Double
    var z: Double
    var visibility: Double
    var presence: Double

    var isVisible: Bool { visibility > 0.35 && presence > 0.35 }

    static func visible(x: Double, y: Double, z: Double = 0) -> WalkingJointSample {
        WalkingJointSample(x: x, y: y, z: z, visibility: 0.95, presence: 0.95)
    }
}

struct WalkingFormSnapshot: Equatable {
    var leftShoulder: WalkingJointSample
    var rightShoulder: WalkingJointSample
    var leftElbow: WalkingJointSample
    var rightElbow: WalkingJointSample
    var leftWrist: WalkingJointSample
    var rightWrist: WalkingJointSample
    var leftHip: WalkingJointSample
    var rightHip: WalkingJointSample
    var leftAnkle: WalkingJointSample
    var rightAnkle: WalkingJointSample
    var leftHeel: WalkingJointSample
    var rightHeel: WalkingJointSample
    var leftFootIndex: WalkingJointSample
    var rightFootIndex: WalkingJointSample
}

enum WalkingFormAdvisor {
    static let corePrincipleLines: [String] = [
        "手臂自然下垂，隨著對側腳步前後擺動，幅度不宜過大。",
        "腳跟先著地，力量順勢平穩地過渡到腳掌，最後由腳尖蹬地推動身體前進。"
    ]

    static func snapshot(from landmarks: QuickPose.Landmarks) -> WalkingFormSnapshot {
        func sample(_ joint: QuickPose.Landmarks.Body) -> WalkingJointSample {
            let p = landmarks.landmark(forBody: joint)
            return WalkingJointSample(x: p.x, y: p.y, z: p.z, visibility: p.visibility, presence: p.presence)
        }
        return WalkingFormSnapshot(
            leftShoulder: sample(.shoulder(side: .left)),
            rightShoulder: sample(.shoulder(side: .right)),
            leftElbow: sample(.elbow(side: .left)),
            rightElbow: sample(.elbow(side: .right)),
            leftWrist: sample(.wrist(side: .left)),
            rightWrist: sample(.wrist(side: .right)),
            leftHip: sample(.hip(side: .left)),
            rightHip: sample(.hip(side: .right)),
            leftAnkle: sample(.ankle(side: .left)),
            rightAnkle: sample(.ankle(side: .right)),
            leftHeel: sample(.heel(side: .left)),
            rightHeel: sample(.heel(side: .right)),
            leftFootIndex: sample(.footIndex(side: .left)),
            rightFootIndex: sample(.footIndex(side: .right))
        )
    }

    static func evaluate(from landmarks: QuickPose.Landmarks) -> (lines: [String], issues: Set<PoseIssueCode>) {
        evaluate(snapshot(from: landmarks))
    }

    static func evaluate(_ s: WalkingFormSnapshot) -> (lines: [String], issues: Set<PoseIssueCode>) {
        var lines: [String] = []
        var issues = Set<PoseIssueCode>()

        let arms = evaluateArms(s)
        lines.append(contentsOf: arms.lines)
        issues.formUnion(arms.issues)

        let feet = evaluateFeet(s)
        lines.append(contentsOf: feet.lines)
        issues.formUnion(feet.issues)

        if lines.isEmpty, arms.evaluated || feet.evaluated {
            if arms.evaluated, arms.issues.isEmpty {
                lines.append("對側擺臂自然，幅度適中。")
            }
            if feet.evaluated, feet.issues.isEmpty {
                lines.append("腳跟先著地，力量平穩過渡到腳掌，再由腳尖蹬地。")
            }
        }
        if lines.isEmpty {
            lines = corePrincipleLines
        }
        return (lines, issues)
    }

    static func accumulate(metrics: inout GaitSessionMetrics, snapshot s: WalkingFormSnapshot) {
        let result = evaluate(s)
        let arms = evaluateArms(s)
        let feet = evaluateFeet(s)
        guard arms.evaluated || feet.evaluated else { return }
        metrics.walkingFormFrames += 1
        if result.issues.contains(.raisedArms) { metrics.raisedArmFrames += 1 }
        if result.issues.contains(.excessiveArmSwing) { metrics.excessiveArmFrames += 1 }
        if result.issues.contains(.ipsilateralArmSwing) { metrics.ipsilateralArmFrames += 1 }
        if result.issues.contains(.forefootStrike) { metrics.forefootStrikeFrames += 1 }
        if result.issues.contains(.incompleteToeOff) { metrics.incompleteToeOffFrames += 1 }
        if arms.evaluated, arms.issues.isEmpty { metrics.goodArmSwingFrames += 1 }
        if feet.evaluated, feet.issues.isEmpty { metrics.goodFootRollFrames += 1 }
    }

    static func sessionSummary(metrics: GaitSessionMetrics) -> [String] {
        var out = ["走路正確姿勢核心："] + corePrincipleLines
        let frames = metrics.walkingFormFrames
        guard frames >= 8 else { return out }

        func pct(_ count: Int) -> Double {
            Double(count) / Double(frames) * 100
        }

        if pct(metrics.raisedArmFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段手臂未自然下垂，請把手臂放鬆垂在身體兩側。", pct(metrics.raisedArmFrames)))
        }
        if pct(metrics.excessiveArmFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段擺臂幅度偏大，前後擺動即可，不必用力甩。", pct(metrics.excessiveArmFrames)))
        }
        if pct(metrics.ipsilateralArmFrames) >= 15 {
            out.append(String(format: "約 %.0f%% 時段同側手腳一起向前，請改為對側擺臂（左手配右腳）。", pct(metrics.ipsilateralArmFrames)))
        }
        if pct(metrics.forefootStrikeFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段腳尖或前腳掌先著地，請改為腳跟先落地。", pct(metrics.forefootStrikeFrames)))
        }
        if pct(metrics.incompleteToeOffFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段後腳缺少腳尖蹬地，離地前請把力量推到腳尖。", pct(metrics.incompleteToeOffFrames)))
        }

        let armGood = pct(metrics.goodArmSwingFrames) >= 70
            && pct(metrics.ipsilateralArmFrames) < 12
            && pct(metrics.excessiveArmFrames) < 12
            && pct(metrics.raisedArmFrames) < 12
        let footGood = pct(metrics.goodFootRollFrames) >= 70
            && pct(metrics.forefootStrikeFrames) < 12
            && pct(metrics.incompleteToeOffFrames) < 12
        if armGood, footGood {
            out.append("本段擺臂與腳跟→腳掌→腳尖滾動符合正確走路姿勢。")
        }
        return out
    }

    private struct PartResult {
        var lines: [String]
        var issues: Set<PoseIssueCode>
        var evaluated: Bool
    }

    private static func evaluateArms(_ s: WalkingFormSnapshot) -> PartResult {
        let joints = [s.leftShoulder, s.rightShoulder, s.leftWrist, s.rightWrist, s.leftHip, s.rightHip]
        guard joints.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()
        let hipMidX = (s.leftHip.x + s.rightHip.x) / 2

        let leftHang = s.leftWrist.y - s.leftShoulder.y
        let rightHang = s.rightWrist.y - s.rightShoulder.y
        if leftHang < 0.03, rightHang < 0.03 {
            lines.append("手臂未自然下垂，請放鬆肩膀，讓手臂垂在身體兩側再隨步伐擺動。")
            issues.insert(.raisedArms)
        }

        let leftAmp = abs(s.leftWrist.x - hipMidX)
        let rightAmp = abs(s.rightWrist.x - hipMidX)
        if leftAmp > 0.24 || rightAmp > 0.24 {
            lines.append("擺臂幅度偏大，前後自然擺動即可，不必過度前甩或後甩。")
            issues.insert(.excessiveArmSwing)
        }

        if s.leftAnkle.isVisible, s.rightAnkle.isVisible {
            let footX = abs(s.leftAnkle.x - s.rightAnkle.x)
            let footZ = abs(s.leftAnkle.z - s.rightAnkle.z)
            if max(footX, footZ) > 0.045 {
                let footLead: Double
                let armLead: Double
                if footZ > footX {
                    footLead = s.rightAnkle.z - s.leftAnkle.z
                    armLead = s.rightWrist.z - s.leftWrist.z
                } else {
                    footLead = s.leftAnkle.x - s.rightAnkle.x
                    armLead = s.leftWrist.x - s.rightWrist.x
                }
                if abs(footLead) > 0.03, abs(armLead) > 0.02, footLead * armLead > 0.0008 {
                    lines.append("手腳同側一起向前，請改為對側擺臂：左手配合右腳、右手配合左腳。")
                    issues.insert(.ipsilateralArmSwing)
                }
            }
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }

    private static func evaluateFeet(_ s: WalkingFormSnapshot) -> PartResult {
        let needed = [s.leftAnkle, s.rightAnkle, s.leftHeel, s.rightHeel, s.leftFootIndex, s.rightFootIndex]
        guard needed.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()

        let leftLower = s.leftAnkle.y >= s.rightAnkle.y
        let stanceHeel = leftLower ? s.leftHeel : s.rightHeel
        let stanceToe = leftLower ? s.leftFootIndex : s.rightFootIndex
        let rearHeel = leftLower ? s.rightHeel : s.leftHeel
        let rearToe = leftLower ? s.rightFootIndex : s.leftFootIndex
        let stanceLabel = leftLower ? "左" : "右"
        let rearLabel = leftLower ? "右" : "左"

        if stanceToe.y - stanceHeel.y > 0.018 {
            lines.append("\(stanceLabel)腳尖或前腳掌先著地，請改為腳跟先落地，再把力量平穩過渡到腳掌。")
            issues.insert(.forefootStrike)
        }

        let footSep = abs(s.leftAnkle.x - s.rightAnkle.x)
        let depthSep = abs(s.leftAnkle.z - s.rightAnkle.z)
        if max(footSep, depthSep) > 0.05, rearHeel.y + 0.008 >= rearToe.y {
            lines.append("\(rearLabel)腳離地前缺少腳尖蹬地，請讓力量走到腳尖再推動身體前進。")
            issues.insert(.incompleteToeOff)
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }
}
