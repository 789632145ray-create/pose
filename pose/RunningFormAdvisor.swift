//
//  RunningFormAdvisor.swift
//  pose
//
//  跑步正確姿勢核心：
//  1) 由腳踝發力，身體整體微微前傾約 10 度，不是從腰彎。
//  2) 收腹、骨盆穩定，避免左右搖晃或上下起伏過大。
//  3) 腳掌落在重心正下方，避免過度跨步，以中足著地。
//  4) 手肘約 90 度，輕握拳，手臂前後擺、不越過中線。
//

import Foundation
import QuickPoseCore

enum RunningFormAdvisor {
    static let corePrincipleLines: [String] = [
        "由腳踝發力，讓身體整體微微前傾（約 10 度），而非從腰部彎曲。利用地心引力協助身體重心前移。",
        "腹部微微收緊，保持骨盆穩定，避免跑步時身體左右過度搖晃或上下起伏過大。",
        "腳掌應落在「身體重心的正下方」，避免過度跨步（Overstriding）。多數建議以中足（腳掌中部）著地，能最有效利用足弓緩衝對膝蓋的衝擊。",
        "手肘彎曲約 90 度，雙手輕握拳。手臂前後擺動，擺動軌跡不要越過身體中線。"
    ]

    static func evaluate(from landmarks: QuickPose.Landmarks) -> (lines: [String], issues: Set<PoseIssueCode>) {
        evaluate(WalkingFormAdvisor.snapshot(from: landmarks))
    }

    static func evaluate(_ s: WalkingFormSnapshot) -> (lines: [String], issues: Set<PoseIssueCode>) {
        var lines: [String] = []
        var issues = Set<PoseIssueCode>()

        let lean = evaluateLean(s)
        lines.append(contentsOf: lean.lines)
        issues.formUnion(lean.issues)

        let core = evaluateCore(s)
        lines.append(contentsOf: core.lines)
        issues.formUnion(core.issues)

        let feet = evaluateFeet(s)
        lines.append(contentsOf: feet.lines)
        issues.formUnion(feet.issues)

        let arms = evaluateArms(s)
        lines.append(contentsOf: arms.lines)
        issues.formUnion(arms.issues)

        if lines.isEmpty, lean.evaluated || core.evaluated || feet.evaluated || arms.evaluated {
            if lean.evaluated, lean.issues.isEmpty {
                lines.append("身體整體微前傾，不是從腰部彎曲。")
            }
            if core.evaluated, core.issues.isEmpty {
                lines.append("腹部收緊、骨盆穩定。")
            }
            if feet.evaluated, feet.issues.isEmpty {
                lines.append("中足落在重心正下方。")
            }
            if arms.evaluated, arms.issues.isEmpty {
                lines.append("手肘約 90 度，手臂前後擺、未越過中線。")
            }
        }
        if lines.isEmpty {
            lines = corePrincipleLines
        }
        return (lines, issues)
    }

    static func accumulate(metrics: inout GaitSessionMetrics, snapshot s: WalkingFormSnapshot) {
        let result = evaluate(s)
        let lean = evaluateLean(s)
        let core = evaluateCore(s)
        let feet = evaluateFeet(s)
        let arms = evaluateArms(s)
        guard lean.evaluated || core.evaluated || feet.evaluated || arms.evaluated else { return }
        metrics.runningFormFrames += 1
        if result.issues.contains(.waistHingeLean) { metrics.waistHingeFrames += 1 }
        if result.issues.contains(.insufficientForwardLean) { metrics.insufficientLeanFrames += 1 }
        if result.issues.contains(.excessiveForwardLean) { metrics.excessiveLeanFrames += 1 }
        if result.issues.contains(.pelvisInstability) { metrics.pelvisUnstableFrames += 1 }
        if result.issues.contains(.overStriding) { metrics.runningOverstrideFrames += 1 }
        if result.issues.contains(.heelStrikeWhileRunning) { metrics.heelStrikeRunFrames += 1 }
        if result.issues.contains(.elbowAngleOff) { metrics.elbowAngleOffFrames += 1 }
        if result.issues.contains(.armCrossMidline) { metrics.armCrossMidlineFrames += 1 }
        if lean.evaluated, lean.issues.isEmpty { metrics.goodRunningLeanFrames += 1 }
        if core.evaluated, core.issues.isEmpty { metrics.goodRunningCoreFrames += 1 }
        if feet.evaluated, feet.issues.isEmpty { metrics.goodRunningFootFrames += 1 }
        if arms.evaluated, arms.issues.isEmpty { metrics.goodRunningArmFrames += 1 }
    }

    static func sessionSummary(metrics: GaitSessionMetrics) -> [String] {
        var out = ["跑步正確姿勢核心："] + corePrincipleLines
        let frames = metrics.runningFormFrames
        guard frames >= 8 else { return out }

        func pct(_ count: Int) -> Double {
            Double(count) / Double(frames) * 100
        }

        if pct(metrics.waistHingeFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段像從腰部往前折，請改由腳踝發力、讓全身一起微前傾。", pct(metrics.waistHingeFrames)))
        }
        if pct(metrics.insufficientLeanFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段身體太直立，可整體再前傾約 10 度，讓重心跟著地心引力前移。", pct(metrics.insufficientLeanFrames)))
        }
        if pct(metrics.excessiveLeanFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段前傾過大，收回約 10 度，避免從腰往下栽。", pct(metrics.excessiveLeanFrames)))
        }
        if pct(metrics.pelvisUnstableFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段骨盆或軀幹左右晃，請微收腹、穩定骨盆。", pct(metrics.pelvisUnstableFrames)))
        }
        if pct(metrics.runningOverstrideFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段腳掌落在重心前方（過度跨步），請讓腳落在身體正下方。", pct(metrics.runningOverstrideFrames)))
        }
        if pct(metrics.heelStrikeRunFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段用腳跟著地，跑步請改中足著地，讓足弓緩衝膝蓋。", pct(metrics.heelStrikeRunFrames)))
        }
        if pct(metrics.elbowAngleOffFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段手肘角度偏離 90 度，請彎曲手肘、雙手輕握拳。", pct(metrics.elbowAngleOffFrames)))
        }
        if pct(metrics.armCrossMidlineFrames) >= 12 {
            out.append(String(format: "約 %.0f%% 時段手臂越過身體中線，請改為前後擺動。", pct(metrics.armCrossMidlineFrames)))
        }

        let good = pct(metrics.goodRunningLeanFrames) >= 65
            && pct(metrics.goodRunningCoreFrames) >= 65
            && pct(metrics.goodRunningFootFrames) >= 65
            && pct(metrics.goodRunningArmFrames) >= 65
            && pct(metrics.waistHingeFrames) < 12
            && pct(metrics.runningOverstrideFrames) < 12
            && pct(metrics.heelStrikeRunFrames) < 12
        if good {
            out.append("本段前傾、骨盆、中足著地與擺臂符合正確跑步姿勢。")
        }
        return out
    }

    static func elbowAngleDegrees(
        shoulder: WalkingJointSample,
        elbow: WalkingJointSample,
        wrist: WalkingJointSample
    ) -> Double? {
        let ux = shoulder.x - elbow.x
        let uy = shoulder.y - elbow.y
        let vx = wrist.x - elbow.x
        let vy = wrist.y - elbow.y
        let du = hypot(ux, uy)
        let dv = hypot(vx, vy)
        guard du > 0.01, dv > 0.01 else { return nil }
        let cos = max(-1, min(1, (ux * vx + uy * vy) / (du * dv)))
        return acos(cos) * 180 / .pi
    }

    static func forwardLeanDegrees(_ s: WalkingFormSnapshot) -> Double? {
        let needed = [s.leftShoulder, s.rightShoulder, s.leftAnkle, s.rightAnkle]
        guard needed.allSatisfy(\.isVisible) else { return nil }
        let shoulderMidX = (s.leftShoulder.x + s.rightShoulder.x) / 2
        let shoulderMidY = (s.leftShoulder.y + s.rightShoulder.y) / 2
        let ankleMidX = (s.leftAnkle.x + s.rightAnkle.x) / 2
        let ankleMidY = (s.leftAnkle.y + s.rightAnkle.y) / 2
        let dy = ankleMidY - shoulderMidY
        guard dy > 0.08 else { return nil }
        return abs(atan((shoulderMidX - ankleMidX) / dy)) * 180 / .pi
    }

    private struct PartResult {
        var lines: [String]
        var issues: Set<PoseIssueCode>
        var evaluated: Bool
    }

    private static func evaluateLean(_ s: WalkingFormSnapshot) -> PartResult {
        let needed = [s.leftShoulder, s.rightShoulder, s.leftHip, s.rightHip, s.leftAnkle, s.rightAnkle]
        guard needed.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()
        let shoulderMidX = (s.leftShoulder.x + s.rightShoulder.x) / 2
        let hipMidX = (s.leftHip.x + s.rightHip.x) / 2
        let ankleMidX = (s.leftAnkle.x + s.rightAnkle.x) / 2
        let torsoDx = shoulderMidX - hipMidX
        let hipDx = hipMidX - ankleMidX

        if abs(torsoDx) > 0.055, abs(hipDx) < 0.02 {
            lines.append("前傾是從腰部彎下去，請改由腳踝發力，讓身體整體一起微微前傾約 10 度。")
            issues.insert(.waistHingeLean)
        } else if let lean = forwardLeanDegrees(s) {
            if lean < 4 {
                lines.append("身體太直立，可由腳踝帶動全身再前傾約 10 度，讓地心引力幫忙把重心前移。")
                issues.insert(.insufficientForwardLean)
            } else if lean > 22 {
                lines.append("前傾角度偏大，收回大約 10 度，避免從腰往下栽。")
                issues.insert(.excessiveForwardLean)
            }
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }

    private static func evaluateCore(_ s: WalkingFormSnapshot) -> PartResult {
        let needed = [s.leftShoulder, s.rightShoulder, s.leftHip, s.rightHip]
        guard needed.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()
        let hipTilt = abs(s.leftHip.y - s.rightHip.y)
        let shoulderMidX = (s.leftShoulder.x + s.rightShoulder.x) / 2
        let hipMidX = (s.leftHip.x + s.rightHip.x) / 2
        let sway = abs(shoulderMidX - hipMidX)

        if hipTilt > 0.05 || sway > 0.075 {
            lines.append("骨盆或軀幹左右晃偏大，請微收腹、穩定骨盆，減少左右搖晃與上下彈跳。")
            issues.insert(.pelvisInstability)
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }

    private static func evaluateFeet(_ s: WalkingFormSnapshot) -> PartResult {
        let needed = [s.leftHip, s.rightHip, s.leftAnkle, s.rightAnkle, s.leftHeel, s.rightHeel, s.leftFootIndex, s.rightFootIndex]
        guard needed.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()
        let hipMidX = (s.leftHip.x + s.rightHip.x) / 2
        let leftAhead = s.leftAnkle.x - hipMidX
        let rightAhead = s.rightAnkle.x - hipMidX
        if max(leftAhead, rightAhead) > 0.075 {
            lines.append("腳掌落在重心前方，這是過度跨步。請縮步，讓腳落在身體正下方。")
            issues.insert(.overStriding)
        }

        let leftLower = s.leftAnkle.y >= s.rightAnkle.y
        let stanceHeel = leftLower ? s.leftHeel : s.rightHeel
        let stanceToe = leftLower ? s.leftFootIndex : s.rightFootIndex
        let heelFirst = stanceHeel.y - stanceToe.y
        if heelFirst > 0.022 {
            lines.append("跑步時腳跟先著地，請改以中足著地，用足弓緩衝膝蓋衝擊。")
            issues.insert(.heelStrikeWhileRunning)
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }

    private static func evaluateArms(_ s: WalkingFormSnapshot) -> PartResult {
        let needed = [s.leftShoulder, s.rightShoulder, s.leftElbow, s.rightElbow, s.leftWrist, s.rightWrist]
        guard needed.allSatisfy(\.isVisible) else {
            return PartResult(lines: [], issues: [], evaluated: false)
        }

        var lines: [String] = []
        var issues = Set<PoseIssueCode>()

        let leftAngle = elbowAngleDegrees(shoulder: s.leftShoulder, elbow: s.leftElbow, wrist: s.leftWrist)
        let rightAngle = elbowAngleDegrees(shoulder: s.rightShoulder, elbow: s.rightElbow, wrist: s.rightWrist)
        let leftOff = leftAngle.map { $0 < 68 || $0 > 118 } ?? false
        let rightOff = rightAngle.map { $0 < 68 || $0 > 118 } ?? false
        if leftOff || rightOff {
            lines.append("手肘角度偏離約 90 度，請彎曲手肘、雙手輕握拳，再做前後擺臂。")
            issues.insert(.elbowAngleOff)
        }

        let shoulderWidth = abs(s.leftShoulder.x - s.rightShoulder.x)
        if shoulderWidth > 0.10 {
            let mid = (s.leftShoulder.x + s.rightShoulder.x) / 2
            if s.leftWrist.x > mid + 0.018 || s.rightWrist.x < mid - 0.018 {
                lines.append("手臂擺動越過身體中線，請改為前後擺，不要橫向甩過胸口。")
                issues.insert(.armCrossMidline)
            }
        }

        return PartResult(lines: lines, issues: issues, evaluated: true)
    }
}
