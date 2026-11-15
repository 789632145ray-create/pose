//
//  BackViewPosePipeline.swift
//  pose
//
//  背面視角（Back View）姿勢處理與特徵提取：
//  1) 關鍵點分級信心過濾（核心 >= 0.7，手臂 >= 0.25／可忽略）
//  2) 中值濾波抑制左右互換與單幀抖動
//  3) 骨盆原點＋軀幹尺度正規化
//  4) 骨盆傾斜／軀幹側傾／肩水平／膝內夾代理／步態對稱
//  5) 滑動視窗防抖後才觸發異常提醒
//
//  左右定義：使用模型解剖學 left/right（非自拍鏡像）。背面拍攝時勿再做左右翻轉標籤。
//  頭部：不依賴鼻／眼，改用雙耳中點相對雙肩中點推估歪斜。
//

import Foundation
import QuickPoseCore

// MARK: - 幀特徵輸出

struct BackViewFrameFeatures: Equatable {
    var timestamp: TimeInterval
    /// 骨盆左右傾斜角（度）；左髖高於右髖為正。
    var pelvicTiltDeg: Double?
    /// 軀幹左右側傾角（度）；肩中點在髖中點右側為正。
    var trunkLateralLeanDeg: Double?
    /// 肩膀水平度（左肩 y − 右肩 y，已除以軀幹尺度；影像 y 向下為正）。
    var shoulderAlignmentNorm: Double?
    /// 左／右膝相對髖–踝連線的橫向偏移（正規化）。
    var leftKneeValgusProxy: Double?
    var rightKneeValgusProxy: Double?
    /// 左右腳踝水平距離（步寬，正規化）。
    var stepWidthNorm: Double?
    /// 左／右腳踝相對骨盆的垂直抬腳高度（正規化；越大越高）。
    var leftAnkleClearanceNorm: Double?
    var rightAnkleClearanceNorm: Double?
    /// 頭部相對肩中點的水平偏移（耳中點 − 肩中點，正規化）。
    var headLateralOffsetNorm: Double?
    /// 本幀核心關節是否足夠可信。
    var coreReliable: Bool
    /// 防抖後的異常代碼。
    var debouncedIssues: Set<PoseIssueCode>
    /// 給 HUD／摘要用的提示句。
    var adviceLines: [String]
}

// MARK: - 管線

@MainActor
final class BackViewPosePipeline {
    /// 核心穩定區：肩、髖、膝、踝。
    private static let coreJoints: Set<String> = [
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]
    /// 易遮擋區：肘、腕與手部。
    private static let armJoints: Set<String> = [
        "left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb",
        "right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"
    ]

    var coreConfidenceThreshold: Double = 0.7
    var armConfidenceThreshold: Double = 0.25
    /// 中值濾波視窗（建議 3~5）。
    var medianWindow: Int = 5
    /// 連續超過門檻的幀數才觸發異常。
    var debounceFrames: Int = 5

    // 門檻（正規化座標／度數）
    var pelvicTiltAlertDeg: Double = 8
    var trunkLeanAlertDeg: Double = 10
    var shoulderAlignAlert: Double = 0.08
    var kneeValgusAlert: Double = 0.12
    var headOffsetAlert: Double = 0.12
    var gaitAsymmetryAlert: Double = 0.25

    private var history: [[String: PoseNode]] = []
    private var lastTrusted: [String: PoseNode] = [:]
    private var issueStreak: [PoseIssueCode: Int] = [:]
    private var recentClearanceL: [Double] = []
    private var recentClearanceR: [Double] = []
    private var recentStepWidth: [Double] = []
    private var lastFeatures: BackViewFrameFeatures?

    var latestFeatures: BackViewFrameFeatures? { lastFeatures }

    func reset() {
        history.removeAll(keepingCapacity: true)
        lastTrusted.removeAll(keepingCapacity: true)
        issueStreak.removeAll()
        recentClearanceL.removeAll()
        recentClearanceR.removeAll()
        recentStepWidth.removeAll()
        lastFeatures = nil
    }

    /// 從 QuickPose landmarks 走完整背面管線。
    func process(landmarks: QuickPose.Landmarks, timestamp: TimeInterval = Date().timeIntervalSince1970) -> BackViewFrameFeatures {
        process(nodes: PoseNodeExtractor.extractAll(from: landmarks), timestamp: timestamp)
    }

    /// 從已抽取的 PoseNode 走完整背面管線。
    func process(nodes: [PoseNode], timestamp: TimeInterval = Date().timeIntervalSince1970) -> BackViewFrameFeatures {
        let filtered = applyConfidenceFilter(nodes)
        let smoothed = pushMedianFilter(filtered)
        let normalized = normalizeToPelvis(smoothed)
        let features = extractFeatures(normalized, timestamp: timestamp)
        lastFeatures = features
        return features
    }

    // MARK: 1) 信心過濾

    private func confidence(of node: PoseNode) -> Double {
        min(node.visibility, node.presence)
    }

    private func applyConfidenceFilter(_ nodes: [PoseNode]) -> [String: PoseNode] {
        var out: [String: PoseNode] = [:]
        for node in nodes {
            let conf = confidence(of: node)
            let joint = node.joint

            if Self.coreJoints.contains(joint) {
                if conf >= coreConfidenceThreshold {
                    out[joint] = node
                    lastTrusted[joint] = node
                } else if let prev = lastTrusted[joint] {
                    // 核心區低信心：forward fill，避免整幀崩潰。
                    out[joint] = prev
                }
                continue
            }

            if Self.armJoints.contains(joint) {
                // 步態／軀幹特徵不依賴手臂；僅為骨架連續性保留。
                if conf >= armConfidenceThreshold {
                    out[joint] = node
                    lastTrusted[joint] = node
                } else if let prev = lastTrusted[joint] {
                    out[joint] = prev
                }
                continue
            }

            // 其餘（耳、口等）：略寬門檻，供頭部推估。
            if conf >= 0.25 {
                out[joint] = node
                lastTrusted[joint] = node
            } else if let prev = lastTrusted[joint] {
                out[joint] = prev
            }
        }
        return out
    }

    // MARK: 2) 中值濾波

    private func pushMedianFilter(_ frame: [String: PoseNode]) -> [String: PoseNode] {
        history.append(frame)
        let w = max(1, medianWindow)
        if history.count > w { history.removeFirst(history.count - w) }

        var smoothed: [String: PoseNode] = [:]
        let joints = Set(history.flatMap(\.keys))
        for joint in joints {
            let samples = history.compactMap { $0[joint] }
            guard !samples.isEmpty else { continue }
            smoothed[joint] = medianNode(samples, joint: joint)
        }
        return smoothed
    }

    private func medianNode(_ samples: [PoseNode], joint: String) -> PoseNode {
        func med(_ values: [Double]) -> Double {
            let s = values.sorted()
            let n = s.count
            if n == 0 { return 0 }
            if n % 2 == 1 { return s[n / 2] }
            return (s[n / 2 - 1] + s[n / 2]) / 2
        }
        return PoseNode(
            joint: joint,
            x: med(samples.map(\.x)),
            y: med(samples.map(\.y)),
            z: med(samples.map(\.z)),
            visibility: med(samples.map(\.visibility)),
            presence: med(samples.map(\.presence))
        )
    }

    // MARK: 3) 正規化

    private struct NormalizedSkeleton {
        var points: [String: (x: Double, y: Double, z: Double)]
        var scale: Double
        var coreReliable: Bool
    }

    private func normalizeToPelvis(_ frame: [String: PoseNode]) -> NormalizedSkeleton? {
        guard let lh = frame["left_hip"], let rh = frame["right_hip"] else { return nil }
        let pelvisX = (lh.x + rh.x) / 2
        let pelvisY = (lh.y + rh.y) / 2
        let pelvisZ = (lh.z + rh.z) / 2

        let ls = frame["left_shoulder"]
        let rs = frame["right_shoulder"]
        var scale: Double
        if let ls, let rs {
            let shoulderMidX = (ls.x + rs.x) / 2
            let shoulderMidY = (ls.y + rs.y) / 2
            let torso = hypot(shoulderMidX - pelvisX, shoulderMidY - pelvisY)
            let pelvisWidth = hypot(lh.x - rh.x, lh.y - rh.y)
            scale = max(torso, pelvisWidth, 1e-6)
        } else {
            scale = max(hypot(lh.x - rh.x, lh.y - rh.y), 1e-6)
        }

        var points: [String: (x: Double, y: Double, z: Double)] = [:]
        for (joint, node) in frame {
            points[joint] = (
                (node.x - pelvisX) / scale,
                (node.y - pelvisY) / scale,
                (node.z - pelvisZ) / scale
            )
        }

        let coreOK = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
            .allSatisfy { frame[$0] != nil }

        return NormalizedSkeleton(points: points, scale: scale, coreReliable: coreOK)
    }

    // MARK: 4) 特徵

    private func extractFeatures(_ skeleton: NormalizedSkeleton?, timestamp: TimeInterval) -> BackViewFrameFeatures {
        guard let skeleton else {
            return BackViewFrameFeatures(
                timestamp: timestamp,
                pelvicTiltDeg: nil, trunkLateralLeanDeg: nil, shoulderAlignmentNorm: nil,
                leftKneeValgusProxy: nil, rightKneeValgusProxy: nil,
                stepWidthNorm: nil, leftAnkleClearanceNorm: nil, rightAnkleClearanceNorm: nil,
                headLateralOffsetNorm: nil, coreReliable: false,
                debouncedIssues: [], adviceLines: ["背面視角：核心關節信心不足，暫不計算特徵。"]
            )
        }

        let p = skeleton.points
        let lh = p["left_hip"]
        let rh = p["right_hip"]
        let ls = p["left_shoulder"]
        let rs = p["right_shoulder"]
        let lk = p["left_knee"]
        let rk = p["right_knee"]
        let la = p["left_ankle"]
        let ra = p["right_ankle"]

        // 骨盆傾斜：atan2(Δy, Δx)，影像 y 向下。
        var pelvicTiltDeg: Double?
        if let lh, let rh {
            pelvicTiltDeg = atan2(lh.y - rh.y, lh.x - rh.x) * 180 / .pi
        }

        // 軀幹側傾：肩中點相對髖中點（原點）與垂直線夾角。
        var trunkLeanDeg: Double?
        if let ls, let rs {
            let shoulderMidX = (ls.x + rs.x) / 2
            let shoulderMidY = (ls.y + rs.y) / 2
            // 垂直向上在影像座標為 -Y。
            trunkLeanDeg = atan2(shoulderMidX, -shoulderMidY) * 180 / .pi
        }

        var shoulderAlign: Double?
        if let ls, let rs {
            shoulderAlign = ls.y - rs.y
        }

        func kneeValgus(hip: (x: Double, y: Double, z: Double)?,
                        knee: (x: Double, y: Double, z: Double)?,
                        ankle: (x: Double, y: Double, z: Double)?) -> Double? {
            guard let hip, let knee, let ankle else { return nil }
            let dx = ankle.x - hip.x
            let dy = ankle.y - hip.y
            let len2 = dx * dx + dy * dy
            guard len2 > 1e-8 else { return nil }
            // 膝點到髖–踝線的有號橫向距離（正規化座標）。
            let cross = (knee.x - hip.x) * dy - (knee.y - hip.y) * dx
            return cross / sqrt(len2)
        }

        let leftValgus = kneeValgus(hip: lh, knee: lk, ankle: la)
        let rightValgus = kneeValgus(hip: rh, knee: rk, ankle: ra)

        var stepWidth: Double?
        if let la, let ra {
            stepWidth = abs(la.x - ra.x)
            recentStepWidth.append(stepWidth!)
            if recentStepWidth.count > 30 { recentStepWidth.removeFirst(recentStepWidth.count - 30) }
        }

        // 抬腳高度：骨盆為原點，ankle.y 越小（越負）代表越高。
        var leftClear: Double?
        var rightClear: Double?
        if let la {
            leftClear = -la.y
            recentClearanceL.append(leftClear!)
            if recentClearanceL.count > 30 { recentClearanceL.removeFirst(recentClearanceL.count - 30) }
        }
        if let ra {
            rightClear = -ra.y
            recentClearanceR.append(rightClear!)
            if recentClearanceR.count > 30 { recentClearanceR.removeFirst(recentClearanceR.count - 30) }
        }

        // 頭部：雙耳中點 vs 雙肩中點（不使用鼻／眼）。
        var headOffset: Double?
        let le = p["left_ear"]
        let re = p["right_ear"]
        if let le, let re, let ls, let rs {
            let earMidX = (le.x + re.x) / 2
            let shoulderMidX = (ls.x + rs.x) / 2
            headOffset = earMidX - shoulderMidX
        }

        let rawIssues = evaluateRawIssues(
            pelvicTiltDeg: pelvicTiltDeg,
            trunkLeanDeg: trunkLeanDeg,
            shoulderAlign: shoulderAlign,
            leftValgus: leftValgus,
            rightValgus: rightValgus,
            headOffset: headOffset,
            coreReliable: skeleton.coreReliable
        )
        let debounced = debounce(rawIssues)
        let lines = advice(from: debounced, features: (
            pelvicTiltDeg, trunkLeanDeg, shoulderAlign, leftValgus, rightValgus, headOffset
        ))

        return BackViewFrameFeatures(
            timestamp: timestamp,
            pelvicTiltDeg: pelvicTiltDeg,
            trunkLateralLeanDeg: trunkLeanDeg,
            shoulderAlignmentNorm: shoulderAlign,
            leftKneeValgusProxy: leftValgus,
            rightKneeValgusProxy: rightValgus,
            stepWidthNorm: stepWidth,
            leftAnkleClearanceNorm: leftClear,
            rightAnkleClearanceNorm: rightClear,
            headLateralOffsetNorm: headOffset,
            coreReliable: skeleton.coreReliable,
            debouncedIssues: debounced,
            adviceLines: lines
        )
    }

    private func evaluateRawIssues(
        pelvicTiltDeg: Double?,
        trunkLeanDeg: Double?,
        shoulderAlign: Double?,
        leftValgus: Double?,
        rightValgus: Double?,
        headOffset: Double?,
        coreReliable: Bool
    ) -> Set<PoseIssueCode> {
        var issues = Set<PoseIssueCode>()
        if !coreReliable {
            issues.insert(.lowVisibility)
        }
        if let v = pelvicTiltDeg, abs(v) >= pelvicTiltAlertDeg {
            issues.insert(.hipTilt)
            issues.insert(.pelvisInstability)
        }
        if let v = trunkLeanDeg, abs(v) >= trunkLeanAlertDeg {
            issues.insert(.trunkSideBend)
            issues.insert(.trunkInstability)
        }
        if let v = shoulderAlign, abs(v) >= shoulderAlignAlert {
            issues.insert(.shoulderTilt)
        }
        if let v = leftValgus, abs(v) >= kneeValgusAlert {
            issues.insert(.insufficientKneeFlexion)
        }
        if let v = rightValgus, abs(v) >= kneeValgusAlert {
            issues.insert(.insufficientKneeFlexion)
        }
        if let v = headOffset, abs(v) >= headOffsetAlert {
            issues.insert(.headOffMidline)
        }
        // 步態對稱：比較近期抬腳高度振幅。
        if recentClearanceL.count >= 10, recentClearanceR.count >= 10 {
            let ampL = (recentClearanceL.max() ?? 0) - (recentClearanceL.min() ?? 0)
            let ampR = (recentClearanceR.max() ?? 0) - (recentClearanceR.min() ?? 0)
            let denom = max(ampL, ampR, 1e-6)
            let asymmetry = abs(ampL - ampR) / denom
            if asymmetry >= gaitAsymmetryAlert {
                issues.insert(.pelvisInstability)
            }
        }
        return issues
    }

    // MARK: 5) 防抖

    private func debounce(_ raw: Set<PoseIssueCode>) -> Set<PoseIssueCode> {
        var confirmed = Set<PoseIssueCode>()
        let all = Set(issueStreak.keys).union(raw)
        for code in all {
            if raw.contains(code) {
                issueStreak[code, default: 0] += 1
            } else {
                issueStreak[code] = 0
            }
            if (issueStreak[code] ?? 0) >= debounceFrames {
                confirmed.insert(code)
            }
        }
        return confirmed
    }

    private func advice(
        from issues: Set<PoseIssueCode>,
        features: (Double?, Double?, Double?, Double?, Double?, Double?)
    ) -> [String] {
        var lines: [String] = ["背面視角分析："]
        if let tilt = features.0 {
            lines.append(String(format: "骨盆左右傾斜 %.1f°", tilt))
        }
        if let lean = features.1 {
            lines.append(String(format: "軀幹側傾 %.1f°", lean))
        }
        if let sh = features.2 {
            lines.append(String(format: "肩膀水平差（正規化）%.3f", sh))
        }
        if let head = features.5 {
            lines.append(String(format: "頭頸相對肩中線偏移 %.3f（耳中點推估）", head))
        }

        if issues.contains(.hipTilt) || issues.contains(.pelvisInstability) {
            lines.append("骨盆左右晃動偏大，單腳支撐時試著保持骨盆水平。")
        }
        if issues.contains(.trunkSideBend) || issues.contains(.trunkInstability) {
            lines.append("軀幹左右側傾偏大，請維持肩－髖中線直立。")
        }
        if issues.contains(.shoulderTilt) {
            lines.append("雙肩高低差偏大，注意肩帶是否放鬆對稱。")
        }
        if issues.contains(.insufficientKneeFlexion) {
            lines.append("膝蓋相對髖－踝連線橫向偏移偏大（內夾／外翻趨勢），著地時對齊膝與腳尖。")
        }
        if issues.contains(.headOffMidline) {
            lines.append("頭頸相對雙肩中線偏移（由雙耳推估），請輕收下巴、視線朝前。")
        }
        if issues.contains(.lowVisibility) {
            lines.append("核心關節信心不足，請確保背面全身入鏡且光線充足。")
        }
        if issues.isEmpty, lines.count == 1 {
            lines.append("背面核心指標穩定。")
        }
        return lines
    }

    /// 整段 session 摘要用的統計句。
    func sessionSummaryLines() -> [String] {
        guard let last = lastFeatures else {
            return ["背面視角：尚無有效特徵。"]
        }
        var out: [String] = ["背面視角特徵摘要："]
        if let v = last.pelvicTiltDeg {
            out.append(String(format: "最近骨盆傾斜 %.1f°", v))
        }
        if let v = last.trunkLateralLeanDeg {
            out.append(String(format: "最近軀幹側傾 %.1f°", v))
        }
        if let v = last.stepWidthNorm {
            out.append(String(format: "最近步寬（正規化）%.3f", v))
        }
        if !recentClearanceL.isEmpty, !recentClearanceR.isEmpty {
            let ampL = (recentClearanceL.max() ?? 0) - (recentClearanceL.min() ?? 0)
            let ampR = (recentClearanceR.max() ?? 0) - (recentClearanceR.min() ?? 0)
            out.append(String(format: "抬腳振幅 L %.3f / R %.3f", ampL, ampR))
        }
        if !last.debouncedIssues.isEmpty {
            out.append("持續異常：" + last.debouncedIssues.map(\.summaryDescription).joined(separator: "、"))
        }
        return out
    }
}
