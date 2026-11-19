//
//  SideViewPreprocessor.swift
//  pose
//
//  側面視角（Side View）前處理與特徵提取——訓練／推論共用同一套邏輯：
//  1) 朝向判斷＋水平翻轉（統一為朝右）
//  2) 近側肢體篩選（忽略遠側遮擋關節）
//  3) 中值濾波＋膝角運動學約束（抗 ID Switch）
//  4) 近側髖原點＋近側軀幹尺度正規化
//  5) 前傾／頸部前引／膝屈曲／髖屈伸／跨步過度／垂直振幅
//

import Foundation
import QuickPoseCore

// MARK: - 輸出

enum SideFacingDirection: String, Equatable {
    case right
    case left
    case unknown
}

struct SideViewFrameFeatures: Equatable {
    var timestamp: TimeInterval
    var facing: SideFacingDirection
    /// 近側為解剖學哪一側（翻轉正規化後固定為 right）。
    var nearSide: String
    /// 身體前傾角（度）：近側肩－髖連線相對垂直線；前傾為正。
    var trunkForwardLeanDeg: Double?
    /// 頸部前引角（度）：近側耳－肩連線相對垂直線；頭部前移為正。
    var forwardHeadDeg: Double?
    /// 膝屈曲角（度）：髖－膝－踝；伸直 ≈ 180，屈曲越小。
    var kneeFlexionDeg: Double?
    /// 髖屈伸角（度）：肩－髖－膝。
    var hipFlexionDeg: Double?
    /// 著地瞬間腳踝相對髖的水平超前量（軀幹長度正規化）；越大越 overstride。
    var overstrideNorm: Double?
    /// 近側髖 Y 振幅（軀幹長度正規化）。
    var verticalOscillationNorm: Double?
    var coreReliable: Bool
    var debouncedIssues: Set<PoseIssueCode>
    var adviceLines: [String]

    /// 固定順序特徵向量（訓練／推論一致；缺值填 0）。
    var featureVector: [Double] {
        [
            trunkForwardLeanDeg ?? 0,
            forwardHeadDeg ?? 0,
            kneeFlexionDeg ?? 0,
            hipFlexionDeg ?? 0,
            overstrideNorm ?? 0,
            verticalOscillationNorm ?? 0,
            coreReliable ? 1 : 0,
            facing == .right ? 1 : (facing == .left ? -1 : 0),
        ]
    }

    static let featureColumnNames: [String] = [
        "trunk_forward_lean_deg",
        "forward_head_deg",
        "knee_flexion_deg",
        "hip_flexion_deg",
        "overstride_norm",
        "vertical_oscillation_norm",
        "core_reliable",
        "facing_sign",
    ]
}

// MARK: - Preprocessor（訓練／推論唯一入口）

/// 側面視角前處理器。Live HUD、摘要與未來側視特徵訓練都必須呼叫此類，避免前處理落差。
@MainActor
final class SideViewPreprocessor {
    /// 中值濾波視窗（3~5）。
    var medianWindow: Int = 5
    /// 單幀膝角變化上限（度／幀，約 1/30s）；超過視為 ID Switch。
    var maxKneeAngleDeltaDeg: Double = 45
    var debounceFrames: Int = 5

    // 提醒門檻
    var trunkLeanAlertDeg: Double = 25
    var forwardHeadAlertDeg: Double = 20
    var kneeHyperextensionAlertDeg: Double = 185
    var overstrideAlertNorm: Double = 0.35
    var verticalOscillationAlertNorm: Double = 0.18

    private var history: [[String: PoseNode]] = []
    private var lastTrusted: [String: PoseNode] = [:]
    private var lastKneeFlexionDeg: Double?
    private var recentHipY: [Double] = []
    private var recentAnkleY: [Double] = []
    private var pelvisXHistory: [Double] = []
    private var issueStreak: [PoseIssueCode: Int] = [:]
    private var lastFeatures: SideViewFrameFeatures?
    private var lastFacing: SideFacingDirection = .unknown

    var latestFeatures: SideViewFrameFeatures? { lastFeatures }

    func reset() {
        history.removeAll(keepingCapacity: true)
        lastTrusted.removeAll(keepingCapacity: true)
        lastKneeFlexionDeg = nil
        recentHipY.removeAll()
        recentAnkleY.removeAll()
        pelvisXHistory.removeAll()
        issueStreak.removeAll()
        lastFeatures = nil
        lastFacing = .unknown
    }

    func process(landmarks: QuickPose.Landmarks, timestamp: TimeInterval = Date().timeIntervalSince1970) -> SideViewFrameFeatures {
        process(nodes: PoseNodeExtractor.extractAll(from: landmarks), timestamp: timestamp)
    }

    /// 訓練與推論共用：原始節點 → 朝向正規化 → 近側篩選 → 濾波 → 尺度正規化 → 特徵。
    func process(nodes: [PoseNode], timestamp: TimeInterval = Date().timeIntervalSince1970) -> SideViewFrameFeatures {
        let byJoint = Dictionary(uniqueKeysWithValues: nodes.map { ($0.joint, $0) })
        let facing = detectFacing(byJoint)
        lastFacing = facing == .unknown ? lastFacing : facing

        let oriented = normalizeOrientation(byJoint, facing: lastFacing)
        let nearSide = "right" // 翻轉＋左右對調後，近側固定為解剖 right
        let nearOnly = selectNearSide(oriented, nearSide: nearSide)
        let smoothed = pushMedianFilter(nearOnly)
        let constrained = applyKinematicConstraints(smoothed)
        let normalized = normalizeToNearHip(constrained, nearSide: nearSide)
        let features = extractFeatures(
            normalized,
            facing: lastFacing,
            nearSide: nearSide,
            timestamp: timestamp
        )
        lastFeatures = features
        return features
    }

    // MARK: 1) 朝向判斷與水平翻轉

    private func detectFacing(_ frame: [String: PoseNode]) -> SideFacingDirection {
        // 腳尖／鼻子／耳朵相對腳跟／髖的 X；並參考骨盆水平速度。
        var votesRight = 0
        var votesLeft = 0

        func vote(frontX: Double?, backX: Double?) {
            guard let frontX, let backX else { return }
            if frontX > backX { votesRight += 1 }
            else if frontX < backX { votesLeft += 1 }
        }

        let nose = frame["nose"]?.x
        let leftEar = frame["left_ear"]?.x
        let rightEar = frame["right_ear"]?.x
        let ear = average(leftEar, rightEar)
        let leftHeel = frame["left_heel"]?.x
        let rightHeel = frame["right_heel"]?.x
        let heel = average(leftHeel, rightHeel)
        let leftToe = frame["left_foot_index"]?.x
        let rightToe = frame["right_foot_index"]?.x
        let toe = average(leftToe, rightToe)
        let leftHip = frame["left_hip"]?.x
        let rightHip = frame["right_hip"]?.x
        let hip = average(leftHip, rightHip)

        vote(frontX: nose, backX: heel)
        vote(frontX: ear, backX: heel)
        vote(frontX: toe, backX: heel)
        vote(frontX: nose, backX: hip)
        vote(frontX: toe, backX: hip)

        if let hip {
            pelvisXHistory.append(hip)
            if pelvisXHistory.count > 8 {
                pelvisXHistory.removeFirst(pelvisXHistory.count - 8)
            }
            if pelvisXHistory.count >= 3 {
                let dx = pelvisXHistory.last! - pelvisXHistory.first!
                if dx > 0.01 { votesRight += 2 }
                else if dx < -0.01 { votesLeft += 2 }
            }
        }

        if votesRight == votesLeft { return .unknown }
        return votesRight > votesLeft ? .right : .left
    }

    /// 朝左時：`x → 1 - x`（影像正規化座標），並對調 left_/right_ 標籤，使近側永遠是 right。
    private func normalizeOrientation(_ frame: [String: PoseNode], facing: SideFacingDirection) -> [String: PoseNode] {
        guard facing == .left else { return frame }
        var flipped: [String: PoseNode] = [:]
        for (joint, node) in frame {
            let mirrored = PoseNode(
                joint: joint,
                x: 1.0 - node.x,
                y: node.y,
                z: -node.z,
                visibility: node.visibility,
                presence: node.presence
            )
            flipped[swapSideLabel(joint)] = PoseNode(
                joint: swapSideLabel(joint),
                x: mirrored.x,
                y: mirrored.y,
                z: mirrored.z,
                visibility: mirrored.visibility,
                presence: mirrored.presence
            )
        }
        return flipped
    }

    private func swapSideLabel(_ joint: String) -> String {
        if joint.hasPrefix("left_") {
            return "right_" + joint.dropFirst("left_".count)
        }
        if joint.hasPrefix("right_") {
            return "left_" + joint.dropFirst("right_".count)
        }
        return joint
    }

    // MARK: 2) 近側肢體篩選

    private static let nearSideSuffixes = [
        "ear", "shoulder", "elbow", "wrist",
        "hip", "knee", "ankle", "heel", "foot_index"
    ]

    private func selectNearSide(_ frame: [String: PoseNode], nearSide: String) -> [String: PoseNode] {
        var out: [String: PoseNode] = [:]
        // 保留中線參考點（翻轉後仍可用）。
        for key in ["nose", "shoulder_mid", "hip_mid"] {
            if let node = frame[key] { out[key] = node }
        }
        for suffix in Self.nearSideSuffixes {
            let key = "\(nearSide)_\(suffix)"
            if let node = frame[key] {
                out[key] = node
                lastTrusted[key] = node
            } else if let prev = lastTrusted[key] {
                out[key] = prev
            }
        }
        return out
    }

    // MARK: 3) 中值濾波＋運動學約束

    private func pushMedianFilter(_ frame: [String: PoseNode]) -> [String: PoseNode] {
        history.append(frame)
        let w = max(3, min(5, medianWindow))
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

    private func applyKinematicConstraints(_ frame: [String: PoseNode]) -> [String: PoseNode] {
        let side = "right"
        guard
            let hip = frame["\(side)_hip"],
            let knee = frame["\(side)_knee"],
            let ankle = frame["\(side)_ankle"]
        else { return frame }

        let kneeAngle = angleDegrees(
            ax: hip.x, ay: hip.y,
            bx: knee.x, by: knee.y,
            cx: ankle.x, cy: ankle.y
        )

        var out = frame
        if let prev = lastKneeFlexionDeg, abs(kneeAngle - prev) > maxKneeAngleDeltaDeg {
            // ID Switch：本幀近側腿改用上一可信幀（即時無法取「後幀」，等同 hold／前向填補）。
            for suffix in ["hip", "knee", "ankle", "heel", "foot_index"] {
                let key = "\(side)_\(suffix)"
                if let trusted = lastTrusted[key] {
                    out[key] = trusted
                }
            }
            return out
        }

        lastKneeFlexionDeg = kneeAngle
        for suffix in ["hip", "knee", "ankle", "heel", "foot_index"] {
            let key = "\(side)_\(suffix)"
            if let node = out[key] { lastTrusted[key] = node }
        }
        return out
    }

    // MARK: 4) 尺度與位置正規化

    private struct NormalizedSideSkeleton {
        var points: [String: (x: Double, y: Double, z: Double)]
        var scale: Double
        var coreReliable: Bool
        /// 正規化前近側髖的影像 y（供垂直振幅用同一尺度）。
        var rawHipY: Double
    }

    private func normalizeToNearHip(_ frame: [String: PoseNode], nearSide: String) -> NormalizedSideSkeleton? {
        guard let hip = frame["\(nearSide)_hip"] else { return nil }
        let shoulder = frame["\(nearSide)_shoulder"]
        let scale: Double
        if let shoulder {
            scale = max(hypot(shoulder.x - hip.x, shoulder.y - hip.y), 1e-6)
        } else {
            scale = 1e-6
        }

        var points: [String: (x: Double, y: Double, z: Double)] = [:]
        for (joint, node) in frame {
            points[joint] = (
                (node.x - hip.x) / scale,
                (node.y - hip.y) / scale,
                (node.z - hip.z) / scale
            )
        }

        let coreOK = ["\(nearSide)_shoulder", "\(nearSide)_hip", "\(nearSide)_knee", "\(nearSide)_ankle"]
            .allSatisfy { frame[$0] != nil }

        return NormalizedSideSkeleton(points: points, scale: scale, coreReliable: coreOK, rawHipY: hip.y)
    }

    // MARK: 5) 特徵

    private func extractFeatures(
        _ skeleton: NormalizedSideSkeleton?,
        facing: SideFacingDirection,
        nearSide: String,
        timestamp: TimeInterval
    ) -> SideViewFrameFeatures {
        guard let skeleton else {
            return SideViewFrameFeatures(
                timestamp: timestamp,
                facing: facing,
                nearSide: nearSide,
                trunkForwardLeanDeg: nil,
                forwardHeadDeg: nil,
                kneeFlexionDeg: nil,
                hipFlexionDeg: nil,
                overstrideNorm: nil,
                verticalOscillationNorm: nil,
                coreReliable: false,
                debouncedIssues: [],
                adviceLines: ["側面視角：近側核心關節不足，暫不計算特徵。"]
            )
        }

        let p = skeleton.points
        let shoulder = p["\(nearSide)_shoulder"]
        let hip = p["\(nearSide)_hip"] // 正規化後應接近 (0,0)
        let knee = p["\(nearSide)_knee"]
        let ankle = p["\(nearSide)_ankle"]
        let ear = p["\(nearSide)_ear"]
        let heel = p["\(nearSide)_heel"]

        // 身體前傾：肩相對髖（原點）與垂直線夾角；影像 -Y 為上，前傾時肩 X 通常 > 0（朝右後）。
        var trunkLean: Double?
        if let shoulder {
            trunkLean = atan2(shoulder.x, -shoulder.y) * 180 / .pi
        }

        // 頸部前引：耳相對肩與垂直線夾角。
        var headDeg: Double?
        if let ear, let shoulder {
            let dx = ear.x - shoulder.x
            let dy = ear.y - shoulder.y
            headDeg = atan2(dx, -dy) * 180 / .pi
        }

        var kneeFlex: Double?
        if let hip, let knee, let ankle {
            kneeFlex = angleDegrees(ax: hip.x, ay: hip.y, bx: knee.x, by: knee.y, cx: ankle.x, cy: ankle.y)
        }

        var hipFlex: Double?
        if let shoulder, let hip, let knee {
            hipFlex = angleDegrees(ax: shoulder.x, ay: shoulder.y, bx: hip.x, by: hip.y, cx: knee.x, cy: knee.y)
        }

        // 垂直振幅：近側髖影像 y 的峰－谷（再除以軀幹尺度）。
        recentHipY.append(skeleton.rawHipY / skeleton.scale)
        if recentHipY.count > 45 { recentHipY.removeFirst(recentHipY.count - 45) }
        var verticalOsc: Double?
        if recentHipY.count >= 8 {
            verticalOsc = (recentHipY.max() ?? 0) - (recentHipY.min() ?? 0)
        }

        // 跨步過度：腳踝 Y 局部最高（影像座標＝最低點）視為著地，量測踝－髖水平差。
        var overstride: Double?
        if let ankle {
            recentAnkleY.append(ankle.y)
            if recentAnkleY.count > 9 { recentAnkleY.removeFirst(recentAnkleY.count - 9) }
            if recentAnkleY.count >= 3 {
                let mid = recentAnkleY.count / 2
                let yMid = recentAnkleY[mid]
                let isLocalMax = yMid >= recentAnkleY[mid - 1] && yMid >= recentAnkleY[mid + 1]
                if isLocalMax {
                    // 正規化後髖在原點；朝右時踝 X > 0 表示落點在身體前方。
                    overstride = ankle.x
                }
            }
            // 若有腳跟，優先用腳跟 X 作為著地代理。
            if let heel, overstride != nil {
                overstride = heel.x
            }
        }

        let rawIssues = evaluateIssues(
            trunkLean: trunkLean,
            headDeg: headDeg,
            kneeFlex: kneeFlex,
            overstride: overstride,
            verticalOsc: verticalOsc,
            coreReliable: skeleton.coreReliable
        )
        let debounced = debounce(rawIssues)
        let lines = advice(
            facing: facing,
            trunkLean: trunkLean,
            headDeg: headDeg,
            kneeFlex: kneeFlex,
            hipFlex: hipFlex,
            overstride: overstride,
            verticalOsc: verticalOsc,
            issues: debounced
        )

        return SideViewFrameFeatures(
            timestamp: timestamp,
            facing: facing,
            nearSide: nearSide,
            trunkForwardLeanDeg: trunkLean,
            forwardHeadDeg: headDeg,
            kneeFlexionDeg: kneeFlex,
            hipFlexionDeg: hipFlex,
            overstrideNorm: overstride,
            verticalOscillationNorm: verticalOsc,
            coreReliable: skeleton.coreReliable,
            debouncedIssues: debounced,
            adviceLines: lines
        )
    }

    private func evaluateIssues(
        trunkLean: Double?,
        headDeg: Double?,
        kneeFlex: Double?,
        overstride: Double?,
        verticalOsc: Double?,
        coreReliable: Bool
    ) -> Set<PoseIssueCode> {
        var issues = Set<PoseIssueCode>()
        if !coreReliable { issues.insert(.lowVisibility) }
        if let v = trunkLean {
            if v >= trunkLeanAlertDeg { issues.insert(.excessiveForwardLean) }
            if v < 0 && abs(v) >= 8 { issues.insert(.insufficientForwardLean) }
            if abs(v) >= trunkLeanAlertDeg { issues.insert(.forwardStoop) }
        }
        if let v = headDeg, abs(v) >= forwardHeadAlertDeg {
            issues.insert(.headOffMidline)
            issues.insert(.forwardStoop)
        }
        if let v = kneeFlex, v >= kneeHyperextensionAlertDeg {
            issues.insert(.insufficientKneeFlexion)
        }
        if let v = overstride, v >= overstrideAlertNorm {
            issues.insert(.overStriding)
        }
        if let v = verticalOsc, v >= verticalOscillationAlertNorm {
            issues.insert(.trunkInstability)
        }
        return issues
    }

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
        facing: SideFacingDirection,
        trunkLean: Double?,
        headDeg: Double?,
        kneeFlex: Double?,
        hipFlex: Double?,
        overstride: Double?,
        verticalOsc: Double?,
        issues: Set<PoseIssueCode>
    ) -> [String] {
        var lines: [String] = ["側面視角分析："]
        switch facing {
        case .right: lines.append("朝向：右（已對齊）")
        case .left: lines.append("朝向：左 → 已水平翻轉為朝右")
        case .unknown: lines.append("朝向：判斷中")
        }
        if let v = trunkLean { lines.append(String(format: "身體前傾 %.1f°", v)) }
        if let v = headDeg { lines.append(String(format: "頸部前引 %.1f°", v)) }
        if let v = kneeFlex { lines.append(String(format: "近側膝屈曲 %.1f°", v)) }
        if let v = hipFlex { lines.append(String(format: "近側髖屈伸 %.1f°", v)) }
        if let v = overstride { lines.append(String(format: "跨步超前（正規化）%.3f", v)) }
        if let v = verticalOsc { lines.append(String(format: "垂直振幅（正規化）%.3f", v)) }

        if issues.contains(.excessiveForwardLean) || issues.contains(.forwardStoop) {
            lines.append("身體／上背前傾偏大，試著讓胸腔開展、由腳踝帶出身體微傾。")
        }
        if issues.contains(.insufficientForwardLean) {
            lines.append("跑步前傾不足，可讓肩略在髖前方，避免直立煞車。")
        }
        if issues.contains(.headOffMidline) {
            lines.append("頸部前引偏大（烏龜頸趨勢），請輕收下巴、視線平行前方。")
        }
        if issues.contains(.insufficientKneeFlexion) {
            lines.append("近側膝接近超伸，著地時保持微屈吸震。")
        }
        if issues.contains(.overStriding) {
            lines.append("著地點過於超前髖部（跨步過度），縮短步幅、提高步頻。")
        }
        if issues.contains(.trunkInstability) {
            lines.append("重心上下振幅偏大，放鬆肩膀、縮短騰空時間。")
        }
        if issues.contains(.lowVisibility) {
            lines.append("近側肩／髖／膝／踝信心不足，請讓側面全身入鏡。")
        }
        if issues.isEmpty, lines.count <= 2 {
            lines.append("側面核心指標穩定。")
        }
        return lines
    }

    func sessionSummaryLines() -> [String] {
        guard let last = lastFeatures else {
            return ["側面視角：尚無有效特徵。"]
        }
        var out: [String] = ["側面視角特徵摘要："]
        if let v = last.trunkForwardLeanDeg { out.append(String(format: "最近身體前傾 %.1f°", v)) }
        if let v = last.forwardHeadDeg { out.append(String(format: "最近頸部前引 %.1f°", v)) }
        if let v = last.kneeFlexionDeg { out.append(String(format: "最近膝屈曲 %.1f°", v)) }
        if let v = last.overstrideNorm { out.append(String(format: "最近跨步超前 %.3f", v)) }
        if let v = last.verticalOscillationNorm { out.append(String(format: "垂直振幅 %.3f", v)) }
        if !last.debouncedIssues.isEmpty {
            out.append("持續異常：" + last.debouncedIssues.map(\.summaryDescription).joined(separator: "、"))
        }
        return out
    }

    // MARK: - Math

    private func average(_ a: Double?, _ b: Double?) -> Double? {
        switch (a, b) {
        case let (a?, b?): return (a + b) / 2
        case let (a?, nil): return a
        case let (nil, b?): return b
        default: return nil
        }
    }

    /// 三點夾角 ∠ABC（度），頂點為 B。
    private func angleDegrees(ax: Double, ay: Double, bx: Double, by: Double, cx: Double, cy: Double) -> Double {
        let bax = ax - bx
        let bay = ay - by
        let bcx = cx - bx
        let bcy = cy - by
        let dot = bax * bcx + bay * bcy
        let mag = sqrt(bax * bax + bay * bay) * sqrt(bcx * bcx + bcy * bcy)
        guard mag > 1e-9 else { return 180 }
        let cos = max(-1, min(1, dot / mag))
        return acos(cos) * 180 / .pi
    }
}
