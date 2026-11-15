//
//  JointConfidenceFilter.swift
//  pose
//
//  背後擺臂等遮擋場景：用 visibility / presence 當信心分數。
//  低於閾值的關節不直接採計，改以前後可信幀做線性插值（即時路徑延遲一幀）。
//  骨架來源為 QuickPose / MediaPipe Full 的 3D 全身關鍵點（含 z）。
//

import Foundation

extension PoseNode {
    /// 信心分數：取可見度與存在度的較小值。
    var confidenceScore: Double {
        min(visibility, presence)
    }

    func replacingPose(
        x: Double,
        y: Double,
        z: Double,
        visibility: Double? = nil,
        presence: Double? = nil
    ) -> PoseNode {
        PoseNode(
            joint: joint,
            x: x,
            y: y,
            z: z,
            visibility: visibility ?? self.visibility,
            presence: presence ?? self.presence
        )
    }
}

/// 即時關節穩定器：低信心關節延遲一幀，用前後可信座標線性插值補齊。
@MainActor
final class JointConfidenceFilter {
    /// 低於此分數視為遮擋／不可信（預設 0.5）。
    var threshold: Double

    /// 手臂與末端關節較容易被軀幹遮擋。
    private static let armJoints: Set<String> = [
        "left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb",
        "right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"
    ]

    private var buffered: [PoseNode]?
    private var lastTrusted: [String: PoseNode] = [:]

    init(threshold: Double = 0.5) {
        self.threshold = threshold
    }

    func reset() {
        buffered = nil
        lastTrusted.removeAll(keepingCapacity: true)
    }

    /// 推入目前幀；回傳已定稿的「上一幀」（延遲一幀才能做前後插值）。首幀回傳 nil。
    func push(_ nodes: [PoseNode]) -> [PoseNode]? {
        let outgoing = buffered.map { finalize(frame: $0, nextFrame: nodes) }
        buffered = nodes
        // 更新最近可信點，供後續 hold／插值使用。
        for node in nodes where isTrusted(node) {
            lastTrusted[node.joint] = node
        }
        return outgoing
    }

    /// 暫停／結束時吐出緩衝中尚未定稿的最後一幀。
    func flush() -> [PoseNode]? {
        guard let last = buffered else { return nil }
        let out = finalize(frame: last, nextFrame: [])
        reset()
        return out
    }

    private func isTrusted(_ node: PoseNode) -> Bool {
        let score = node.confidenceScore
        if Self.armJoints.contains(node.joint) {
            return score >= threshold
        }
        // 軀幹／下肢略寬，避免過度補點。
        return score >= min(threshold, 0.35)
    }

    private func finalize(frame: [PoseNode], nextFrame: [PoseNode]) -> [PoseNode] {
        let nextTrusted = Dictionary(
            uniqueKeysWithValues: nextFrame.filter(isTrusted).map { ($0.joint, $0) }
        )
        var result: [PoseNode] = []
        result.reserveCapacity(frame.count)

        for node in frame {
            if isTrusted(node) {
                lastTrusted[node.joint] = node
                result.append(node)
                continue
            }

            let before = lastTrusted[node.joint]
            let after = nextTrusted[node.joint]

            if let before, let after {
                result.append(Self.lerp(before, after, t: 0.5))
            } else if let before {
                // 尚無後幀：沿用最近可信座標，信心標成閾值表示補點。
                result.append(
                    before.replacingPose(
                        x: before.x,
                        y: before.y,
                        z: before.z,
                        visibility: threshold,
                        presence: threshold
                    )
                )
            } else if let after {
                result.append(
                    after.replacingPose(
                        x: after.x,
                        y: after.y,
                        z: after.z,
                        visibility: threshold,
                        presence: threshold
                    )
                )
            }
            // 前後都沒有可信點：不採計該關節。
        }
        return result
    }

    private static func lerp(_ a: PoseNode, _ b: PoseNode, t: Double) -> PoseNode {
        let u = max(0, min(1, t))
        return PoseNode(
            joint: a.joint,
            x: a.x + (b.x - a.x) * u,
            y: a.y + (b.y - a.y) * u,
            z: a.z + (b.z - a.z) * u,
            visibility: min(a.visibility, b.visibility),
            presence: min(a.presence, b.presence)
        )
    }

    // MARK: - 離線整段補齊（上傳／品質辨識前）

    /// 對已錄製影格做雙向線性插值：低信心關節用時間軸上前後可信點補齊。
    static func interpolateSession(
        _ frames: [PoseFrameRecord],
        threshold: Double = 0.5
    ) -> [PoseFrameRecord] {
        guard !frames.isEmpty else { return frames }

        let joints = Array(Set(frames.flatMap { $0.nodes.map(\.joint) })).sorted()
        var rebuilt = frames

        for joint in joints {
            let series: [(index: Int, node: PoseNode?)] = rebuilt.enumerated().map { idx, frame in
                (idx, frame.nodes.first(where: { $0.joint == joint }))
            }
            let trustedIdx = series.compactMap { item -> Int? in
                guard let node = item.node, min(node.visibility, node.presence) >= threshold else { return nil }
                return item.index
            }
            guard !trustedIdx.isEmpty else { continue }

            for (i, item) in series.enumerated() {
                if let node = item.node, min(node.visibility, node.presence) >= threshold {
                    continue
                }
                let before = trustedIdx.last(where: { $0 < i })
                let after = trustedIdx.first(where: { $0 > i })
                let filled: PoseNode?
                if let before, let after,
                   let a = series[before].node,
                   let b = series[after].node {
                    let t = Double(i - before) / Double(after - before)
                    filled = lerp(a, b, t: t)
                } else if let before, let a = series[before].node {
                    filled = a.replacingPose(x: a.x, y: a.y, z: a.z, visibility: threshold, presence: threshold)
                } else if let after, let b = series[after].node {
                    filled = b.replacingPose(x: b.x, y: b.y, z: b.z, visibility: threshold, presence: threshold)
                } else {
                    filled = nil
                }
                guard let filled else { continue }
                var nodes = rebuilt[i].nodes.filter { $0.joint != joint }
                nodes.append(filled)
                rebuilt[i] = PoseFrameRecord(
                    frameIndex: rebuilt[i].frameIndex,
                    timestamp: rebuilt[i].timestamp,
                    nodes: nodes
                )
            }
        }
        return rebuilt
    }
}
