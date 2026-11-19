//
//  PoseAdvice.swift
//  pose
//
//  左右對稱／中線姿勢建議；供 QuickPose 與原生 MediaPipe 共用。
//

import Foundation
import QuickPoseCore

struct PoseFrameAdvice {
    var lines: [String]
    var issues: Set<PoseIssueCode>
}

enum PoseAdvice {
    static func evaluate(from landmarks: QuickPose.Landmarks) -> PoseFrameAdvice {
        let ls = landmarks.landmark(forBody: .shoulder(side: .left))
        let rs = landmarks.landmark(forBody: .shoulder(side: .right))
        let lh = landmarks.landmark(forBody: .hip(side: .left))
        let rh = landmarks.landmark(forBody: .hip(side: .right))
        let nose = landmarks.landmark(forBody: .nose)
        return evaluate(
            leftShoulder: (ls.x, ls.y, ls.visibility, ls.presence),
            rightShoulder: (rs.x, rs.y, rs.visibility, rs.presence),
            leftHip: (lh.x, lh.y, lh.visibility, lh.presence),
            rightHip: (rh.x, rh.y, rh.visibility, rh.presence),
            nose: (nose.x, nose.y, nose.visibility, nose.presence)
        )
    }

    /// MediaPipe 原生路徑：由 PoseNode 評估左右對齊。
    static func evaluate(nodes: [PoseNode]) -> PoseFrameAdvice {
        let by = MediaPipeSkeletonGraph.indexed(nodes)
        func xyvp(_ name: String) -> (Double, Double, Double, Double)? {
            guard let n = by[name] else { return nil }
            return (n.x, n.y, n.visibility, n.presence)
        }
        guard let ls = xyvp("left_shoulder"),
              let rs = xyvp("right_shoulder"),
              let lh = xyvp("left_hip"),
              let rh = xyvp("right_hip"),
              let nose = xyvp("nose") else {
            return PoseFrameAdvice(
                lines: ["關節可見度不足，請後退一點、提高光線，或讓全身入鏡。"],
                issues: [.lowVisibility]
            )
        }
        return evaluate(
            leftShoulder: ls,
            rightShoulder: rs,
            leftHip: lh,
            rightHip: rh,
            nose: nose
        )
    }

    private static func evaluate(
        leftShoulder ls: (Double, Double, Double, Double),
        rightShoulder rs: (Double, Double, Double, Double),
        leftHip lh: (Double, Double, Double, Double),
        rightHip rh: (Double, Double, Double, Double),
        nose: (Double, Double, Double, Double)
    ) -> PoseFrameAdvice {
        func ok(_ p: (Double, Double, Double, Double)) -> Bool {
            p.2 > 0.4 && p.3 > 0.4
        }

        guard ok(ls), ok(rs), ok(lh), ok(rh), ok(nose) else {
            return PoseFrameAdvice(
                lines: ["關節可見度不足，請後退一點、提高光線，或讓全身入鏡。"],
                issues: [.lowVisibility]
            )
        }

        var out: [String] = []
        var issues: Set<PoseIssueCode> = []
        let shoulderTilt = abs(ls.1 - rs.1)
        let hipTilt = abs(lh.1 - rh.1)
        let shoulderMidX = (ls.0 + rs.0) / 2
        let hipMidX = (lh.0 + rh.0) / 2
        let trunkOffset = abs(shoulderMidX - hipMidX)
        let headOffset = abs(nose.0 - shoulderMidX)

        if shoulderTilt > 0.05 {
            out.append("肩膀有明顯高低差，試著放鬆並讓雙肩保持水平。")
            issues.insert(.shoulderTilt)
        } else {
            out.append("肩膀對齊良好。")
        }

        if hipTilt > 0.05 {
            out.append("骨盆有些傾斜，重心可再平均分配到雙腳。")
            issues.insert(.hipTilt)
        } else {
            out.append("骨盆穩定度不錯。")
        }

        if trunkOffset > 0.06 {
            out.append("上半身有側傾，建議收核心，讓肩膀回到髖部正上方。")
            issues.insert(.trunkSideBend)
        } else {
            out.append("軀幹中線穩定。")
        }

        if headOffset > 0.08 {
            out.append("頭部偏離身體中線，請把視線與頭部回正。")
            issues.insert(.headOffMidline)
        } else {
            out.append("頭部位置良好。")
        }

        return PoseFrameAdvice(lines: out, issues: issues)
    }
}
