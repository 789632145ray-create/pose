//
//  MediaPipeSkeletonOverlay.swift
//  pose
//
//  用目前幀的關節座標畫 MediaPipe 風格節點與連線。
//  QuickPose 的 overlay 圖在切換引擎後可能空白；這層不依賴那張圖。
//

import SwiftUI

enum MediaPipeSkeletonGraph {
    static let minVisibility = 0.35

    static let connections: [(String, String)] = [
        ("nose", "left_eye_inner"),
        ("left_eye_inner", "left_eye"),
        ("left_eye", "left_eye_outer"),
        ("left_eye_outer", "left_ear"),
        ("nose", "right_eye_inner"),
        ("right_eye_inner", "right_eye"),
        ("right_eye", "right_eye_outer"),
        ("right_eye_outer", "right_ear"),
        ("left_mouth", "right_mouth"),

        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "shoulder_mid"),
        ("right_shoulder", "shoulder_mid"),
        ("left_hip", "right_hip"),
        ("left_hip", "hip_mid"),
        ("right_hip", "hip_mid"),
        ("shoulder_mid", "hip_mid"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),

        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_wrist", "left_thumb"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_pinky"),
        ("left_index", "left_pinky"),

        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_wrist", "right_thumb"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_pinky"),
        ("right_index", "right_pinky"),

        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_foot_index"),
        ("left_heel", "left_foot_index"),

        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_foot_index"),
        ("right_heel", "right_foot_index")
    ]

    static func indexed(_ nodes: [PoseNode]) -> [String: PoseNode] {
        Dictionary(nodes.map { ($0.joint, $0) }, uniquingKeysWith: { _, latest in latest })
    }

    static func isVisible(_ node: PoseNode) -> Bool {
        node.visibility >= minVisibility && node.presence >= minVisibility
    }
}

struct MediaPipeSkeletonOverlay: View {
    let nodes: [PoseNode]
    var flipHorizontally: Bool

    var body: some View {
        Canvas { context, size in
            let byName = MediaPipeSkeletonGraph.indexed(nodes)
            let lineColor = Color.mint
            let pointColor = Color.white

            for (a, b) in MediaPipeSkeletonGraph.connections {
                guard let na = byName[a], let nb = byName[b],
                      MediaPipeSkeletonGraph.isVisible(na),
                      MediaPipeSkeletonGraph.isVisible(nb)
                else { continue }
                var path = Path()
                path.move(to: point(na, in: size))
                path.addLine(to: point(nb, in: size))
                context.stroke(path, with: .color(lineColor), lineWidth: 3)
            }

            for node in nodes where MediaPipeSkeletonGraph.isVisible(node) {
                let p = point(node, in: size)
                let r: CGFloat = 4
                let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
                context.fill(Path(ellipseIn: rect), with: .color(pointColor))
                context.stroke(Path(ellipseIn: rect), with: .color(lineColor), lineWidth: 1.5)
            }
        }
        .allowsHitTesting(false)
    }

    private func point(_ node: PoseNode, in size: CGSize) -> CGPoint {
        let x = flipHorizontally ? 1 - node.x : node.x
        return CGPoint(x: x * size.width, y: node.y * size.height)
    }
}
