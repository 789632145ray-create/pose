//
//  MediaPipeLandmarkMapper.swift
//  pose
//
//  將 MediaPipe Pose Landmarker 的 33 點 BlazePose 輸出轉成專案共用的 PoseNode 鍵名。
//

import Foundation

enum MediaPipeLandmarkMapper {
    /// BlazePose / MediaPipe Pose 標準 33 點順序 → 本專案 joint 鍵名。
    static let blazePoseJointNames: [String] = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer", "left_ear",
        "right_eye_inner", "right_eye", "right_eye_outer", "right_ear",
        "left_mouth", "right_mouth",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    struct LandmarkSample {
        var x: Double
        var y: Double
        var z: Double
        var visibility: Double
        var presence: Double
    }

    /// 由正規化座標（0~1）陣列建立 PoseNode；不足 33 點時盡量對齊。
    static func nodes(from samples: [LandmarkSample]) -> [PoseNode] {
        var nodes: [PoseNode] = []
        nodes.reserveCapacity(35)
        let count = min(samples.count, blazePoseJointNames.count)
        for i in 0..<count {
            let s = samples[i]
            nodes.append(
                PoseNode(
                    joint: blazePoseJointNames[i],
                    x: s.x,
                    y: s.y,
                    z: s.z,
                    visibility: s.visibility,
                    presence: s.presence
                )
            )
        }
        let byName = MediaPipeSkeletonGraph.indexed(nodes)
        if let ls = byName["left_shoulder"], let rs = byName["right_shoulder"] {
            nodes.append(midpointNode(joint: "shoulder_mid", a: ls, b: rs))
        }
        if let lh = byName["left_hip"], let rh = byName["right_hip"] {
            nodes.append(midpointNode(joint: "hip_mid", a: lh, b: rh))
        }
        return nodes
    }

    private static func midpointNode(joint: String, a: PoseNode, b: PoseNode) -> PoseNode {
        PoseNode(
            joint: joint,
            x: (a.x + b.x) / 2,
            y: (a.y + b.y) / 2,
            z: (a.z + b.z) / 2,
            visibility: min(a.visibility, b.visibility),
            presence: min(a.presence, b.presence)
        )
    }

    static func walkingSnapshot(from nodes: [PoseNode]) -> WalkingFormSnapshot {
        let by = MediaPipeSkeletonGraph.indexed(nodes)
        func sample(_ name: String) -> WalkingJointSample {
            guard let n = by[name] else {
                return WalkingJointSample(x: 0, y: 0, z: 0, visibility: 0, presence: 0)
            }
            return WalkingJointSample(
                x: n.x, y: n.y, z: n.z,
                visibility: n.visibility, presence: n.presence
            )
        }
        return WalkingFormSnapshot(
            leftShoulder: sample("left_shoulder"),
            rightShoulder: sample("right_shoulder"),
            leftElbow: sample("left_elbow"),
            rightElbow: sample("right_elbow"),
            leftWrist: sample("left_wrist"),
            rightWrist: sample("right_wrist"),
            leftHip: sample("left_hip"),
            rightHip: sample("right_hip"),
            leftAnkle: sample("left_ankle"),
            rightAnkle: sample("right_ankle"),
            leftHeel: sample("left_heel"),
            rightHeel: sample("right_heel"),
            leftFootIndex: sample("left_foot_index"),
            rightFootIndex: sample("right_foot_index")
        )
    }

    static func keypointFrame(from nodes: [PoseNode]) -> PoseKeypointFrame {
        let by = MediaPipeSkeletonGraph.indexed(nodes)
        func joint(_ name: String) -> JointSample? {
            guard let n = by[name] else { return nil }
            return JointSample(
                x: n.x, y: n.y, z: n.z,
                visibility: n.visibility, presence: n.presence
            )
        }
        return PoseKeypointFrame(
            leftHip: joint("left_hip"),
            rightHip: joint("right_hip"),
            leftAnkle: joint("left_ankle"),
            rightAnkle: joint("right_ankle")
        )
    }
}
