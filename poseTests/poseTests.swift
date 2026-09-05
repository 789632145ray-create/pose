//
//  poseTests.swift
//  poseTests
//

import XCTest
@testable import pose

final class poseTests: XCTestCase {

    func testAssessmentEngineHasMediaPipe() {
        XCTAssertEqual(PoseAssessmentEngine.allCases.count, 3)
        XCTAssertEqual(PoseAssessmentEngine.allCases.map(\.title), ["QuickPose", "MediaPipe", "自訓模型"])
        XCTAssertTrue(PoseAssessmentEngine.allCases.contains(.mediaPipe))
        XCTAssertEqual(PoseAssessmentEngine.mediaPipe.title, "MediaPipe")
        XCTAssertEqual(PoseAssessmentEngine.mediaPipe.hudBadgeTitle, "MediaPipe Full")
    }

    func testMediaPipeUsesLocalRulesNotBackendPredict() {
        XCTAssertTrue(PoseAssessmentEngine.mediaPipe.usesFullDetectionPipeline)
        XCTAssertTrue(PoseAssessmentEngine.mediaPipe.usesRuleBasedAdvice)
        XCTAssertFalse(PoseAssessmentEngine.mediaPipe.usesTrainedModelPredict)

        XCTAssertTrue(PoseAssessmentEngine.trainedModel.usesTrainedModelPredict)
        XCTAssertFalse(PoseAssessmentEngine.trainedModel.usesRuleBasedAdvice)

        XCTAssertTrue(PoseAssessmentEngine.quickPose.usesFullDetectionPipeline)
        XCTAssertTrue(PoseAssessmentEngine.quickPose.usesRuleBasedAdvice)
        XCTAssertFalse(PoseAssessmentEngine.quickPose.usesTrainedModelPredict)
        XCTAssertTrue(PoseDatabase.store(for: .quickPose) === PoseDatabase.shared)
    }

    func testMediaPipeUsesDedicatedDatabase() {
        XCTAssertEqual(MediaPipeRealm.fileName, "mediapipe.realm")
        XCTAssertEqual(MediaPipeRealm.fileURL.lastPathComponent, "mediapipe.realm")
        XCTAssertEqual(PoseRealm.fileURL.lastPathComponent, "pose.realm")
        XCTAssertNotEqual(PoseDatabase.shared.fileURL, PoseDatabase.mediaPipe.fileURL)
        XCTAssertTrue(PoseDatabase.store(for: .mediaPipe) === PoseDatabase.mediaPipe)
        XCTAssertTrue(PoseDatabase.store(for: .trainedModel) === PoseDatabase.shared)
        XCTAssertTrue(PoseDatabase.store(for: .quickPose) === PoseDatabase.shared)
        XCTAssertEqual(PoseNodeStoreKind.mediaPipe.uploadPath, "/mediapipe/poses")
        XCTAssertEqual(PoseNodeStoreKind.pose.uploadPath, "/poses")
        XCTAssertTrue(PoseNodeStoreKind.mediaPipe.uploadPath.contains("mediapipe"))
    }

    func testMediaPipeSkeletonConnectionsUseExtractedJoints() {
        let known = Set([
            "nose", "shoulder_mid", "hip_mid",
            "left_eye_inner", "left_eye", "left_eye_outer", "left_ear", "left_mouth",
            "right_eye_inner", "right_eye", "right_eye_outer", "right_ear", "right_mouth",
            "left_shoulder", "left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb",
            "right_shoulder", "right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb",
            "left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index",
            "right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"
        ])
        XCTAssertFalse(MediaPipeSkeletonGraph.connections.isEmpty)
        for (a, b) in MediaPipeSkeletonGraph.connections {
            XCTAssertTrue(known.contains(a), "unknown joint \(a)")
            XCTAssertTrue(known.contains(b), "unknown joint \(b)")
        }
        XCTAssertTrue(MediaPipeSkeletonGraph.isVisible(
            PoseNode(joint: "nose", x: 0.5, y: 0.2, z: 0, visibility: 0.9, presence: 0.9)
        ))
        XCTAssertFalse(MediaPipeSkeletonGraph.isVisible(
            PoseNode(joint: "nose", x: 0.5, y: 0.2, z: 0, visibility: 0.1, presence: 0.9)
        ))
    }

    func testSwitchingEnginesKeepsSharedPoseRuntime() {
        XCTAssertTrue(PoseAssessmentEngine.quickPose.usesFullPoseModel)
        XCTAssertTrue(PoseAssessmentEngine.mediaPipe.usesFullPoseModel)
        XCTAssertTrue(PoseAssessmentEngine.trainedModel.usesFullPoseModel)
        for from in PoseAssessmentEngine.allCases {
            for to in PoseAssessmentEngine.allCases where from != to {
                XCTAssertFalse(
                    PoseAssessmentEngine.requiresPoseRuntimeRestart(from: from, to: to),
                    "\(from.rawValue) → \(to.rawValue) must keep the running overlay session"
                )
            }
        }
    }

    func testLegacyEngineStorageMigration() {
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained_model"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "quickpose"), .quickPose)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "mediapipe"), .mediaPipe)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "unknown"), .trainedModel)
    }
}
