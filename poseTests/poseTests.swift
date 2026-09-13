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

    func testWalkingFormCorePrincipleLines() {
        XCTAssertEqual(WalkingFormAdvisor.corePrincipleLines, [
            "手臂自然下垂，隨著對側腳步前後擺動，幅度不宜過大。",
            "腳跟先著地，力量順勢平穩地過渡到腳掌，最後由腳尖蹬地推動身體前進。"
        ])
        XCTAssertEqual(PoseIssueCode.ipsilateralArmSwing.summaryDescription, "同側擺臂（未對側配合）")
        XCTAssertEqual(PoseIssueCode.forefootStrike.summaryDescription, "腳尖／前腳掌先著地")
        XCTAssertEqual(PoseIssueCode.incompleteToeOff.summaryDescription, "腳尖蹬地不足")
    }

    func testWalkingFormDetectsContralateralSwingAndHeelToe() {
        let result = WalkingFormAdvisor.evaluate(Self.goodSideViewStride())
        XCTAssertTrue(result.issues.isEmpty, "\(result.issues)")
        XCTAssertTrue(result.lines.contains(where: { $0.contains("對側擺臂") }))
        XCTAssertTrue(result.lines.contains(where: { $0.contains("腳跟先著地") }))
    }

    func testWalkingFormDetectsIpsilateralArmSwing() {
        var snapshot = Self.goodSideViewStride()
        snapshot.leftWrist.x = 0.38
        snapshot.rightWrist.x = 0.66
        let result = WalkingFormAdvisor.evaluate(snapshot)
        XCTAssertTrue(result.issues.contains(.ipsilateralArmSwing))
        XCTAssertTrue(result.lines.contains(where: { $0.contains("對側擺臂") || $0.contains("同側") }))
    }

    func testWalkingFormDetectsRaisedArmsAndForefootStrike() {
        var snapshot = Self.goodSideViewStride()
        snapshot.leftWrist.y = 0.22
        snapshot.rightWrist.y = 0.22
        snapshot.rightHeel.y = 0.70
        snapshot.rightFootIndex.y = 0.82
        let result = WalkingFormAdvisor.evaluate(snapshot)
        XCTAssertTrue(result.issues.contains(.raisedArms))
        XCTAssertTrue(result.issues.contains(.forefootStrike))
    }

    func testWalkingFormSessionSummaryIncludesCore() {
        var metrics = GaitSessionMetrics()
        for _ in 0..<12 {
            WalkingFormAdvisor.accumulate(metrics: &metrics, snapshot: Self.goodSideViewStride())
        }
        let summary = WalkingFormAdvisor.sessionSummary(metrics: metrics)
        XCTAssertTrue(summary.contains(WalkingFormAdvisor.corePrincipleLines[0]))
        XCTAssertTrue(summary.contains(WalkingFormAdvisor.corePrincipleLines[1]))
        XCTAssertTrue(summary.contains(where: { $0.contains("符合正確走路姿勢") }))
    }

    private static func goodSideViewStride() -> WalkingFormSnapshot {
        WalkingFormSnapshot(
            leftShoulder: .visible(x: 0.47, y: 0.30),
            rightShoulder: .visible(x: 0.53, y: 0.30),
            leftElbow: .visible(x: 0.44, y: 0.42),
            rightElbow: .visible(x: 0.56, y: 0.42),
            leftWrist: .visible(x: 0.62, y: 0.52),
            rightWrist: .visible(x: 0.40, y: 0.52),
            leftHip: .visible(x: 0.48, y: 0.55),
            rightHip: .visible(x: 0.52, y: 0.55),
            leftAnkle: .visible(x: 0.38, y: 0.76),
            rightAnkle: .visible(x: 0.64, y: 0.80),
            leftHeel: .visible(x: 0.36, y: 0.70),
            rightHeel: .visible(x: 0.62, y: 0.82),
            leftFootIndex: .visible(x: 0.41, y: 0.80),
            rightFootIndex: .visible(x: 0.66, y: 0.78)
        )
    }

    func testLegacyEngineStorageMigration() {
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained_model"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "quickpose"), .quickPose)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "mediapipe"), .mediaPipe)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "unknown"), .trainedModel)
    }
}
