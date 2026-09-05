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
