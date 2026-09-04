//
//  poseTests.swift
//  poseTests
//

import XCTest
@testable import pose

final class poseTests: XCTestCase {

    func testAssessmentEngineHasMediaPipe() {
        XCTAssertEqual(PoseAssessmentEngine.allCases.count, 3)
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

        XCTAssertFalse(PoseAssessmentEngine.quickPose.usesFullDetectionPipeline)
    }

    func testLegacyEngineStorageMigration() {
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "trained_model"), .trainedModel)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "quickpose"), .quickPose)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "mediapipe"), .mediaPipe)
        XCTAssertEqual(PoseAssessmentEngine.resolved(fromStored: "unknown"), .trainedModel)
    }
}
