//
//  poseTests.swift
//  poseTests
//
//  Created by 郭家豪 on 2026/4/15.
//

import XCTest
@testable import pose

final class poseTests: XCTestCase {

    func testGaitStatsParsedFromSummaryLines() throws {
        let lines = [
            "模型辨識：姿勢「好」（信心 95%）",
            "總步數：24（左 13 / 右 11）",
            "平均步頻：120 bpm",
            "個人化評估：青壯年（22 歲）",
            "左右平衡：左 54% / 右 46%"
        ]

        let stats = try XCTUnwrap(SavedSummary.gaitStats(from: lines))
        XCTAssertEqual(stats.totalSteps, 24)
        XCTAssertEqual(stats.leftSteps, 13)
        XCTAssertEqual(stats.rightSteps, 11)
        XCTAssertEqual(stats.avgCadenceBPM, 120)
    }

    func testDisplayStepsFallbackWhenStoredFieldsAreZero() {
        let summary = SavedSummary(
            id: UUID(),
            date: Date(),
            sourceLabel: "影片：demo.mov",
            totalSteps: 0,
            leftSteps: 0,
            rightSteps: 0,
            avgCadenceBPM: nil,
            lines: [
                "總步數：24（左 13 / 右 11）",
                "平均步頻：120 bpm"
            ]
        )

        XCTAssertEqual(summary.displayTotalSteps, 24)
        XCTAssertEqual(summary.displayLeftSteps, 13)
        XCTAssertEqual(summary.displayRightSteps, 11)
        XCTAssertEqual(summary.displayAvgCadenceBPM, 120)
    }

    func testDisplayStepsPreferStoredFields() {
        let summary = SavedSummary(
            id: UUID(),
            date: Date(),
            sourceLabel: "相機（即時）",
            totalSteps: 48,
            leftSteps: 22,
            rightSteps: 26,
            avgCadenceBPM: 125,
            lines: [
                "總步數：24（左 13 / 右 11）",
                "平均步頻：120 bpm"
            ]
        )

        XCTAssertEqual(summary.displayTotalSteps, 48)
        XCTAssertEqual(summary.displayLeftSteps, 22)
        XCTAssertEqual(summary.displayRightSteps, 26)
        XCTAssertEqual(summary.displayAvgCadenceBPM, 125)
    }
}
