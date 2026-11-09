//
//  SummaryStore.swift
//  pose
//
//  將每次的分析摘要持久化到 Realm Database，提供新增、清除與讀取歷史。
//

import Combine
import Foundation
import RealmSwift
import SwiftUI

struct SavedSummary: Identifiable, Codable, Equatable {
    let id: UUID
    let date: Date
    /// 來源說明，例如「影片：xxx.mov」或「相機（即時）」。
    let sourceLabel: String
    let totalSteps: Int
    let leftSteps: Int
    let rightSteps: Int
    let avgCadenceBPM: Double?
    let lines: [String]

    /// 列表／詳情顯示用：欄位若因 pipeline reset 被存成 0，改從摘要文案回填。
    var displayTotalSteps: Int {
        if totalSteps > 0 { return totalSteps }
        return Self.gaitStats(from: lines)?.totalSteps ?? totalSteps
    }

    var displayLeftSteps: Int {
        if leftSteps > 0 { return leftSteps }
        return Self.gaitStats(from: lines)?.leftSteps ?? leftSteps
    }

    var displayRightSteps: Int {
        if rightSteps > 0 { return rightSteps }
        return Self.gaitStats(from: lines)?.rightSteps ?? rightSteps
    }

    var displayAvgCadenceBPM: Double? {
        if let avgCadenceBPM { return avgCadenceBPM }
        return Self.gaitStats(from: lines)?.avgCadenceBPM
    }

    /// 從「總步數：24（左 13 / 右 11）」與「平均步頻：120 bpm」解析步態統計。
    static func gaitStats(from lines: [String]) -> (totalSteps: Int, leftSteps: Int, rightSteps: Int, avgCadenceBPM: Double?)? {
        let stepPattern = #"總步數[：:]\s*(\d+)\s*[（(]\s*左\s*(\d+)\s*/\s*右\s*(\d+)\s*[）)]"#
        let cadencePattern = #"平均步頻[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*bpm"#
        var total: Int?
        var left: Int?
        var right: Int?
        var cadence: Double?

        for line in lines {
            if total == nil, let values = captureInts(in: line, pattern: stepPattern, count: 3) {
                total = values[0]
                left = values[1]
                right = values[2]
            }
            if cadence == nil, let values = captureDoubles(in: line, pattern: cadencePattern, count: 1) {
                cadence = values[0]
            }
        }

        guard let total, let left, let right else { return nil }
        return (total, left, right, cadence)
    }

    private static func captureInts(in text: String, pattern: String, count: Int) -> [Int]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              match.numberOfRanges == count + 1 else { return nil }
        let ns = text as NSString
        var values: [Int] = []
        for i in 1...count {
            guard let value = Int(ns.substring(with: match.range(at: i))) else { return nil }
            values.append(value)
        }
        return values
    }

    private static func captureDoubles(in text: String, pattern: String, count: Int) -> [Double]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              match.numberOfRanges == count + 1 else { return nil }
        let ns = text as NSString
        var values: [Double] = []
        for i in 1...count {
            guard let value = Double(ns.substring(with: match.range(at: i))) else { return nil }
            values.append(value)
        }
        return values
    }
}

@MainActor
final class SummaryStore: ObservableObject {
    @Published private(set) var items: [SavedSummary] = []

    init() {
        PoseRealm.migrateLegacyIfNeeded()
        reload()
    }

    func add(_ summary: SavedSummary) {
        guard let realm = try? PoseRealm.open() else { return }
        let obj = RLMSavedSummary()
        obj.id = summary.id.uuidString
        obj.date = summary.date
        obj.sourceLabel = summary.sourceLabel
        obj.totalSteps = summary.totalSteps
        obj.leftSteps = summary.leftSteps
        obj.rightSteps = summary.rightSteps
        obj.avgCadenceBPM = summary.avgCadenceBPM
        obj.lines.append(objectsIn: summary.lines)
        try? realm.write {
            realm.add(obj, update: .modified)
        }
        reload()
    }

    func remove(at offsets: IndexSet) {
        guard let realm = try? PoseRealm.open() else { return }
        let ids = offsets.compactMap { items.indices.contains($0) ? items[$0].id.uuidString : nil }
        try? realm.write {
            for id in ids {
                if let obj = realm.object(ofType: RLMSavedSummary.self, forPrimaryKey: id) {
                    realm.delete(obj)
                }
            }
        }
        reload()
    }

    func clear() {
        guard let realm = try? PoseRealm.open() else { return }
        try? realm.write {
            realm.delete(realm.objects(RLMSavedSummary.self))
        }
        reload()
    }

    private func reload() {
        guard let realm = try? PoseRealm.open() else {
            items = []
            return
        }
        let objects = Array(realm.objects(RLMSavedSummary.self).sorted(byKeyPath: "date", ascending: false))
        var needsRepair = false
        for obj in objects where obj.totalSteps == 0 {
            if let parsed = SavedSummary.gaitStats(from: Array(obj.lines)), parsed.totalSteps > 0 {
                needsRepair = true
                break
            }
        }
        if needsRepair {
            try? realm.write {
                for obj in objects {
                    let lines = Array(obj.lines)
                    guard obj.totalSteps == 0,
                          let parsed = SavedSummary.gaitStats(from: lines),
                          parsed.totalSteps > 0 else { continue }
                    obj.totalSteps = parsed.totalSteps
                    obj.leftSteps = parsed.leftSteps
                    obj.rightSteps = parsed.rightSteps
                    if obj.avgCadenceBPM == nil {
                        obj.avgCadenceBPM = parsed.avgCadenceBPM
                    }
                }
            }
        }

        items = objects.map { obj in
            SavedSummary(
                id: UUID(uuidString: obj.id) ?? UUID(),
                date: obj.date,
                sourceLabel: obj.sourceLabel,
                totalSteps: obj.totalSteps,
                leftSteps: obj.leftSteps,
                rightSteps: obj.rightSteps,
                avgCadenceBPM: obj.avgCadenceBPM,
                lines: Array(obj.lines)
            )
        }
    }
}
