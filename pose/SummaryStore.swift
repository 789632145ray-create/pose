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
        items = realm.objects(RLMSavedSummary.self)
            .sorted(byKeyPath: "date", ascending: false)
            .map { obj in
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
