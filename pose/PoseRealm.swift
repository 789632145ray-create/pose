//
//  PoseRealm.swift
//  pose
//
//  本機 Realm Database：姿勢節點 session 與分析摘要歷史。
//

import Foundation
import RealmSwift
import SQLite3

// MARK: - Realm 模型

final class RLMPoseNode: EmbeddedObject {
    @Persisted var joint: String = ""
    @Persisted var x: Double = 0
    @Persisted var y: Double = 0
    @Persisted var z: Double = 0
    @Persisted var visibility: Double = 0
    @Persisted var presence: Double = 0
}

final class RLMPoseFrame: EmbeddedObject {
    @Persisted var frameIndex: Int = 0
    @Persisted var timestamp: Double = 0
    @Persisted var nodes = List<RLMPoseNode>()
}

final class RLMPoseSession: Object {
    @Persisted(primaryKey: true) var id: String = ""
    @Persisted var startedAt: Date = Date()
    @Persisted var endedAt: Date?
    @Persisted var sourceLabel: String = ""
    @Persisted var totalSteps: Int = 0
    @Persisted var leftSteps: Int = 0
    @Persisted var rightSteps: Int = 0
    @Persisted var avgCadenceBPM: Double?
    @Persisted var trainingLabel: String?
    @Persisted var cloudUploadedAt: Date?
    @Persisted var frames = List<RLMPoseFrame>()
}

final class RLMSavedSummary: Object {
    @Persisted(primaryKey: true) var id: String = ""
    @Persisted var date: Date = Date()
    @Persisted var sourceLabel: String = ""
    @Persisted var totalSteps: Int = 0
    @Persisted var leftSteps: Int = 0
    @Persisted var rightSteps: Int = 0
    @Persisted var avgCadenceBPM: Double?
    @Persisted var lines = List<String>()
}

// MARK: - 開檔與舊版資料遷移

enum PoseRealm {
    private static let schemaVersion: UInt64 = 1
    private static let legacyMigratedKey = "pose_realm_legacy_migrated_v1"

    static var fileURL: URL {
        let fm = FileManager.default
        let base: URL
        if let dir = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true) {
            base = dir
        } else {
            base = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        }
        return base.appendingPathComponent("pose.realm")
    }

    static func open() throws -> Realm {
        let config = Realm.Configuration(
            fileURL: fileURL,
            schemaVersion: schemaVersion,
            objectTypes: [
                RLMPoseSession.self,
                RLMSavedSummary.self,
                RLMPoseFrame.self,
                RLMPoseNode.self,
            ]
        )
        return try Realm(configuration: config)
    }

    /// 首次啟動時，將舊版 SQLite / JSON 匯入 Realm 並備份原檔。
    static func migrateLegacyIfNeeded() {
        guard !UserDefaults.standard.bool(forKey: legacyMigratedKey) else { return }
        guard let realm = try? open() else { return }

        let supportDir = fileURL.deletingLastPathComponent()
        let sqliteURL = supportDir.appendingPathComponent("pose.sqlite3")
        let jsonURL = supportDir.appendingPathComponent("pose_summaries.json")

        try? realm.write {
            migrateSQLite(from: sqliteURL, into: realm)
            migrateSummariesJSON(from: jsonURL, into: realm)
        }

        backupIfExists(sqliteURL)
        backupIfExists(jsonURL)
        UserDefaults.standard.set(true, forKey: legacyMigratedKey)
    }

    // MARK: 私有遷移

    private static func backupIfExists(_ url: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return }
        let backup = url.deletingPathExtension().appendingPathExtension("bak")
        try? fm.removeItem(at: backup)
        try? fm.moveItem(at: url, to: backup)
    }

    private static func migrateSummariesJSON(from url: URL, into realm: Realm) {
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([LegacySavedSummary].self, from: data) else { return }

        for item in decoded {
            let id = item.id.uuidString
            if realm.object(ofType: RLMSavedSummary.self, forPrimaryKey: id) != nil { continue }
            let obj = RLMSavedSummary()
            obj.id = id
            obj.date = item.date
            obj.sourceLabel = item.sourceLabel
            obj.totalSteps = item.totalSteps
            obj.leftSteps = item.leftSteps
            obj.rightSteps = item.rightSteps
            obj.avgCadenceBPM = item.avgCadenceBPM
            obj.lines.append(objectsIn: item.lines)
            realm.add(obj)
        }
    }

    private static func migrateSQLite(from url: URL, into realm: Realm) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK, let db else { return }
        defer { sqlite3_close(db) }

        guard realm.objects(RLMPoseSession.self).isEmpty else { return }

        var sessions: [(id: String, started: Double, ended: Double?, source: String, total: Int, left: Int, right: Int, bpm: Double?, training: String?, uploaded: Double?)] = []
        var stmt: OpaquePointer?
        let sessionSQL = """
            SELECT id, started_at, ended_at, source_label, total_steps, left_steps, right_steps,
                   avg_cadence_bpm, training_label, cloud_uploaded_at
            FROM sessions ORDER BY started_at;
        """
        if sqlite3_prepare_v2(db, sessionSQL, -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = String(cString: sqlite3_column_text(stmt, 0))
                let started = sqlite3_column_double(stmt, 1)
                let ended: Double? = sqlite3_column_type(stmt, 2) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 2)
                let source = sqlite3_column_text(stmt, 3).map { String(cString: $0) } ?? ""
                let total = Int(sqlite3_column_int(stmt, 4))
                let left = Int(sqlite3_column_int(stmt, 5))
                let right = Int(sqlite3_column_int(stmt, 6))
                let bpm: Double? = sqlite3_column_type(stmt, 7) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 7)
                let training = sqlite3_column_text(stmt, 8).map { String(cString: $0) }
                let uploaded: Double? = sqlite3_column_type(stmt, 9) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 9)
                sessions.append((id, started, ended, source, total, left, right, bpm, training, uploaded))
            }
        }
        sqlite3_finalize(stmt)

        for s in sessions {
            let session = RLMPoseSession()
            session.id = s.id
            session.startedAt = Date(timeIntervalSince1970: s.started)
            session.endedAt = s.ended.map { Date(timeIntervalSince1970: $0) }
            session.sourceLabel = s.source
            session.totalSteps = s.total
            session.leftSteps = s.left
            session.rightSteps = s.right
            session.avgCadenceBPM = s.bpm
            session.trainingLabel = s.training
            session.cloudUploadedAt = s.uploaded.map { Date(timeIntervalSince1970: $0) }

            var frameMap: [Int: RLMPoseFrame] = [:]
            let nodeSQL = """
                SELECT frame_index, timestamp, joint, x, y, z, visibility, presence
                FROM pose_nodes WHERE session_id = ? ORDER BY frame_index, id;
            """
            if sqlite3_prepare_v2(db, nodeSQL, -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_text(stmt, 1, s.id, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let frameIdx = Int(sqlite3_column_int(stmt, 0))
                    let ts = sqlite3_column_double(stmt, 1)
                    let frame = frameMap[frameIdx] ?? {
                        let f = RLMPoseFrame()
                        f.frameIndex = frameIdx
                        f.timestamp = ts
                        frameMap[frameIdx] = f
                        return f
                    }()
                    let node = RLMPoseNode()
                    node.joint = String(cString: sqlite3_column_text(stmt, 2))
                    node.x = sqlite3_column_double(stmt, 3)
                    node.y = sqlite3_column_double(stmt, 4)
                    node.z = sqlite3_column_double(stmt, 5)
                    node.visibility = sqlite3_column_double(stmt, 6)
                    node.presence = sqlite3_column_double(stmt, 7)
                    frame.nodes.append(node)
                }
            }
            sqlite3_finalize(stmt)

            for idx in frameMap.keys.sorted() {
                if let frame = frameMap[idx] {
                    session.frames.append(frame)
                }
            }
            realm.add(session)
        }
    }

    private struct LegacySavedSummary: Codable {
        let id: UUID
        let date: Date
        let sourceLabel: String
        let totalSteps: Int
        let leftSteps: Int
        let rightSteps: Int
        let avgCadenceBPM: Double?
        let lines: [String]
    }
}
