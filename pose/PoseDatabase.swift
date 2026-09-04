//
//  PoseDatabase.swift
//  pose
//
//  以 Realm Database 建立姿勢偵測資料庫，將每一幀偵測到的「全部身體節點」
//  連同座標寫入本機。QuickPose／自訓模型寫入 pose.realm；MediaPipe 寫入 mediapipe.realm。
//
//  資料結構：
//    RLMPoseSession  一次偵測（相機或影片）為一筆，含起訖時間、來源與最終步態統計。
//    RLMPoseFrame   每一幀的節點集合（嵌入 session）。
//
//  寫入皆在獨立序列佇列上以 write transaction 完成，避免阻塞 QuickPose 影格回呼。
//

import Foundation
import QuickPoseCore
import RealmSwift

// MARK: - 純值節點

/// 單一身體節點（已從 QuickPose.Point3d 取出，可跨執行緒安全傳遞）。
struct PoseNode {
    let joint: String
    let x: Double
    let y: Double
    let z: Double
    let visibility: Double
    let presence: Double
}

/// 從 QuickPose 的 Landmarks 取出「整副骨架」全部節點，並給予穩定的英文鍵名。
enum PoseNodeExtractor {
    private static let sides: [(QuickPose.Side, String)] = [(.left, "left"), (.right, "right")]

    static func extractAll(from landmarks: QuickPose.Landmarks) -> [PoseNode] {
        var nodes: [PoseNode] = []
        nodes.reserveCapacity(35)

        func add(_ name: String, _ joint: QuickPose.Landmarks.Body) {
            let p = landmarks.landmark(forBody: joint)
            nodes.append(PoseNode(joint: name, x: p.x, y: p.y, z: p.z, visibility: p.visibility, presence: p.presence))
        }

        add("nose", .nose)
        add("shoulder_mid", .shoulderMid)
        add("hip_mid", .hipMid)

        for (side, label) in sides {
            add("\(label)_eye_inner", .eyeInner(side: side))
            add("\(label)_eye", .eye(side: side))
            add("\(label)_eye_outer", .eyeOuter(side: side))
            add("\(label)_ear", .ear(side: side))
            add("\(label)_mouth", .mouth(side: side))
            add("\(label)_shoulder", .shoulder(side: side))
            add("\(label)_elbow", .elbow(side: side))
            add("\(label)_wrist", .wrist(side: side))
            add("\(label)_pinky", .pinky(side: side))
            add("\(label)_index", .index(side: side))
            add("\(label)_thumb", .thumb(side: side))
            add("\(label)_hip", .hip(side: side))
            add("\(label)_knee", .knee(side: side))
            add("\(label)_ankle", .ankle(side: side))
            add("\(label)_heel", .heel(side: side))
            add("\(label)_foot_index", .footIndex(side: side))
        }

        return nodes
    }
}

// MARK: - 查詢用結構

/// 一個影格的全部節點（供上傳到後端）。
struct PoseFrameRecord {
    let frameIndex: Int
    let timestamp: Double
    let nodes: [PoseNode]
}

/// 一次偵測 session 的概要（供歷史 / 資料庫瀏覽顯示）。
struct PoseSessionRecord: Identifiable {
    let id: String
    let startedAt: Date
    let endedAt: Date?
    let sourceLabel: String
    let frameCount: Int
    let nodeCount: Int
    let totalSteps: Int
    let leftSteps: Int
    let rightSteps: Int
    let avgCadenceBPM: Double?
    /// 已上傳至雲端訓練庫的標籤：good / bad
    let trainingLabel: String?
    let cloudUploadedAt: Date?
}

// MARK: - Realm 資料庫

/// 本機節點庫種類：QuickPose／自訓模型走 `pose.realm`，MediaPipe 走獨立的 `mediapipe.realm`。
enum PoseNodeStoreKind: String {
    case pose
    case mediaPipe = "mediapipe"

    var sheetTitle: String {
        switch self {
        case .pose: return "節點資料庫"
        case .mediaPipe: return "MediaPipe 資料庫"
        }
    }

    var buttonTitle: String {
        switch self {
        case .pose: return "姿勢節點資料庫"
        case .mediaPipe: return "MediaPipe 資料庫"
        }
    }

    var uploadPath: String {
        switch self {
        case .pose: return "/poses"
        case .mediaPipe: return "/mediapipe/poses"
        }
    }

    var emptyHint: String {
        switch self {
        case .pose:
            return "開始相機或影片偵測後，節點會存入 pose.realm；完成後可在此標「好 / 壞」上傳訓練。"
        case .mediaPipe:
            return "MediaPipe 偵測的骨架會存入獨立的 mediapipe.realm，與 QuickPose／自訓模型資料分開。"
        }
    }
}

final class PoseDatabase {
    /// QuickPose／自訓模型節點 + 與歷史摘要同一套 pose.realm。
    static let shared = PoseDatabase(
        kind: .pose,
        queueLabel: "ai.pose.database",
        fileURL: PoseRealm.fileURL,
        open: PoseRealm.open,
        migrate: { PoseRealm.migrateLegacyIfNeeded() }
    )

    /// MediaPipe 專用獨立 Realm（mediapipe.realm）。
    static let mediaPipe = PoseDatabase(
        kind: .mediaPipe,
        queueLabel: "ai.pose.mediapipe.database",
        fileURL: MediaPipeRealm.fileURL,
        open: MediaPipeRealm.open
    )

    static func store(for engine: PoseAssessmentEngine) -> PoseDatabase {
        engine == .mediaPipe ? .mediaPipe : .shared
    }

    let kind: PoseNodeStoreKind
    let fileURL: URL

    private let queue: DispatchQueue
    private let openRealm: () throws -> Realm

    /// 目前進行中的 session（僅在 queue 上存取）。
    private var activeSessionID: String?
    private var frameCounter: Int = 0

    private init(
        kind: PoseNodeStoreKind,
        queueLabel: String,
        fileURL: URL,
        open: @escaping () throws -> Realm,
        migrate: (() -> Void)? = nil
    ) {
        self.kind = kind
        self.fileURL = fileURL
        self.queue = DispatchQueue(label: queueLabel, qos: .utility)
        self.openRealm = open
        migrate?()
    }

    // MARK: Session 生命週期

    @discardableResult
    func beginSession(sourceLabel: String) -> String {
        let id = UUID().uuidString
        let now = Date()
        queue.async { [weak self] in
            guard let self, let realm = try? self.openRealm() else { return }
            self.activeSessionID = id
            self.frameCounter = 0
            let session = RLMPoseSession()
            session.id = id
            session.startedAt = now
            session.sourceLabel = sourceLabel
            try? realm.write {
                realm.add(session)
            }
        }
        return id
    }

    func recordFrame(nodes: [PoseNode], timestamp: Date = Date()) {
        guard !nodes.isEmpty else { return }
        let ts = timestamp.timeIntervalSince1970
        queue.async { [weak self] in
            guard let self, let sid = self.activeSessionID, let realm = try? self.openRealm() else { return }
            let frame = self.frameCounter
            self.frameCounter += 1

            let frameObj = RLMPoseFrame()
            frameObj.frameIndex = frame
            frameObj.timestamp = ts
            for n in nodes {
                let node = RLMPoseNode()
                node.joint = n.joint
                node.x = n.x
                node.y = n.y
                node.z = n.z
                node.visibility = n.visibility
                node.presence = n.presence
                frameObj.nodes.append(node)
            }

            try? realm.write {
                if let session = realm.object(ofType: RLMPoseSession.self, forPrimaryKey: sid) {
                    session.frames.append(frameObj)
                }
            }
        }
    }

    func endSession(totalSteps: Int, leftSteps: Int, rightSteps: Int, avgCadenceBPM: Double?) {
        let now = Date()
        queue.async { [weak self] in
            guard let self, let sid = self.activeSessionID, let realm = try? self.openRealm() else { return }
            try? realm.write {
                if let session = realm.object(ofType: RLMPoseSession.self, forPrimaryKey: sid) {
                    session.endedAt = now
                    session.totalSteps = totalSteps
                    session.leftSteps = leftSteps
                    session.rightSteps = rightSteps
                    session.avgCadenceBPM = avgCadenceBPM
                }
            }
            self.activeSessionID = nil
            self.frameCounter = 0
        }
    }

    // MARK: 查詢

    func sessions() -> [PoseSessionRecord] {
        queue.sync {
            guard let realm = try? openRealm() else { return [] }
            return realm.objects(RLMPoseSession.self)
                .sorted(byKeyPath: "startedAt", ascending: false)
                .map { session in
                    let nodeCount = session.frames.reduce(0) { $0 + $1.nodes.count }
                    return PoseSessionRecord(
                        id: session.id,
                        startedAt: session.startedAt,
                        endedAt: session.endedAt,
                        sourceLabel: session.sourceLabel,
                        frameCount: session.frames.count,
                        nodeCount: nodeCount,
                        totalSteps: session.totalSteps,
                        leftSteps: session.leftSteps,
                        rightSteps: session.rightSteps,
                        avgCadenceBPM: session.avgCadenceBPM,
                        trainingLabel: session.trainingLabel,
                        cloudUploadedAt: session.cloudUploadedAt
                    )
                }
        }
    }

    func totalNodeCount() -> Int {
        queue.sync {
            guard let realm = try? openRealm() else { return 0 }
            return realm.objects(RLMPoseSession.self).reduce(0) { total, session in
                total + session.frames.reduce(0) { $0 + $1.nodes.count }
            }
        }
    }

    func firstFrameNodes(sessionID: String) -> [PoseNode] {
        queue.sync {
            guard let realm = try? openRealm(),
                  let session = realm.object(ofType: RLMPoseSession.self, forPrimaryKey: sessionID),
                  let first = session.frames.min(by: { $0.frameIndex < $1.frameIndex }) else { return [] }
            return first.nodes.map {
                PoseNode(joint: $0.joint, x: $0.x, y: $0.y, z: $0.z, visibility: $0.visibility, presence: $0.presence)
            }
        }
    }

    func allFrames(sessionID: String) -> [PoseFrameRecord] {
        queue.sync {
            guard let realm = try? openRealm(),
                  let session = realm.object(ofType: RLMPoseSession.self, forPrimaryKey: sessionID) else { return [] }
            return session.frames
                .sorted(by: { $0.frameIndex < $1.frameIndex })
                .map { frame in
                    PoseFrameRecord(
                        frameIndex: frame.frameIndex,
                        timestamp: frame.timestamp,
                        nodes: frame.nodes.map {
                            PoseNode(joint: $0.joint, x: $0.x, y: $0.y, z: $0.z, visibility: $0.visibility, presence: $0.presence)
                        }
                    )
                }
        }
    }

    func markTrainingUpload(sessionID: String, label: String) {
        let now = Date()
        queue.sync {
            guard let realm = try? openRealm() else { return }
            try? realm.write {
                if let session = realm.object(ofType: RLMPoseSession.self, forPrimaryKey: sessionID) {
                    session.trainingLabel = label
                    session.cloudUploadedAt = now
                }
            }
        }
    }

    func clearAll() {
        queue.sync {
            guard let realm = try? openRealm() else { return }
            try? realm.write {
                realm.delete(realm.objects(RLMPoseSession.self))
            }
            activeSessionID = nil
            frameCounter = 0
        }
    }
}
