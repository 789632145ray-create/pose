//
//  MediaPipeRealm.swift
//  pose
//
//  獨立 Realm：只存放 MediaPipe BlazePose 偵測的骨架 session。
//  與 pose.realm（QuickPose／自訓模型／歷史摘要）完全分開，避免資料混在一起。
//

import Foundation
import RealmSwift

enum MediaPipeRealm {
    nonisolated static let schemaVersion: UInt64 = 1
    nonisolated static let fileName = "mediapipe.realm"

    nonisolated static var fileURL: URL {
        let fm = FileManager.default
        let base: URL
        if let dir = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true) {
            base = dir
        } else {
            base = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        }
        return base.appendingPathComponent(fileName)
    }

    nonisolated static func open() throws -> Realm {
        let config = Realm.Configuration(
            fileURL: fileURL,
            schemaVersion: schemaVersion,
            objectTypes: [
                RLMPoseSession.self,
                RLMPoseFrame.self,
                RLMPoseNode.self,
            ]
        )
        return try Realm(configuration: config)
    }
}
