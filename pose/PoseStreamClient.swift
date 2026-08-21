//
//  PoseStreamClient.swift
//  pose
//
//  將姿勢節點「每幀即時串流」上傳到後端 NoSQL（MongoDB），並向 /predict 取得即時好/壞預測。
//  - begin：開一個串流 session（/sessions）。
//  - enqueue：每幀快速暫存（不做網路），由背景迴圈批次上傳。
//  - flush：把暫存的幀批次送到 /sessions/{id}/frames。
//  - predictIfReady：用最近視窗呼叫 /predict。
//  - end：送出剩餘幀並結束 session（/sessions/{id}/finish）。
//  權杖取自 Keychain（FastAPI 登入後寫入，與 AuthManager 共用）。
//
//  說明：本類別標記為 @MainActor，與專案預設隔離一致；網路皆為 async，await 期間不會卡住主執行緒。
//

import Foundation

/// 即時串流上傳（預測模式，不帶訓練標籤；標好/壞改在節點資料庫操作）。
enum PoseStreamMode: String, CaseIterable, Identifiable {
    case predict

    var id: String { rawValue }

    var title: String { "預測" }

    var backendLabel: String? { nil }
}

/// 即時預測結果（或狀態訊息）。
struct LivePrediction {
    let label: String?          // "good" / "bad" / nil
    let probabilityGood: Double
    let note: String?           // 例如「尚未訓練模型」

    var hudText: String {
        if let note { return "即時預測：\(note)" }
        guard let label else { return "即時預測：—" }
        let pct = Int(confidencePercent)
        return "即時預測：\(label == "good" ? "好" : "壞") \(pct)%"
    }

    var isGood: Bool { label == "good" }

    /// 預測為該標籤的信心百分比（0–100）。
    var confidencePercent: Double {
        guard let label else { return 0 }
        let p = label == "good" ? probabilityGood : (1 - probabilityGood)
        return max(0, min(100, p * 100))
    }

    var qualityTitle: String {
        guard let label else { return "無法辨識" }
        return label == "good" ? "好" : "壞"
    }
}

@MainActor
final class PoseStreamClient {
    // 與 AuthManager / PoseDatabase 共用的儲存鍵。
    private static let tokenAccount = "access_token"
    private var baseURL = ""
    private var token = ""
    private var sessionID: String?

    private var frameIndex = 0
    private var pending: [WireFrame] = []   // 待上傳
    private var window: [WireFrame] = []    // 最近視窗（供預測）

    private let batchMaxPending = 600        // 離線時上限，避免無限增長
    private let windowSize = 30

    // MARK: Wire 模型（對應後端 PoseFrame / PoseNode）

    struct WireNode: Codable {
        let joint: String
        let x, y, z, visibility, presence: Double
    }

    struct WireFrame: Codable {
        let frame_index: Int
        let timestamp: Double
        let nodes: [WireNode]
    }

    private struct StartBody: Encodable {
        let label: String?
        let source_label: String
    }

    private struct StartResponse: Decodable {
        let session_id: String
    }

    private struct FramesBody: Encodable {
        let frames: [WireFrame]
    }

    private struct FinishBody: Encodable {
        let total_steps: Int
        let left_steps: Int
        let right_steps: Int
        let avg_cadence_bpm: Double?
    }

    private struct PredictResponse: Decodable {
        let label: String
        let probability_good: Double
        let probability_bad: Double
    }

    private struct APIErrorBody: Decodable {
        let detail: String?
    }

    var isStreaming: Bool { sessionID != nil }

    // MARK: 生命週期

    /// 開一個串流 session。回傳是否成功（失敗時不影響本機偵測）。
    @discardableResult
    func begin(mode: PoseStreamMode, sourceLabel: String) async -> Bool {
        reset()
        guard loadCredentials() else { return false }

        let body = StartBody(label: mode.backendLabel, source_label: sourceLabel)
        guard let data = try? await post(path: "/sessions", body: body),
              let resp = try? JSONDecoder().decode(StartResponse.self, from: data) else {
            return false
        }
        sessionID = resp.session_id
        return true
    }

    /// 每幀呼叫（快速，不做網路）：累積到暫存與視窗。
    func enqueue(nodes: [PoseNode], timestamp: Double) {
        let frame = WireFrame(
            frame_index: frameIndex,
            timestamp: timestamp,
            nodes: nodes.map { WireNode(joint: $0.joint, x: $0.x, y: $0.y, z: $0.z, visibility: $0.visibility, presence: $0.presence) }
        )
        frameIndex += 1

        pending.append(frame)
        if pending.count > batchMaxPending { pending.removeFirst(pending.count - batchMaxPending) }

        window.append(frame)
        if window.count > windowSize { window.removeFirst(window.count - windowSize) }
    }

    /// 批次上傳暫存的幀。
    func flush() async {
        guard let sessionID, !pending.isEmpty else { return }
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        let body = FramesBody(frames: batch)
        if (try? await post(path: "/sessions/\(sessionID)/frames", body: body)) == nil {
            // 上傳失敗：放回暫存等下次重試（保留上限）。
            pending.insert(contentsOf: batch, at: 0)
            if pending.count > batchMaxPending { pending.removeFirst(pending.count - batchMaxPending) }
        }
    }

    /// 用最近視窗向 /predict 取得即時好/壞。
    func predictIfReady() async -> LivePrediction? {
        guard !window.isEmpty else { return nil }
        guard loadCredentials() else {
            return LivePrediction(label: nil, probabilityGood: 0, note: "尚未登入")
        }
        return await predict(frames: window)
    }

    /// 以指定影格向 /predict 辨識姿勢品質（供影片整段分析）。
    func predict(frames: [WireFrame]) async -> LivePrediction? {
        guard !frames.isEmpty else {
            return LivePrediction(label: nil, probabilityGood: 0, note: "尚無節點資料")
        }
        guard loadCredentials() else {
            return LivePrediction(label: nil, probabilityGood: 0, note: "尚未登入")
        }
        let body = FramesBody(frames: frames)
        let timeout: TimeInterval = frames.count > 100 ? 90 : 30
        do {
            let (data, http) = try await postRaw(path: "/predict", body: body, timeout: timeout)
            if http.statusCode == 503 {
                return LivePrediction(label: nil, probabilityGood: 0, note: "尚未訓練模型")
            }
            if http.statusCode == 401 {
                return LivePrediction(label: nil, probabilityGood: 0, note: "登入已過期，請重新登入")
            }
            guard (200...299).contains(http.statusCode),
                  let resp = try? JSONDecoder().decode(PredictResponse.self, from: data) else {
                let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
                return LivePrediction(label: nil, probabilityGood: 0, note: detail ?? "辨識失敗")
            }
            return LivePrediction(label: resp.label, probabilityGood: resp.probability_good, note: nil)
        } catch {
            return LivePrediction(label: nil, probabilityGood: 0, note: "無法連線伺服器")
        }
    }

    /// 將本機影格轉成 API 格式並呼叫 /predict。
    func predict(frames records: [PoseFrameRecord], maxFrames: Int = 400) async -> LivePrediction? {
        let sampled = Self.sampleFrames(records, maxCount: maxFrames)
        guard !sampled.isEmpty else { return LivePrediction(label: nil, probabilityGood: 0, note: "尚無節點資料") }
        let wire = sampled.map { rec in
            WireFrame(
                frame_index: rec.frameIndex,
                timestamp: rec.timestamp,
                nodes: rec.nodes.map { WireNode(joint: $0.joint, x: $0.x, y: $0.y, z: $0.z, visibility: $0.visibility, presence: $0.presence) }
            )
        }
        return await predict(frames: wire)
    }

    /// 均勻抽樣，避免長影片一次送出過多影格。
    static func sampleFrames(_ frames: [PoseFrameRecord], maxCount: Int) -> [PoseFrameRecord] {
        guard frames.count > maxCount, maxCount > 0 else { return frames }
        let step = Double(frames.count) / Double(maxCount)
        return (0..<maxCount).map { i in
            frames[min(Int(Double(i) * step), frames.count - 1)]
        }
    }

    /// 結束 session：送出剩餘幀並回報統計。
    func end(totalSteps: Int, leftSteps: Int, rightSteps: Int, avgCadenceBPM: Double?) async {
        guard let sessionID else { reset(); return }
        await flush()
        let body = FinishBody(total_steps: totalSteps, left_steps: leftSteps, right_steps: rightSteps, avg_cadence_bpm: avgCadenceBPM)
        _ = try? await post(path: "/sessions/\(sessionID)/finish", body: body)
        reset()
    }

    private struct PoseUploadBody: Encodable {
        let label: String
        let source_label: String
        let total_steps: Int
        let left_steps: Int
        let right_steps: Int
        let avg_cadence_bpm: Double?
        let frames: [WireFrame]
    }

    private struct PoseUploadResponse: Decodable {
        let id: String
        let frame_count: Int
        let node_count: Int
    }

    struct LabeledUploadResult {
        let success: Bool
        let message: String
    }

    /// 將本機一次偵測的節點以 good/bad 標籤上傳到後端（供 train.py 訓練）。
    func uploadLabeledSession(
        label: String,
        sourceLabel: String,
        totalSteps: Int,
        leftSteps: Int,
        rightSteps: Int,
        avgCadenceBPM: Double?,
        frames: [PoseFrameRecord],
        maxFrames: Int = 600
    ) async -> LabeledUploadResult {
        guard let token = KeychainHelper.read(Self.tokenAccount), !token.isEmpty else {
            return LabeledUploadResult(success: false, message: "尚未登入")
        }
        self.token = token
        self.baseURL = ServerConfig.baseURL

        let sampled = Self.sampleFrames(frames, maxCount: maxFrames)
        guard !sampled.isEmpty else {
            return LabeledUploadResult(success: false, message: "此 session 尚無節點")
        }

        let wire = sampled.map { rec in
            WireFrame(
                frame_index: rec.frameIndex,
                timestamp: rec.timestamp,
                nodes: rec.nodes.map {
                    WireNode(joint: $0.joint, x: $0.x, y: $0.y, z: $0.z, visibility: $0.visibility, presence: $0.presence)
                }
            )
        }

        let body = PoseUploadBody(
            label: label,
            source_label: sourceLabel,
            total_steps: totalSteps,
            left_steps: leftSteps,
            right_steps: rightSteps,
            avg_cadence_bpm: avgCadenceBPM,
            frames: wire
        )

        do {
            let (data, http) = try await postRaw(path: "/poses", body: body, timeout: 120)
            guard (200...299).contains(http.statusCode) else {
                let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
                return LabeledUploadResult(success: false, message: detail ?? "上傳失敗（\(http.statusCode)）")
            }
            if let resp = try? JSONDecoder().decode(PoseUploadResponse.self, from: data) {
                return LabeledUploadResult(
                    success: true,
                    message: "已上傳 \(resp.frame_count) 幀、\(resp.node_count) 節點"
                )
            }
            return LabeledUploadResult(success: true, message: "已上傳至雲端")
        } catch {
            return LabeledUploadResult(success: false, message: "無法連線：\(error.localizedDescription)")
        }
    }

    // MARK: 私有

    /// 從 Keychain 載入登入權杖與伺服器網址（影片品質辨識等未走 begin 的路徑也需要）。
    @discardableResult
    private func loadCredentials() -> Bool {
        guard let stored = KeychainHelper.read(Self.tokenAccount), !stored.isEmpty else {
            return false
        }
        token = stored
        baseURL = ServerConfig.baseURL
        return true
    }

    private func reset() {
        sessionID = nil
        frameIndex = 0
        pending.removeAll(keepingCapacity: true)
        window.removeAll(keepingCapacity: true)
    }

    @discardableResult
    private func post<T: Encodable>(path: String, body: T) async throws -> Data {
        let (data, http) = try await postRaw(path: path, body: body)
        guard (200...299).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
            throw NSError(domain: "PoseStream", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: detail ?? "HTTP \(http.statusCode)"])
        }
        return data
    }

    private func postRaw<T: Encodable>(path: String, body: T, timeout: TimeInterval = 20) async throws -> (Data, HTTPURLResponse) {
        guard let url = URL(string: baseURL + path), url.scheme != nil, url.host != nil else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = timeout
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        return (data, http)
    }
}
