//
//  PoseAnalysisPipeline.swift
//  pose
//
//  逐步（每一步）偵測管線：
//  1) 5 幀移動平均低通（穩定關鍵點 y 值，抑制抖動造成的偽步）
//  2) 以骨盆中心為原點的相對座標（人移動位置時不影響步偵測）
//  3) 左右腳「腳踝 y」峰／谷偵測；同幀／短間隔以振幅仲裁，並偏好左右交替，避免一腳擺動時對側誤加步
//

import Foundation
import QuickPoseCore

// MARK: - 純值關鍵點

struct JointSample: Equatable {
    var x: Double
    var y: Double
    var z: Double
    var visibility: Double
    var presence: Double

    fileprivate static let zero = JointSample(x: 0, y: 0, z: 0, visibility: 0, presence: 0)

    func shifted(by origin: JointSample) -> JointSample {
        JointSample(
            x: x - origin.x,
            y: y - origin.y,
            z: z - origin.z,
            visibility: min(visibility, origin.visibility),
            presence: min(presence, origin.presence)
        )
    }
}

extension QuickPose.Point3d {
    fileprivate var asJointSample: JointSample {
        JointSample(x: x, y: y, z: z, visibility: visibility, presence: presence)
    }
}

struct PoseKeypointFrame: Equatable {
    var leftHip: JointSample?
    var rightHip: JointSample?
    var leftAnkle: JointSample?
    var rightAnkle: JointSample?

    fileprivate static let empty = PoseKeypointFrame(leftHip: nil, rightHip: nil, leftAnkle: nil, rightAnkle: nil)
}

enum PoseKeypointExtractor {
    static func extract(_ landmarks: QuickPose.Landmarks) -> PoseKeypointFrame {
        PoseKeypointFrame(
            leftHip: landmarks.landmark(forBody: .hip(side: .left)).asJointSample,
            rightHip: landmarks.landmark(forBody: .hip(side: .right)).asJointSample,
            leftAnkle: landmarks.landmark(forBody: .ankle(side: .left)).asJointSample,
            rightAnkle: landmarks.landmark(forBody: .ankle(side: .right)).asJointSample
        )
    }
}

// MARK: - 5 幀低通

final class LandmarkLowPassFilter {
    private var window: [PoseKeypointFrame] = []
    private let maxFrames: Int

    init(windowSize: Int = 5) {
        self.maxFrames = max(1, windowSize)
    }

    func reset() {
        window.removeAll(keepingCapacity: true)
    }

    func pushAndAverage(_ frame: PoseKeypointFrame) -> PoseKeypointFrame {
        window.append(frame)
        if window.count > maxFrames { window.removeFirst() }

        func avg(_ kp: KeyPath<PoseKeypointFrame, JointSample?>) -> JointSample? {
            let xs = window.compactMap { $0[keyPath: kp] }
            guard !xs.isEmpty else { return nil }
            let n = Double(xs.count)
            return JointSample(
                x: xs.map(\.x).reduce(0, +) / n,
                y: xs.map(\.y).reduce(0, +) / n,
                z: xs.map(\.z).reduce(0, +) / n,
                visibility: xs.map(\.visibility).reduce(0, +) / n,
                presence: xs.map(\.presence).reduce(0, +) / n
            )
        }

        return PoseKeypointFrame(
            leftHip: avg(\.leftHip),
            rightHip: avg(\.rightHip),
            leftAnkle: avg(\.leftAnkle),
            rightAnkle: avg(\.rightAnkle)
        )
    }
}

// MARK: - 骨盆中心歸一

enum PosePelvisNormalizer {
    static func pelvisCenter(from frame: PoseKeypointFrame) -> JointSample? {
        guard let lh = frame.leftHip, let rh = frame.rightHip else { return nil }
        return JointSample(
            x: (lh.x + rh.x) / 2,
            y: (lh.y + rh.y) / 2,
            z: (lh.z + rh.z) / 2,
            visibility: min(lh.visibility, rh.visibility),
            presence: min(lh.presence, rh.presence)
        )
    }
}

// MARK: - 步事件

enum PoseFootSide: String {
    case left, right
    var localizedLabel: String { self == .left ? "左" : "右" }
}

struct StepEvent: Identifiable, Equatable {
    let id: UUID
    let index: Int          // 全域第幾步
    let side: PoseFootSide
    let timestamp: Date
    /// 與「上一步」（不分腳側）的時間差，單位秒；首步為 nil。
    let intervalFromPrevious: TimeInterval?

    var cadenceBPM: Double? {
        guard let dt = intervalFromPrevious, dt > 0 else { return nil }
        return 60.0 / dt
    }
}

// MARK: - 左右腳「每一步」偵測（含對側誤觸抑制）

final class PerFootStepDetector {
    private struct FootState {
        var series: [Double] = []
        var lastValleyValue: Double? = nil
        var lastPeakFrameIndex: Int = -10_000
        var frameCounter: Int = 0
    }

    private struct PeakCandidate {
        var side: PoseFootSide
        var amplitude: Double
        var peakFrame: Int
    }

    private(set) var leftSteps: Int = 0
    private(set) var rightSteps: Int = 0
    private(set) var recentEvents: [StepEvent] = []

    var totalSteps: Int { leftSteps + rightSteps }
    var lastEvent: StepEvent? { recentEvents.last }

    private var leftState = FootState()
    private var rightState = FootState()
    private var lastEventTime: Date?
    private var lastAcceptedSide: PoseFootSide?
    private var lastAcceptedGlobalFrame: Int = -10_000
    private var globalFrame: Int = 0
    private var globalIndex = 0

    /// 同腳兩峰最少間隔幀數（30FPS 下約 8 ≈ 0.27s）。
    private let minPeakDistanceFrames: Int
    /// 任一腳被接受後，全域最短再接受間隔（抑制同幀／對側連帶假峰）。
    private let minGlobalPeakDistanceFrames: Int
    /// 接受一步後短暫壓制對側，避免「左腳擺動、右腳也加步」。
    private let crossSideSuppressFrames: Int
    /// 同側連續兩步的最短間隔（跛行等例外才放寬；正常應左右交替）。
    private let minSameSideDistanceFrames: Int
    /// 從谷到峰最小振幅（骨盆歸一 y）。
    private let minPeakValleyDelta: Double
    private let maxHistory: Int = 60
    private let maxRecentEvents: Int = 30

    init(
        minPeakDistanceFrames: Int = 8,
        minGlobalPeakDistanceFrames: Int = 6,
        crossSideSuppressFrames: Int = 7,
        minSameSideDistanceFrames: Int = 18,
        minPeakValleyDelta: Double = 0.028
    ) {
        self.minPeakDistanceFrames = minPeakDistanceFrames
        self.minGlobalPeakDistanceFrames = minGlobalPeakDistanceFrames
        self.crossSideSuppressFrames = crossSideSuppressFrames
        self.minSameSideDistanceFrames = minSameSideDistanceFrames
        self.minPeakValleyDelta = minPeakValleyDelta
    }

    func reset() {
        leftState = FootState()
        rightState = FootState()
        lastEventTime = nil
        lastAcceptedSide = nil
        lastAcceptedGlobalFrame = -10_000
        globalFrame = 0
        globalIndex = 0
        leftSteps = 0
        rightSteps = 0
        recentEvents.removeAll(keepingCapacity: true)
    }

    /// 每幀以「已骨盆歸一的左／右腳踝 y」推進；同幀雙峰只保留振幅較大側。
    @discardableResult
    func ingest(leftAnkleY: Double?, rightAnkleY: Double?, now: Date = Date()) -> [StepEvent] {
        globalFrame += 1

        var leftCandidate: PeakCandidate?
        var rightCandidate: PeakCandidate?
        if let y = leftAnkleY {
            leftCandidate = detectPeak(y: y, side: .left, state: &leftState)
        }
        if let y = rightAnkleY {
            rightCandidate = detectPeak(y: y, side: .right, state: &rightState)
        }

        let chosen: PeakCandidate?
        switch (leftCandidate, rightCandidate) {
        case let (l?, r?):
            // 同幀兩腳都像一步：只算振幅較大的那一腳，避免對側連帶加步。
            chosen = l.amplitude >= r.amplitude ? l : r
        case let (l?, nil):
            chosen = l
        case let (nil, r?):
            chosen = r
        case (nil, nil):
            chosen = nil
        }

        guard let candidate = chosen, shouldAccept(candidate) else { return [] }

        // 確認接受後才鎖定該腳的峰索引，避免被拒的假峰吃掉後續真實步。
        switch candidate.side {
        case .left:
            leftState.lastPeakFrameIndex = candidate.peakFrame
            leftState.lastValleyValue = nil
            leftSteps += 1
        case .right:
            rightState.lastPeakFrameIndex = candidate.peakFrame
            rightState.lastValleyValue = nil
            rightSteps += 1
        }

        lastAcceptedSide = candidate.side
        lastAcceptedGlobalFrame = globalFrame
        globalIndex += 1
        let interval = lastEventTime.map { now.timeIntervalSince($0) }
        lastEventTime = now
        let event = StepEvent(
            id: UUID(),
            index: globalIndex,
            side: candidate.side,
            timestamp: now,
            intervalFromPrevious: interval
        )
        recentEvents.append(event)
        if recentEvents.count > maxRecentEvents { recentEvents.removeFirst() }
        return [event]
    }

    private func shouldAccept(_ candidate: PeakCandidate) -> Bool {
        let sinceGlobal = globalFrame - lastAcceptedGlobalFrame
        guard sinceGlobal >= minGlobalPeakDistanceFrames else { return false }

        if let last = lastAcceptedSide {
            if last == candidate.side {
                // 正常步態應左右交替；同側連計需明顯更長間隔（例如跛行）。
                guard sinceGlobal >= minSameSideDistanceFrames else { return false }
            } else if sinceGlobal < crossSideSuppressFrames {
                // 對側在極短時間內的峰多半是連帶抖動，不是下一步。
                return false
            }
        }
        return true
    }

    /// 偵測峰候選；不寫入 lastPeak／不清 valley，等 ingest 仲裁接受後再鎖定。
    private func detectPeak(y: Double, side: PoseFootSide, state: inout FootState) -> PeakCandidate? {
        state.frameCounter += 1
        state.series.append(y)
        if state.series.count > maxHistory { state.series.removeFirst() }
        guard state.series.count >= 3 else { return nil }

        let n = state.series.count
        let y0 = state.series[n - 3]
        let y1 = state.series[n - 2]
        let y2 = state.series[n - 1]

        let isValleyAtMid = y0 > y1 && y1 < y2
        let isPeakAtMid = y0 < y1 && y1 > y2

        if isValleyAtMid {
            state.lastValleyValue = y1
        }

        guard isPeakAtMid else { return nil }
        let peakFrame = state.frameCounter - 1
        guard peakFrame - state.lastPeakFrameIndex >= minPeakDistanceFrames else { return nil }
        guard let valley = state.lastValleyValue else { return nil }
        let amplitude = y1 - valley
        guard amplitude >= minPeakValleyDelta else { return nil }

        return PeakCandidate(side: side, amplitude: amplitude, peakFrame: peakFrame)
    }
}

// MARK: - 串接

struct PoseStepFrameResult {
    var lines: [String]
    var hudSummary: String
    var emittedSteps: [StepEvent]
    var totalSteps: Int
    var leftSteps: Int
    var rightSteps: Int
    var recentEvents: [StepEvent]
}

/// 每幀姿勢問題的結構化標記，用來累積整支影片的整體建議。
enum PoseIssueCode: Hashable {
    case lowVisibility
    case shoulderTilt
    case hipTilt
    case trunkSideBend
    case headOffMidline
    case overStriding
    case insufficientKneeFlexion
    case footDrag
    case limitedHipExtension
    case trunkInstability
    case forwardStoop
    case raisedArms
    case excessiveArmSwing
    case ipsilateralArmSwing
    case forefootStrike
    case incompleteToeOff
    case waistHingeLean
    case insufficientForwardLean
    case excessiveForwardLean
    case pelvisInstability
    case heelStrikeWhileRunning
    case elbowAngleOff
    case armCrossMidline

    /// 背面視角無法可靠量測的前後向（矢狀面）問題；背面模式應略過。
    static let sagittalPlaneCodes: Set<PoseIssueCode> = [
        .overStriding,
        .footDrag,
        .limitedHipExtension,
        .forwardStoop,
        .forefootStrike,
        .incompleteToeOff,
        .waistHingeLean,
        .insufficientForwardLean,
        .excessiveForwardLean,
        .heelStrikeWhileRunning,
    ]

    var summaryDescription: String {
        switch self {
        case .lowVisibility: return "關節可見度不足"
        case .shoulderTilt:  return "雙肩高低差"
        case .hipTilt:       return "骨盆傾斜"
        case .trunkSideBend: return "上半身側傾"
        case .headOffMidline:return "頭部偏離身體中線"
        case .overStriding: return "步幅過大／腳落點超前"
        case .insufficientKneeFlexion: return "觸地膝屈曲不足"
        case .footDrag: return "拖步（腳尖未充分離地）"
        case .limitedHipExtension: return "髖伸展不足"
        case .trunkInstability: return "軀幹偏移／平衡不穩"
        case .forwardStoop: return "上背或頭部前傾"
        case .raisedArms: return "手臂未自然下垂"
        case .excessiveArmSwing: return "擺臂幅度過大"
        case .ipsilateralArmSwing: return "同側擺臂（未對側配合）"
        case .forefootStrike: return "腳尖／前腳掌先著地"
        case .incompleteToeOff: return "腳尖蹬地不足"
        case .waistHingeLean: return "從腰部彎曲前傾"
        case .insufficientForwardLean: return "跑步前傾不足"
        case .excessiveForwardLean: return "跑步前傾過大"
        case .pelvisInstability: return "骨盆不穩／左右搖晃"
        case .heelStrikeWhileRunning: return "跑步用腳跟著地"
        case .elbowAngleOff: return "手肘未保持約 90 度"
        case .armCrossMidline: return "擺臂越過身體中線"
        }
    }
}

final class PoseAnalysisPipeline {
    private let lowPass = LandmarkLowPassFilter(windowSize: 5)
    private let stepDetector = PerFootStepDetector()
    private var issueCounts: [PoseIssueCode: Int] = [:]
    private var totalEvaluatedFrames: Int = 0
    private var gaitMetrics = GaitSessionMetrics()

    /// 由登入使用者的身高／體重衍生的步態個人化檔案。
    var bodyProfile: BodyGaitProfile?
    /// 走路或跑步，決定用哪一套姿勢核心。
    var activityMode: GaitActivityMode = .walking
    /// 背面拍攝：不累積／不摘要前後向（前傾、跨步、著地等）建議。
    var suppressSagittalFormAdvice = false

    var totalSteps: Int { stepDetector.totalSteps }
    var leftSteps: Int { stepDetector.leftSteps }
    var rightSteps: Int { stepDetector.rightSteps }
    /// 本 session 是否已累積過有效姿勢幀（暫停後 reset 前用來判斷可否產生摘要）。
    var hasEvaluatedFrames: Bool { totalEvaluatedFrames > 0 }
    var avgCadenceBPM: Double? {
        let bpms = stepDetector.recentEvents.compactMap { $0.cadenceBPM }
        guard !bpms.isEmpty else { return nil }
        return bpms.reduce(0, +) / Double(bpms.count)
    }

    func reset() {
        lowPass.reset()
        stepDetector.reset()
        issueCounts.removeAll()
        totalEvaluatedFrames = 0
        gaitMetrics = GaitSessionMetrics()
    }

    /// 對單一幀做：低通 → 骨盆歸一 → 左右腳踝獨立步偵測；同時累積問題次數。
    func process(landmarks: QuickPose.Landmarks, lines: [String], issues: Set<PoseIssueCode>) -> PoseStepFrameResult {
        if !suppressSagittalFormAdvice {
            gaitMetrics.ingest(landmarks: landmarks, profile: bodyProfile, activity: activityMode)
        }
        return process(keypointFrame: PoseKeypointExtractor.extract(landmarks), lines: lines, issues: issues)
    }

    /// MediaPipe 原生路徑：已轉成關鍵點幀，不再依賴 QuickPose.Landmarks。
    func process(
        keypointFrame: PoseKeypointFrame,
        lines: [String],
        issues: Set<PoseIssueCode>,
        formSnapshot: WalkingFormSnapshot? = nil
    ) -> PoseStepFrameResult {
        totalEvaluatedFrames += 1
        let countedIssues = suppressSagittalFormAdvice
            ? issues.subtracting(PoseIssueCode.sagittalPlaneCodes)
            : issues
        for code in countedIssues {
            issueCounts[code, default: 0] += 1
        }
        if !suppressSagittalFormAdvice, let formSnapshot {
            switch activityMode {
            case .walking:
                WalkingFormAdvisor.accumulate(metrics: &gaitMetrics, snapshot: formSnapshot)
            case .running:
                RunningFormAdvisor.accumulate(metrics: &gaitMetrics, snapshot: formSnapshot)
            }
        }
        let smoothed = lowPass.pushAndAverage(keypointFrame)

        var leftRel: Double?
        var rightRel: Double?
        if let pelvis = PosePelvisNormalizer.pelvisCenter(from: smoothed) {
            if let la = smoothed.leftAnkle, la.visibility > 0.35, la.presence > 0.35 {
                leftRel = la.y - pelvis.y
            }
            if let ra = smoothed.rightAnkle, ra.visibility > 0.35, ra.presence > 0.35 {
                rightRel = ra.y - pelvis.y
            }
        }

        let emitted = stepDetector.ingest(leftAnkleY: leftRel, rightAnkleY: rightRel)
        // 固定位數寬度，避免左上角 HUD 隨步數位數增減而伸縮。
        let hud: String
        if let last = stepDetector.lastEvent {
            let bpm = last.cadenceBPM.map { String(format: "%3.0f", $0) } ?? "  —"
            hud = String(
                format: "L:%3d R:%3d 總:%3d ·%@ %@bpm",
                stepDetector.leftSteps,
                stepDetector.rightSteps,
                stepDetector.totalSteps,
                last.side.localizedLabel,
                bpm
            )
        } else {
            hud = String(format: "L:%3d R:%3d 總:%3d ·—   —bpm", 0, 0, 0)
        }

        return PoseStepFrameResult(
            lines: lines,
            hudSummary: hud,
            emittedSteps: emitted,
            totalSteps: stepDetector.totalSteps,
            leftSteps: stepDetector.leftSteps,
            rightSteps: stepDetector.rightSteps,
            recentEvents: stepDetector.recentEvents
        )
    }

    /// 給整支影片用的整體建議（步態 + 姿勢問題累積）。
    func videoSummary() -> [String] {
        guard totalEvaluatedFrames > 0 else {
            return ["尚未取得有效的姿勢資料。請確認影片中有完整人物，再試一次。"]
        }

        var out: [String] = []
        let total = stepDetector.totalSteps
        out.append("總步數：\(total)（左 \(stepDetector.leftSteps) / 右 \(stepDetector.rightSteps)）")

        let bpms = stepDetector.recentEvents.compactMap { $0.cadenceBPM }
        let avgCadence: Double? = bpms.isEmpty ? nil : bpms.reduce(0, +) / Double(bpms.count)
        if !bpms.isEmpty, let avg = avgCadence {
            out.append(String(format: "平均步頻：%.0f bpm", avg))
        }

        if !suppressSagittalFormAdvice {
            switch activityMode {
            case .walking:
                out.append(contentsOf: WalkingFormAdvisor.sessionSummary(metrics: gaitMetrics))
            case .running:
                out.append(contentsOf: RunningFormAdvisor.sessionSummary(metrics: gaitMetrics))
            }

            if let profile = bodyProfile, let avg = avgCadence {
                let personalized = GaitPersonalizationSummary.lines(
                    profile: profile,
                    avgCadence: avg,
                    totalSteps: total,
                    metrics: gaitMetrics
                )
                out.append(contentsOf: personalized)
            }
        } else {
            out.append("背面視角：已略過前後向（前傾／跨步／著地）建議，僅保留左右對稱與背面特徵。")
        }

        if total > 0 {
            let l = Double(stepDetector.leftSteps) / Double(total) * 100
            let r = Double(stepDetector.rightSteps) / Double(total) * 100
            out.append(String(format: "左右平衡：左 %.0f%% / 右 %.0f%%", l, r))
            if abs(l - r) > 25 {
                out.append("步數左右差距偏大，建議放慢節奏並注意對稱性。")
            }
        }

        let frames = Double(totalEvaluatedFrames)
        let issues = issueCounts.sorted { $0.value > $1.value }
        var added = 0
        for (code, count) in issues {
            let pct = Double(count) / frames * 100
            if pct < 8 { continue }
            out.append("\(code.summaryDescription)：影片約 \(Int(pct))% 時段需注意。")
            added += 1
            if added >= 3 { break }
        }

        if added == 0 {
            out.append("整體姿勢穩定，沒有明顯需要改善的部位。")
        } else {
            out.append("可針對上述部位做伸展與意識訓練，再上傳一次比對改善程度。")
        }

        return out
    }
}
