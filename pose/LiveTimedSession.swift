//
//  LiveTimedSession.swift
//  pose
//
//  實時相機：準備倒數 → 定時採樣 → 完成鎖定。
//  秒數為可調常數，避免上／下跑步機過程污染姿勢數據。
//

import AVFoundation
import AudioToolbox
import Foundation

/// 實時定時採樣狀態機。
enum LiveSessionState: Equatable {
    case idle
    case preparing(remaining: Int)
    case analyzing(remaining: Int)
    case completed

    var isBusy: Bool {
        switch self {
        case .preparing, .analyzing: return true
        case .idle, .completed: return false
        }
    }

    var remainingSeconds: Int? {
        switch self {
        case .preparing(let s), .analyzing(let s): return s
        case .idle, .completed: return nil
        }
    }
}

/// 可調整的準備／採樣時長（秒）。
enum LiveTimedSessionConfig {
    /// 準備倒數：走上跑步機並加速至穩定跑速。
    static let prepareSeconds = 15
    /// 正式採樣分析時長。
    static let analyzeSeconds = 15
}

/// 跑步機場景聽覺提示（語音 + 系統短音）。
@MainActor
final class SessionAudioCues {
    private let synthesizer = AVSpeechSynthesizer()

    func stopAll() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    func announcePreparing(seconds: Int = LiveTimedSessionConfig.prepareSeconds) {
        speak("請上跑步機準備，\(seconds) 秒後開始")
    }

    func announceAnalyzing() {
        playHighBeep()
        speak("開始分析")
    }

    func announceCompleted() {
        playCompleteBeep()
        speak("分析完成，可以減速了")
    }

    /// 準備倒數最後 3、2、1 的短提示音。
    func playTickBeep() {
        // 1104：短促 Tock，遠距離可辨。
        AudioServicesPlaySystemSound(1104)
    }

    private func playHighBeep() {
        // 1113：開始錄影提示，音高較明顯。
        AudioServicesPlaySystemSound(1113)
    }

    private func playCompleteBeep() {
        // 1114：結束錄影提示。
        AudioServicesPlaySystemSound(1114)
    }

    private func speak(_ text: String) {
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-TW")
            ?? AVSpeechSynthesisVoice(language: "zh-HK")
            ?? AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.95
        utterance.pitchMultiplier = 1.05
        utterance.preUtteranceDelay = 0
        utterance.postUtteranceDelay = 0
        synthesizer.speak(utterance)
    }
}
