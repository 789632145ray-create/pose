//
//  GaitActivityMode.swift
//  pose
//
//  走路／跑步分開評估：著地方式與軀幹前傾標準不同。
//

import Foundation

enum GaitActivityMode: String, CaseIterable, Identifiable {
    case walking
    case running

    var id: String { rawValue }

    var title: String {
        switch self {
        case .walking: return "走路"
        case .running: return "跑步"
        }
    }

    var coreTitle: String {
        switch self {
        case .walking: return "走路正確姿勢核心"
        case .running: return "跑步正確姿勢核心"
        }
    }

    var systemImage: String {
        switch self {
        case .walking: return "figure.walk"
        case .running: return "figure.run"
        }
    }

    var corePrincipleLines: [String] {
        switch self {
        case .walking: return WalkingFormAdvisor.corePrincipleLines
        case .running: return RunningFormAdvisor.corePrincipleLines
        }
    }

    static func resolved(fromStored raw: String) -> GaitActivityMode {
        GaitActivityMode(rawValue: raw) ?? .walking
    }
}
