//
//  ServerConfig.swift
//  pose
//
//  後端網址：Release 實機用 Info.plist 的雲端 HTTPS；Debug 模擬器用本機。
//

import Foundation

enum ServerConfig {
    private static let infoPlistKey = "PoseServerBaseURL"
    private static let localSimulatorURL = "http://127.0.0.1:8000"

    /// Release 實機：讀 Info.plist 的 PoseServerBaseURL（部署後填入 Railway 等 HTTPS 網址）。
    static var baseURL: String {
        #if DEBUG
        #if targetEnvironment(simulator)
        return localSimulatorURL
        #else
        return productionURL
        #endif
        #else
        return productionURL
        #endif
    }

    /// 是否連到本機開發伺服器（用於 UI 提示）。
    static var isLocalDevelopment: Bool {
        #if DEBUG
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
        #else
        return false
        #endif
    }

    private static var productionURL: String {
        if let url = Bundle.main.object(forInfoDictionaryKey: infoPlistKey) as? String {
            let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, !trimmed.contains("YOUR-CLOUD-URL") {
                return trimmed
            }
        }
        // 尚未設定雲端網址時的占位；部署後請改 Info.plist。
        return "https://YOUR-CLOUD-URL.up.railway.app"
    }
}
