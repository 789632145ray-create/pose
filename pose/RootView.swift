//
//  RootView.swift
//  pose
//
//  App 根畫面：未登入顯示 LoginView；登入後若尚未填個人資料則顯示 UserProfileSetupView；
//  完成後顯示姿勢偵測主畫面。歷史紀錄與會員管理入口在 PoseDetectionView 右上角選單。
//

import SwiftUI

struct RootView: View {
    @StateObject private var auth = AuthManager()
    @State private var didBootstrap = false

    var body: some View {
        Group {
            if !didBootstrap {
                loadingScreen
            } else if auth.isAuthenticated {
                if auth.needsProfileSetup {
                    UserProfileSetupView(auth: auth)
                } else {
                    PoseDetectionShellView()
                        .environmentObject(auth)
                        .safeAreaInset(edge: .top, spacing: 0) {
                            if auth.isOfflineSession {
                                Text("離線模式：可本機偵測，登入／上傳／辨識需有網路")
                                    .font(.caption2)
                                    .foregroundStyle(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 6)
                                    .background(Color.orange.opacity(0.9))
                            }
                        }
                }
            } else {
                LoginView(auth: auth)
            }
        }
        .task {
            await auth.bootstrap()
            didBootstrap = true
        }
    }

    private var loadingScreen: some View {
        ZStack {
            Color(red: 0.10, green: 0.10, blue: 0.13).ignoresSafeArea()
            VStack(spacing: 16) {
                ProgressView().tint(.white)
                Text("載入中…").foregroundStyle(.white.opacity(0.8))
            }
        }
    }
}
