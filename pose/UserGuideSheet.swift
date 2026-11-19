//
//  UserGuideSheet.swift
//  pose
//
//  App 內使用說明：逐步「下一步」導覽（註冊後首次、或選單隨時開啟）。
//

import SwiftUI

private struct UserGuideStep: Identifiable, Equatable {
    let id: Int
    let systemImage: String
    let title: String
    let detail: String
}

private enum UserGuideContent {
    static let steps: [UserGuideStep] = [
        UserGuideStep(
            id: 0,
            systemImage: "hand.wave.fill",
            title: "歡迎使用 Pose 偵測",
            detail: "接下來用幾個步驟說明如何偵測走路／跑步姿勢。新會員註冊後會自動顯示一次；之後也可從右上角選單再開啟。"
        ),
        UserGuideStep(
            id: 1,
            systemImage: "camera.fill",
            title: "先允許相機並選好設定",
            detail: "首次請允許使用相機。底部可選偵測引擎（QuickPose／MediaPipe／自訓）、走路／跑步，以及拍攝視角（正面／側面／背面）。要用 MediaPipe 訓練請選 MediaPipe 引擎，勿再經 QuickPose。"
        ),
        UserGuideStep(
            id: 2,
            systemImage: "figure.run",
            title: "相機：準備 \(LiveTimedSessionConfig.prepareSeconds) 秒＋採樣 \(LiveTimedSessionConfig.analyzeSeconds) 秒",
            detail: "按「開始」後先進入準備倒數（不收集數據），方便走上跑步機並加速；倒數結束自動開始正式採樣。結束後會立刻停止收集並顯示摘要，之後減速下機不會寫入結果。"
        ),
        UserGuideStep(
            id: 3,
            systemImage: "speaker.wave.2.fill",
            title: "聽提示音，不必盯螢幕",
            detail: "準備開始、最後 3 秒滴答、開始分析、分析完成都會有語音或提示音。準備或分析中可按「取消」清空暫存並回待機。"
        ),
        UserGuideStep(
            id: 4,
            systemImage: "camera.rotate",
            title: "前後鏡頭與拍攝視角",
            detail: "右上角或底部可切換前／後鏡頭；跑步機側面／背面建議用後鏡頭。背面模式只給左右對稱相關建議，不會亂給前傾／跨步等前後向建議。"
        ),
        UserGuideStep(
            id: 5,
            systemImage: "film.fill",
            title: "也可以上傳影片分析",
            detail: "點「上傳影片偵測」或「檔案」匯入。讀取時會顯示請稍候。確認後短暫倒數開始偵測，可暫停或按品質辨識與摘要查看結果。"
        ),
        UserGuideStep(
            id: 6,
            systemImage: "checklist",
            title: "看摘要、歷史與資料庫",
            detail: "分析結束會顯示建議（打勾＝達標、警告＝需注意）。選單可看歷史紀錄；底部可開資料庫標好／壞或匯出。清空前都會再確認一次。"
        ),
        UserGuideStep(
            id: 7,
            systemImage: "lightbulb.fill",
            title: "跑步機建議流程",
            detail: "架好後鏡頭 → 選背面或側面 → 按開始 → 準備期上機加速 → 聽到「開始分析」後維持穩定跑姿 → 聽到完成再減速下機。準備／分析中請勿切換引擎、視角或鏡頭。"
        ),
    ]
}

struct UserGuideView: View {
    var onFinished: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var pageIndex = 0

    private var steps: [UserGuideStep] { UserGuideContent.steps }
    private var isFirst: Bool { pageIndex <= 0 }
    private var isLast: Bool { pageIndex >= steps.count - 1 }

    var body: some View {
        VStack(spacing: 0) {
            progressHeader
            TabView(selection: $pageIndex) {
                ForEach(steps) { item in
                    stepPage(item)
                        .tag(item.id)
                        .padding(.horizontal, 24)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(.easeInOut(duration: 0.25), value: pageIndex)

            bottomBar
        }
        .background(Color(uiColor: .systemBackground))
        .navigationTitle("使用說明")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var progressHeader: some View {
        VStack(spacing: 10) {
            Text("步驟 \(pageIndex + 1)／\(steps.count)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ProgressView(value: Double(pageIndex + 1), total: Double(steps.count))
                .tint(.accentColor)
                .padding(.horizontal, 24)
        }
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    private func stepPage(_ item: UserGuideStep) -> some View {
        VStack(spacing: 20) {
            Spacer(minLength: 12)
            Image(systemName: item.systemImage)
                .font(.system(size: 52, weight: .semibold))
                .foregroundStyle(Color.accentColor)
                .frame(width: 96, height: 96)
                .background(Color.accentColor.opacity(0.12), in: Circle())
            Text(item.title)
                .font(.title2.weight(.bold))
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Text(item.detail)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var bottomBar: some View {
        HStack(spacing: 12) {
            if !isFirst {
                Button("上一步") {
                    withAnimation { pageIndex = max(0, pageIndex - 1) }
                }
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)
            }

            Button {
                if isLast {
                    if let onFinished {
                        onFinished()
                    } else {
                        dismiss()
                    }
                } else {
                    withAnimation { pageIndex = min(steps.count - 1, pageIndex + 1) }
                }
            } label: {
                Text(isLast ? "開始使用" : "下一步")
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(.bar)
    }
}

/// 獨立 Sheet（註冊後首次、或外部 present）。
struct UserGuideSheet: View {
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            UserGuideView(onFinished: onClose)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("略過", action: onClose)
                    }
                }
        }
        .interactiveDismissDisabled(false)
    }
}
