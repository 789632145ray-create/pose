//
//  UserGuideSheet.swift
//  pose
//
//  App 內使用說明：相機定時採樣、影片分析、視角與資料庫操作。
//

import SwiftUI

struct UserGuideView: View {
    var body: some View {
        List {
            sectionQuickStart
            sectionCameraTimed
            sectionVideo
            sectionViewModes
            sectionEngines
            sectionResults
            sectionTips
        }
        .navigationTitle("使用說明")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var sectionQuickStart: some View {
        Section {
            guideRow(
                "1. 登入帳號",
                "使用已註冊帳號登入。自訓模型品質辨識需要有效登入。"
            )
            guideRow(
                "2. 允許相機",
                "首次使用請點「允許使用相機」。若曾拒絕，請到系統設定開啟權限。"
            )
            guideRow(
                "3. 選好設定再開始",
                "底部可選：偵測引擎、活動（走路／跑步）、拍攝視角（正面／側面／背面）。自訓模型還可選骨架模型。"
            )
        } header: {
            Text("快速開始")
        }
    }

    private var sectionCameraTimed: some View {
        Section {
            guideRow(
                "開始＝準備 \(LiveTimedSessionConfig.prepareSeconds) 秒＋採樣 \(LiveTimedSessionConfig.analyzeSeconds) 秒",
                "相機模式按「開始」後會先進入準備倒數（不上傳姿勢數據），方便走上跑步機並加速；倒數結束自動進入正式採樣。"
            )
            guideRow(
                "聽覺提示",
                "準備開始、最後 3 秒滴答、開始分析、分析完成都會有語音或提示音，方便你看著前方跑步機而不必盯螢幕。"
            )
            guideRow(
                "採樣結束自動出報告",
                "採樣時間一到會立刻停止收集數據，並顯示分析摘要。之後減速、下機的畫面不會再寫入結果。"
            )
            guideRow(
                "取消／重設",
                "準備或分析中可按「取消」，會清空本次暫存並回到待機，不產生報告。"
            )
            guideRow(
                "前後鏡頭",
                "右上角旋轉圖示或底部「目前：前／後鏡頭」可切換。跑步機側面／背面建議用後鏡頭。切換請在待機時進行。"
            )
        } header: {
            Text("相機即時偵測（跑步機）")
        } footer: {
            Text("準備階段只預覽骨架，不會寫入資料庫；只有「分析中」才會收集節點與統計。")
        }
    }

    private var sectionVideo: some View {
        Section {
            guideRow(
                "上傳影片",
                "點「上傳影片偵測」從相簿選，或「檔案」從檔案 App 匯入。讀取較大影片時會顯示「請稍候」。"
            )
            guideRow(
                "確認後開始",
                "確認影片後會短暫倒數再開始偵測。可按「暫停」產生品質辨識與摘要；也可按「品質辨識與摘要」手動開啟。"
            )
            guideRow(
                "切回相機",
                "影片模式可隨時切回相機繼續使用定時採樣流程。"
            )
        } header: {
            Text("影片分析")
        } footer: {
            Text("影片在手機本機做骨架分析；若使用自訓／MediaPipe 品質模型，才會連伺服器做好／壞辨識。")
        }
    }

    private var sectionViewModes: some View {
        Section {
            guideRow(
                "正面",
                "一般正面拍攝，使用完整走路／跑步姿勢建議。"
            )
            guideRow(
                "側面",
                "適合觀察前傾、膝角、跨步過度等前後向特徵。請讓全身側面入鏡。"
            )
            guideRow(
                "背面",
                "適合觀察骨盆左右傾斜、軀幹側傾、肩膀高低差。背面模式不會給前傾／跨步／著地等前後向建議（背面無法可靠量測）。"
            )
        } header: {
            Text("拍攝視角")
        } footer: {
            Text("請依實際鏡頭位置選擇視角，選錯會讓建議不準。")
        }
    }

    private var sectionEngines: some View {
        Section {
            guideRow(
                "QuickPose",
                "官方 overlay 骨架，適合快速預覽與規則建議。"
            )
            guideRow(
                "MediaPipe",
                "BlazePose Full 骨架，可做品質辨識；節點寫入 mediapipe.realm。"
            )
            guideRow(
                "自訓模型",
                "連雲端模型辨識好／壞。可再選 QuickPose 或 MediaPipe 骨架／品質路徑。需登入且伺服器可用。"
            )
            guideRow(
                "活動：走路／跑步",
                "會套用不同正確姿勢核心與建議邏輯，請依實際運動選擇。"
            )
        } header: {
            Text("偵測引擎與活動")
        }
    }

    private var sectionResults: some View {
        Section {
            guideRow(
                "分析摘要",
                "結束後會顯示建議列表與（若有）模型品質。左側圖示：打勾＝達標、警告＝需注意、資訊＝一般說明。"
            )
            guideRow(
                "歷史紀錄",
                "選單 → 歷史紀錄可查看過去摘要。清空前會再確認一次。"
            )
            guideRow(
                "資料庫",
                "底部可開啟節點資料庫，查看本次／歷史 session，並可標好／壞或匯出 JSON（訓練用）。清空前會再確認。"
            )
            guideRow(
                "左上角狀態",
                "顯示目前引擎、活動、視角、步數與品質結果。鏡頭切換請用右上角或底部按鈕。"
            )
        } header: {
            Text("結果與資料")
        }
    }

    private var sectionTips: some View {
        Section {
            guideRow(
                "光線與入鏡",
                "全身入鏡、光線充足、背景單純，關節信心較穩。"
            )
            guideRow(
                "跑步機建議流程",
                "架好後鏡頭 → 選背面或側面 → 按開始 → 在準備倒數內走上機並加速 → 聽到開始分析後維持穩定跑姿 → 聽到完成後再減速下機。"
            )
            guideRow(
                "不要中途亂切設定",
                "準備／分析中請勿切換引擎、視角或鏡頭，以免資料不一致。需要重來請按取消。"
            )
        } header: {
            Text("拍攝小技巧")
        }
    }

    private func guideRow(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text(detail)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 4)
    }
}

/// 獨立 Sheet（若從其他入口直接 present 時使用）。
struct UserGuideSheet: View {
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            UserGuideView()
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("完成", action: onClose)
                    }
                }
        }
    }
}
