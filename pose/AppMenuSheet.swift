//
//  AppMenuSheet.swift
//  pose
//
//  主選單 Sheet：歷史紀錄、會員資料、偵測引擎切換、開發者資訊、登出。
//

import SwiftUI

struct PoseDeveloperInfo {
    var engineTitle: String
    var activityTitle: String
    var realmFileName: String
    var dbNodeCount: Int
    var frameCallbackCount: Int
    var fpsText: String
    var quickPoseVersion: String
    var mediaPipeModelWeight: String?
    var localVideoNote: String
}

struct AppMenuSheet: View {
    @EnvironmentObject private var auth: AuthManager
    @EnvironmentObject private var modeStore: DetectionModeStore
    @Environment(\.dismiss) private var dismiss

    let developerInfo: PoseDeveloperInfo?
    let onHistory: () -> Void
    let onMemberProfile: () -> Void
    let onSignOut: () -> Void

    @State private var showDeveloperInfo = false

    var body: some View {
        NavigationStack {
            List {
                accountSection
                actionsSection
                engineSection
                developerSection
                signOutSection
            }
            .navigationTitle("選單")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
            .sheet(isPresented: $showDeveloperInfo) {
                developerInfoSheet
            }
        }
    }

    @ViewBuilder
    private var accountSection: some View {
        if let name = auth.username {
            Section {
                LabeledContent("已登入", value: name)
                if let profile = auth.userProfile {
                    LabeledContent("姓名", value: profile.displayName)
                }
            }
        }
    }

    private var actionsSection: some View {
        Section {
            Button(action: onHistory) {
                Label("歷史紀錄", systemImage: "clock.arrow.circlepath")
            }
            Button(action: onMemberProfile) {
                Label("管理會員資料", systemImage: "person.crop.circle")
            }
        }
    }

    private var engineSection: some View {
        Section {
            ForEach(PoseAssessmentEngine.allCases) { engine in
                AppMenuEngineRow(engine: engine, isSelected: modeStore.engine == engine) {
                    PoseChrome.impact(.light)
                    modeStore.engine = engine
                }
            }
        } header: {
            Text("偵測引擎（3 種）")
        } footer: {
            Text("QuickPose 為官方 overlay；MediaPipe 為 BlazePose Full 骨架；自訓模型會連雲端辨識。")
        }
    }

    private var developerSection: some View {
        Section {
            Button {
                showDeveloperInfo = true
            } label: {
                Label("開發者資訊 / Debug", systemImage: "wrench.and.screwdriver")
            }
        } footer: {
            Text("引擎回呼、Realm 檔名、節點計數等技術資訊改集中於此，不佔用運動畫面。")
        }
    }

    private var signOutSection: some View {
        Section {
            Button(role: .destructive, action: onSignOut) {
                Label("登出", systemImage: "rectangle.portrait.and.arrow.right")
            }
        }
    }

    @ViewBuilder
    private var developerInfoSheet: some View {
        NavigationStack {
            List {
                if let info = developerInfo {
                    Section("執行狀態") {
                        LabeledContent("偵測引擎", value: info.engineTitle)
                        LabeledContent("活動模式", value: info.activityTitle)
                        LabeledContent("FPS", value: info.fpsText)
                        LabeledContent("引擎回呼次數", value: "\(info.frameCallbackCount)")
                        LabeledContent("資料庫節點", value: "\(info.dbNodeCount)")
                    }
                    Section("本機儲存") {
                        LabeledContent("寫入 Realm", value: info.realmFileName)
                        Text(info.localVideoNote)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Section("SDK") {
                        LabeledContent("QuickPose", value: "v\(info.quickPoseVersion)")
                        if let weight = info.mediaPipeModelWeight {
                            LabeledContent("MediaPipe", value: weight)
                        }
                    }
                } else {
                    Text("尚無可顯示的偵測狀態。")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("開發者資訊")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { showDeveloperInfo = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

private struct AppMenuEngineRow: View {
    let engine: PoseAssessmentEngine
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(engine.title)
                    Text(engine.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(PoseChrome.accent)
                }
            }
        }
        .foregroundStyle(.primary)
    }
}
