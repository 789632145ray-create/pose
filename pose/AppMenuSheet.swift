//
//  AppMenuSheet.swift
//  pose
//
//  主選單 Sheet：歷史紀錄、會員資料、偵測引擎切換、登出。
//

import SwiftUI

struct AppMenuSheet: View {
    @EnvironmentObject private var auth: AuthManager
    @EnvironmentObject private var modeStore: DetectionModeStore
    @Environment(\.dismiss) private var dismiss

    let onHistory: () -> Void
    let onMemberProfile: () -> Void
    let onSignOut: () -> Void

    var body: some View {
        NavigationStack {
            List {
                accountSection
                actionsSection
                engineSection
                signOutSection
            }
            .navigationTitle("選單")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
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
            NavigationLink {
                UserGuideView()
            } label: {
                Label("使用說明", systemImage: "book.pages")
            }
            Button(action: onHistory) {
                Label("歷史紀錄", systemImage: "clock.arrow.circlepath")
            }
            Button(action: onMemberProfile) {
                Label("管理會員資料", systemImage: "person.crop.circle")
            }
        }
    }

    @ViewBuilder
    private var engineSection: some View {
        Section {
            ForEach(PoseAssessmentEngine.allCases) { engine in
                AppMenuEngineRow(engine: engine, isSelected: modeStore.engine == engine) {
                    modeStore.engine = engine
                }
            }
        } header: {
            Text("偵測引擎（3 種）")
        } footer: {
            Text("QuickPose 為官方 overlay；MediaPipe 以原生 Landmarker 採樣；自訓模型可再選 QuickPose／MediaPipe 兩套已訓練品質模型做辨識。")
        }

        if modeStore.engine == .trainedModel {
            Section {
                ForEach(TrainedQualityModel.allCases) { model in
                    Button {
                        modeStore.trainedQualityModel = model
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.title)
                                Text(model.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if modeStore.trainedQualityModel == model {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(Color.accentColor)
                            }
                        }
                    }
                    .foregroundStyle(.primary)
                }
            } header: {
                Text("自訓辨識模型")
            } footer: {
                Text("請選與訓練資料相同來源的模型：QuickPose 採樣對 QuickPose 模型；MediaPipe 原生採樣對 MediaPipe 模型。")
            }
        }
    }

    private var signOutSection: some View {
        Section {
            Button(role: .destructive, action: onSignOut) {
                Label("登出", systemImage: "rectangle.portrait.and.arrow.right")
            }
        }
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
                        .foregroundStyle(Color.accentColor)
                }
            }
        }
        .foregroundStyle(.primary)
    }
}
