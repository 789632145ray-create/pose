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
            Button(action: onHistory) {
                Label("歷史紀錄", systemImage: "clock.arrow.circlepath")
            }
            Button(action: onMemberProfile) {
                Label("管理會員資料", systemImage: "person.crop.circle")
            }
        }
    }

    private var engineSection: some View {
        Section("偵測引擎") {
            ForEach(PoseDetectionEngine.allCases) { engine in
                AppMenuEngineRow(engine: engine, isSelected: modeStore.engine == engine) {
                    modeStore.engine = engine
                }
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
    let engine: PoseDetectionEngine
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
