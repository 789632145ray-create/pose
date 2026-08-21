//
//  MemberProfileSheet.swift
//  pose
//
//  管理會員資料：顯示帳號與個人基本資料、修改密碼、重新驗證登入狀態與登出。
//

import SwiftUI

struct MemberProfileSheet: View {
    @ObservedObject var auth: AuthManager
    let onClose: () -> Void

    @State private var statusMessage: String?
    @State private var isError = false
    @State private var showChangePassword = false
    @State private var showEditProfile = false
    @State private var currentPassword = ""
    @State private var newPassword = ""
    @State private var confirmPassword = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("帳號資訊") {
                    LabeledContent("使用者名稱", value: auth.username ?? "—")
                    Button {
                        refreshProfile()
                    } label: {
                        HStack {
                            if auth.isWorking && !showChangePassword && !showEditProfile { ProgressView() }
                            Label("重新驗證登入狀態", systemImage: "arrow.clockwise")
                        }
                    }
                    .disabled(auth.isWorking)
                }

                if let profile = auth.userProfile {
                    Section("個人資料") {
                        LabeledContent("姓名", value: profile.displayName)
                        LabeledContent("性別", value: profile.gender.label)
                        LabeledContent("年齡", value: "\(profile.age) 歲")
                        LabeledContent("身高", value: String(format: "%.1f cm", profile.heightCm))
                        LabeledContent("體重", value: String(format: "%.1f kg", profile.weightKg))
                        if let bmi = profile.bmi {
                            LabeledContent("BMI", value: String(format: "%.1f", bmi))
                        }
                        Button {
                            showEditProfile = true
                        } label: {
                            Label("編輯個人資料", systemImage: "pencil")
                        }
                    }
                }

                Section {
                    Button {
                        withAnimation { showChangePassword.toggle() }
                    } label: {
                        Label(showChangePassword ? "收起修改密碼" : "修改密碼", systemImage: "key.fill")
                    }

                    if showChangePassword {
                        SecureField("目前密碼", text: $currentPassword)
                        SecureField("新密碼（至少 6 字元）", text: $newPassword)
                        SecureField("確認新密碼", text: $confirmPassword)
                        Button {
                            submitPasswordChange()
                        } label: {
                            HStack {
                                if auth.isWorking { ProgressView() }
                                Text("儲存新密碼")
                            }
                        }
                        .disabled(auth.isWorking || currentPassword.isEmpty || newPassword.isEmpty || confirmPassword.isEmpty)
                    }
                } header: {
                    Text("密碼")
                } footer: {
                    if showChangePassword {
                        Text("修改成功後請用新密碼重新登入（若在其他裝置有登入）。")
                    }
                }

                if let statusMessage {
                    Section {
                        Text(statusMessage)
                            .font(.footnote)
                            .foregroundStyle(isError ? .red : .green)
                    }
                }

                Section {
                    Button(role: .destructive) {
                        auth.signOut()
                        onClose()
                    } label: {
                        Label("登出", systemImage: "rectangle.portrait.and.arrow.right")
                    }
                }
            }
            .navigationTitle("管理會員資料")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成", action: onClose)
                }
            }
            .sheet(isPresented: $showEditProfile) {
                UserProfileSetupView(
                    auth: auth,
                    title: "編輯個人資料",
                    subtitle: "更新姓名、性別、年齡、身高與體重",
                    buttonTitle: "儲存"
                ) {
                    showEditProfile = false
                    statusMessage = "個人資料已更新"
                    isError = false
                }
            }
        }
    }

    private func refreshProfile() {
        statusMessage = nil
        Task {
            await auth.refreshProfile()
            if let err = auth.errorMessage {
                isError = true
                statusMessage = err
            } else {
                isError = false
                statusMessage = "登入狀態正常"
            }
        }
    }

    private func submitPasswordChange() {
        statusMessage = nil
        guard newPassword == confirmPassword else {
            isError = true
            statusMessage = "兩次輸入的新密碼不一致"
            return
        }
        Task {
            let ok = await auth.changePassword(current: currentPassword, new: newPassword)
            if ok {
                isError = false
                statusMessage = "密碼已更新"
                currentPassword = ""
                newPassword = ""
                confirmPassword = ""
                showChangePassword = false
            } else {
                isError = true
                statusMessage = auth.errorMessage ?? "修改密碼失敗"
            }
        }
    }
}
