//
//  LoginView.swift
//  pose
//
//  登入 / 註冊畫面。
//

import SwiftUI

struct LoginView: View {
    @ObservedObject var auth: AuthManager

    private enum Mode: String, CaseIterable {
        case login = "登入"
        case register = "註冊"
    }

    @State private var mode: Mode = .login
    @State private var username = ""
    @State private var password = ""
    @FocusState private var focusedField: Field?

    private enum Field { case username, password }

    private let background = Color(red: 0.10, green: 0.10, blue: 0.13)

    var body: some View {
        ZStack {
            background.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 22) {
                    header

                    Picker("模式", selection: $mode) {
                        ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 4)

                    credentialFields

                    if let message = auth.errorMessage {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    primaryButton
                }
                .padding(24)
                .frame(maxWidth: 460)
                .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: 子畫面

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "figure.walk.motion")
                .font(.system(size: 52))
                .foregroundStyle(.orange)
            Text("Pose 偵測")
                .font(.largeTitle.weight(.bold))
                .foregroundStyle(.white)
            Text(mode == .login ? "請登入以開始使用" : "建立新帳號")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.top, 24)
    }

    private var credentialFields: some View {
        VStack(spacing: 14) {
            HStack {
                Image(systemName: "person.fill").foregroundStyle(.white.opacity(0.6))
                TextField("帳號（至少 3 字元）", text: $username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .submitLabel(.next)
                    .focused($focusedField, equals: .username)
                    .onSubmit { focusedField = .password }
                    .foregroundStyle(.white)
            }
            .padding(14)
            .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))

            HStack {
                Image(systemName: "lock.fill").foregroundStyle(.white.opacity(0.6))
                SecureField("密碼（至少 6 字元）", text: $password)
                    .submitLabel(.go)
                    .focused($focusedField, equals: .password)
                    .onSubmit { submit() }
                    .foregroundStyle(.white)
            }
            .padding(14)
            .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var primaryButton: some View {
        Button {
            submit()
        } label: {
            HStack {
                if auth.isWorking { ProgressView().tint(.white) }
                Text(mode == .login ? "登入" : "註冊並登入")
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.accentColor)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .disabled(auth.isWorking)
    }

    private func submit() {
        focusedField = nil
        Task {
            switch mode {
            case .login:
                await auth.login(username: username, password: password)
            case .register:
                await auth.register(username: username, password: password)
            }
        }
    }
}
