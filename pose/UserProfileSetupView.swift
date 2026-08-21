//
//  UserProfileSetupView.swift
//  pose
//
//  登入／註冊後填寫個人基本資料：姓名、性別、年齡、身高、體重。
//

import SwiftUI

struct UserProfileSetupView: View {
    @ObservedObject var auth: AuthManager
    var title: String = "建立個人資料"
    var subtitle: String = "請填寫基本資料，以便提供更合適的姿勢分析"
    var buttonTitle: String = "完成並開始使用"
    var onComplete: (() -> Void)? = nil

    @State private var displayName = ""
    @State private var gender: UserProfile.Gender = .male
    @State private var ageText = ""
    @State private var heightText = ""
    @State private var weightText = ""
    @State private var keyboardHeight: CGFloat = 0
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case name, age, height, weight, submit
    }

    private static let focusOrder: [Field] = [.name, .age, .height, .weight]

    private let background = Color(red: 0.10, green: 0.10, blue: 0.13)

    var body: some View {
        NavigationStack {
            ZStack {
                background.ignoresSafeArea()

                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 22) {
                            header
                            formFields

                            if let message = auth.errorMessage {
                                Text(message)
                                    .font(.footnote)
                                    .foregroundStyle(.red)
                                    .multilineTextAlignment(.center)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }

                            submitButton
                                .id(Field.submit)
                        }
                        .padding(24)
                        .padding(.bottom, max(keyboardHeight, 24))
                        .frame(maxWidth: 460)
                        .frame(maxWidth: .infinity)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: focusedField) { _, field in
                        guard let field else { return }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                            withAnimation(.easeOut(duration: 0.25)) {
                                proxy.scrollTo(field, anchor: .center)
                            }
                        }
                    }
                }
            }
            .navigationBarHidden(true)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Button("上一項") {
                        moveFocus(forward: false)
                    }
                    .disabled(focusedField == Self.focusOrder.first)

                    Button("下一項") {
                        moveFocus(forward: true)
                    }
                    .disabled(focusedField == Self.focusOrder.last)

                    Spacer()

                    Button("完成") {
                        dismissKeyboard()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)) { notification in
            guard let frame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect else { return }
            keyboardHeight = frame.height - 20
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillHideNotification)) { _ in
            keyboardHeight = 0
        }
        .onAppear {
            prefillFromExistingProfile()
        }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.text.rectangle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text(title)
                .font(.largeTitle.weight(.bold))
                .foregroundStyle(.white)
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .padding(.top, 16)
    }

    private var formFields: some View {
        VStack(spacing: 14) {
            fieldRow(icon: "person.fill", title: "姓名") {
                TextField("請輸入姓名", text: $displayName)
                    .textInputAutocapitalization(.words)
                    .autocorrectionDisabled()
                    .focused($focusedField, equals: .name)
                    .submitLabel(.next)
                    .onSubmit { focusedField = .age }
                    .foregroundStyle(.white)
            }
            .id(Field.name)

            fieldRow(icon: "figure.stand", title: "性別") {
                Picker("性別", selection: $gender) {
                    ForEach(UserProfile.Gender.allCases) { option in
                        Text(option.label).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            }

            fieldRow(icon: "calendar", title: "年齡") {
                TextField("例如 25", text: $ageText)
                    .keyboardType(.numberPad)
                    .focused($focusedField, equals: .age)
                    .foregroundStyle(.white)
            }
            .id(Field.age)

            fieldRow(icon: "ruler", title: "身高（cm）") {
                TextField("例如 170", text: $heightText)
                    .keyboardType(.decimalPad)
                    .focused($focusedField, equals: .height)
                    .foregroundStyle(.white)
            }
            .id(Field.height)

            fieldRow(icon: "scalemass", title: "體重（kg）") {
                TextField("例如 65", text: $weightText)
                    .keyboardType(.decimalPad)
                    .focused($focusedField, equals: .weight)
                    .foregroundStyle(.white)
            }
            .id(Field.weight)
        }
    }

    private func fieldRow<Content: View>(icon: String, title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white.opacity(0.85))
            content()
        }
        .padding(14)
        .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }

    private var submitButton: some View {
        Button {
            submit()
        } label: {
            HStack {
                if auth.isWorking { ProgressView().tint(.white) }
                Text(buttonTitle)
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

    private func dismissKeyboard() {
        focusedField = nil
    }

    private func moveFocus(forward: Bool) {
        guard let current = focusedField else {
            focusedField = forward ? Self.focusOrder.first : Self.focusOrder.last
            return
        }
        guard let index = Self.focusOrder.firstIndex(of: current) else { return }
        if forward {
            if index < Self.focusOrder.count - 1 {
                focusedField = Self.focusOrder[index + 1]
            } else {
                dismissKeyboard()
            }
        } else if index > 0 {
            focusedField = Self.focusOrder[index - 1]
        }
    }

    private func prefillFromExistingProfile() {
        guard let profile = auth.userProfile else { return }
        displayName = profile.displayName
        gender = profile.gender
        ageText = String(profile.age)
        heightText = profile.heightCm.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.0f", profile.heightCm)
            : String(format: "%.1f", profile.heightCm)
        weightText = profile.weightKg.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.0f", profile.weightKg)
            : String(format: "%.1f", profile.weightKg)
    }

    private func submit() {
        dismissKeyboard()
        auth.errorMessage = nil

        let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            auth.errorMessage = "請輸入姓名"
            return
        }
        guard let age = Int(ageText.trimmingCharacters(in: .whitespaces)), (1...120).contains(age) else {
            auth.errorMessage = "年齡請輸入 1～120 的整數"
            return
        }
        guard let height = Double(heightText.trimmingCharacters(in: .whitespaces)), (50...250).contains(height) else {
            auth.errorMessage = "身高請輸入 50～250 cm"
            return
        }
        guard let weight = Double(weightText.trimmingCharacters(in: .whitespaces)), (20...300).contains(weight) else {
            auth.errorMessage = "體重請輸入 20～300 kg"
            return
        }

        Task {
            await auth.saveProfile(
                displayName: name,
                gender: gender,
                age: age,
                heightCm: height,
                weightKg: weight
            )
            if auth.errorMessage == nil {
                onComplete?()
            }
        }
    }
}
