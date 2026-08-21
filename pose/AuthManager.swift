//
//  AuthManager.swift
//  pose
//
//  帳號系統：FastAPI + MongoDB Atlas `users` collection。
//  權杖存於 Keychain；伺服器網址由 ServerConfig 自動設定。
//

import Combine
import Foundation
import SwiftUI

// MARK: - 使用者個人資料

struct UserProfile: Codable, Equatable {
    enum Gender: String, Codable, CaseIterable, Identifiable {
        case male
        case female
        case other

        var id: String { rawValue }

        var label: String {
            switch self {
            case .male: return "男"
            case .female: return "女"
            case .other: return "其他"
            }
        }
    }

    var displayName: String
    var gender: Gender
    var age: Int
    var heightCm: Double
    var weightKg: Double

    var bmi: Double? {
        let meters = heightCm / 100
        guard meters > 0 else { return nil }
        return weightKg / (meters * meters)
    }
}

// MARK: - API 資料模型

private struct Credentials: Encodable {
    let username: String
    let password: String
}

private struct ChangePasswordBody: Encodable {
    let current_password: String
    let new_password: String
}

private struct ProfileUpdateBody: Encodable {
    let display_name: String
    let gender: String
    let age: Int
    let height_cm: Double
    let weight_kg: Double
}

private struct TokenResponse: Decodable {
    let access_token: String
    let token_type: String
    let username: String
}

private struct MeResponse: Decodable {
    let username: String
    let display_name: String?
    let gender: String?
    let age: Int?
    let height_cm: Double?
    let weight_kg: Double?
    let profile_completed: Bool?

    var isProfileCompleted: Bool {
        if let profile_completed { return profile_completed }
        guard let display_name, let gender, let age, let height_cm, let weight_kg else {
            return false
        }
        return !display_name.isEmpty && !gender.isEmpty
            && age > 0 && height_cm > 0 && weight_kg > 0
    }
}

private struct APIErrorBody: Decodable {
    let detail: String?
}

enum AuthError: LocalizedError {
    case invalidURL
    case unauthorized
    case server(String)
    case network(String)
    case decoding
    case profileAPIUnavailable

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "伺服器網址格式不正確"
        case .unauthorized: return "登入已過期，請重新登入"
        case .server(let msg): return localizedServerMessage(msg)
        case .network(let msg): return "無法連線到伺服器：\(msg)"
        case .decoding: return "伺服器回應格式錯誤"
        case .profileAPIUnavailable:
            return "雲端後端尚未更新個人資料功能，已先儲存於本機"
        }
    }

    private func localizedServerMessage(_ msg: String) -> String {
        if msg == "Not Found" || msg.contains("not found") {
            return "雲端後端尚未支援個人資料儲存，請更新後端後再試"
        }
        return msg
    }
}

// MARK: - AuthManager

@MainActor
final class AuthManager: ObservableObject {
    @Published private(set) var isAuthenticated = false
    @Published private(set) var username: String?
    @Published private(set) var userProfile: UserProfile?
    @Published private(set) var needsProfileSetup = false
    @Published private(set) var isOfflineSession = false
    @Published var isWorking = false
    @Published var errorMessage: String?

    private static let tokenAccount = "access_token"
    private static let usernameKey = "saved_username"
    private static let profileKeyPrefix = "user_profile_"

    private var token: String?

    init() {
        self.token = KeychainHelper.read(Self.tokenAccount)
        self.username = UserDefaults.standard.string(forKey: Self.usernameKey)
        if let username {
            userProfile = Self.loadCachedProfile(for: username)
            needsProfileSetup = userProfile == nil
        }
    }

    /// App 啟動時檢查既有權杖；連不上伺服器時保留本機登入（離線可用本機偵測）。
    func bootstrap() async {
        guard let token, !token.isEmpty else {
            isAuthenticated = false
            isOfflineSession = false
            needsProfileSetup = false
            return
        }
        do {
            let me = try await requestMe(token: token)
            applyMeResponse(me)
            isAuthenticated = true
            isOfflineSession = false
        } catch AuthError.unauthorized {
            signOut()
        } catch {
            isAuthenticated = username != nil
            isOfflineSession = isAuthenticated
            if userProfile == nil, let username {
                userProfile = Self.loadCachedProfile(for: username)
            }
            needsProfileSetup = userProfile == nil
        }
    }

    func register(username rawUsername: String, password: String) async {
        await authenticate(path: "/auth/register", username: rawUsername, password: password)
    }

    func login(username rawUsername: String, password: String) async {
        await authenticate(path: "/auth/login", username: rawUsername, password: password)
    }

    func saveProfile(displayName: String, gender: UserProfile.Gender, age: Int, heightCm: Double, weightKg: Double) async {
        errorMessage = nil
        guard let token = KeychainHelper.read(Self.tokenAccount), !token.isEmpty else {
            errorMessage = "尚未登入"
            return
        }

        isWorking = true
        defer { isWorking = false }

        let body = ProfileUpdateBody(
            display_name: displayName,
            gender: gender.rawValue,
            age: age,
            height_cm: heightCm,
            weight_kg: weightKg
        )

        do {
            let me = try await putProfile(token: token, body: body)
            applyMeResponse(me)
        } catch AuthError.profileAPIUnavailable {
            saveProfileLocally(
                displayName: displayName,
                gender: gender,
                age: age,
                heightCm: heightCm,
                weightKg: weightKg
            )
        } catch let error as AuthError {
            errorMessage = error.errorDescription
        } catch {
            saveProfileLocally(
                displayName: displayName,
                gender: gender,
                age: age,
                heightCm: heightCm,
                weightKg: weightKg
            )
        }
    }

    private func saveProfileLocally(
        displayName: String,
        gender: UserProfile.Gender,
        age: Int,
        heightCm: Double,
        weightKg: Double
    ) {
        let profile = UserProfile(
            displayName: displayName,
            gender: gender,
            age: age,
            heightCm: heightCm,
            weightKg: weightKg
        )
        userProfile = profile
        needsProfileSetup = false
        if let username {
            Self.cacheProfile(profile, for: username)
        }
        isOfflineSession = true
        errorMessage = nil
    }

    func signOut() {
        if let username {
            UserDefaults.standard.removeObject(forKey: Self.profileCacheKey(for: username))
        }
        token = nil
        username = nil
        userProfile = nil
        needsProfileSetup = false
        isAuthenticated = false
        isOfflineSession = false
        KeychainHelper.delete(Self.tokenAccount)
        UserDefaults.standard.removeObject(forKey: Self.usernameKey)
    }

    func changePassword(current: String, new: String) async -> Bool {
        errorMessage = nil
        guard new.count >= 6 else { errorMessage = "新密碼至少需 6 個字元"; return false }
        guard current != new else { errorMessage = "新密碼不可與目前密碼相同"; return false }
        guard let token = KeychainHelper.read(Self.tokenAccount), !token.isEmpty else {
            errorMessage = "尚未登入"
            return false
        }

        isWorking = true
        defer { isWorking = false }

        do {
            let url = try makeURL("/auth/change-password")
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.timeoutInterval = 15
            request.httpBody = try JSONEncoder().encode(ChangePasswordBody(current_password: current, new_password: new))

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw AuthError.decoding }
            guard (200...299).contains(http.statusCode) else {
                let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
                throw AuthError.server(detail ?? "伺服器錯誤（\(http.statusCode)）")
            }
            return true
        } catch let error as AuthError {
            errorMessage = error.errorDescription
            return false
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }

    func refreshProfile() async {
        errorMessage = nil
        guard let token = KeychainHelper.read(Self.tokenAccount), !token.isEmpty else {
            errorMessage = "尚未登入"
            return
        }
        self.token = token
        isWorking = true
        defer { isWorking = false }
        do {
            let me = try await requestMe(token: token)
            applyMeResponse(me)
            isOfflineSession = false
        } catch let error as AuthError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    var accessToken: String? {
        KeychainHelper.read(Self.tokenAccount)
    }

    // MARK: 私有

    private func authenticate(path: String, username rawUsername: String, password: String) async {
        let user = rawUsername.trimmingCharacters(in: .whitespacesAndNewlines)
        errorMessage = nil

        guard user.count >= 3 else { errorMessage = "帳號至少需 3 個字元"; return }
        guard password.count >= 6 else { errorMessage = "密碼至少需 6 個字元"; return }

        isWorking = true
        defer { isWorking = false }

        do {
            let result = try await postCredentials(path: path, username: user, password: password)
            token = result.access_token
            username = result.username
            KeychainHelper.save(result.access_token, for: Self.tokenAccount)
            UserDefaults.standard.set(result.username, forKey: Self.usernameKey)
            isAuthenticated = true
            isOfflineSession = false
            await loadUserProfile(token: result.access_token)
        } catch let error as AuthError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadUserProfile(token: String) async {
        do {
            let me = try await requestMe(token: token)
            applyMeResponse(me)
        } catch {
            userProfile = username.flatMap { Self.loadCachedProfile(for: $0) }
            needsProfileSetup = userProfile == nil
        }
    }

    private func applyMeResponse(_ me: MeResponse) {
        username = me.username
        UserDefaults.standard.set(me.username, forKey: Self.usernameKey)

        if me.isProfileCompleted,
           let displayName = me.display_name,
           let genderRaw = me.gender,
           let gender = UserProfile.Gender(rawValue: genderRaw),
           let age = me.age,
           let height = me.height_cm,
           let weight = me.weight_kg {
            let profile = UserProfile(
                displayName: displayName,
                gender: gender,
                age: age,
                heightCm: height,
                weightKg: weight
            )
            userProfile = profile
            needsProfileSetup = false
            Self.cacheProfile(profile, for: me.username)
        } else {
            userProfile = nil
            needsProfileSetup = true
            UserDefaults.standard.removeObject(forKey: Self.profileCacheKey(for: me.username))
        }
    }

    private static func profileCacheKey(for username: String) -> String {
        profileKeyPrefix + username
    }

    private static func cacheProfile(_ profile: UserProfile, for username: String) {
        if let data = try? JSONEncoder().encode(profile) {
            UserDefaults.standard.set(data, forKey: profileCacheKey(for: username))
        }
    }

    private static func loadCachedProfile(for username: String) -> UserProfile? {
        guard let data = UserDefaults.standard.data(forKey: profileCacheKey(for: username)),
              let profile = try? JSONDecoder().decode(UserProfile.self, from: data) else {
            return nil
        }
        return profile
    }

    private func makeURL(_ path: String) throws -> URL {
        let base = ServerConfig.baseURL
        guard let url = URL(string: base + path), url.scheme != nil, url.host != nil else {
            throw AuthError.invalidURL
        }
        return url
    }

    private func postCredentials(path: String, username: String, password: String) async throws -> TokenResponse {
        let url = try makeURL(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 15
        request.httpBody = try JSONEncoder().encode(Credentials(username: username, password: password))

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthError.network(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else { throw AuthError.decoding }
        guard (200...299).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
            throw AuthError.server(detail ?? "伺服器錯誤（\(http.statusCode)）")
        }

        guard let token = try? JSONDecoder().decode(TokenResponse.self, from: data) else {
            throw AuthError.decoding
        }
        return token
    }

    private func requestMe(token: String) async throws -> MeResponse {
        let url = try makeURL("/auth/me")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 15

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthError.network(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else { throw AuthError.decoding }
        if http.statusCode == 401 {
            throw AuthError.unauthorized
        }
        guard (200...299).contains(http.statusCode) else {
            throw AuthError.server("權杖驗證失敗（\(http.statusCode)）")
        }
        guard let me = try? JSONDecoder().decode(MeResponse.self, from: data) else {
            throw AuthError.decoding
        }
        return me
    }

    private func putProfile(token: String, body: ProfileUpdateBody) async throws -> MeResponse {
        let url = try makeURL("/auth/profile")
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 15
        request.httpBody = try JSONEncoder().encode(body)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthError.network(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else { throw AuthError.decoding }
        if http.statusCode == 401 {
            throw AuthError.unauthorized
        }
        if http.statusCode == 404 {
            throw AuthError.profileAPIUnavailable
        }
        guard (200...299).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(APIErrorBody.self, from: data))?.detail
            throw AuthError.server(detail ?? "儲存失敗（\(http.statusCode)）")
        }
        guard let me = try? JSONDecoder().decode(MeResponse.self, from: data) else {
            throw AuthError.decoding
        }
        return me
    }
}
