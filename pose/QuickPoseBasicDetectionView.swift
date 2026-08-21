//
//  QuickPoseBasicDetectionView.swift
//  pose
//
//  對照官方 BasicDemo：camera 區塊 onAppear 即 start、onDisappear 才 stop。
//  用於驗證 QuickPose SDK 是否正常回呼影格（與完整偵測管線分離）。
//

import AVFoundation
import Combine
import PhotosUI
import SwiftUI
import QuickPoseCore
import QuickPoseSwiftUI

private struct BasicPickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dst = URL.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "-" + received.file.lastPathComponent)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: received.file, to: dst)
            return BasicPickedMovie(url: dst)
        }
    }
}

/// 持有單一 QuickPose 實例（class），onFrame 直接更新 @Published，不經 SwiftUI struct 閉包。
@MainActor
final class BasicQuickPoseRunner: ObservableObject {
    let quickPose: QuickPose

    @Published var overlayImage: UIImage?
    @Published var fpsText = "FPS: —"
    @Published var statusLine = "尚未啟動"
    @Published private(set) var frameCount = 0
    private(set) var loopActive = false

    init() {
        let raw = (Bundle.main.object(forInfoDictionaryKey: "QuickPoseSDKKey") as? String) ?? ""
        quickPose = QuickPose(sdkKey: raw.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func start() {
        guard !loopActive else { return }
        loopActive = true
        frameCount = 0
        statusLine = "引擎已 start，等待影格…"

        quickPose.start(features: [.overlay(.wholeBody)]) { [weak self] status, image, _, _, _ in
            Task { @MainActor in
                self?.apply(status: status, image: image)
            }
        }
    }

    func stop() {
        if loopActive {
            quickPose.stop()
            loopActive = false
        }
    }

    private func apply(status: QuickPose.Status, image: UIImage?) {
        frameCount += 1
        overlayImage = image

        switch status {
        case .success(let info):
            fpsText = "FPS: \(info.fps)"
            statusLine = "成功 · 累計 \(frameCount) 幀"
        case .noPersonFound:
            fpsText = "FPS: —"
            statusLine = "無人物 · 累計 \(frameCount) 幀"
        case .sdkValidationError:
            fpsText = "FPS: —"
            statusLine = "SDK 驗證失敗"
        @unknown default:
            statusLine = "未知狀態 · 累計 \(frameCount) 幀"
        }
    }
}

struct QuickPoseBasicDetectionView: View {
    @EnvironmentObject private var modeStore: DetectionModeStore
    @StateObject private var runner = BasicQuickPoseRunner()

    @State private var inputMode: BasicInputMode = .liveCamera
    @State private var videoURL: URL?
    @State private var pickedItem: PhotosPickerItem?
    @State private var isLoadingVideo = false
    @State private var cameraAuthorized = false

    private enum BasicInputMode: String, CaseIterable {
        case liveCamera = "相機"
        case video = "影片"
    }

    var body: some View {
        ZStack {
            Color(red: 0.10, green: 0.10, blue: 0.12).ignoresSafeArea()

            if inputMode == .liveCamera, !cameraAuthorized {
                cameraGatePanel
            } else {
                basicCameraStack
            }
        }
        .onAppear {
            refreshCameraAuthorization()
        }
        .onChange(of: pickedItem) { _, item in
            Task { await loadVideo(from: item) }
        }
    }

    private var cameraGatePanel: some View {
        VStack(spacing: 16) {
            Text("QuickPose 原生模式")
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            Text("需要相機權限。請允許後會自動開始偵測。")
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.85))
                .padding(.horizontal, 24)
            Button("允許相機並開始") {
                requestCameraAccess()
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
        }
        .padding()
    }

    private var basicCameraStack: some View {
        GeometryReader { geometry in
            let w = max(geometry.size.width, 1)
            let h = max(geometry.size.height, 1)

            ZStack(alignment: .top) {
                Group {
                    switch inputMode {
                    case .liveCamera:
                        QuickPoseCameraView(useFrontCamera: true, delegate: runner.quickPose, videoGravity: .resizeAspectFill)
                    case .video:
                        if let url = videoURL {
                            QuickPoseSimulatedCameraView(useFrontCamera: false, delegate: runner.quickPose, video: url)
                                .id(url)
                        } else {
                            Color.black
                            Text("請上傳影片")
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    }
                }
                .frame(width: w, height: h)
                .clipped()

                QuickPoseOverlayView(overlayImage: $runner.overlayImage, contentMode: .fill)
                    .frame(width: w, height: h)
                    .allowsHitTesting(false)

                VStack {
                    hudPanel
                        .padding(.horizontal, 12)
                        .padding(.top, geometry.safeAreaInsets.top + 8)

                    Spacer()

                    controlPanel
                        .padding(.horizontal, 12)
                        .padding(.bottom, geometry.safeAreaInsets.bottom + 12)
                }
            }
        }
        .onAppear {
            runner.start()
        }
        .onDisappear {
            runner.stop()
        }
    }

    private var hudPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("QuickPose 原生")
                .font(.caption.weight(.bold))
                .foregroundStyle(.cyan)
            Text(runner.fpsText)
                .font(.system(size: 17, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
            Text(runner.statusLine)
                .font(.caption)
                .foregroundStyle(.yellow)
            Text("v\(runner.quickPose.quickPoseVersion())")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.6))
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private var controlPanel: some View {
        VStack(spacing: 10) {
            DetectionEnginePickerRow()

            Picker("輸入來源", selection: $inputMode) {
                ForEach(BasicInputMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: inputMode) { _, newMode in
                runner.stop()
                if newMode == .liveCamera {
                    videoURL = nil
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                    runner.start()
                }
            }

            if inputMode == .video {
                PhotosPicker(selection: $pickedItem, matching: .videos, photoLibrary: .shared()) {
                    HStack {
                        if isLoadingVideo { ProgressView().tint(.white) }
                        Label(videoURL == nil ? "選擇影片" : "重新選擇影片", systemImage: "film.fill")
                            .font(.subheadline.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .disabled(isLoadingVideo)
            }

            Button {
                runner.stop()
                runner.start()
            } label: {
                Label("重新 start 引擎", systemImage: "arrow.clockwise")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.bordered)
            .tint(.orange)

            Text("此模式對照官方 demo：無倒數、無暫停邏輯。若這裡有 FPS，代表 SDK 正常。")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.55))
                .multilineTextAlignment(.center)
        }
        .padding(12)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }

    private func refreshCameraAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraAuthorized = true
        default:
            cameraAuthorized = false
        }
    }

    private func requestCameraAccess() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                cameraAuthorized = granted
            }
        }
    }

    private func loadVideo(from item: PhotosPickerItem?) async {
        guard let item else { return }
        isLoadingVideo = true
        defer {
            isLoadingVideo = false
            pickedItem = nil
        }
        do {
            if let movie = try await item.loadTransferable(type: BasicPickedMovie.self) {
                await MainActor.run {
                    runner.stop()
                    videoURL = movie.url
                    runner.start()
                }
            }
        } catch {
            await MainActor.run {
                runner.statusLine = "影片載入失敗"
            }
        }
    }
}
