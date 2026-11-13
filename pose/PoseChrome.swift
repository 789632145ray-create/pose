//
//  PoseChrome.swift
//  pose
//
//  偵測畫面共用視覺語彙：單一強調色、深色毛玻璃、緊湊控制元件。
//

import SwiftUI
import UIKit

enum PoseChrome {
    /// 運動螢光綠 — 全畫面唯一主強調色。
    static let accent = Color(red: 50 / 255, green: 215 / 255, blue: 75 / 255)
    static let canvas = Color(red: 0.07, green: 0.07, blue: 0.09)
    static let glassFill = Color.black.opacity(0.38)
    static let secondaryFill = Color.white.opacity(0.12)
    static let label = Color.secondary
    static let primaryText = Color.white

    static let cardRadius: CGFloat = 16
    static let capsuleRadius: CGFloat = 10
    static let spacingXS: CGFloat = 8
    static let spacingSM: CGFloat = 12
    static let spacingMD: CGFloat = 16

    static let controlSpring = Animation.spring(response: 0.32, dampingFraction: 0.82)

    static func impact(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .medium) {
        UIImpactFeedbackGenerator(style: style).impactOccurred()
    }
}

// MARK: - Glass surfaces

struct PoseGlassCard<Content: View>: View {
    var cornerRadius: CGFloat = PoseChrome.cardRadius
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(PoseChrome.spacingSM)
            .background {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .background(
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .fill(PoseChrome.glassFill)
                    )
            }
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

// MARK: - Circular icon control

struct PoseCircleIconButton: View {
    let systemImage: String
    var label: String?
    var isProminent: Bool = false
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button {
            PoseChrome.impact(isProminent ? .medium : .light)
            action()
        } label: {
            VStack(spacing: 4) {
                Image(systemName: systemImage)
                    .font(.system(size: isProminent ? 22 : 16, weight: .semibold))
                    .foregroundStyle(isProminent ? Color.black : PoseChrome.primaryText)
                    .frame(width: isProminent ? 64 : 44, height: isProminent ? 64 : 44)
                    .background {
                        Circle()
                            .fill(isProminent ? PoseChrome.accent : PoseChrome.secondaryFill)
                    }
                    .overlay {
                        Circle()
                            .strokeBorder(Color.white.opacity(isProminent ? 0 : 0.14), lineWidth: 1)
                    }
                if let label {
                    Text(label)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(PoseChrome.primaryText.opacity(0.78))
                        .lineLimit(1)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.35 : 1)
    }
}

// MARK: - Compact segmented chip row

struct PoseSegmentChipRow<T: Hashable & Identifiable>: View where T: Equatable {
    let options: [T]
    @Binding var selection: T
    let title: (T) -> String
    var systemImage: ((T) -> String)? = nil

    var body: some View {
        HStack(spacing: 4) {
            ForEach(options) { option in
                let selected = selection == option
                Button {
                    guard selection != option else { return }
                    PoseChrome.impact(.light)
                    withAnimation(PoseChrome.controlSpring) {
                        selection = option
                    }
                } label: {
                    HStack(spacing: 4) {
                        if let systemImage {
                            Image(systemName: systemImage(option))
                                .font(.caption2.weight(.bold))
                        }
                        Text(title(option))
                            .font(.caption.weight(.bold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }
                    .foregroundStyle(selected ? Color.black : PoseChrome.primaryText.opacity(0.9))
                    .frame(maxWidth: .infinity)
                    .frame(height: 32)
                    .background {
                        Capsule(style: .continuous)
                            .fill(selected ? PoseChrome.accent : PoseChrome.secondaryFill)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .frame(height: 36)
    }
}

// MARK: - Grid stat cell

struct PoseStatCell: View {
    let title: String
    let value: String
    var subtitle: String? = nil
    var badge: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.system(size: 10, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(PoseChrome.label)
                if let badge {
                    Text(badge)
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(PoseChrome.accent)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(PoseChrome.accent.opacity(0.16), in: Capsule())
                }
            }
            Text(value)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(PoseChrome.primaryText)
                .minimumScaleFactor(0.7)
                .lineLimit(1)
            if let subtitle {
                Text(subtitle)
                    .font(.caption2.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(PoseChrome.label)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
