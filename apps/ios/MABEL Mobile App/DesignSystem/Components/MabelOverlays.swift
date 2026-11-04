//
//  MabelOverlays.swift
//  MABEL Mobile App
//
//  ONE presentation language for everything that floats above the page.
//
//  Before this file there were three: the system `.sheet` with its grabber and
//  detents (timeline clip editing), the system `Menu` (the timeline's project
//  switcher), and the hand-built Control-Center dropdown (StatusDetailPanel /
//  the robot switcher). The first two are UIKit chrome — they carry their own
//  corner radius, their own material, their own typography — so the app changed
//  visual identity the moment you edited a clip.
//
//  The dropdown was already right, so it is the reference. Its geometry is the
//  contract every overlay here keeps:
//
//      .regularMaterial, RoundedRectangle(cornerRadius: 20, .continuous)
//      1 pt strokeBorder in Palette.hairlineSoft
//      13 pt padding; inner rows on Palette.surface at Radius.md
//
//  Two shapes are provided:
//
//      MabelDropdown  — anchored to a control, for picking one of a few things.
//                       Floats in a popover so it never pushes page content.
//      MabelModal     — centred over a scrim, for editing. Replaces `.sheet`.
//
//  Both are plain SwiftUI containers: callers supply content, never chrome.
//

import SwiftUI

// MARK: - Shared geometry

/// The one place the overlay look is defined. Changing a number here changes
/// every floating surface in the app, which is the point.
enum MabelOverlay {
    static let corner: CGFloat = 20          // matches StatusDetailPanel
    static let pad: CGFloat = 13
    static let rowCorner: CGFloat = Theme.Radius.md
    static let dropdownWidth: CGFloat = 290  // matches StatusDetailPanel
    static let modalMaxWidth: CGFloat = 520
    static let scrimOpacity: Double = 0.28

    /// The panel treatment itself — material, radius, hairline. Applied by both
    /// shapes so they cannot drift apart.
    struct Panel: ViewModifier {
        var corner: CGFloat = MabelOverlay.corner
        func body(content: Content) -> some View {
            content
                .background(.regularMaterial,
                            in: RoundedRectangle(cornerRadius: corner, style: .continuous))
                // CLIP, then border. Without the clip a hosted NavigationStack draws
                // its own bar background right over the rounded corners and the panel
                // reads as a square box with a rounded outline floating on top of it —
                // which is exactly what a Form-based editor looked like in here.
                .clipShape(RoundedRectangle(cornerRadius: corner, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: corner, style: .continuous)
                        .strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1)
                }
        }
    }
}

extension View {
    /// The shared floating-surface treatment. Use for anything that sits above
    /// the page: dropdowns, modals, inspectors.
    func mabelPanel(corner: CGFloat = MabelOverlay.corner) -> some View {
        modifier(MabelOverlay.Panel(corner: corner))
    }
}

// MARK: - Dropdown

/// A Control-Center-style dropdown, anchored to whatever opened it.
///
/// Use instead of `Menu`. `Menu` renders UIKit's own context menu — different
/// radius, different material, different type — so a picker in the timeline
/// looked nothing like the status dropdown two rows above it.
struct MabelDropdown<Content: View>: View {
    var width: CGFloat = MabelOverlay.dropdownWidth
    var title: String? = nil
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let title {
                Text(title.uppercased())
                    .font(.system(size: 10, weight: .semibold))
                    .kerning(0.6)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 2)
            }
            content()
        }
        .padding(MabelOverlay.pad)
        .frame(width: width)
        .mabelPanel()
    }
}

/// One selectable line in a dropdown. Mirrors the tiles inside StatusDetailPanel:
/// `Palette.surface` at `Radius.md`, hairline border, accent wash when active.
struct MabelDropdownRow: View {
    var icon: String? = nil
    var title: String
    var subtitle: String? = nil
    var isActive: Bool = false
    var isDestructive: Bool = false
    var action: () -> Void

    /// Icon tint. Named explicitly because `tint` alone resolves to SwiftUI's
    /// `View.tint(_:)` and fails with a ShapeStyle conformance error.
    private var iconTint: Color {
        if isDestructive { return Theme.Palette.danger }
        return isActive ? Theme.Palette.accent : Theme.Palette.textSecondary
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 18)
                        .foregroundStyle(iconTint)
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(.system(size: 13.5, weight: isActive ? .semibold : .medium))
                        .foregroundStyle(isDestructive ? Theme.Palette.danger : Theme.Palette.textPrimary)
                        .lineLimit(1)
                    if let subtitle {
                        Text(subtitle)
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
                if isActive {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Theme.Palette.accent)
                }
            }
            .padding(9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isActive ? Theme.Palette.accent.opacity(0.10) : Theme.Palette.surface,
                        in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                    .strokeBorder(isActive ? Theme.Palette.accent.opacity(0.45) : Theme.Palette.hairlineSoft,
                                  lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
    }
}

/// A hairline divider sized for the inside of a dropdown.
struct MabelDropdownDivider: View {
    var body: some View {
        Divider().overlay(Theme.Palette.hairlineSoft).padding(.horizontal, 2)
    }
}

extension View {
    /// Present a dropdown anchored to this view. Wraps `.popover` with the
    /// compact adaptation, exactly as the status bar does, so it stays a
    /// floating panel on iPhone rather than becoming a pull-up sheet.
    func mabelDropdown<C: View>(isPresented: Binding<Bool>,
                                arrowEdge: Edge = .top,
                                @ViewBuilder content: @escaping () -> C) -> some View {
        popover(isPresented: isPresented, arrowEdge: arrowEdge) {
            content().presentationCompactAdaptation(.popover)
        }
    }
}

// MARK: - Dismissal

/// `@Environment(\.dismiss)` only works inside a system presentation, so a view
/// hosted in a MabelModal has to be handed its own way out. Swapping the key is a
/// one-line change at each call site and leaves every `dismiss()` body untouched.
private struct MabelDismissKey: EnvironmentKey {
    static let defaultValue: () -> Void = {}
}

extension EnvironmentValues {
    var mabelDismiss: () -> Void {
        get { self[MabelDismissKey.self] }
        set { self[MabelDismissKey.self] = newValue }
    }
}

// MARK: - Modal

/// A centred editing surface over a dimmed scrim.
///
/// Replaces `.sheet` for clip / note / library editing. The system sheet brought
/// a grab handle, detents and a navigation bar that matched nothing else in the
/// app, and on a short screen the medium detent clipped the very controls it was
/// opened for. This keeps the dropdown's material and radius, puts the title and
/// the close affordance in a header the app styles itself, and scrolls its body
/// rather than resizing the window.
struct MabelModal<Content: View>: View {
    var title: String
    var subtitle: String? = nil
    /// Set false when the hosted view already carries its own bar. Forms that use
    /// `List`/`.searchable` need their NavigationStack, and its bar holds the
    /// Save/Cancel actions — so the modal steps back rather than stacking a second
    /// header on top of it.
    var showsHeader: Bool = true
    /// Scrolling belongs to the content when it is a Form or List; a ScrollView
    /// wrapped around one of those breaks its own scrolling.
    var scrolls: Bool = true
    /// Short content (a note editor) should hug its height instead of padding out to
    /// the cap; a Form must not, or it collapses.
    var shrinkToFit: Bool = false
    var onClose: () -> Void
    @ViewBuilder var content: () -> Content

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            if showsHeader {
                header
                Divider().overlay(Theme.Palette.hairlineSoft)
            }
            if scrolls {
                ScrollView { content().padding(MabelOverlay.pad) }
            } else {
                content()
            }
        }
        // Sized, not stretched. A Form or List will happily grow to fill whatever it
        // is given, so without an explicit cap the "modal" became a full-height slab
        // and stopped reading as a centred window at all.
        .frame(maxWidth: MabelOverlay.modalMaxWidth)
        .frame(maxHeight: maxHeight)
        .fixedSize(horizontal: false, vertical: shrinkToFit)
        .mabelPanel()
        .mabelShadow(Theme.Shadow.card)
        .padding(.horizontal, Theme.Space.lg)
    }

    /// Cap the panel well inside the screen so it always reads as a window floating
    /// over the timeline rather than a page that replaced it.
    private var maxHeight: CGFloat {
        #if os(visionOS)
        return 640
        #else
        let screen = UIScreen.main.bounds.height
        return min(640, screen * 0.74)
        #endif
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .lineLimit(1)
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 0)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .padding(7)
                    .background(Theme.Palette.surface, in: Circle())
                    .overlay { Circle().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            }
            .buttonStyle(PressableStyle())
            .accessibilityLabel("Close")
            .accessibilityIdentifier("modal.close")
        }
        .padding(.horizontal, MabelOverlay.pad)
        .padding(.vertical, 11)
    }
}

/// Hosts a `MabelModal` over a dimmed, tappable scrim.
private struct MabelModalHost<Item: Identifiable, C: View>: ViewModifier {
    @Binding var item: Item?
    @ViewBuilder var sheet: (Item) -> C
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content.overlay {
            ZStack {
                if item != nil {
                    // Scrim fades on its own track. Animating it WITH the panel made
                    // the whole thing feel like one flat cross-fade; separating them
                    // reads as a window arriving over a page that dims behind it.
                    Rectangle()
                        .fill(.black.opacity(MabelOverlay.scrimOpacity))
                        .ignoresSafeArea()
                        .contentShape(Rectangle())
                        .onTapGesture { dismiss() }
                        .transition(.opacity)
                        .accessibilityIdentifier("modal.scrim")
                        .accessibilityLabel("Dismiss")
                        .accessibilityAddTraits(.isButton)
                }
                if let value = item {
                    sheet(value)
                        .environment(\.mabelDismiss) { dismiss() }
                        // In from slightly small and slightly low, out straight down.
                        // Asymmetric because a dismissal that merely un-scales looks
                        // like a glitch, where a small drop reads as "put away".
                        .transition(reduceMotion
                                    ? .opacity
                                    : .asymmetric(
                                        insertion: .opacity
                                            .combined(with: .scale(scale: 0.92))
                                            .combined(with: .offset(y: 14)),
                                        removal: .opacity
                                            .combined(with: .scale(scale: 0.96))
                                            .combined(with: .offset(y: 10))))
                }
            }
            // Fills the window, so the panel is centred on the SCREEN rather than on
            // whatever slice of it the timeline happened to occupy.
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
            // SCOPED TO THE OVERLAY. This used to sit on the modified view, i.e. on
            // `content` as well — so the animation applied to the whole timeline
            // canvas and every edit underneath sprang. Adding a clip visibly juddered
            // the canvas. An implicit .animation covers the entire subtree it is
            // attached to, which here has to be the overlay and nothing else.
            .animation(reduceMotion ? Theme.Motion.gentle : Self.pop, value: item?.id)
            .zIndex(10)
        }
    }

    /// A touch looser than Theme.Motion.sheet: enough overshoot to feel alive,
    /// short enough that it never delays the edit you opened it for.
    /// Computed, not stored: this type is generic, and Swift allows no stored
    /// static in a generic type (it would need one instance per specialisation).
    private static var pop: Animation { .spring(response: 0.34, dampingFraction: 0.78) }

    private func dismiss() { item = nil }
}

extension View {
    /// Drop-in replacement for `.sheet(item:)` that presents a `MabelModal`
    /// instead of the system pull-up sheet. The closure supplies the modal.
    func mabelModal<Item: Identifiable, C: View>(
        item: Binding<Item?>,
        @ViewBuilder content: @escaping (Item) -> C
    ) -> some View {
        modifier(MabelModalHost(item: item, sheet: content))
    }
}

// MARK: - Form pieces

/// A titled group inside a modal — the flat equivalent of `Form`'s `Section`,
/// without the inset-grouped list background that fights the panel material.
struct MabelModalSection<Content: View>: View {
    var title: String? = nil
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let title {
                Text(title.uppercased())
                    .font(.system(size: 10, weight: .semibold))
                    .kerning(0.6)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 2)
            }
            VStack(spacing: 8) { content() }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One row inside a modal section, on the same tile treatment as a dropdown row.
struct MabelModalRow<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.Palette.surface,
                        in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                    .strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1)
            }
    }
}
