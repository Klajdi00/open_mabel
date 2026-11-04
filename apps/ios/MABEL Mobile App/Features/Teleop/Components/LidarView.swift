//
//  LidarView.swift
//  MABEL Mobile App
//
//  Top-down 2D LiDAR tile for the pilot monitor — an at-a-glance "is the LiDAR
//  alive" check. Renders the latest planar sweep (robot frame, forward = up)
//  with range rings; shows a NO DATA badge until a sweep arrives (real robot:
//  RPLIDAR via /scan; sim: the bridge's 360-ray ray-cast). Pure SwiftUI Canvas —
//  360 points at ~5 Hz is trivially cheap.
//

import SwiftUI

struct LidarView: View {
    let scan: RobotState.LidarScan?
    var tint: Color = Theme.Palette.accent

    /// Furthest ring shown (m). Points at or beyond the sensor's `rangeMax` are
    /// non-returns and are not drawn.
    private let viewRange: Double = 8.0

    var body: some View {
        ZStack {
            Canvas { ctx, size in
                let c = CGPoint(x: size.width / 2, y: size.height / 2)
                let scale = min(size.width, size.height) / 2 / viewRange   // pt per metre

                // Range rings + crosshair — the static backdrop.
                for r in [2.0, 4.0, 6.0, 8.0] {
                    let rad = r * scale
                    let ring = Path(ellipseIn: CGRect(x: c.x - rad, y: c.y - rad,
                                                      width: rad * 2, height: rad * 2))
                    ctx.stroke(ring, with: .color(.white.opacity(r == 8.0 ? 0.16 : 0.09)),
                               lineWidth: 1)
                }
                var cross = Path()
                cross.move(to: CGPoint(x: c.x, y: 0)); cross.addLine(to: CGPoint(x: c.x, y: size.height))
                cross.move(to: CGPoint(x: 0, y: c.y)); cross.addLine(to: CGPoint(x: size.width, y: c.y))
                ctx.stroke(cross, with: .color(.white.opacity(0.07)), lineWidth: 1)

                // The sweep. Robot frame: angle 0 = +x = forward = screen UP,
                // CCW positive → screen left. (x_r, y_r) → (cx - y_r·s, cy - x_r·s).
                if let scan, !scan.ranges.isEmpty {
                    var pts = Path()
                    let cutoff = min(scan.rangeMax, viewRange) - 1e-3
                    for (i, r) in scan.ranges.enumerated() {
                        guard r > 0.05, r < cutoff else { continue }   // drop non-returns
                        let a = scan.angleMin + Double(i) * scan.angleInc
                        let x = r * cos(a), y = r * sin(a)
                        let p = CGPoint(x: c.x - y * scale, y: c.y - x * scale)
                        pts.addEllipse(in: CGRect(x: p.x - 1.2, y: p.y - 1.2, width: 2.4, height: 2.4))
                    }
                    ctx.fill(pts, with: .color(tint.opacity(0.9)))
                }

                // Robot marker: a small forward-pointing wedge at the centre.
                var robot = Path()
                robot.move(to: CGPoint(x: c.x, y: c.y - 5))
                robot.addLine(to: CGPoint(x: c.x - 3.6, y: c.y + 4))
                robot.addLine(to: CGPoint(x: c.x + 3.6, y: c.y + 4))
                robot.closeSubpath()
                ctx.fill(robot, with: .color(.white.opacity(0.85)))
            }

            // Label chip, same grammar as the camera thumbnails.
            VStack {
                HStack {
                    HStack(spacing: 4) {
                        Circle().fill(scan == nil ? Theme.Palette.danger : Theme.Palette.live)
                            .frame(width: 5, height: 5)
                        Text("LIDAR").font(.system(size: 8, weight: .bold, design: .rounded))
                            .foregroundStyle(Theme.Palette.textPrimary)
                    }
                    .padding(.horizontal, 5).padding(.vertical, 3)
                    .background(.black.opacity(0.4), in: Capsule())
                    Spacer()
                }
                Spacer()
            }
            .padding(5)

            if scan == nil {
                ZStack {
                    Color.black.opacity(0.55)
                    VStack(spacing: 2) {
                        Image(systemName: "dot.radiowaves.left.and.right")
                            .font(.system(size: 11, weight: .regular))
                        Text("NO DATA")
                            .font(.system(size: 7, weight: .bold, design: .monospaced))
                            .tracking(1.4)
                    }
                    .foregroundStyle(.white.opacity(0.55))
                }
                .allowsHitTesting(false)
            }
        }
        .background(Color(hex: 0x07060A))
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        }
    }
}
