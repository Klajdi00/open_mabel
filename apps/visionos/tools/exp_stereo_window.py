#!/usr/bin/env python3
"""exp_stereo_window.py — why full immersion's head window needs a per-eye shift.

Analytic model of the head-locked stereo window (StereoWindowPanel.swift):
parallel stereo cameras with baseline b, shown on ONE flat window at distance D
centred between the eyes (IPD e). The window subtends the camera FOV exactly
(W = 2·D·tan(hfov/2)), so each image is true-scale; what goes wrong without a
correction is VERGENCE: each eye's image is IPD/2 off that eye's axis, adding
2·atan(e/2 / D) of convergence to every point. The material's per-eye texture
shift s = (e/2)/W cancels it exactly (proved per-pixel by
MabelTeleopTests/StereoWindowGeometryTests).

Same equations as the Swift; no simulation — this is the derivation, plotted.

    python3 exp_stereo_window.py --out ../../../ios_apps/iOS_v2/docs/figures/stereo_window_geometry.pdf
"""

import argparse
import math

E = 0.063            # IPD = camera baseline (MJCF zed_left/right: 63 mm)
FOVY = 83.3          # MJCF head camera
ASPECT = 4 / 3


def hfov():
    return 2 * math.atan(math.tan(math.radians(FOVY) / 2) * ASPECT)


def window(d):
    return 2 * d * math.tan(hfov() / 2), 2 * d * math.tan(math.radians(FOVY) / 2)


def vergence_error(d):
    return 2 * math.atan((E / 2) / d)


def perceived(z, d, shifted):
    """Distance the eyes converge to for a point straight ahead at depth z."""
    v = 2 * math.atan((E / 2) / z) + (0 if shifted else vergence_error(d))
    return (E / 2) / math.tan(v / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"camera: fovy {FOVY}°  aspect {ASPECT:.4f}  hfov {math.degrees(hfov()):.2f}°  IPD {E*1000:.0f} mm")
    print(f"{'D (m)':>6} {'W (m)':>7} {'H (m)':>7} {'shift s':>8} {'verg err':>9} {'1 m reads':>10} {'3 m reads':>10} {'inf reads':>10}")
    for d in (1.0, 1.5, 2.0, 3.0, 5.0):
        w, h = window(d)
        print(f"{d:6.1f} {w:7.3f} {h:7.3f} {(E/2)/w:8.5f} {math.degrees(vergence_error(d)):8.3f}° "
              f"{perceived(1.0, d, False):9.3f}m {perceived(3.0, d, False):9.3f}m {perceived(1e9, d, False):9.3f}m")
    print("with the per-eye shift every row reads TRUE depth (1.000 / 3.000 / inf).")

    if not a.out:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    zs = [0.3 + 0.05 * i for i in range(155)]
    ax1.plot(zs, zs, "k-", lw=2, label="per-eye shift (orthostereo)")
    for d, c in ((1.0, "#c0392b"), (2.0, "#e67e22"), (3.0, "#2980b9")):
        ax1.plot(zs, [perceived(z, d, False) for z in zs], "--", color=c, lw=1.4,
                 label=f"no shift, window at {d:.0f} m")
    ax1.set_xlabel("true distance (m)")
    ax1.set_ylabel("perceived distance (m)")
    ax1.set_title("(a) depth read from vergence", fontsize=10)
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3)
    ds = [0.5 + 0.05 * i for i in range(90)]
    ax2.plot(ds, [math.degrees(vergence_error(d)) for d in ds], "k-")
    ax2.axvline(2.0, color="0.5", ls=":")
    ax2.annotate(f"chosen D = 2 m: {math.degrees(vergence_error(2.0)):.2f}° cancelled",
                 (2.0, math.degrees(vergence_error(2.0))), xytext=(2.4, 4.0), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.4"))
    ax2.set_xlabel("window distance D (m)")
    ax2.set_ylabel("vergence error without shift (deg)")
    ax2.set_title("(b) the error the shift cancels", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
