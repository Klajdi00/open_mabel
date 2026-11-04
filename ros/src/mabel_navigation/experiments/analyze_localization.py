#!/usr/bin/env python3
"""Benchmark localization estimators vs ground truth + tune a fused EKF.

Compares, against /odom_ground_truth:
  WHEEL  = swerve wheel odometry (/odom)
  VISUAL = cuVSLAM (/visual_slam/tracking/odometry)
  FUSED  = offline 2D EKF: predict from wheel body-twist, correct with visual pose.
The EKF sweeps the wheel process noise (= "weight on the wheels") to find the
trust setting that minimises fused trajectory error.

Metrics (after rigid R+t alignment to GT -- standard ATE, NO scale fudge so a
scale error counts as error):
  ATE   = RMS position error over the whole run (m)
  FINAL = position error at the end (m)
  YAWerr= RMS heading error (deg)
"""
import numpy as np

d = np.load("/tmp/loc_data.npz")
t = d["t"]; gt = d["gt"]; wh = d["wh"]; vo = d["vo"]; tw = d["tw"]
dt = np.diff(t, prepend=t[0]); dt[0] = dt[1] if len(dt) > 1 else 0.05
N = len(t)


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def rel_frame(traj):
    """Express a trajectory relative to its own start pose (start-frame coords)."""
    p0 = traj[0, :2]; th0 = traj[0, 2]
    c, s = np.cos(-th0), np.sin(-th0)
    R = np.array([[c, -s], [s, c]])
    xy = (traj[:, :2] - p0) @ R.T
    th = wrap(traj[:, 2] - th0)
    return np.column_stack([xy, th])


def kabsch(P, Q):
    """R,t minimising ||R P + t - Q|| (2D, rotation+translation, no scale)."""
    muP = P.mean(0); muQ = Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = muQ - R @ muP
    return R, t


def metrics(est_xytheta, gt_rel):
    """Direct comparison in the shared start frame (both estimator and GT are
    expressed relative to the same physical start pose, so no rotation alignment --
    that is ill-conditioned on a near-1D path and would hide real error)."""
    e = np.linalg.norm(est_xytheta[:, :2] - gt_rel[:, :2], axis=1)
    yawerr = wrap(est_xytheta[:, 2] - gt_rel[:, 2])
    return dict(ate=float(np.sqrt((e**2).mean())), final=float(e[-1]),
                maxe=float(e.max()), yaw=float(np.degrees(np.sqrt((yawerr**2).mean()))),
                aligned=est_xytheta[:, :2])


gt_rel = rel_frame(gt)
wh_rel = rel_frame(wh)
vo_rel = rel_frame(vo)

print(f"=== dataset: {N} samples, {t[-1]:.1f}s, GT path "
      f"{np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)):.2f} m ===\n")

m_wheel = metrics(wh_rel, gt_rel)
m_vis = metrics(vo_rel, gt_rel)


def run_ekf(sig_w, sig_wth, sig_v=0.04, sig_vth=0.03, vis_meas=None, valid=None):
    """2D EKF in GT frame. Predict from wheel twist (tw), update from visual pose.
    sig_w  : wheel position process noise growth (m / sqrt(s)) -> BIG = trust wheels less
    sig_wth: wheel yaw process noise (rad / sqrt(s))
    sig_v  : visual position meas noise (m), sig_vth: visual yaw meas noise (rad)
    Returns the fused trajectory (Nx3) in GT frame."""
    x = np.array([gt_rel[0, 0], gt_rel[0, 1], gt_rel[0, 2]])
    P = np.diag([1e-3, 1e-3, 1e-3])
    R = np.diag([sig_v**2, sig_v**2, sig_vth**2])
    out = np.zeros((N, 3))
    for i in range(N):
        h = dt[i] if dt[i] > 1e-3 else 0.05
        vx, vy, w = tw[i]
        thm = x[2] + 0.5*w*h
        x = x + np.array([(vx*np.cos(thm) - vy*np.sin(thm))*h,
                          (vx*np.sin(thm) + vy*np.cos(thm))*h, w*h])
        x[2] = wrap(x[2])
        Q = np.diag([(sig_w**2)*h, (sig_w**2)*h, (sig_wth**2)*h])
        P = P + Q
        if valid is None or valid[i]:           # correct only when visual is available
            z = vis_meas[i]
            y = np.array([z[0]-x[0], z[1]-x[1], wrap(z[2]-x[2])])
            S = P + R
            K = P @ np.linalg.inv(S)
            x = x + K @ y; x[2] = wrap(x[2])
            P = (np.eye(3) - K) @ P
        out[i] = x
    return out


# visual measurement lives in the shared start frame already (cuVSLAM odom starts
# aligned to base_link) -> use vo_rel directly as the EKF correction.
vo_in_gt = vo_rel

# sweep wheel process noise: small = trust wheels a lot, large = trust visual a lot
sweep = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
best = None
fused_rows = []
for sw in sweep:
    fused = run_ekf(sw, sw*0.5, vis_meas=vo_in_gt)
    mf = metrics(fused, gt_rel)
    fused_rows.append((sw, mf))
    if best is None or mf['ate'] < best[1]['ate']:
        best = (sw, mf, fused)

print("ESTIMATOR            ATE(m)   FINAL(m)  MAX(m)  YAW(deg)")
print(f"  wheel only        {m_wheel['ate']:7.3f} {m_wheel['final']:8.3f} "
      f"{m_wheel['maxe']:7.3f} {m_wheel['yaw']:7.2f}")
print(f"  visual only       {m_vis['ate']:7.3f} {m_vis['final']:8.3f} "
      f"{m_vis['maxe']:7.3f} {m_vis['yaw']:7.2f}")
print("  --- fused EKF, sweeping wheel process noise sig_w (BIG = less wheel weight) ---")
for sw, mf in fused_rows:
    tag = "  <-- BEST" if sw == best[0] else ""
    print(f"  fused sig_w={sw:4.2f}   {mf['ate']:7.3f} {mf['final']:8.3f} "
          f"{mf['maxe']:7.3f} {mf['yaw']:7.2f}{tag}")

# crude wheel-only drift rate (how fast wheel diverges from GT per metre travelled)
gtpath = np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))
print(f"\nwheel over-read: wheel path {np.sum(np.linalg.norm(np.diff(wh[:,:2],axis=0),axis=1)):.2f} m "
      f"vs GT path {gtpath:.2f} m  (ratio {np.sum(np.linalg.norm(np.diff(wh[:,:2],axis=0),axis=1))/gtpath:.2f})")
print(f"visual path {np.sum(np.linalg.norm(np.diff(vo[:,:2],axis=0),axis=1)):.2f} m  "
      f"(ratio {np.sum(np.linalg.norm(np.diff(vo[:,:2],axis=0),axis=1))/gtpath:.2f})")
print(f"\nBEST fused sig_w={best[0]} -> ATE {best[1]['ate']:.3f} m")

# --- robustness test: simulate VISUAL DROPOUTS (cuVSLAM loses tracking on blank
# walls / fast turns). visual-only freezes; a fused EKF coasts on the wheels. This
# is WHY you keep the wheels even though they slip -- they bridge the gaps. ---
valid = np.ones(N, bool)
for a, b in [(0.25, 0.38), (0.55, 0.68), (0.82, 0.95)]:   # 3 dropout windows
    valid[int(a*N):int(b*N)] = False
# visual-only under dropout: hold last good pose (frozen)
vo_drop = vo_in_gt.copy()
last = vo_in_gt[0]
for i in range(N):
    if valid[i]:
        last = vo_in_gt[i]
    else:
        vo_drop[i] = last
m_vis_drop = metrics(vo_drop, gt_rel)
# fused under dropout, with a MODERATE wheel weight (so wheels can bridge gaps)
for sw_try in [3.2, 0.2, 0.1, 0.05]:
    fused_drop = run_ekf(sw_try, sw_try*0.5, vis_meas=vo_in_gt, valid=valid)
    md = metrics(fused_drop, gt_rel)
    print(f"  [dropout 39% of run] fused sig_w={sw_try:4.2f}: ATE {md['ate']:.3f} m  final {md['final']:.3f} m")
print(f"  [dropout 39% of run] visual-only (frozen): ATE {m_vis_drop['ate']:.3f} m  final {m_vis_drop['final']:.3f} m")
np.savez("/tmp/loc_result.npz", gt=gt_rel, wheel=m_wheel['aligned'],
         visual=m_vis['aligned'], fused=best[2], best_sig_w=best[0])
