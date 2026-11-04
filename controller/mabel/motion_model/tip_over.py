"""
Tip-over safety motion model for MABEL's mobile base (lift-height aware).

This is THE safety envelope every base command must pass through before it
reaches `swerve_ik` — per-direction speed/deceleration caps that shrink as the
lift raises the CoM, plus the full coupled ZMP (support-polygon) criterion for
combined translation + rotation. `safe_base_step(...)` is the one entry point:
it clips, projects, ramps, and transient-clamps a target twist so the zero-
moment point stays inside the wheel-contact triangle.

NumPy-only on purpose: the ROS navigation node imports this on the robot
without MuJoCo/scipy. All constants here are MEASURED from the canonical MJCF
(`simulation/mabel_mujoco/models/mabel_full.xml`) and go stale when masses,
the lift, or link origins change — re-derive with
`python3 controller/experiments/wbc_stability.py` (E4 CoM ACCURACY) and paste
the printed values below. `mabel.config` re-exports everything in this module,
so `from mabel.config import safe_base_step` keeps working everywhere.
"""
import math
import numpy as np

__all__ = [
    # base motion limits / lift hardware
    "MAX_LIN_SPEED", "MAX_ANG_SPEED", "MAX_LIN_ACCEL", "MAX_ANG_ACCEL",
    "LIFT_RANGE", "LIFT_HW_MAX_SPEED",
    # tip-over constants (measured from the MJCF)
    "GRAVITY", "TIP_LEVER_FWD", "TIP_LEVER_LAT", "TIP_LEVER_BACK",
    "TIP_SF_FWD", "TIP_SF_AGILE", "STOP_TIME_FWD", "STOP_TIME_AGILE",
    "COM_Z0", "COM_DZDLIFT", "V_SPEED_FLOOR", "V_FWD_CEIL", "V_AGILE_CEIL",
    "SUPPORT_PTS", "COM_XY",
    # the motion-model API
    "com_height", "max_decel_dir", "max_speed_dir", "max_decel", "max_speed",
    "com_accel", "tip_usage", "accel_safe_scale", "safe_base_step",
    "tip_speed_scale",
]

# ── Base motion limits / smoothing ───────────────────────────────────────────
MAX_LIN_SPEED   = 0.8     # m/s  (absolute ceiling; the tip-over model lowers it)
MAX_ANG_SPEED   = 2.0     # rad/s
MAX_LIN_ACCEL   = 2.0     # m/s^2 (ceiling; the tip-over model lowers it with lift)
MAX_ANG_ACCEL   = 5.0     # rad/s^2

# ── Lift travel & hardware speed ceiling ─────────────────────────────────────
LIFT_RANGE        = (0.0, 0.635)   # metres of cascaded lift travel
# Real-hardware ceiling: the lift firmware caps cruise speed at 60 mm/s
# (firmware/lift/firmware/config.py: MAX_VELOCITY_MM_S = 60.0). Every lift
# velocity limit derives from this so no software layer (WBC / IK / lazy
# retarget / nav / jog) can command a speed the hardware cannot deliver.
# KEEP IN SYNC with the firmware if that ceiling changes.
LIFT_HW_MAX_SPEED = 0.06           # m/s  (= 60 mm/s, real-hardware cruise cap)

# ── Tip-over safety: DIRECTION-AWARE speed / deceleration vs. lift height ─────
# When the base brakes at |a|, the inertial force m*a at the CoM (height h)
# tips the robot over the support edge it is pushing against (lever d_dir):
#     m*a*h <= m*g*d_dir   =>   a_tip(h) = g * d_dir / h     (mass cancels)
# Raising the lift raises h (~linearly), so the safe accel/speed shrink as ~1/h.
# MABEL's triangular base (single REAR wheel) + slightly forward CoM makes the
# FRONT the binding direction; lateral is only marginally larger and rearward is
# generous. Agile strafing therefore comes mostly from FAST STEERING (low-latency
# module reorientation), not from a high lateral drive speed. Levers measured
# from mabel_sim.xml (CoM via subtree_com, support polygon from wheel contacts).
GRAVITY        = 9.81    # m/s^2
# Re-derived 2026-06 after the lift_base origin was lowered 0.10 m in mabel_full.xml
# (default height dropped): the robot subtree CoM moved ~1.6 cm forward, so the
# forward lever shrank. Measured (settled, robot subtree CoM) by experiments/
# wbc_stability.py against the live wheel-contact support triangle.
TIP_LEVER_FWD  = 0.078   # m; CoM -> front edge (fl-fr): the only real tip risk (was 0.094)
TIP_LEVER_LAT  = 0.108   # m; CoM -> side edges (fl-b / fr-b) (was 0.103)
TIP_LEVER_BACK = 0.241   # m; CoM -> rear wheel (b) (was 0.225)
# Per-direction safety factor: forward tipping is the genuine risk, so it stays
# conservative; lateral/rearward tips have a bit more margin AND are far less
# dangerous, so we run them near the tip limit for snappy direction changes.
TIP_SF_FWD        = 2.0  # forward braking: keep |a| <= a_tip / 2 (conservative)
TIP_SF_AGILE      = 1.2  # lateral & rearward: |a| <= a_tip / 1.2 (agile reversals)
# Nominal time to shed full speed (sets the speed cap, v = a * STOP_TIME, and
# the full-speed reversal time ~= 2 * STOP_TIME). Forward keeps a longer, gentle
# stop & higher top speed; lateral/rear use a short one for snappy reversals.
STOP_TIME_FWD     = 0.80 # s; forward
STOP_TIME_AGILE   = 0.55 # s; lateral / rearward
# CoM height vs. lift command z in metres (linear fit):  h = z0 + k*z
# Re-fit 2026-06 after the lift_base drop (0.10 m lower): h(0)=0.384 m, h(0.635)=0.658 m
# (measured, settled robot subtree CoM; was z0=0.443, k=0.432).
# Fit error (measured 2026-07, exp_wbc_validation/wbc_stability E4): max 1.5 cm,
# UNDER-estimating h at z=0 (the settled z=0 CoM sits off the linear trend), i.e.
# a_safe is over-estimated by <~4% there — comfortably inside the 1.2–2.0 safety
# factors. Mid-range the fit over-estimates h (conservative).
COM_Z0       = 0.369
COM_DZDLIFT  = 0.451
V_SPEED_FLOOR = 0.15            # m/s; always allow at least a crawl, fully raised
V_FWD_CEIL    = MAX_LIN_SPEED   # m/s; forward speed ceiling (conservative)
V_AGILE_CEIL  = 1.50            # m/s; lateral / rearward ceiling (agile; physics still binds)

def com_height(z_lift):
    """CoM height [m] for a lift command z_lift in metres."""
    z = min(max(float(z_lift), LIFT_RANGE[0]), LIFT_RANGE[1])
    return COM_Z0 + COM_DZDLIFT * z

def _axis_decel(lever, sf, z_lift):
    return GRAVITY * lever / (sf * com_height(z_lift))

def max_decel_dir(z_lift):
    """Tip-safe |acceleration| caps [m/s^2] as (fwd, back, lat) at lift z.
    Forward stays conservative; lateral/rear run near the tip limit for snappy
    braking and direction reversals."""
    return (_axis_decel(TIP_LEVER_FWD,  TIP_SF_FWD,   z_lift),
            _axis_decel(TIP_LEVER_BACK, TIP_SF_AGILE, z_lift),
            _axis_decel(TIP_LEVER_LAT,  TIP_SF_AGILE, z_lift))

def max_speed_dir(z_lift):
    """Tip-safe speed caps [m/s] as (fwd, back, lat): a stop takes ~STOP_TIME_*."""
    af, ab, al = max_decel_dir(z_lift)
    clamp = lambda v, hi: min(max(v, V_SPEED_FLOOR), hi)
    return (clamp(af * STOP_TIME_FWD,   V_FWD_CEIL),
            clamp(ab * STOP_TIME_AGILE, V_AGILE_CEIL),
            clamp(al * STOP_TIME_AGILE, V_AGILE_CEIL))

# Back-compat scalars (forward = most conservative direction).
def max_decel(z_lift): return _axis_decel(TIP_LEVER_FWD, TIP_SF_FWD, z_lift)
def max_speed(z_lift):
    return min(max(_axis_decel(TIP_LEVER_FWD, TIP_SF_FWD, z_lift) * STOP_TIME_FWD, V_SPEED_FLOOR), V_FWD_CEIL)

# ── Coupled tip-over for COMBINED motion (ZMP / support-polygon) ──────────────
# The per-axis caps above are the pure-axis special case. Under combined
# translation + rotation the CoM acceleration picks up Coriolis (omega x v),
# centripetal (omega^2 r_c) and tangential (domega x r_c) terms, so a turn while
# driving tips the robot even at "safe" per-axis speeds. We therefore enforce the
# full criterion: the zero-moment point x_zmp = x_com - (h/g) a_c must stay inside
# the wheel-contact triangle. Equivalently, for every support edge k with outward
# unit normal o_k and CoM->edge lever d_k, the CoM accel obeys
#     (-o_k) . a_c  <=  g d_k / (SF_k h).        (Papadopoulos-Rey / ZMP)
# Contact points + CoM projection are measured from mabel_sim.xml (chassis frame).
SUPPORT_PTS = np.array([[-0.161, -0.174],   # fl wheel
                        [-0.147,  0.166],   # fr wheel
                        [ 0.165, -0.017]])  # b  wheel (rear)
COM_XY = np.array([-0.076, -0.001])         # CoM ground projection (front-biased; re-measured 2026-06 after the lift drop, was [-0.060,-0.005])

def _support_edges():
    """(outward unit normal, CoM->edge lever d, per-edge safety factor) per edge."""
    pts = SUPPORT_PTS; ctr = pts.mean(0); fwd = np.array([-1.0, 0.0])  # front = -X
    edges = []
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        e = b - a
        n = np.array([e[1], -e[0]]); n /= np.linalg.norm(n)
        if n @ (a - ctr) < 0:        # orient outward (away from polygon centroid)
            n = -n
        d = float(n @ (a - COM_XY))  # CoM -> edge distance along outward normal
        sf = TIP_SF_FWD if (n @ fwd) > 0.5 else TIP_SF_AGILE   # front edge is strict
        edges.append((n, d, sf))
    return edges

_SUPPORT_EDGES = _support_edges()

def com_accel(vx, vy, omega, ax, ay, aw):
    """Horizontal CoM acceleration (body frame) for body velocity (vx,vy,omega)
    and its rate (ax,ay,aw): transport/Coriolis + tangential + centripetal."""
    rx, ry = COM_XY
    a_cx = (ax - omega * vy) - aw * ry - omega * omega * rx
    a_cy = (ay + omega * vx) + aw * rx - omega * omega * ry
    return a_cx, a_cy

def tip_usage(vx, vy, omega, ax, ay, aw, z_lift):
    """Support-polygon (ZMP) usage in [0, inf): max over edges of the toward-edge
    CoM acceleration divided by that edge's tip-safe limit. <=1 is stable."""
    a_cx, a_cy = com_accel(vx, vy, omega, ax, ay, aw)
    h = com_height(z_lift); u = 0.0
    for n, d, sf in _SUPPORT_EDGES:
        toward = -(n[0] * a_cx + n[1] * a_cy)      # accel pushing ZMP across edge k
        if toward > 0.0:
            u = max(u, toward / (GRAVITY * d / (sf * h)))
    return u

def accel_safe_scale(vx, vy, omega, ax, ay, aw, z_lift):
    """Scale in [0,1] on the COMMAND acceleration (ax,ay,aw) so the full
    instantaneous CoM acceleration (command + centripetal/Coriolis at the current
    twist) keeps the ZMP in the support polygon. This catches the transient
    coupling the per-axis ramp misses (accelerating while turning).

    Never dead-locks: the cap is max(1, u0) where u0 is the usage at zero command
    accel (centripetal only), so scale 0 is always feasible, and any *braking*
    command (which lowers usage) is passed through unscaled."""
    u0 = tip_usage(vx, vy, omega, 0.0, 0.0, 0.0, z_lift)   # centripetal-only floor
    cap = max(1.0, u0)
    if tip_usage(vx, vy, omega, ax, ay, aw, z_lift) <= cap:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        um = tip_usage(vx, vy, omega, mid * ax, mid * ay, mid * aw, z_lift)
        lo, hi = (mid, hi) if um <= cap else (lo, mid)
    return lo

def safe_base_step(vx, vy, omega, tvx, tvy, tw, dt, z_lift):
    """One tip-safe smoothing step: clip target to the per-axis ceilings, project
    the combined target into the steady ZMP envelope, ramp per-axis toward it,
    then clamp the transient so the full a_c keeps the ZMP in the support polygon.
    Returns the new (vx, vy, omega). Shared by the teleop smoother and the
    stability test harness so they exercise identical logic."""
    a_fwd, a_back, a_lat = max_decel_dir(z_lift)
    v_fwd, v_back, v_lat = max_speed_dir(z_lift)
    tvx = max(-v_fwd, min(v_back, tvx))
    tvy = max(-v_lat, min(v_lat, tvy))
    s = tip_speed_scale(tvx, tvy, tw, z_lift)              # steady-envelope project
    tvx *= s; tvy *= s; tw *= s
    ax_cap = a_fwd if vx < tvx else a_back                 # +a_x = front tip
    cvx = min(tvx, vx + ax_cap*dt) if vx < tvx else max(tvx, vx - ax_cap*dt)
    cvy = min(tvy, vy + a_lat*dt)  if vy < tvy else max(tvy, vy - a_lat*dt)
    cw  = min(tw, omega + MAX_ANG_ACCEL*dt) if omega < tw else max(tw, omega - MAX_ANG_ACCEL*dt)
    al = accel_safe_scale(vx, vy, omega, (cvx-vx)/dt, (cvy-vy)/dt, (cw-omega)/dt, z_lift)
    return vx + al*(cvx-vx), vy + al*(cvy-vy), omega + al*(cw-omega)

def tip_speed_scale(vx, vy, omega, z_lift):
    """Scale in (0,1] that shrinks a *steady* twist (vx,vy,omega) into the
    tip-safe envelope (ZMP inside the support polygon at zero command accel).

    The steady CoM acceleration a_c = (-omega*vy - omega^2 rx, omega*vx - omega^2 ry)
    is degree-2 homogeneous in the twist, and tip_usage is linear in a_c, so
    usage(s*twist) = s^2 usage(twist). Hence the projection is closed-form,
    s = 1/sqrt(u). Crucially u(0)=0 -> s=1, so a stop command is NEVER throttled;
    an over-limit state is pulled back in (not frozen), so the robot can always
    decelerate. This replaces the earlier blend, which dead-locked (fell back to
    'hold velocity') when the current twist already violated the constraint --
    e.g. a fast spin mixed with linear velocity could not be stopped."""
    u = tip_usage(vx, vy, omega, 0.0, 0.0, 0.0, z_lift)
    return 1.0 if u <= 1.0 else u ** -0.5
