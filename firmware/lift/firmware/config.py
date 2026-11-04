# config.py - Hardware pinout, mechanical constants, and control gains.
#
# This is the single source of truth for tuning the lift. Nothing here
# performs any action; every other module imports the values it needs.
#
# NOTE: GP numbers are *GPIO* numbers, NOT the physical pin numbers printed
# on the Pico board. e.g. GP27 is physical pin 32. Check a pinout diagram.

# ----------------------------------------------------------------------------
# BTS7960 H-bridge motor driver (logic side, 3V3)
# ----------------------------------------------------------------------------
# MABEL and NOVA run the SAME lift, so this file is kept identical in both repos apart
# from ENC_A_PIN, which is per-board wiring.
RPWM_PIN = 2          # GP2 (phys 4) -> RPWM  (drives the lift UP)
LPWM_PIN = 3          # GP3 (phys 5) -> LPWM  (drives the lift DOWN)
R_EN_PIN = 4          # GP4 (phys 6) -> R_EN  (driver enable, held HIGH)
L_EN_PIN = 5          # GP5 (phys 7) -> L_EN  (driver enable, held HIGH)

# The module ALSO needs, and these are not Pico pins:
#   B+ / B-   motor supply (the big screw terminals), 12-24 V for a desk column
#   M+ / M-   motor leads
#   VCC       logic supply — 5 V. Use the Pico's VBUS (phys 40), NOT 3V3.
#   GND       common with the Pico GND *and* with the motor supply ground.

MOTOR_PWM_FREQ = 20000   # Hz. 20 kHz keeps PWM whine out of the audible band.
DEADTIME_MS    = 10      # break-before-make pause when reversing direction.

# Effort below this magnitude can't overcome stiction, so the controller
# snaps small commands either to zero or up to MIN_EFFORT. Tune on hardware.
MIN_EFFORT = 0.12        # 0.0 - 1.0
# Global ceiling on duty cycle. FULL — speed is the priority on this column.
#
# MEASURED duty -> steady speed (open-loop jog, both directions, 2026-09-17):
#   duty  0.45   0.55   0.65   0.75   0.85   0.97
#   up    12.8   15.9   19.8   23.7   30.1   36.1  mm/s
#   down   6.3   17.9   31.5   33.0   33.0   37.5  mm/s
# Speed is roughly LINEAR in duty all the way to 0.97 — the motor is nowhere near its
# no-load speed, so every point of duty is worth real speed (the last 12 % is worth
# 20 % more speed going up). An earlier note here claimed the opposite, that 0.90 duty
# already gave 99 % of the speed; that was wrong, from comparing a closed-loop velocity
# PEAK (a noisy estimate) against an open-loop STEADY value. Do not cap this for
# thermal headroom without re-measuring — it costs speed proportionally.
MAX_EFFORT = 1.00

# ----------------------------------------------------------------------------
# Quadrature encoder (3V3)
# ----------------------------------------------------------------------------
# MABEL keeps phase A on GP27; NOVA moved it to GP22 to free ADC1 for L_IS. Per-board
# wiring, not a property of the shared mechanism — this line is expected to differ.
ENC_A_PIN = 27        # GP27 (phys 32) -> encoder phase A
ENC_B_PIN = 28        # GP28 (phys 34) -> encoder phase B

# Counts -> millimetres of travel. This is only the INITIAL/default scale; once
# you calibrate, the firmware measures it exactly (LIFT_TRAVEL_MM / count span)
# and persists it. Measure roughly for your leadscrew/gearbox.
MM_PER_COUNT = 0.20018

# Non-volatile state file on the Pico flash (survives power-off): stores the
# measured scale, the travel limits, and the last known absolute height so the
# lift remembers where it is across a power cycle.
STATE_FILE = "lift_state.json"

# True quadrature: read phase B on each phase-A edge to get the REAL direction
# of travel, so position is absolute and encoder-only — immune to coasting,
# back-driving, reversals, or controller windup. Requires phase B (GP28) wired.
# Set False only if B is unavailable (falls back to commanded-direction, which
# can drift). Default True.
USE_QUADRATURE = True
# Flip to -1 if driving "up" makes the count/height DECREASE (i.e. going DOWN
# made the height go UP). Set to -1 for this lift's encoder/B-phase wiring.
ENCODER_SIGN = -1

# ----------------------------------------------------------------------------
# Physical travel geometry (the hard stops are fixed in hardware)
# ----------------------------------------------------------------------------
# Height reported at the fully-retracted bottom stop (the home datum).
LIFT_MIN_HEIGHT_MM = 317.5          # 0.3175 m
# Total mechanical stroke between the bottom and top hard stops.
LIFT_TRAVEL_MM     = 635.0          # 0.635 m
# Height at the fully-extended top stop = min + stroke (= 0.9525 m).
LIFT_MAX_HEIGHT_MM = LIFT_MIN_HEIGHT_MM + LIFT_TRAVEL_MM

# Height assigned to the floor when 'home' finishes (the bottom datum).
HOME_HEIGHT_MM = LIFT_MIN_HEIGHT_MM

# The RESTING pose — where the column sits when nothing else asks for a height.
# Mid-stroke, so a bring-up never dives into either hard stop. This is a target,
# NOT a datum: 'home'/'calibrate' still stall-seek the FLOOR, because absolute
# position can only be recovered from a physical stop.
HOME_POSE_MM = LIFT_MIN_HEIGHT_MM + LIFT_TRAVEL_MM / 2.0     # 635.0 mm

# Exponential moving-average factor for the velocity estimate (0..1).
# Higher = more responsive but noisier. Lower = smoother but laggy.
VEL_FILTER_ALPHA = 0.35

# ----------------------------------------------------------------------------
# Inner loop: PI velocity controller (regulates effort to hold a target mm/s)
# ----------------------------------------------------------------------------
# Feedforward: nominal effort per mm/s. A DC motor's speed is ~linear in duty,
# so this does most of the work; the PI terms only trim the residual error.
# Feedforward = the INVERSE PLANT GAIN, so it has to match the measured motor.
# NOVA's column does 37.0 mm/s at 100 % duty (see MAX_VELOCITY_MM_S above), so one
# unit of effort buys 37 mm/s and the feedforward is 1/37 = 0.027 effort per mm/s.
#
# This was 0.0090, inherited from MABEL, which implies full duty = 111 mm/s — three
# times NOVA's column. The symptom was subtle: with the old inflated vmax of 60 the
# FF term came out near the right VALUE by accident (0.009*60 = 0.54) and the loop
# reached 41 mm/s. Setting an honest vmax of 35 exposed it — FF dropped to 0.315,
# the PI trim plus the 0.22 anti-windup clamp could only add ~0.3, and the loop sat
# in equilibrium at 0.63 duty / 22 mm/s, unable to reach its own 35 mm/s setpoint.
VEL_FF       = 0.0270    # effort per (mm/s) = 1 / 37 mm/s measured at full duty
# Kp and Ki scale with the same plant gain (x3 from MABEL's values), so the trim
# keeps comparable authority now that the feedforward does most of the work.
VEL_KP       = 0.0180    # effort per (mm/s) of error
VEL_KI       = 0.0450    # effort per (mm/s * s) of accumulated error
VEL_I_LIMIT  = 0.25      # anti-windup clamp: also caps how hard a jam is pushed

# ----------------------------------------------------------------------------
# Position control mode
# ----------------------------------------------------------------------------
# BANG_BANG: drive at MAX_EFFORT toward the target and cut power inside STOP_BAND.
# This is the mode NOVA runs, by request: "just do 100 % duty going up and down".
#
# It suits this column specifically. Speed is roughly LINEAR in duty (measured:
# 0.65 -> 19.8 mm/s, 0.85 -> 30.1, 0.97 -> 36.1 going up), so anything less than full
# duty is simply slower, and the leadscrew is NOT back-drivable, so cutting power
# stops the column almost immediately instead of coasting. The cascaded
# profile+PI loop bought smoothness this mechanism does not need and cost speed.
#
# Set False to fall back to the trapezoidal profile + PI velocity loop below.
BANG_BANG = True
# Bang-bang alone chatters at the target: the column arrives at FULL speed, overshoots
# the stop band, and the next command drives it back at full duty — a limit cycle of
# incremental steps, each with a current spike. That is what the operator saw when
# pressing Home. Two things fix it, and both are needed.
#
# 1. TAPER. Inside TAPER_MM of the target, effort ramps down from MAX_EFFORT to
#    TAPER_MIN_EFFORT, so the column arrives slowly instead of slamming into the band.
#    Speed is ~linear in duty here, so 0.35 duty is roughly 8 mm/s — a gentle approach
#    whose stopping distance is well inside the band.
# 2. HYSTERESIS. Once stopped, the error must exceed RESTART_BAND_MM before driving
#    again. Without it, settling 1.6 mm from a 1.5 mm band is enough to re-trigger.
# TAPER_MM was 25.0, which crippled SHORT moves: a 18 mm correction lies entirely
# inside the taper, so the whole move ran at reduced effort and lagged 8-19 mm behind
# the glide. 10 mm keeps full duty for the bulk of any move while still braking the
# final approach. TAPER_MIN_EFFORT 0.45 is ~12 mm/s — slow enough that the stopping
# distance stays well inside the stop band, fast enough to beat stiction.
TAPER_MM = 10.0
TAPER_MIN_EFFORT = 0.45
# RESTART_BAND was 4.0, and since errors under it are never corrected that WAS the
# steady-state error (measured +4.75 mm). 2.5 mm against a 1.5 mm stop band leaves
# 1 mm of hysteresis — enough to break the limit cycle, because the tapered approach
# arrives at ~12 mm/s and overshoots the band edge by well under half a millimetre.
RESTART_BAND_MM = 2.5
# Distance from the target at which power is cut. Must cover the stopping distance
# at full speed; the leadscrew's own friction does the rest. Measure it: command a
# move and read the final error. Too small overshoots, too large stops short.
STOP_BAND_MM = 1.5

# ----------------------------------------------------------------------------
# Outer loop: trapezoidal motion profile (turns a position goal into a vel cmd)
# ----------------------------------------------------------------------------
# MEASURED CEILING — the single most important number for NOVA's lift.
# Full 635 mm stroke at 100 % duty, open loop, 2 trials each direction (2026-09-17):
#   UP    steady 37.0 mm/s (36.8, 37.3), peak 41.8, full-traverse average 34.4
#   DOWN  steady 36.4 mm/s (35.0, 37.7), peak 42.0, full-traverse average 33.7
# Down/up = 0.98 — effectively SYMMETRIC, because the leadscrew is not
# back-drivable, so gravity barely changes the speed. A full stroke takes ~18.3 s.
#
# 35 mm/s is the design limit: below the 36.4 mm/s slower direction, so the
# velocity loop keeps authority to hold it under load instead of saturating.
# This is the profile's honest cruise ceiling, and enforcing it HERE is what makes
# it impossible for any animation, policy or studio to command a speed the column
# cannot reach. It was 60.0, which promised 71 % more than the motor can deliver.
# Set ABOVE the plant's measured ceiling (36.1 mm/s up / 37.5 down steady at full
# duty) so the velocity loop always saturates on a large move and the column runs as
# fast as it physically can. The trapezoid still decelerates correctly: brake_v is
# derived from the distance remaining, not from this number. Measured overshoot with
# this arrangement is under 1 mm.
#
# 35.0 remains the AUTHORING limit for animations (motions.js LIFT_MAX_VEL) — a
# conservative, round figure just under the slower direction. This constant governs
# the hardware, which should never be the thing holding the column back.
MAX_VELOCITY_MM_S = 40.0    # cruise speed ceiling for position moves
MAX_ACCEL_MM_S2   = 200.0   # ramp rate for the velocity setpoint
POSITION_TOL_MM   = 0.5     # "arrived" window
ARRIVE_VEL_MM_S   = 2.0     # must also be slower than this to declare arrival

# ----------------------------------------------------------------------------
# Current sensing (BTS7960 R_IS -> GP26 / ADC0)
# ----------------------------------------------------------------------------
# The BTS7960 mirrors its load current out of the IS pin at roughly I_load / 8500
# (the datasheet's differential sense ratio dk). A resistor from IS to ground turns
# that into a voltage the Pico's ADC can read.
#
# IMPORTANT: R_IS is the sense for the RIGHT half-bridge, which this module drives
# from RPWM — i.e. UP. Driving DOWN uses LPWM and L_IS, so with only R_IS wired the
# current reading is valid for UPWARD motion and reads ~0 going down. To sense both,
# L_IS needs its own ADC; the Pico's other ADC pins are GP27/GP28, currently the
# encoder, so that means moving the encoder to two other GPIOs first.
CURRENT_SENSE = True
CURRENT_SENSE_PIN = 26          # GP26 = ADC0 -> R_IS (the UP half-bridge)
# Second channel for the DOWN half-bridge. None = not wired, and load reads None while
# descending. To enable it: move the encoder off the ADC pins and put L_IS on GP27.
#   encoder A  GP27 -> GP14 (phys 19)     ENC_A_PIN below
#   encoder B  GP28 -> GP15 (phys 20)     ENC_B_PIN below
#   L_IS       new  -> GP27 (phys 32, ADC1)
# Any GPIO serves the encoder — it only needs interrupt capability — whereas only
# GP26/27/28 can do analogue, which is why the encoder is the thing that moves.
CURRENT_SENSE_PIN_DOWN = None   # set to 27 once L_IS is wired there
# Sense resistor from IS to GND, in ohms. MUST match the hardware — the amp figure
# scales directly off it. 1k with the 1/8500 ratio puts 10 A at ~1.18 V.
# Set to 0 if the module carries its own resistor and the value is unknown; the raw
# ADC count is still published and is usable as a relative load signal.
CURRENT_SENSE_R = 1000.0
CURRENT_SENSE_RATIO = 8500.0    # I_load / I_IS
CURRENT_FILTER_ALPHA = 0.20     # EMA on the ADC; the IS pin is noisy under 20 kHz PWM
ADC_VREF = 3.3

# BASELINE: median current while running free UPWARD at each duty, measured
# 2026-09-17 **with the upper body mounted** — i.e. the robot as it normally stands.
# So load_a reads ZERO for the robot carrying nothing extra (confirmed: mean -0.076 A
# over five identical runs) and positive for anything ADDED. Re-measure this table if
# the upper body's mass changes. Extra load shows up as the DEVIATION from this, which
# is the quantity that matters — it does not need an absolute force calibration, and it
# is unaffected by whatever the sense resistor actually is.
#
# DUTY-CORRECTED (see Motor.amps). The first table here was measured without the
# correction and was contaminated by duty itself; these are the real figures.
#   duty  0.55   0.65   0.75   0.85   1.00
#   amps  6.21   5.86   8.90  10.79  11.55
#   count 14510  13680  20801  25216  26986      (speed 16.2 21.1 23.7 33.5 37.4 mm/s)
#
# What this data does NOT support: extracting an absolute weight. The hope was that low
# duty would show a weight-dominated plateau with friction only appearing at speed, but
# the low-duty points spread 43.6 % (14510 / 13680 / 20801) against a run-to-run noise
# of only ~1464 counts — so the variation is real and friction scales with speed across
# the whole range. There is no weight-only regime to exploit.
#
# Fitting V*duty = I*R + k*v fails for the same underlying reason: current and speed are
# collinear over a duty sweep, the fit is ill-conditioned, and it returns a NEGATIVE
# resistance. An absolute N/A constant needs either one known mass or the up-vs-down
# current difference at matched speed (which needs L_IS on a second ADC).
#
# What it DOES support, and what load_a is for: at a FIXED duty the reading is
# repeatable to ~1464 counts, so the CHANGE tracks added load. Sensitivity is at most
# ~933 counts/kg (optimistic — that figure assumes the whole low-duty reading is the
# ~17.5 kg upper body when much of it is friction), giving a realistic resolution of
# about 2-3 kg.
CURRENT_BASE_UP = ((0.55, 6.21), (0.65, 5.86), (0.75, 8.90), (0.85, 10.79), (1.00, 11.55))
# Newtons per amp of DEVIATION from the baseline. None until calibrated with a known
# mass; while None, load is reported in amps only.
LOAD_N_PER_A = None

# ----------------------------------------------------------------------------
# Contact / external-load detection
# ----------------------------------------------------------------------------
# The column has no force sensor, but it does have a MEASURED speed-vs-duty curve,
# and that is enough: at a known duty the free-running speed is known, so a speed
# DEFICIT is external load. Measured free speed (open-loop jog, 2026-09-17):
#   duty  0.45   0.55   0.65   0.75   0.85   0.97
#   up    12.8   15.9   19.8   23.7   30.1   36.1  mm/s
#   down   6.3   17.9   31.5   33.0   33.0   37.5  mm/s
# Both directions fit v_free ~= 46*duty - 7.6 above the stiction regime (duty >= 0.5).
# (Down is SLOWER than up at 0.45 — consistent with a gas-spring assist helping the
# column rise, which is normal for a standing-desk column.)
# Detection is REPORT-ONLY by default. The robot may legitimately be carrying an
# object or a weight, and extra load is information, not a reason to stop. Set
# CONTACT_STOP True only if you want contact to cut power.
CONTACT_DETECT = True
CONTACT_STOP = False
CONTACT_V_SLOPE = 46.0      # mm/s per unit duty
CONTACT_V_OFFSET = -7.6     # mm/s
# Contact is declared only once the column has PROVEN it was running free — reaching
# CONTACT_ARM_FRAC of the predicted speed — and then falls under CONTACT_TRIP_FRAC for
# CONTACT_TRIP_MS. Requiring the free-running proof first is what keeps acceleration
# from rest, and the stiction regime, from looking like a collision.
CONTACT_ARM_FRAC = 0.70
CONTACT_TRIP_FRAC = 0.55
CONTACT_TRIP_MS = 120
CONTACT_MIN_EFFORT = 0.50   # below this the free-speed model is not trustworthy

# ----------------------------------------------------------------------------
# Soft limits & stall-based calibration (no physical limit switches)
# ----------------------------------------------------------------------------
LIMIT_ENABLED    = True
GRACE_MS         = 100   # ignore stall detection for this long after a new move
NO_TICK_LIMIT_MS = 80    # once moving, no encoder tick this long = jam/end stop
                         # (short => cuts power fast on a jam, minimal flex)
# Any commanded motion needs encoder feedback. If we drive but the encoder never
# ticks at all within this window, treat it as a wiring/power fault and stop
# (protects the motor from grinding into a stop).
#
# This applies to the open-loop jog (up/down/gpio) TOO. It used to be exempt, and
# that is a runaway: with a dead encoder, 'up' drove at effort 1.0 forever, with
# telemetry frozen at count=0, until the firmware was killed by hand. A jog with no
# feedback now cuts out after MANUAL_NO_MOTION_FAULT_MS. Set 'limit off' to disable
# every stall/limit cut when you deliberately need to jog a broken encoder.
# 600 ms, not 2500. At full duty the column reaches 30+ mm/s quickly and the encoder
# resolves 0.2376 mm/count, so the FIRST tick arrives within ~10-20 ms of real motion —
# 600 ms is already generous for a slow start. The old 2500 ms meant that starting a
# move while sitting ON a hard stop ground the motor at full duty for two and a half
# seconds before the fault fired, which is exactly the behaviour that has to stop.
NO_MOTION_FAULT_MS = 600
# Longer for the jog: it is the tool you reach for when something is already wrong,
# and it is the only way to drive a column that has not been calibrated yet.
MANUAL_NO_MOTION_FAULT_MS = 600
# Calibration / homing drive OPEN-LOOP at a fixed effort (like the original
# firmware and like manual up/down) — robust and not dependent on velocity-loop
# tuning. 0.0-1.0 PWM duty. The UP sweep is near-full to "jam-bust" free of the
# bottom dead-stop (the original used 100%). Lower if it slams the stops.
CALIB_DOWN_EFFORT = 0.80   # seek the floor (fast)
CALIB_UP_EFFORT   = 1.00   # seek the ceiling (full = break free from the floor)

# Stall detection is more forgiving DURING a calibration/home sweep so a slow
# patch can't false-trip it; the fast NO_TICK_LIMIT_MS above is only for normal
# moves (quick jam cut). These apply only in CALIB/HOME modes.
CALIB_GRACE_MS         = 400
CALIB_NO_TICK_LIMIT_MS = 300

# Resting height for the "park" command. Was 50.0 mm, which is BELOW the 317.5 mm
# bottom stop, so every park clamped to the floor and drove the column all the way
# down. Mid-stroke instead.
DEFAULT_PARK_MM  = HOME_POSE_MM

# ----------------------------------------------------------------------------
# Loop timing & telemetry
# ----------------------------------------------------------------------------
CONTROL_HZ   = 100       # closed-loop update rate
# Telemetry rate. 50 Hz matches the arms and the gateway's control rate, so a host
# streaming setpoints at 50 Hz sees fresh feedback for every frame it sends.
# 'rate <hz>' retunes it live; the protocol clamps to 1-50.
TELEMETRY_HZ = 50
