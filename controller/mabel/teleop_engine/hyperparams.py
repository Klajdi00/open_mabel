"""ALL tunable hyperparameters for the Vision Pro -> MABEL teleop, in one place.
Edit here to tune the system; every module reads these."""
import os
import numpy as np


ZMQ_SUB_SRC = "tcp://localhost:5556"   # from vision_pro_stream.py (PUB)
ZMQ_CMD_SRC = "tcp://localhost:5558"   # optional reset/stop/start commands
ZMQ_PUB_PORT = 5555                    # solved targets -> controller / robot

# Package home is controller/mabel/teleop_engine; the scene tree lives under
# the repo's simulation/ dir (3 levels up from this file's directory).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
XML_FILE = os.path.join(_REPO_ROOT, "simulation", "mabel_mujoco",
                        "scenes", "mabel_sim.xml")

# avp_stream / vision_pro_stream publishes RAW ARKit (Y-up). This rotates an
# ARKit vector into the robot frame (x=forward, y=left, z=up). Matches
# mabel.config.AVP_TO_ROBOT — the single source of truth for the app frame.
AVP_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])
# Head "forward" in the ARKit head frame (the gaze axis); -z, like the headset.
AVP_GAZE_LOCAL = np.array([0.0, 0.0, -1.0])

# Per-finger scale of the operator's keypoints in the canonical palm frame
# (human hand is larger / proportioned differently than the ORCA hand). Tune
# here, nowhere else. Keys are the finger group.
SCALES = {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}

# Finger-IK cost weights (cosine = shape/orientation, relative = pinch geometry).
W_COSINE, W_RELATIVE, W_SMOOTH, W_REG = 20.0, 150.0, 0.5, 0.02

# Arm DLS-IK tuning.
# --- Frames -----------------------------------------------------------------
# ROBOT base_link = the chassis body (moves + yaws with the mobile base); world =
# odom. HUMAN base frame is derived from the Vision Pro head (see BodyFrame):
# origin = live head (walking cancels), heading = operator yaw captured at
# calibration (head look-around does NOT move the arms — that is the neck channel).
# Every arm/hand target is built in these base frames, then placed into the world.
ROBOT_BASE_BODY = "mabel_chassis"
# REACH_SCALE = robot reach / human reach, AUTO-CALIBRATED at startup (operator
# extends arms forward, press Enter): the human head->wrist extension is measured
# from the stream and compared to the robot's URDF head->wrist reach, so the robot
# always covers the operator's workspace. This is the fallback if calibration is
# skipped.
REACH_SCALE = 1.2
# The MABEL chassis frame has its FRONT along -x (front camera / gaze at -x), so the
# robot body frame (x=fwd, y=left, z=up) is the chassis rotated 180° about z. The
# operator body-frame coords are mapped through this so "operator forward" lands in
# FRONT of the robot (not behind) and right/left are not crossed.
R_BASE_ALIGN = np.diag([-1.0, -1.0, 1.0])

# NECK DATUM: latched (default) vs live. The neck's zero is the operator's
# torso frame; taking it from the body-frame estimator EVERY FRAME made it a
# live hand-derived quantity, so hand motion and estimator noise moved the neck
# with a stationary head ("the neck acts super erratic even when I am
# stationary", 2026-07-31). Latched keeps the gravity-aligned datum — the reason
# the torso-relative neck was introduced — and re-latches on recalibrate / Set
# Forward. =live restores the per-frame behaviour for A/B.
NECK_DATUM_LIVE = os.environ.get("MABEL_NECK_DATUM", "latched").lower() == "live"

# --- ARM MAPPING ---------------------------------------------------------------
# "gmr" (default, 2026-07-24): the ROOT-UNIFORM SCALED-KEYPOINT map of
# Retargeting Matters / GMR (Araujo, Ze, Xu, Wu, Liu — arXiv:2510.02252,
# ICRA 2026), adapted to the sparse headset stream. All operator keypoints
# (wrists, synthesized elbows) are expressed relative to ONE root — the
# mid-shoulder point — and scaled by the per-side arm-length ratio
# sigma = L_rob / L_hum before being re-anchored at the ROBOT's root:
#     t = r_rob + sigma · M · (p − r_hum).
# One similarity transform for the whole upper body ⇒ the head–wrist–wrist
# TRIANGLE (the pose shape) is preserved by construction — unlike the
# per-shoulder "extension" map, whose anchors sit on the robot's 3×-narrower
# shoulders and geometrically collapse inward-angled bimanual poses (that map
# needed an anti-collapse floor to hide it). Near full extension the target
# magnitude is RENORMALIZED about the robot's own shoulder to rho·L_rob·od
# (smoothstep-gated on rho ≥ ARM_GMR_EXT_GATE), so "full human extension =
# full robot extension" — the extension map's one guarantee — is kept exactly.
# MABEL_ARM_MAP=extension restores the per-shoulder map; =head the legacy map.
ARM_MAP = os.environ.get("MABEL_ARM_MAP", "gmr").lower()
# Extension-renormalization gate: below this human extension fraction the GMR
# map is pure (pose-shape-preserving); above it the target magnitude blends to
# rho·L_rob·od(rho) about the robot shoulder (full extension exact).
ARM_GMR_EXT_GATE = float(os.environ.get("MABEL_ARM_GMR_EXT_GATE", "0.85"))
# Slight radial overdrive so rho = 1 pushes the target marginally past the reach
# boundary: the position term then dominates the posture regularizers and the
# elbow straightens fully instead of holding a few residual degrees of bend.
ARM_EXT_OVERDRIVE = float(os.environ.get("MABEL_ARM_EXT_OVERDRIVE", "1.04"))
# SHAPE GATE (2026-10-18). GMR keeps the target's DIRECTION about the robot shoulder from the root-uniform map,
# and the robot's shoulders are 3x narrower than a human's, so a straight human arm came out splayed outward:
# 16-24 deg of direction error forward / up / down / diagonals, ~2 deg only for a T-pose
# (retarget_lab/experiments/exp_arm_shape.py). An extended arm's SHAPE is its shoulder->wrist direction, so the
# direction blends from GMR's to the operator's own (through the body map M) as rho rises from LO to HI
# (smoothstep). Mid-workspace manipulation (rho < LO) stays pure GMR — it is the only map that keeps the two
# hands' scaled separation (test_gmr_scale_map: a 0.55 gate dropped the rho-0.75 separation ratio to 0.81) — and
# the bimanual blend and the anti-collapse floor still run after. Straight arms with it: robot direction error
# 1.5-8.8 deg (was 7.6-21). MABEL_ARM_SHAPE_GATE=0 turns it off.
ARM_SHAPE_GATE = os.environ.get("MABEL_ARM_SHAPE_GATE", "1").lower() not in ("0", "false", "no")
ARM_SHAPE_GATE_LO = float(os.environ.get("MABEL_ARM_SHAPE_GATE_LO", "0.70"))
ARM_SHAPE_GATE_HI = float(os.environ.get("MABEL_ARM_SHAPE_GATE_HI", "0.95"))
ARM_EXT_RHO_MAX   = 1.0   # extension fraction clamp (hyperextension saturates)
# DIRECTIONAL reach for the extension-exact blend (2026-08-01). L_rob was ONE
# scalar measured on forward-ish rays, so full human extension asked for that
# same distance in every direction — and MABEL reaches 0.65 m forward but 0.77 m
# laterally. A T-pose therefore requested 0.70 m into a direction good for
# 0.77 m and the arm answered with 46 deg of elbow bend: a pure artefact of the
# single number, not a limit of the arm. Sampling the reachable sphere once at
# construction (64 directions, ~0.6 s/side) makes "full human extension = full
# robot extension" true in EVERY direction, which is what the operator means by
# the robot matching their shape. =0 restores the scalar.
# SYMMETRIC REACH (2026-10-27, operator: "the left and right should be symmetrical"). The scalar L_rob is measured
# per side from the arm-MOUNT body, and the canonical MJCF is not mirrored there: the right mount sits 6 cm from its
# shoulder joint, the left one on it, and the left elbow's upper limit (+1.542) stops 0.17 rad short of its straight
# angle (+1.710; the right's -1.440 is inside its range) — so R 0.669 / L 0.660 m from links that are identical
# (shoulder->wrist straight 0.6539 m both sides). Both sides take the SMALLER value, so sigma = L_rob/L_hum is the
# same for both arms and full human extension never asks the shorter side for more than it has (the 4 % overdrive
# still straightens the longer one). The directional tables are per side. MABEL_ARM_REACH_SYMMETRIC=0 = per side.
ARM_REACH_SYMMETRIC = os.environ.get("MABEL_ARM_REACH_SYMMETRIC", "1").lower() not in ("0", "false", "no")
ARM_REACH_DIRECTIONAL = os.environ.get("MABEL_ARM_REACH_DIR", "1").lower() \
    not in ("0", "false", "no")

# Bimanual proximity correction: per-shoulder anchoring is exact per arm but the
# two hands only meet exactly if the robot's shoulder-width/arm-length ratio
# matches the operator's. When the operator's wrists come CLOSE to each other,
# blend in a symmetric correction that sets the robot inter-wrist vector to the
# operator's 1:1 (about the pair's midpoint), so touching hands touch. Fades in
# between NEAR (start) and FULL (fully corrected); MABEL_ARM_BIMANUAL=0 disables.
ARM_BIMANUAL_BLEND = os.environ.get("MABEL_ARM_BIMANUAL", "1").lower() not in ("0", "false", "no")
ARM_BIMANUAL_NEAR = 0.35  # m; operator inter-wrist distance where correction starts
ARM_BIMANUAL_FULL = 0.12  # m; …and where it is fully applied
# GLOBAL (body-frame) mapping — 2026-07-22, from a live operator report ("walk
# around, come back facing forward, the retargeting fails to reproduce the
# pose"). The wrist-from-shoulder DIRECTION, the palm orientation axes and the
# synthesized elbow are decomposed in the OPERATOR BODY frame (right/up/fwd
# from the body-frame estimator — wrist-consensus-grounded, so it does not
# drift) and reconstructed in the ROBOT'S CHASSIS body frame, instead of the
# fixed ARKit-world AVP_TO_ROBOT rotation. The map then depends ONLY on
# body-relative geometry: any world position/heading of the operator (and any
# chassis heading of the robot) produces the same robot config for the same
# body-relative pose. MABEL_ARM_BODY_FRAME=0 restores the world-fixed map.
ARM_BODY_FRAME_MAP = os.environ.get("MABEL_ARM_BODY_FRAME", "1").lower() \
    not in ("0", "false", "no")
# ANTI-COLLAPSE wrist-separation floor (beyond the near-contact 1:1 blend):
# the robot inter-wrist distance is floored at min(HAND_CLEAR, the operator's
# own metric separation) — absolute, so a one-arm task with the other arm
# hanging is never touched, yet clearly-separated human wrists can never
# produce overlapping robot wrists, and touching hands still touch (the floor
# fades to the operator's 1:1 separation near contact). ~2 palm widths.
ARM_HAND_CLEAR = float(os.environ.get("MABEL_ARM_HAND_CLEAR", "0.18"))
# Synthesized ELBOW hint for headset clients (extension map only): ARKit tracks
# no elbow, so the arm IK's elbow-direction cosine (ARM_W_ELBOW) gets a
# two-link elbow computed from the estimated shoulder + wrist + HAND ROLL —
# stabilises the redundant arm's configuration (no flip-flops near shoulder-
# singular targets) and shapes it like the operator's arm. Tracked elbows
# (webcam clients) always win. MABEL_ARM_SYNTH_ELBOW=0 disables.
ARM_SYNTH_ELBOW = os.environ.get("MABEL_ARM_SYNTH_ELBOW", "1").lower() \
    not in ("0", "false", "no")
# TRACKED elbow for headset clients (2026-07-29): consume ARKit's forearm
# joint DIRECTION-ONLY (wrist→elbow line re-anchored at the anatomical forearm
# length) as the arm IK's elbow hint — the operator's real forearm, so the
# robot's elbow bends like the human's (T palms-down → elbow forward,
# palms-up → elbow up, inward flex → elbow down-forward). The old
# hand-anchored POSITION is never used (the 2026-07-11 left-arm-twist defect).
# =0 restores null-and-synthesize (the gravity/palm-roll model below).
ARM_TRACK_ELBOW = os.environ.get("MABEL_ARM_TRACK_ELBOW", "1").lower() \
    not in ("0", "false", "no")
# EMA on the reconstructed elbow point (per NEW frame, ~30 Hz): hand-tracking
# wobble rolls the extrapolated forearm slightly frame to frame; 0.25 settles
# a step in ~0.4 s without visible lag on deliberate elbow motion.
ARM_ELBOW_TRACK_EMA = float(os.environ.get("MABEL_ARM_ELBOW_TRACK_EMA", "0.25"))
# Bend-proportional elbow-hint strength (tracked-elbow path): full ARM_W_ELBOW
# at an operator elbow flexion of BEND_REF rad and above, tapering to W_MIN
# for a straight arm — a straight arm's elbow direction is geometrically weak
# and a full-weight hint measurably traded wrist-position accuracy for it.
ARM_ELBOW_BEND_REF = float(os.environ.get("MABEL_ARM_ELBOW_BEND_REF", "0.6"))
# 0.15 until 2026-08-01. The de-weighting existed because a full-weight elbow
# hint on a near-straight arm "measurably traded wrist-position accuracy for it
# (T-pose errors tripled)" — but that was measured against a target placed by
# ONE direction-independent L_rob, i.e. in the wrong place for a lateral reach.
# The elbow hint was not fighting the wrist, it was fighting a bad target.
# With ARM_REACH_DIRECTIONAL the two agree, and re-measuring the same trade
# finds no cost at all: bimanual tracking |r| 0.999/0.990 -> 0.999/0.988 and the
# solve time unchanged, while the T-pose elbow goes 148.4 -> 175.4 deg (its
# kinematic best is 173.8). So the arm now straightens when the operator's arm
# is straight, which is what "match the shape" asks for.
# 2026-08-23: NOTE THAT AT 1.0 THIS KNOB IS INERT. The live scale is
# clip(bend/BEND_REF, W_MIN, 1.0), so W_MIN = 1.0 collapses it to clip(x, 1, 1) —
# a constant 1.0 at every elbow bend, i.e. the bend-proportional de-weighting has
# not run since 2026-08-01. That is left as it is, deliberately: the 08-01 change
# was made to fix the T-pose and it did, and the reason a de-weighting was wanted
# in the first place — a full-weight elbow hint costing palm/wrist accuracy — is
# now addressed at the source by ARM_W_ELBOW = 15 rather than by a schedule.
# Raising this above 1.0 does nothing; lowering it re-arms the taper.
ARM_ELBOW_W_MIN = float(os.environ.get("MABEL_ARM_ELBOW_W_MIN", "1.0"))
# Upper-arm standoff from the body (engine._retarget, 2026-10-16): the elbow hint is
# shifted this far OUTWARD (m, horizontal, away from the other shoulder) — a
# direction-only abduction bias so the upper arm does not hug the torso. 0.15 from the
# E9 sweep (web_gui/retarget_lab/experiments/exp_elbow_body.py): elbow-torso p50 75 -> 137 mm,
# upper arm within 20 mm of the torso 9.0 % -> 0.5 % of frames, wrist tracking p50 32.6 -> 28.8 mm.
# 0 = the operator's elbow direction copied unchanged (before 2026-10-16).
ARM_ELBOW_ABDUCT = float(os.environ.get("MABEL_ARM_ELBOW_ABDUCT", "0.15"))
# RELATIVE INCREMENT map (TeleopEngine.rel_track, Retarget Lab variant "rel", 2026-10-23): the robot's tracked point
# moves by REL_SCALE x the operator's wrist increments (world frame, through the body-heading map; 1 = metres 1:1)
# and turns by the wrist's rotation increments; a hand untracked for more than REL_RELATCH_GAP s re-latches at the
# robot's current pose when it comes back (no jump).
REL_SCALE = float(os.environ.get("MABEL_REL_SCALE", "1.0"))
REL_RELATCH_GAP = float(os.environ.get("MABEL_REL_RELATCH_GAP", "0.25"))
REL_REACH = float(os.environ.get("MABEL_REL_REACH", "1.05"))       # anti-windup radius, x the arm's reach
# Swivel-from-palm-roll (geometry.synth_elbow): raising the elbow rotates the
# humerus and the whole distal arm rolls with it, so the palm's roll about the
# arm axis carries the swivel — while pronation/supination can roll the palm
# ±~60° with NO elbow motion. The measured roll maps through that DEADBAND
# (rad), then 1:1, clamped to [MIN, MAX] about gravity-down (0 = elbow down,
# +π/2 = toward the body-tangent, ±π = elbow up).
SWIVEL_DEADBAND = float(os.environ.get("MABEL_SWIVEL_DEADBAND", "0.45"))  # ~26°
# 1.05 (~60°) until 2026-07-24: it hid EVERY moderate elbow raise (the elbow
# stayed gravity-down below 60° of palm roll — "the arms don't bend like
# mine"). A human forearm pronates the palm ~±25° with the elbow truly still
# once the hand is already palm-down mid-reach; 0.45 covers that slack while
# letting deliberate raises through (validated by the E7 sweep + the new
# pronation false-positive check).
# Symmetric swivel clamp: elbow-up is toward the OUTWARD tangent, whose sign
# is chiral (right arm: negative φ in the down/tangent basis) — the inward
# side is bounded by the torso-clearance clamp, not by this range.
SWIVEL_MAX_ABS = 2.6     # rad; up-and-over limit (~149°) either way
# Torso-clearance clamp on the hint: the elbow's body-frame lateral offset may
# not cross INWARD past the shoulder by more than this fraction of the upper
# arm (an inward elbow on a cross-body reach pressed the upper arm into the
# chest — now also guarded by the l/r_upper↔torso capsules).
SWIVEL_LAT_MIN = 0.15
# Winter / Drillis–Contini anthropometric fallback ratios (fraction of stature H;
# mirror mabel.config.ANTHRO_* — the shoulder estimator uses them only until a
# measured OperatorProfile / live wrist evidence replaces them).
ANTHRO_ARM_RATIO          = 0.332  # shoulder→wrist length / H
ANTHRO_BIACROMIAL         = 0.259  # biacromial shoulder width / H
ANTHRO_ACROMION_BELOW_EYE = 0.118  # eye line → acromion vertical drop / H
ANTHRO_EYE_HEIGHT_RATIO   = 0.936  # eye height / H (stature from the HMD height)

# --- Arm retargeting: COSINE SIMILARITY (same idea as the hand) -------------
# The arm is retargeted by matching, SCALE-INDEPENDENTLY, the DIRECTIONS of the
# operator arm keypoints (head->elbow, head->wrist) and the palm orientation axes
# to the robot's, plus relative-distance (reach) and a per-joint regularization.
# Solved with SLSQP -> smooth, never erratic at singularities (no Jacobian inverse,
# unlike DLS). This is exactly the finger retargeter's approach, lifted to the arm.
# Only the ARM uses REACH_SCALE; the finger optimization is fully DECOUPLED (its own
# per-finger SCALES). With --lock the OpenArm J7 is fixed -> 6-DoF arm + ORCA wrist.
ARM_LOCK_J7  = False     # locked by the --lock CLI flag
# ARM POSITION TARGET = the operator WRIST, not the middle fingertip. Tracking the
# fingertip (h_mats[12]) coupled FINGER FLEXION into ARM motion: closing the hand
# retracts the fingertip, moving the arm target and swinging the whole arm even though
# the operator's wrist never moved. Tracking the wrist (h_mats[0]) makes the arm true
# to the operator's ARM movement and independent of the fingers, while the head-anchored
# reach map still holds each wrist's position relative to the head + the other wrist.
# MABEL_ARM_TRACK_WRIST=0 restores the fingertip-contact mode (precise two-hand-touch).
ARM_TRACK_WRIST = os.environ.get("MABEL_ARM_TRACK_WRIST", "1").lower() not in ("0", "false", "no")
ARM_W_WRIST  = 600.0     # precise WRIST position (tracking + lets the two hands meet)
# 30.0 until 2026-08-23. THE ELBOW TERM OVER-CONSTRAINS THE ARM, and the arithmetic
# says so before any measurement does: 7 arm joints + the ORCA wrist = 8 DoF, and
# position (3) + orientation (3) + an elbow DIRECTION (2) = 8 constraints. No
# slack — so whichever residual is weighted heaviest wins, and the palm loses.
#
# MEASURED two ways over the front grid (exp_front_workspace.py). First, with a
# REALISTIC operator elbow — swivelled 35 deg off the relaxed prior, because an
# operator who is holding their elbow somewhere deliberate is the only case this
# term exists for (right/left):
#
#     w_elbow   poses achieved      elbow vs the COMMANDED elbow
#        0      96.7 / 96.7 %         104 / 115 mm
#       15      82.6 / 89.1           78 / 68            <- taken
#       30      69.6 / 64.1           74 / 50
#
# 15 -> 30 costs 13/25 points of workspace and buys 4/18 mm of elbow. Second, the
# posture check with the natural elbow as the hint — elbow-vs-natural 37.6/36.8 mm
# at w = 0, 18.4/15.3 at 15, 15.4/13.6 at 30: the last step buys 3.0/1.7 mm.
#
# Both say the same thing, and the third column says why: even at full weight the
# robot's elbow lands 50-74 mm from the operator's. It CANNOT follow a 35 deg
# swivel while holding the wrist pose — MABEL's arm proportions are not a human's
# — so the weight was paying a fifth of the workspace for an elbow match that was
# never on offer. 15 keeps the elbow following that IS achievable.
ARM_W_ELBOW  = float(os.environ.get("MABEL_ARM_W_ELBOW", "15.0"))
                         # ELBOW direction FROM THE SHOULDER -> the arm bends like the
                         # human's (this is what stops the redundant arm twisting into
                         # weird configs). Balanced against the wrist so both hold.
                         # KEPT AT 30 (2026-07-24 sweep, exp_retarget_fidelity E9):
                         # raising it does NOT improve elbow-direction fidelity — the
                         # residual is kinematic (robot arm proportions ≠ human), p95
                         # is flat at 65° for W in [30,120] and the mean moves only
                         # 28.1°→25.7°. But W ≥ 45 opens a SECOND equal-cost IK basin
                         # for the redundant arm and the solver limit-cycles between
                         # them (held-pose command std 0→44 mrad, a 0.1 rad/tick
                         # oscillation = the reported "very unstable"). The elbow
                         # HUMAN-LIKENESS lever is SWIVEL_DEADBAND (below) + the GMR
                         # scaled-keypoint map, NOT this weight.
# ELBOW FLEXION ANGLE — match the SHAPE, not just the wrist position
# (operator, 2026-08-01: "when the human does a T shape the robot needs to do a
# T shape as well ... right now the arms are still slightly bent").
#
# The wrist-position term is satisfiable with a bent elbow whenever the target
# is inside the arm's reach along that ray, so tracking the position perfectly
# says nothing about the arm LOOKING like the operator's. Measured before this
# term: at the operator's full forward extension the robot tracked the wrist to
# 0.996 of full reach while holding 17 deg of elbow bend, and in a T-pose it held
# 46 deg.
#
# The existing elbow cue is a DIRECTION cosine, and it is deliberately
# de-weighted exactly where shape matters most: ARM_ELBOW_W_MIN drops it to 15%
# for a near-straight arm, because a straight arm's elbow direction is
# geometrically weak and a full-weight direction hint costs wrist accuracy. That
# reasoning is right, and it is why a direction cue alone can never fix this.
#
# The elbow ANGLE does not degenerate: it is well conditioned at every extension,
# it is exactly the quantity the operator is looking at, and it is available from
# the same reconstructed elbow the direction cue already uses. So this term asks
# the robot elbow to carry the operator's own flexion, straight arm included.
ARM_W_ELBOW_ANGLE = float(os.environ.get("MABEL_ARM_W_ELBOW_ANGLE", "18.0"))
# PALM ORIENTATION, SPLIT (2026-08-23). It was ONE weight on both cosine axes.
# "Palm facing up / down / in / out" is a constraint on the palm NORMAL — two
# degrees of freedom. The third, the SPIN about that normal (which way the
# fingers point inside the palm plane), is a far weaker preference: a human
# reaching to a point palm-up takes whatever finger azimuth is comfortable.
# Weighting them equally over-constrains a 7-DoF arm that also has to hold a
# position, and MABEL has exactly ONE forearm roll (Arm 5, +/-90 deg) to spend.
# Measured over the front manipulation grid
# (controller/experiments/exp_front_workspace.py, W1/W2): splitting them is
# worth tens of percent of achievable palm directions at no cost in position.
# Palm-NORMAL cosine weight — how hard the arm holds the direction the palm
# faces. It was 20 against a position weight of 600: a 30:1 ratio, so the solver
# traded palm angle away for millimetres, which is the operator-visible half of
# "the wrist is the main issue". The offline table weights the same pair 4:1 and
# solves the poses this one misses.
#
# Measured over the front grid, split by what the table proved reachable
# (exp_front_workspace.py, sub-sampled):
#
#   on REACHABLE poses (~76 % of the box)    palm tilt p90   wrist pos p90
#       w_orient = 20  (shipped)                 9.5 deg        12.7 mm
#       w_orient = 60  (taken)                   4.3 deg        14.8 mm
#       w_orient = 120                           2.4 deg        18.8 mm
#
# FEASIBILITY GATE — rejected once, then reinstated when the SCOPE changed, and
# both decisions were right on their own evidence.
#
# The table records how close its own global solve got in each cell, so it knows
# which (position, palm-direction) pairs this arm cannot actually hold. Backing
# the orientation weight off there ought to be free: where the palm angle is not
# deliverable, leaning on it does not deliver it, it only drags the WRIST off the
# point the operator is pointing at. Measured across all FOUR palm directions it
# was not free — it cost ~1 deg of palm angle to buy ~2.5 mm of wrist — so it was
# removed.
#
# What changed is which poses count. Palm-OUTWARD is out of scope (operator,
# 2026-08-23), and it is also where nearly all the infeasibility lives. Measured
# per direction at a held target, the arm's JOINT TRAVEL to reach the pose:
#
#     down 3.1 rad | inward 2.7 | up 3.3 | OUTWARD 5.6      (~2x the others)
#     settled palm error p90: 1.0 / 1.9 / 2.7 deg | OUTWARD 5.3-7.3 deg
#     settled wrist error p90: 28 / 34 / 24 mm   | OUTWARD 44-56 mm
#
# That is the "very erratic" report, and it is not oscillation — the arm settles
# perfectly still (wander over the last second: 0.000 rad). It is a long,
# large-amplitude excursion spent chasing an angle it then fails to hold. The
# gate ought to be what stops the arm paying 5.6 rad of travel for it. MEASURED,
# IT IS NOT, so it is not shipped (ARM_W_ORIENT_MIN defaults to the same value as
# ARM_W_ORIENT, i.e. no gate). With the gate at 12:
#
#     palm outward   travel 5.61 -> 5.57 rad, wrist p90 56.3 -> 56.3 mm   (nothing)
#     palm down      settled palm error p90 1.0 -> 2.6 deg               (worse)
#     palm up                                2.7 -> 3.2 deg               (worse)
#
# The reason is that outward's cost is not the solver over-chasing an angle — the
# arm settles perfectly still there (wander 0.000 rad over the last second). It
# is that palm-outward IS a distant configuration on this arm: one forearm roll
# driven to its stop, reached by a genuinely long path. No reweighting shortens a
# path that kinematics makes long. The three in-scope directions cost 2.2-3.3 rad
# and land within 2 deg; outward costs 5.6 rad and lands 5-9 deg out, and the
# honest answer is that it is out of scope rather than that it is mis-tuned.
ARM_W_ORIENT = float(os.environ.get("MABEL_ARM_W_ORIENT", "60.0"))
ARM_W_ORIENT_MIN = float(os.environ.get("MABEL_ARM_W_ORIENT_MIN",
                                        os.environ.get("MABEL_ARM_W_ORIENT", "60.0")))
# Palm FINGER-AXIS (twist) weight. Default raised 12 -> 60 on 2026-08-26.
#
# At 12 this was a fifth of ARM_W_ORIENT (60), so the solver traded the hand's
# twist ABOUT its palm normal away — correct behaviour, wrong price. The
# operator reported it by eye ("the hand orientation looks weird") after every
# metric here had scored the palm NORMAL only, which is blind by construction to
# rotation about that normal.
#
# Swept over 5 clips including two already-healthy controls, with the 2026-08-25
# target clamps and the mobile-base workspace assist in place:
#   twist p50      13.3 -> 6.6 deg
#   palm normal    4.54 -> 4.02 deg   (slightly BETTER — not traded away)
#   wrist position   37 -> 37 mm      (unchanged)
# Both controls improved too (drawer 7.2 -> 1.0 deg, pour 15.4 -> 5.6) at no
# positional cost, so at 60 this is not a trade. 90 buys only 6.6 -> 6.0 deg and
# starts costing position again. An EARLIER sweep, before the clamps and the
# assist, measured a 1-7 mm position cost for the same change; those two fixes
# absorbed it — which is why this is being raised now and was not then.
ARM_W_ORIENT_FWD = float(os.environ.get("MABEL_ARM_W_ORIENT_FWD", "60.0"))
                         # finger-axis cosine (the spin inside the palm plane)

# ── TABULAR ARM IK (2026-08-23) ───────────────────────────────────────────────
# The live solve is warm-started and LOCAL: right for tracking, wrong for
# choosing which configuration a pose belongs to. The palm flip is where that
# bites — rolling the palm from down to up sweeps Arm 5 across its whole range,
# and because Arm 5 is not collinear with the ORCA-wrist anchor the POSITION
# residual (600) fights the roll while the orientation residual that wants it is
# weighted 20. The solver settles half-flipped with a perfect wrist position, and
# _basin_escape never fires because it only ever watched POSITION error.
#
# So the global structure is precomputed: mabel/teleop_engine/workspace_table.py
# holds a coherent atlas of postures over the front manipulation volume crossed
# with the palm rolls, built offline by a multi-seed global solve under a human
# elbow prior. The live solver looks the pose up, uses it as the posture
# REFERENCE (so "naturally extended" is workspace-aware rather than a pull toward
# the straight-forward qpos0), and STEERS to it through the existing trust region
# whenever the table's configuration is a clearly better answer than the one the
# warm start is sitting in. Continuous, rate-limited, no snap.
#
# MABEL_ARM_TABLE=0 restores the pure local solve.
ARM_USE_TABLE = os.environ.get("MABEL_ARM_TABLE", "1").lower() not in ("0", "false", "no")
# Steer to the table's configuration when its TASK cost (position + orientation
# only — smoothness always favours the incumbent and would veto every switch) is
# below this fraction of the warm solve's. 0.75 = "a clearly better answer",
# which is what a genuine branch change looks like; near-ties are left alone so a
# tracking solve is never disturbed.
ARM_TABLE_MARGIN = float(os.environ.get("MABEL_ARM_TABLE_MARGIN", "0.75"))
# Refinement iterations on the looked-up posture before it is compared: the table
# is sampled on grid points with a canonical finger direction, so it needs a few
# GN steps to become the answer to the ACTUAL command.
ARM_TABLE_REFINE = int(os.environ.get("MABEL_ARM_TABLE_REFINE", "12"))
# Weight of the table posture as the IK's regularisation reference. 1.0 = the
# table fully replaces qpos0 as "where the arm wants to be".
ARM_TABLE_REG = float(os.environ.get("MABEL_ARM_TABLE_REG", "1.0"))
# Temporal smoothness (anti-jitter). It is fair to ask whether this still earns
# its place now that the trust region rate-limits the output anyway. Measured
# over the front grid (right/left):
#
#     W_SMOOTH   poses ok      palm p90     joint path   flips never settled
#        0.5    72.2/67.6 %  8.7/7.7 deg   3.90/3.94     2.8/3.7 %
#        2.0    73.1/69.4    5.9/5.3       4.13/3.78     0.0/3.7      <- kept
#        5.0    73.1/71.3    4.6/5.7       4.42/4.04     0.0/1.9
#
# It does earn it: the trust region bounds how FAR a solve may move, and this
# term is what stops it wandering inside that bound. 5.0 is arguably a hair
# better on accuracy but buys it with 7 % more joint travel per flip, which is
# the opposite of the "minimise the movement of the arms" ask. 2.0 stays.
ARM_W_SMOOTH = float(os.environ.get("MABEL_ARM_W_SMOOTH", "2.0"))
                         # LOWER = SNAPPIER wrist
                         # tracking (less lag), higher = smoother but laggier.
ARM_MAXITER = 35         # SLSQP iterations per frame (the jointly-optimised wrist DoF
                         # converges well, so 35 matches 45's quality and is faster)
# ARM SOLVER: "gn" (default) = damped Gauss–Newton with MuJoCo's ANALYTIC
# Jacobians — same cost, ~0.5 ms/side vs SLSQP's ~7 ms of finite differencing
# (each SLSQP gradient = 9 full-model mj_kinematics; measured 2026-07-22, the
# whole retarget frame dropped 33 ms → <15 ms). "slsqp" = the scipy reference
# (MABEL_ARM_SOLVER=slsqp), kept for parity checks.
ARM_SOLVER = os.environ.get("MABEL_ARM_SOLVER", "gn").lower()
ARM_GN_ITERS = int(os.environ.get("MABEL_ARM_GN_ITERS", "8"))   # early-stop usually exits sooner
ARM_GN_DAMPING = 1e-2
# BASIN ESCAPE (arm.py _basin_escape): a quasi-static target held for
# ESCAPE_FRAMES frames with contact error > ESCAPE_ERR triggers a cold re-solve
# from nominal; a genuinely better basin is steered into through the trust
# region (continuous — no snap). 0 disables.
ARM_ESCAPE_ERR = float(os.environ.get("MABEL_ARM_ESCAPE_ERR", "0.025"))   # m
ARM_ESCAPE_FRAMES = int(os.environ.get("MABEL_ARM_ESCAPE_FRAMES", "12"))
# TRUST REGION: a warm-started solve is bounded to last_q ± a step, so the
# redundant arm can NEVER hop between IK configuration basins frame-to-frame
# (measured: 0.36–0.52 rad/tick shoulder-yaw flips with near-face wrist targets
# — the smoothness term is powerless against a distant basin with equal
# position error). The FIRST solve (cold start) uses the full joint range.
#
# DEFINED PER SECOND, NOT PER FRAME. It was 0.10 rad per SOLVE, justified in its
# own comment as "0.10 rad @ 60 Hz frames = a 6 rad/s per-joint slew — far above
# human teleop speed, so tracking feel is unchanged". The Vision Pro app then
# dropped to 30 Hz (2026-07-19, to kill a SwiftUI invalidation storm) and the
# slew silently halved to 3 rad/s, which IS inside human teleop speed: a 25 cm
# hand jump took 455 ms to travel 63 % of the way, and the arm's peak speed
# measured 2.62 rad/s — the cap exactly (server/tools/exp_retarget_latency.py).
# As a RATE the guarantee is the one the constant always claimed, at whatever
# rate the client happens to stream. The clamp is what actually gets applied per
# solve; ARM_TRUST_MAX bounds it so a long frame gap cannot open the basin door.
# 6.0 until 2026-07-31 ("the re-targeting snaps into the desired position way
# too fast and sometimes oscillates"). MEASURED over a 20 cm wrist step with a
# synthetic ARKit operator (exp_vp_bimanual_latency.py, E3): the peak commanded
# joint speed and the step overshoot fall with the rate while the TRACKING LAG
# does not move at all --
#     rate   peak       overshoot   sweep lag
#     6.0    3.81 rad/s  +39.3 %     -7.6 ms   (old default)
#     4.0    3.81        +39.7 %     -7.2 ms
#     2.5    3.30        +33.4 %     -6.3 ms
#     2.0    2.57        +27.1 %     -7.1 ms   (new default)
#     1.5    2.27        +23.3 %     -3.9 ms
# -- because normal tracking never reaches the limit; only a JUMP does (an
# engage, a re-anchor, a hand coming back into view). So the slam is bought down
# for free, which is why this can be lowered without trading away the operator's
# stated first priority. Above ~4 rad/s the region is not even binding (6.0 and
# 4.0 give the identical 3.81 peak) — the old default was doing nothing except
# on the largest jumps, where it was doing too little.
ARM_TRUST_RATE = float(os.environ.get("MABEL_ARM_TRUST_RATE", "2.0"))   # rad/s
# PER-JOINT trust rate (2026-08-23). One rate for all seven joints is the same
# mistake as one PD gain for all seven: the shoulder carries the whole arm and
# the distal joints carry a hand. The 2.0 rad/s cap above was chosen to stop the
# SHOULDER slamming on a re-anchor, and it is right for that — but it also caps
# the FOREARM ROLL, and a palm flip is a pi-rad sweep of exactly that joint, so
# the cap turned every wrist flip into a 1.6 s crawl (measured, W5). The distal
# joints have ~1/6 the reflected inertia of the shoulder (0.053 vs 0.536 kg m^2,
# measured off the MJCF) and their own +/-7 N.m actuators; letting them slew
# faster costs nothing the operator complained about.
#
# MEASURED (exp_front_workspace.py W5, palm flip down->in->up->out->down over
# the front grid). Settle time, and the joint path actually travelled — the
# second column is the check that a faster cap is not just buying speed with
# overshoot:
#     distal rate   settle med   settle p90   joint path
#        2.0 rad/s    0.63 s       1.60 s      3.99 rad   (uniform = shipped)
#        4.0          0.33         1.03        3.93
#        5.0          0.29         1.05        3.91       <- taken
#        8.0          0.22         1.03        3.99
#   (sub-sampled grid; on the FULL grid the taken value gives settle med 0.30 s,
#    p90 0.63/0.70 s, against the shipped 0.63 s median and a p90 past the 4 s
#    measurement window.)
# The path is flat across the whole sweep, so nothing is oscillating; p90 stops
# improving at 4 rad/s because past that the remaining slow flips are limited by
# the SOLVE, not by the cap. 5.0 is the knee. The proximal rate is untouched, so
# the 2026-07-31 "snaps into position too fast" fix stands exactly as measured.
ARM_TRUST_RATE_DISTAL = float(os.environ.get("MABEL_ARM_TRUST_RATE_DISTAL", "5.0"))
#: index of the first DISTAL joint in the optimisation vector (Arm 5 = forearm
#: roll; everything from here out is wrist/hand-scale inertia)
ARM_TRUST_DISTAL_FROM = 4
ARM_TRUST_MAX = float(os.environ.get("MABEL_ARM_TRUST_MAX", "0.30"))    # rad/solve
# Fallback step for a solve with no usable dt (the cold path). 0 disables the
# trust region entirely.
ARM_TRUST_STEP = float(os.environ.get("MABEL_ARM_TRUST_STEP", "0.10"))
# ORCA wrist (1-DoF pitch): when True it is a JOINTLY-OPTIMISED 8th DoF (solved from
# the palm-orientation objective, not hard-coded from the human flexion), which adds
# redundancy that unlocks the natural flip up/down and lets the arm avoid twisting.
ARM_OPTIMIZE_WRIST = True
ARM_REG_WRIST = 0.03     # light reg on the optimised ORCA wrist (free to flip, returns to rest)
ARM_QP_VMAX = 4.0        # rad/s per-joint velocity cap used ONLY inside the active
                         # self-collision QP (no effect on free motion)
SELFCOL_INFLUENCE = 0.10 # m; the barrier starts braking a closing pair below this gap
                         # (larger = reacts EARLIER -> holds the margin under fast approach)
SELFCOL_ALPHA     = 8.0  # 1/s; self-collision barrier gain (higher = firmer wall)
# Extra margin applied to the MEASURED-pose barrier rows, for the HAND capsules
# only (`SelfCollisionGuard.lag_capsules`). It covers the servo following error
# between the command the QP bounds and the metal that actually moves: the hand
# capsule turns with the ORCA wrist, a ~2.9 Hz joint that cannot be commanded
# aside quickly, and without a pad the palm reached 6.8 mm inside the torso
# capsule on 0.15 % of frames.
#
# The wrist is soft ON PURPOSE and stays that way: only 0.217 kg hangs off it and
# it sags 1.9 deg, while stiffening it measurably breaks external-contact force
# estimation (see the MJCF actuator comment). So the barrier leaves room for the
# joint instead of demanding authority the joint should not have.
#
# PER-CAPSULE, not global, and that distinction is the whole point. Measured with
# a blanket pad, the FOREARM standoff went 10 -> 28 mm to buy a margin only the
# hand needed — 3x the thin layer the operator asked for, spent on links that
# track perfectly well. Scoped to the hand capsules the forearm holds at 11.9 mm
# and the hand gets the room it needs.
SELFCOL_LAG_PAD = float(os.environ.get("MABEL_SELFCOL_LAG_PAD", "0.020"))  # m
# Position task point = the hand CONTACT point (~the middle fingertip), HAND_REACH
# forward of the ORCA wrist along the palm. Tracking the CONTACT point (not the
# wrist) is what lets the two robot hands TOUCH when the operator's fingertips touch:
# two coincident operator fingertips map to ONE robot point regardless of scale, so
# the gap goes 0->0. Computed by FORWARD KINEMATICS (no privileged sim state).
HAND_REACH = 0.14        # m; ORCA wrist -> middle-fingertip along the palm forward
# FRONT-OF-BODY clamp: the wrist/contact target is never allowed behind this chassis-x
# line (robot front = -x), so the arm can't be stretched/twisted to the back of the
# robot when the operator moves/reaches back faster than the base can follow. The base
# then backs up (eager, see BASE_BOX_BACK) to bring the operator's intent forward of it.
#: Minimum wrist-target radius about that arm's own shoulder [m]. A human
#: working against their own chest maps, through the root-anchored GMR
#: similarity, to a point inside MABEL's much narrower torso -- measured 82 % of
#: one clip's targets within 0.16 m of the torso axis, which the arm cannot
#: reach (it folds against the chest and sits 173 mm short). Only ~1 % of
#: random in-limit arm poses get inside 0.15 m, and the posture table's box
#: starts at 0.25 m, so anything below this is a pose the solver will not find.
#: The correction is radial, so the operator's indicated DIRECTION is kept.
#: Set to 0.0 to disable.
ARM_MIN_SHOULDER_RADIUS = 0.20

#: Minimum HORIZONTAL distance from the robot's torso axis for a wrist target
#: [m]. This is the constraint that actually matters for hands held against the
#: chest: measured on `screw_unscrew_allen_fixture/ep003` the left target sat
#: 0.023-0.063 m from the torso axis at shoulder height -- on the centreline,
#: between the two shoulders (half-width 0.074 m), inside the body. Pushing such
#: a target away from the AXIS moves it to the side the arm can reach; pushing
#: it radially from the SHOULDER (ARM_MIN_SHOULDER_RADIUS) sends it along an
#: almost arbitrary bearing and made that clip 44 mm worse. Set 0.0 to disable.
#: 0.10 was measured best over a 4-clip probe (mean wrist error 104 -> 94 mm,
#: chopsticks 141 -> 84, control clip unchanged at 20). 0.14 and 0.18 over-correct:
#: they move the target further from what the operator indicated than the arm
#: gains by being outside the body.
ARM_MIN_TORSO_RADIUS = 0.10

# ── MOBILE-BASE WORKSPACE ASSIST ─────────────────────────────────────────────
#
# MABEL is a mobile manipulator, and until 2026-08-25 the retargeting used none
# of that: the base sat still while the arms failed to reach. The measured
# failure is that an operator working against their own chest maps to a target
# INSIDE the robot's torso (0.023-0.063 m from the torso axis against a 0.074 m
# shoulder half-width) -- unreachable by any arm configuration.
#
# THE SUBTLETY THAT DICTATES THE DESIGN: the GMR target is anchored at the
# robot's LIVE mid-shoulder (`t = root_r + sigma*M*(w - root_h)`), so driving
# the base moves the target with it and reachability does not change at all.
# Base motion alone is a no-op. What works is offsetting the map's ANCHOR --
# which instantly gives the arm room -- and then driving the base to *pay off*
# that offset, so the offset is not a lie about where the robot is: the hands
# stay put in the world while the robot backs out from under them.
#
# The offset is therefore a debt: it grows when the workspace is crowded, and
# shrinks by exactly as much as the base actually moves.
ARM_WS_ASSIST = True

#: Shoulder radius below which targets are "crowded" and the base should back
#: off [m]. The arm's usable shell starts around here: only ~1 % of random
#: in-limit poses reach inside 0.15 m, and the posture table's box starts at
#: 0.25 m.
ARM_WS_NEAR = 0.30

#: Shoulder radius above which targets are out of reach and the base should
#: close in [m]. Measured arm reach is 0.66-0.77 m; leave margin for posture.
ARM_WS_FAR = 0.60

#: Mean lateral target offset beyond which the base strafes to re-centre [m].
ARM_WS_LATERAL = 0.22

#: Cap on the anchor offset [m] — the most the base can be "in debt" by.
ARM_WS_MAX_OFFSET = 0.35

#: How fast the offset accumulates toward what the workspace error asks for
#: [m/s]. Slow, because it is a whole-body reposition, not a tracking channel.
ARM_WS_RATE = 0.25

#: Deadband on the workspace error before any assist engages [m]. Without it the
#: base creeps continuously on ordinary motion.
ARM_WS_DEADBAND = 0.04

#: Base speed commanded per metre of outstanding offset [1/s].
ARM_WS_BASE_GAIN = 1.2

#: Low-pass on the HEAD POSITION that anchors the operator's shoulder estimate
#: (EMA coefficient per frame; 1.0 = off). ShoulderEstimator places the
#: shoulders at a rigid offset from `head_pos`, so it rejects head ROTATION (its
#: body frame is yaw-only) but passes head TRANSLATION straight through -- and
#: that lands on every wrist target. Measured on EgoDex
#: `wipe_kitchen_surfaces/ep3`: the operator's left wrist moves 2 mm/s while its
#: target moves 26.6 mm/s, and freezing the head alone removes 66 % of that.
#: A real torso translation is slow; head bob, lean and glance are not, so a
#: long time constant keeps the former and rejects the latter. 0.05 ~ 0.67 s at
#: 30 Hz.
ARM_HEAD_ANCHOR_LP = 0.05

ARM_WRIST_FRONT = -0.05  # m; chassis-x of the body's front vertical line (more -x = further front)
# per-joint regularization toward rest (7 entries; sliced [:6] when J7 is locked).
# Tuned to the HUMAN arm's per-joint mobility so the redundant 7-DoF arm resolves
# the way a human arm does:
#   Arm1/2 (shoulder pos)  light  -> free to POSITION the arm for reach
#   Arm3   (shoulder roll) HARD   -> the upper-arm roll is done by the FOREARM, not
#                                    the shoulder (keeps the top of the arm steady)
#   Arm4   (elbow)         light
#   Arm5   (forearm roll)  VERY LIGHT -> this is the "flip" DoF; let it move freely so
#                                        the hand can flip up (hardware range permitting)
#   Arm6   (wrist pitch)   light
#   Arm7   (wrist YAW)     HARD   -> a human wrist barely yaws (radial/ulnar deviation
#                                    ~ +/-25 deg); held near rest AND hard-bounded below
# Arm6 (0.20) is DELIBERATELY stiffer than its neighbours, and not a typo. Measured off the
# MJCF, Arm6's joint axis and the ORCA WRIST's are PARALLEL — both
# (0, 0.174, -0.985) at rest — so wrist PITCH is redundant between them and the
# IK is free to spend it on either. Regularising Arm6 toward nominal makes the
# ORCA wrist the cheap one, so pitch lands in the hand where the operator feels
# it (operator, 2026-07-28: "the wrist pitch motion should be prioritised
# mapped to the ORCA hand wrist joint"). Arm5 stays nearly free (0.02) for the
# same reason in reverse: it is the forearm ROLL, and that is where a wrist
# flip belongs.
#
# 0.20, not more: at 0.45 the redundancy was won just as decisively (ORCA 23.7 deg
# against Arm6 0.1 deg) but the ARM went deaf to small motions — a +-4.5 cm hand
# wave produced 1.3 deg of arm travel where it had produced 99.6 deg, because the
# IK could satisfy it in the hand alone. Winning a redundancy must not cost the
# arm its responsiveness; both are measured by server/tools/exp_arm_fidelity.py
# and exp_tracking_channels.py, and they trade against each other here.
ARM_REG_W = np.array([0.08, 0.08, 0.40, 0.08, 0.02, 0.20, 0.60])
# When an ELBOW objective is present (tracked webcam elbow, or the palm-roll
# synthesised hint), the Arm3 shoulder-roll regulariser relaxes by this factor:
# it exists to resolve REDUNDANCY when nothing constrains the swivel, but
# elbow-up requires exactly the humerus roll it pins (measured: swivel
# following saturated at ~40° of the commanded 115° with the full 0.40 weight).
ARM_REG_ELBOW_RELAX = 0.2
# Swivel-adaptive ELBOW weight: W_eff = ARM_W_ELBOW·(1 + BOOST·|φ|). At a big
# deliberate elbow raise the rolled palm orientation is UNREACHABLE (J5/J6
# saturate) and its cost exactly cancelled the elbow gain — two basins with
# equal total cost, the arm stalling between them. When the operator steers
# the elbow, its look wins the tie; neutral swivel keeps the original balance.
ARM_ELBOW_SWIVEL_BOOST = 1.5
# PALM ROLL -> ELBOW SWIVEL, as a GAIN. 0 = the synthesised elbow keeps only its
# gravity base, i.e. the elbow hangs DOWN and IN — the relaxed human swivel.
#
# It was 1.0, added so "when I push my elbow up the sim follows". The cost is
# that a hand and an elbow cannot be told apart from one palm orientation:
# pronating the forearm rolls the palm exactly as raising the elbow does, so
# every wrist flip swung the elbow. Measured over the wire on a 172 deg operator
# forearm roll with the hand held still: the elbow travelled 13.3 cm and the
# commanded wrist point drifted 9.0 cm — and the elbow flared OUTWARD, which is
# both unhuman and the direction that walks the arm toward a singularity
# (operator, 2026-07-28). Roll now goes where it anatomically belongs, to J5.
#
# The trade: a deliberate elbow raise no longer follows. Raise this toward 1.0
# to get it back, and re-run server/tools/exp_arm_fidelity.py to see the cost.
ARM_SWIVEL_PALM_GAIN = float(os.environ.get("MABEL_SWIVEL_PALM_GAIN", "0.0"))
# Arm 7's optimisation bound, clipped to rest +/- this (rad) and intersected with
# the model range. Only applies in the 7-DoF (no --lock) case.
#
# 0.45 rad (~26 deg) until 2026-08-23, on the argument that "a human wrist barely
# yaws (radial/ulnar deviation ~ +/-25 deg)". The premise is right about a human
# and wrong about THIS joint: measured off the MJCF, Arm 7's axis is
# perpendicular to the forearm AND 217 mm proximal of the ORCA-wrist anchor, so
# it is not a wrist deviation at all — it is the last arm joint swinging the
# whole hand tower. Clamping it to a human wrist's range throws away orientation
# authority the arm has and the operator was asking for.
#
# MEASURED over the full front manipulation grid (150 points x 4 palm directions
# per arm, experiments/exp_front_workspace.py W2a) — poses achieved, and the
# palm-direction error at the 90th percentile:
#
#     clamp            right ok / tilt p90      left ok / tilt p90
#     0.45 rad (26 deg)   73.0 % / 4.5 deg      72.3 % / 4.7 deg   (shipped)
#     0.70     (40)       73.0   / 3.9          74.3   / 4.0
#     0.90     (52)       74.7   / 3.8          75.2   / 4.0
#     1.20     (69)       76.5   / 2.9          77.2   / 3.4       <- taken
#     1.57     (90)       78.5   / 2.7          77.5   / 3.9
#
# 1.57 is marginally ahead on the right and behind on the left; 1.20 is the
# middle that wins on both palm-error columns, and it still keeps SOME of the
# original intent (the joint is not free to spin a full quarter turn).
ARM_J7_YAW_LIM = float(os.environ.get("MABEL_ARM_J7_YAW_LIM", "1.20"))   # rad (~69 deg)

# Human wrist flexion -> ORCA 1-DoF wrist joint. The operator's wrist bend (angle
# between forearm and hand about the hand's flexion axis) maps to the single ORCA
# wrist joint, clamped to its range. Flip the sign / scale the gain to taste.
WRIST_FLEX_GAIN = 1.0
WRIST_FLEX_SIGN = 1.0

# --- FINGER solver selection --------------------------------------------------
# "keypoint" (default): DexPilot-style fingertip keypoint IK — the dexterous_hand
# planner lineage (teleop/planners/planner_avp.py / planner_per_finger.py →
# mabel.hand_retargeting.keypoint_ik): canonical palm frame, per-finger length
# calibration, shape + bone-direction rows, gated pinch, damped Gauss–Newton with
# an analytic Jacobian (~1 ms/hand). "anatomical" restores the webcam-parity
# angle decomposition (hand_anatomical.py) — also the automatic fallback when the
# controller package is absent.  MABEL_FINGER_SOLVER=anatomical to switch back;
# MABEL_HAND_SOLVER=slsqp runs the reference planner_avp cost inside keypoint.
FINGER_SOLVER = os.environ.get("MABEL_FINGER_SOLVER", "keypoint").lower()
HAND_KP_SOLVER = os.environ.get("MABEL_HAND_SOLVER", "gn").lower()   # gn | slsqp

# --- Webcam-parity anatomical FINGER retargeting (hand_anatomical.py) --------
# The engine's fingers run the SAME anatomical decomposition as the control-center
# GUI's webcam hand teleop (bodyteleop.decomposeHand / dexterity.retargetLandmarks).
HAND_ANAT_SMOOTH = 0.45   # (legacy EMA fallback, MABEL_HAND_ONEEURO=0) EMA weight of the
                          # NEW sample per joint; 0.45 matched the shipped ARKit filter but
                          # let ~0.7 rad/frame finger-tracking spikes through as jitter.
# One-euro filter on the finger command — the SOTA smoother for noisy interactive
# tracking. Unlike a fixed EMA (which must trade lag for smoothness), it strongly
# smooths at low speed (kills the measured ~0.7 rad jitter spikes when the operator
# holds/moves slowly) yet raises its cutoff with joint speed so a real open/close is
# NOT laggy. Params env-tunable for live tuning; MABEL_HAND_ONEEURO=0 reverts to EMA.
HAND_ONEEURO = os.environ.get("MABEL_HAND_ONEEURO", "1").lower() not in ("0", "false", "no")
# Responsive by default (min_cutoff 2.5 Hz ≈ 60 ms rest lag, was 1.5≈106 ms and felt
# laggy): a higher beta keeps fast motion snappy. Lower MABEL_HAND_MINCUTOFF for MORE
# smoothing, raise it for LESS lag.
HAND_ONEEURO_MIN_CUTOFF = float(os.environ.get("MABEL_HAND_MINCUTOFF", "2.5"))  # Hz — rest smoothing (lower=smoother)
HAND_ONEEURO_BETA       = float(os.environ.get("MABEL_HAND_BETA", "0.20"))      # speed→cutoff (higher=less lag when fast)
HAND_ONEEURO_DCUTOFF    = float(os.environ.get("MABEL_HAND_DCUTOFF", "1.0"))    # Hz — derivative smoothing

# Finger SPLAY (abduction): was hard-pinned to mid-range (untracked). Track the measured
# abduction so spreading the fingers spreads the robot's. Gain trims the mapping; the
# one-euro filter smooths it like every other joint. MABEL_HAND_ABDUCTION=0 restores neutral.
HAND_TRACK_ABDUCTION = os.environ.get("MABEL_HAND_ABDUCTION", "1").lower() not in ("0", "false", "no")
HAND_ABD_GAIN = float(os.environ.get("MABEL_HAND_ABD_GAIN", "1.0"))
# THUMB flexion neutral offset [rad]: a relaxed human thumb is slightly bent, which the
# raw metacarpal→phalanx angle read as flexion → the robot thumb "closed too early".
# Subtract a resting bend so a relaxed thumb maps to OPEN. Tunable per operator.
HAND_THUMB_FLEX_OFFSET = float(os.environ.get("MABEL_HAND_THUMB_OFFSET", "0.35"))
HAND_THUMB_FLEX_GAIN   = float(os.environ.get("MABEL_HAND_THUMB_GAIN", "1.0"))

# --- Operator torso heading for headset clients (base yaw follow) -------------
# The desktop streamer derives body_yaw from the two-elbow line (BodyOdom); headset
# clients get the same signal from the ARKit forearm joints, low-passed here.
BODY_YAW_TAU = 0.25       # s; heading low-pass time constant (streamer default)

# --- ARKit ORIGIN-JUMP guard / anchor stability (headset clients) -------------
# When the immersive space opens, ARKit's world origin can relocalise moments after
# tracking reports "normal" — the head teleports ~1.6 m (floor snap). Anchors taken
# before that (world_ref, lift head_h0, base box) go stale and everything maps sky-
# high. So: anchor only after the head has streamed steadily for HEAD_ANCHOR_HOLD,
# and a per-frame head displacement above HEAD_JUMP_M re-anchors everything.
HEAD_JUMP_M      = 0.35   # m between consecutive frames = origin relocalisation
HEAD_ANCHOR_HOLD = 0.40   # s of continuous head tracking before anchors may latch

# --- 1-pose reach calibration capture rule (engine._calibrate_step) -----------
# "The arm-length number stabilised" == success: median head→wrist reach over a
# rolling window, judged by TIME SPAN + STANDARD DEVIATION (both rate-independent
# — the old ≥5-samples-in-0.6s + max−min rule could never complete below ~8 Hz
# and got stricter with rate under tremor; tools/exp_calibration_robustness.py).
# MIRRORED by the app's CalibrationCoach (CalibrationProfile.swift) — change BOTH.
CAL_WINDOW   = 1.00   # s, rolling sample window
CAL_MIN_SPAN = 0.50   # s, the window must cover this much time (any rate ≥ ~4 Hz)
CAL_STD      = 0.02   # m, reach standard deviation that counts as "held still"
CAL_HOLD     = 0.25   # s of continuous stillness before the median is captured

# Neck mapping (relative head rotation -> neck joints). Flip a sign if the head
# turns the wrong way; magnitude scales how much the neck follows the head.
NECK_YAW_SIGN, NECK_PITCH_SIGN, NECK_ROLL_SIGN = 1.0, -1.0, 1.0
NECK_GAIN = 1.0
# NECK-YAW -> BASE-YAW HANDOFF (operator, 2026-10-23: "I need to have 360 view as I
# move my head"). Past NECK_YAW_HANDOFF the chassis turns to carry the excess, so the
# head camera follows a look anywhere around the robot; the neck never leaves its
# +-90 deg range. Rate = gain * excess, capped, through the tip-safe motion model.
# Latches off if the MEASURED chassis yaw lags the commanded turn by STALL (stale
# odometry / blocked base). Law + guard: neck_handoff.py; measured by
# controller/experiments/exp_head_gaze_coverage.py. MABEL_NECK_YAW_HANDOFF=0 disables.
NECK_YAW_HANDOFF_ON = os.environ.get("MABEL_NECK_YAW_HANDOFF", "1") != "0"
NECK_YAW_HANDOFF = np.radians(float(os.environ.get("MABEL_NECK_YAW_HANDOFF_DEG", "80")))
NECK_HANDOFF_GAIN = 2.0                  # 1/s  excess yaw -> base yaw rate
NECK_HANDOFF_WMAX = 0.8                  # rad/s cap (below BASE_WMAX 1.5)
NECK_HANDOFF_STALL = np.radians(45.0)    # commanded-minus-measured turn that latches off

SHOW_OVERLAY = True                  # ghost hands + arm balls + gaze ray

# --- Whole-body control (dynamic teleop with gravity compensation) ----------
# The robot is now PHYSICALLY simulated: retargeted joint angles drive position
# actuators, with a feed-forward gravity torque g(q) added so the arm/lift/torso
# hold their commanded pose against gravity (no PD sag). The mobile base follows
# the operator's odom translation/turn (swerve), and the lift follows head height.
# Re-exported from mabel.dynamic_models.gravity (the ONE definition, shared with
# the torque engine so Drive and Joint/Task cannot compensate differently).
from ..dynamic_models.gravity import GRAV_GAIN, GRAV_HZ   # noqa: E402,F401
# --- RETARGET SCOPE: which DoFs are in the follow loop -----------------------
# LOCAL (default)  arms + hands + torso + lift + neck. The mobile BASE is NOT in
#                  the loop at all — the operator drives it with the joystick.
#                  This is deliberately the easier problem: a fixed root, so the
#                  arm IK is solving against a base that cannot move under it.
# GLOBAL           the base joins the follow, taking the overflow beyond the
#                  comfort box (translation) and the yaw deadband (orientation).
# The two differ ONLY in whether the base follows; torso/lift/arms/neck/hands are
# identical, so a switch cannot change how the upper body tracks.
RETARGET_SCOPE = os.environ.get("MABEL_RETARGET_SCOPE", "local").lower()

# --- SET FORWARD must land, and land SOON ------------------------------------
# The full datum wants a head AND both wrists in the same frame, because that is
# what lets `ForwardConsensus` learn the wrist offsets it uses afterwards. But
# the operator presses the button with their hands wherever they happen to be,
# and a hand outside the headset's field of view simply is not there — so
# "wait for the good frame" can wait a very long time, which is what the
# operator felt as "the recenter forward function is too slow" (2026-08-01).
#
# So: prefer the full frame, but BOUND the wait. Past this deadline the heading
# is zeroed from the HEAD ALONE, which is the same quantity the full path uses
# for the datum itself (`_geom`'s `hyaw` -> `psi`); what a head-only zero cannot
# do is learn the wrist offsets, and those are relearned by the next full
# `calibrate_facing`. Well under the operator's 3 s ceiling on purpose — the
# guarantee is worth more here than the last of the precision.
FORWARD_DEADLINE = float(os.environ.get("MABEL_FORWARD_DEADLINE", "1.0"))   # s

# --- LAZINESS of the slow reach extenders ------------------------------------
# The torso and the lift exist to EXTEND reach, not to mirror sway. Without a
# deadband they answer every centimetre of the operator's natural micro-motion,
# which reads as the robot fidgeting and (because the head rides the torso) as
# the gaze drifting. These are SOFT deadbands — the response is shifted, not
# clipped, so crossing the threshold does not step.
TORSO_DEADBAND = float(os.environ.get("MABEL_TORSO_DEADBAND", "0.06"))  # m of lean
LIFT_DEADBAND = float(os.environ.get("MABEL_LIFT_DEADBAND", "0.05"))    # m of head rise

BASE_FOLLOW = True       # HEAD horizontal motion -> swerve base (beyond the comfort box)
# Base-engage BOUNDING BOX (referenced to the operator's calibrated HEAD position, in
# the robot frame). Head motion INSIDE the box is absorbed by the torso (+lift); only
# the part that OVERFLOWS the box drives the base. The box is asymmetric -- DEEP in
# FRONT (so reaching/leaning forward a little does NOT locomote; the torso/lift handle
# it) and SHALLOW behind (so stepping back engages the base quickly). Lateral is in
# between. Raise a box value -> that direction is LESS sensitive (more head travel
# before the base moves); lower it -> MORE sensitive.
BASE_BOX_FWD  = 0.18     # m; forward comfort box (deep -> forward reach uses torso/lift)
BASE_BOX_BACK = 0.02     # m; backward box ~0 -> the base backs up THE INSTANT you step
                         # back, so the wrist never gets left behind the body
BASE_BOX_LAT  = 0.12     # m; lateral comfort box
# Per-direction base SENSITIVITY WEIGHTS: base position offset per metre of head
# overflow beyond the box (NOT separate objectives -- just gains). Backward weighted
# >1 so the base LEADS slightly when backing up (keeps the hands in front).
BASE_W_FWD  = 1.0        # forward overflow -> base offset gain
BASE_W_BACK = 1.3        # backward overflow -> base offset gain (leads when backing up)
BASE_W_LAT  = 1.0        # lateral overflow -> base offset gain
BASE_KP     = 3.5        # 1/s   base position-error -> velocity (high -> catches up fast;
                         # the tip-over motion model still caps the realized accel/speed)
# ORIENTATION comfort box, the yaw twin of BASE_BOX_*. The base turns only once
# the operator's heading differs by more than this, so ordinary shoulder/torso
# sway never spins the chassis (translation already had a box; yaw did not, so
# the base chased every degree). Soft-deadbanded like the translation overflow.
BASE_YAW_BOX = float(os.environ.get("MABEL_BASE_YAW_BOX", "0.26"))   # rad (~15 deg)

# HEADING + BASE FOLLOW MODEL (2026-10-15, Retarget Lab whole-body takes). Two
# defects measured on real Vision Pro data (controller/experiments/exp_heading.py,
# web_gui/retarget_lab/docs §E6):
#  * the head+wrist BodyFrameEstimator latched 30-75 deg heading offsets after a
#    hand dropout and never returned (return-to-facing error p90 51-73 deg); the
#    base yaw AND the arm map's body frame followed it — "everything drifts";
#  * the base target was not 1:1 with the operator (backward gain 1.3, a reach
#    multiplier up to 2x, the box in WORLD axes so a lean after a turn read as a
#    sidestep), so after walking back to the start the robot ended elsewhere.
# HEADING_MODE "head" = whole_body_retargeting/heading.LazyHeadHeading (drift-free:
# |trunk - head| <= slack always, -> 0 whenever the head is held); the base then
# tracks it with NO yaw deadband (the laziness is in the estimator). "bodyframe" =
# the shipped estimator.  BASE_TRANSLATION_MODE "body" = the operator's head
# displacement taken 1:1, the comfort box and the lean read in the operator's BODY
# frame, the overflow rotated back to the world; "shipped" = the box-gain law above.
# Robot defaults unchanged; the Retarget Lab (and so NOVA's gateway) runs head/body.
HEADING_MODE = os.environ.get("MABEL_HEADING_MODE", "bodyframe").strip().lower()
HEADING_SLACK = float(os.environ.get("MABEL_HEADING_SLACK", "0.436"))   # rad (25 deg), swept H1-H3
HEADING_TAU = float(os.environ.get("MABEL_HEADING_TAU", "2.0"))         # s
BASE_TRANSLATION_MODE = os.environ.get("MABEL_BASE_TRANSLATION", "shipped").strip().lower()

# PHYSICAL RATE LIMITS on the follow targets (operator, 2026-10-15: "the lift is
# moving so fast — the max speed is around 3 cm/s"). The lift target was a
# per-FRAME low-pass (LIFT_RATE) with no speed bound at all: ~60 ms time constant
# at 60 Hz, faster still at 90 Hz. The torso lean had no rate bound either, though
# config.MAX_TORSO_IK_SPEED (0.5 rad/s) exists precisely because a fast torso
# carries the CoM past the support edge. Both are now slew-limited per second.
LIFT_FOLLOW_SPEED = float(os.environ.get("MABEL_LIFT_FOLLOW_SPEED", "0.03"))    # m/s
TORSO_FOLLOW_SPEED = float(os.environ.get("MABEL_TORSO_FOLLOW_SPEED", "0.5"))   # rad/s

# REACH-AWARE base engagement (Haviland, Sunderhauf, Corke, "A Holistic Approach
# to Reactive Mobile Manipulation", RA-L 2022, arXiv:2109.04749). Their result:
# in a whole-body QP, weighting the BASE DoFs by 1/||e|| — cheap when the
# end-effector error is large, expensive when it is small — makes the base take
# the gross motion and the arm the fine motion with NO threshold and no
# hysteresis, and is 2-3x faster than plan-base-then-move-arm (35 vs 71 failures
# in 1000 reaching tasks).
#
# MABEL's base is a separate cascade rather than one QP, so we take the INSIGHT
# rather than the formulation: the head-displacement overflow that drives the
# base is additionally scaled by how badly the ARMS actually need help. A
# stretched-out operator (both wrists near the robot's reach limit) gets a base
# that leans in eagerly; an operator working comfortably close gets one that
# stays put even outside the box, because the arms have it covered. That is the
# behaviour the box alone cannot express — the box only knows where the HEAD is,
# not whether the ARMS are struggling.
#
# 0 disables and restores the pure geometric box.
BASE_REACH_ASSIST = float(os.environ.get("MABEL_BASE_REACH_ASSIST", "1.0"))
# Extension fraction (wrist distance / robot arm reach) at which the assist is
# fully on. Below REACH_FREE the arms are comfortable and the assist is 0.
BASE_REACH_FREE = float(os.environ.get("MABEL_BASE_REACH_FREE", "0.70"))
BASE_REACH_FULL = float(os.environ.get("MABEL_BASE_REACH_FULL", "0.95"))

BASE_KP_YAW = 2.0        # 1/s   base heading-error -> yaw rate
BASE_YAW_SIGN = 1.0      # flip if turning your body turns the robot the wrong way
BASE_VMAX   = 0.8        # m/s   base linear speed cap
BASE_WMAX   = 1.5        # rad/s base yaw rate cap
# LIFT: the operator head's WORLD height maps 1:1 to lift travel, so the robot head
# rises/falls exactly as much as the human head does, with the lift range as the
# hard constraint (the robot cannot go higher/lower than the lift allows).
LIFT_FOLLOW = True       # head world height -> lift travel
LIFT_GAIN   = 1.0        # m of lift per m of head rise (1:1 = the robot head matches)
LIFT_RANGE  = (0.0, 0.635)

# --- TORSO: the fast/agile reach extender — bends FIRST when you lean over -----
# The torso (a -60..+15 deg forward-lean joint) is much more AGILE than the lift
# (speed-limited) and the base (not agile, slow to change direction). So in the
# whole-body cascade the torso absorbs the fast forward LEAN immediately, while the
# lift / base only handle the slow vertical / translation (rate-limited below). The
# lean is the head's forward offset from the body odom (so bending in place moves the
# torso, not the base; walking moves the base, not the torso).
TORSO_FOLLOW = True      # operator forward lean -> torso bend (first responder)
TORSO_GAIN   = 2.5       # rad of torso bend per m of operator forward lean
# LEAN MODEL (2026-10-15, Retarget Lab: "my torso is not bending forward a lot
# when I am bending forward by a lot"). The "box" law above feeds the torso the
# forward head offset CLIPPED TO BASE_BOX_FWD, so the bend saturates at
# TORSO_GAIN * (0.18 - 0.06) = 0.30 rad (17 deg) however far the operator bends —
# measured on a live recording: the torso sat at exactly -0.300 while the head
# went 0.24 m forward and 0.16 m down. The joint itself reaches -1.047 (60 deg).
# "angle" reads the operator's TRUNK PITCH instead, from a hip-pivot model: a
# trunk of length L (hip -> eyes) pitched by theta puts the head forward by
# L sin(theta) and down by L (1 - cos(theta)), so theta = atan2(d, L - h) for the
# head's forward offset d and drop h — or, when the head dropped more than a
# rigid hip pivot explains (a real deep bend flexes the neck and lower spine
# too), acos(1 - h/L) from the drop, capped at asin(d / TORSO_LEAN_PIVOT_MIN) so
# a squat (drop, no forward travel) is never a bend. The robot torso bends by the same angle
# (gain TORSO_LEAN_GAIN, soft deadband TORSO_LEAN_DEADBAND), clipped to the joint.
#   * LOCAL scope: every forward offset is lean (the base is not in the loop).
#   * GLOBAL scope: the base still takes the box overflow, but only the part a
#     bend does NOT explain — d minus L sin(theta_drop), with theta_drop from the
#     head DROP alone. Walking keeps the head level (theta_drop ~ 0: base moves,
#     torso stays within the box law); bending lowers it (torso takes it).
#   * The lift follows head height MINUS the drop the lean itself produces, so a
#     bend is not also read as "operator got shorter" (the robot's own torso
#     bend already lowers its head).
# Default "box" = the shipped real-robot behaviour, unchanged; the Retarget Lab
# (kinematic, no hardware) runs "angle". A 60 deg real torso lean moves the CoM
# far forward — validate with wbc_stability.py before enabling it on hardware.
TORSO_LEAN_MODE = os.environ.get("MABEL_TORSO_LEAN_MODE", "box").lower()
TORSO_LEAN_PIVOT = float(os.environ.get("MABEL_TORSO_LEAN_PIVOT", "0.70"))   # m, hip -> eyes
# shortest plausible bend radius: caps the head-DROP reading of the lean by the
# forward travel that actually happened (theta <= asin(d / this))
TORSO_LEAN_PIVOT_MIN = float(os.environ.get("MABEL_TORSO_LEAN_PIVOT_MIN", "0.35"))  # m
TORSO_LEAN_GAIN = float(os.environ.get("MABEL_TORSO_LEAN_GAIN", "1.0"))      # rad / rad
TORSO_LEAN_DEADBAND = float(os.environ.get("MABEL_TORSO_LEAN_DEADBAND", "0.08"))  # rad (~4.6 deg)
# UPPER BODY (LOCAL scope) tracks the operator in a frame that moves WITH them: walking is not a lean.
# A hip bend drops the head by L(1 - cos th); walking does not, so in LOCAL the lean is read from the
# head DROP (past this deadband, which swallows the ~2 cm gait bob) and the horizontal anchor is kept
# within the L sin th the bend explains — the rest of the head's travel drags the anchor along.
TORSO_LOCAL_DROP_DEADBAND = float(os.environ.get("MABEL_TORSO_LOCAL_DROP_DEADBAND", "0.02"))  # m

# --- LOCAL-MODE upper body: torso + lift from the operator's OWN GEOMETRY ------
# In LOCAL retargeting the base is held and the world estimate is refused, so the
# global laws above cannot be used: they read the head's ARKit-anchor DISPLACEMENT
# (torso) and its world HEIGHT (lift), and in a moving vehicle both track the
# vehicle rather than the operator.
#
# The local law reads ONE quantity instead: the operator's hand centroid relative
# to their OWN HEAD, expressed in the HEAD'S OWN FRAME. That is a device-frame
# measurement — invariant to world translation AND to the tilt/yaw drift of the
# world estimate, because both endpoints and the basis come from the same rigid
# body. Reaching UP raises the lift; reaching OUT leans the torso.
#
# IT IS A STATIC MAP FROM AN ANCHORED REFERENCE, NOT AN INTEGRATING COMFORT LOOP.
# That is load-bearing. The first version integrated a comfort-box overflow on the
# commanded WRIST TARGETS in the chassis frame, and ran both actuators to their
# stops in 5 s (measured: lift 52.9 cm of 63.5, torso pinned at −60°). The arm map
# anchors the target at the robot's mid arm-mount — which RIDES the lift and the
# torso — so raising the lift raises the target by exactly as much and the error
# never closes: positive feedback. Here the input depends only on the OPERATOR, so
# the output is bounded by their arm length no matter what the robot does.
# --- NEUTRAL poses: where a SWITCHED-OFF tracking channel eases to ------------
# Switching a channel off must not FREEZE it. A frozen limb reads as a crash, and
# a frozen neck leaves the cameras pointed wherever the operator last looked. So
# each channel eases, slowly, to a pose that is safe and legible at a glance.
#
# ARMS: the gravity EQUILIBRIUM — arms hanging straight down at the sides, wrists
# at their MJCF ref so the hands are level. Not the model's qpos0, which is all
# zeros: that pose costs 38.6 Nm of holding torque against 0.004 Nm here, so on
# the real robot it is the difference between a resting arm and a straining one.
# Same numbers the apps ship as their rest pose (RobotController.restBodyPose).
NEUTRAL_ARM = {
    "left_arm_1": -0.0044, "left_arm_2": -0.1745, "left_arm_3": 0.0811,
    "left_arm_4": 1.5795, "left_arm_5": 0.0125, "left_arm_6": -0.0215,
    "left_arm_7": 0.0077,
    "right_arm_1": 0.0087, "right_arm_2": 0.1732, "right_arm_3": -0.0805,
    "right_arm_4": -1.5812, "right_arm_5": -0.0099, "right_arm_6": 0.0161,
    "right_arm_7": -0.0054,
}
NEUTRAL_WRIST = {"left_wrist": 0.15, "right_wrist": -0.15}   # MJCF ref = level hand
NEUTRAL_TORSO = 0.0      # rad; upright
NEUTRAL_LIFT = 0.0       # m;   column down
NEUTRAL_NECK = 0.0       # rad; level, looking straight ahead
NEUTRAL_GRIP = 0.0       # fingers open
# Seconds to ease a switched-off channel to neutral. Slow on purpose: this is a
# limb moving on its own while the operator is not driving it, and it has to read
# as deliberate. Matches SAFE_RETREAT_TIME's intent.
PARK_TIME = 2.5
# s; a re-enabled channel blends from its held command to the live one over
# this long (smoothstep) — no command step on release (safety.ChannelPark).
PARK_RELEASE_TIME = 1.0

LIFT_RATE    = 0.25      # 0..1 lift target low-pass (the lift is speed-limited -> sluggish)
BASE_VEL_LP  = 0.20      # 0..1 base velocity low-pass (the base is not agile -> smooth)
MAX_CATCHUP = 0.05       # s; physics advances by wall-time, capped (realtime track)

# --- SAFETY / engagement fail-safe (tracking-loss -> ease HOME -> ease back) ----
# If the operator removes the headset (the VR stream stalls) or a hand / arm drops
# out of tracking, the robot SLOWLY eases its whole upper body to a safe HOME pose
# and freezes the base; when tracking resumes it SLOWLY eases from HOME back to the
# operator's CURRENT live pose (no snap). Governed by EngagementSupervisor.
SAFE_STREAM_TIMEOUT = 0.40   # s with NO VR payload -> headset off / link down
SAFE_EASE_MAX_DT    = 0.04   # s; max ease-clock advance per tick — a wall-clock
                             # stall must never fast-forward the safety ease.
                             # 0.04/2.5 s ease × a ~3 rad span ≤ 0.05 rad/tick,
                             # which sums with the arm trust step (0.10) to stay
                             # inside the 0.15 rad/tick real-dispatch step bound
                             # under ANY load (exp_real_retarget_chain)
SAFE_SIGNAL_TIMEOUT = 0.50   # s a required signal (head / hand / elbow) stale -> lost
SAFE_RETREAT_TIME   = 4.0    # s to ease from the live pose -> HOME (slow, gentle)
SAFE_ENGAGE_TIME    = 2.5    # s to ease from HOME -> live pose on re-acquire (slow)
# SAFE PARK POSE (operator request 2026-07-27): the HOME the robot eases to must be
# the pose where cutting power is harmless. Lift = HALF the 0.635 m travel (the worm
# gear is non-back-drivable, so it holds at any height unpowered; mid-travel keeps the
# arms at a useful working height and clear of the base). Torso = its +15 deg lean-BACK
# end stop (MJCF ctrlrange +0.2618 rad): resting against the backstop means a torque
# cut lets gravity settle it INTO the stop instead of flopping the chest forward.
SAFE_HOME_LIFT      = 0.3175  # m; lift height at HOME (half of the 0..0.635 travel)
SAFE_HOME_TORSO_DEG = 15.0    # deg; torso lean-BACK at HOME (+ = back; clipped to the
                              # actuator ctrlrange, which caps at exactly +15 deg)
SAFE_REQUIRE_BOTH_HANDS = False  # True: both hands required; False: either hand keeps it live

# --- CLUTCH / re-indexing (single-hand DOUBLE-PINCH toggles a whole-body freeze) ---
# Pinch EITHER hand twice quickly (like a double-click) to FREEZE the robot and
# recenter your arms/body; double-pinch again (either hand) to RELEASE — the world
# transform re-anchors so the robot eases back from its held pose to your CURRENT pose
# (no jump). A deliberate double-pinch is far less twitchy than holding a pose, and a
# single pinch is left free for grasping.
CLUTCH_ENABLE    = True
CLUTCH_PINCH_ON  = 0.030  # m; thumb-index gap below this = pinched (closing edge)
CLUTCH_PINCH_OFF = 0.055  # m; above this = released (hysteresis, so it can't chatter)
CLUTCH_DOUBLE_WINDOW = 0.60  # s; the two pinches must land within this to count as a double
CLUTCH_COOLDOWN  = 0.80   # s; lockout after a toggle (one gesture = one toggle)

# MediaPipe 21-landmark layout the stream sends (index 0 = wrist; the four
# metacarpals are dropped, so each finger is knuckle->...->tip).
TIP_INDICES = {4, 8, 12, 16, 20}
RELATIVE_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20)]   # thumb tip <-> each fingertip
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4),          # thumb
              (0, 5), (5, 6), (6, 7), (7, 8),          # index
              (0, 9), (9, 10), (10, 11), (11, 12),     # middle
              (0, 13), (13, 14), (14, 15), (15, 16),   # ring
              (0, 17), (17, 18), (18, 19), (19, 20)]   # pinky
# Which 21-array index belongs to which finger group (for per-finger scaling).
_GROUP = ({1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb"}
          | {i: "index" for i in (5, 6, 7, 8)}
          | {i: "middle" for i in (9, 10, 11, 12)}
          | {i: "ring" for i in (13, 14, 15, 16)}
          | {i: "pinky" for i in (17, 18, 19, 20)})
