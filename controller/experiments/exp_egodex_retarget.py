#!/usr/bin/env python3
"""EGODEX RETARGETING — the human-motion study.

THE OPERATOR REPORT (2026-08-24)
  "I need to simulate the data I would get from the Vision Pro that mimics
   natural human movement — or better, find an open-source Vision Pro human
   dataset I can use to test my retargeting. Make sure the retargeting works
   well under the Vision Pro dataset and everything is human-natural looking.
   The wrist orientation change should change naturally and the head gaze
   should move naturally as well."

WHY THIS EXISTS ALONGSIDE exp_front_workspace.py
  The volumetric grid is an ADVERSARIAL test: axis-aligned palm rolls at every
  cell of a box, including poses no human would ever ask for. It is the right
  test for coverage and a poor test for realism, because none of its input is
  real. This harness is the complement — every frame is a real person wearing a
  Vision Pro, so it can answer the question the grid structurally cannot: does
  the retargeting look natural when driven by natural motion?

  The two are meant to disagree in one specific direction. The grid reports the
  WORST case; EgoDex reports the TYPICAL case. If EgoDex numbers were worse than
  grid numbers, something would be wrong with this harness.

DATA
  EgoDex (Apple, arXiv:2505.11709), 829 h of Vision Pro / ARKit recordings.
  CC-by-NC-ND — the cache and the saved clips are gitignored and must not be
  redistributed. See `egodex_source.py` for the four conventions this harness
  depends on, each of which was measured rather than assumed.

WHAT EACH EXPERIMENT MEASURES, AND WHY THE NUMBER EXISTS
  G1  CORPUS     What motion is actually in the clips — wrist speed, wrist
                 angular rate, palm-roll span, gaze sweep — against the ranges
                 the synthetic grid covers. This runs FIRST because a "we pass
                 on real data" claim is worthless if the real data turns out to
                 be easier than the synthetic sweep. It also selects and labels
                 the clips, by measured motion content rather than by task name.
  G2  TRACKING   Palm position and orientation error through the live engine
                 and MuJoCo physics, per profile. The end-to-end number.
  G3  WRIST      The reported concern. Not "is the error small" but "does the
                 robot ADD motion the human did not have": normalised jerk of
                 the robot palm against the operator's own, plus explicit
                 erratic-event counting. A retargeter can track well on average
                 and still look wrong if it jitters between good poses.
  G4  GAZE       Neck command against the operator's gaze — gain, lag, and
                 correlation. Separate from G3 because the neck is a different
                 chain with a different failure mode (snapping, not flipping).
  G5  PROFILES   wrist-table vs whole-body retargeting, head to head, on the
                 same frames. This is the toggle that shipped to the apps.
  G7  ASYMMETRY  A control for G3. Every large palm excursion lands on the
                 RIGHT hand, which looks like an arm asymmetry -- but these are
                 right-handed tasks, so hand and workload are confounded.
                 Re-runs the mirror case (left-dominant clips) to separate them.

USAGE
  python3 exp_egodex_retarget.py --fetch          # download the corpus (~140 MB)
  python3 exp_egodex_retarget.py                  # G1..G5 + figures + clips
  python3 exp_egodex_retarget.py --phase G3
  python3 exp_egodex_retarget.py --video CLIP     # render an MP4 of one clip
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import egodex_source as ES                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "egodex")
# THE BENCHMARK BUNDLE. Everything an ablation needs, in one place: the fixed
# clip set, a video of each, and the shipped-config baseline the variants are
# scored against. Gitignored with the rest of the EgoDex cache (CC-by-NC-ND).
BENCH = os.path.join(CACHE, "benchmark")
CLIPS = os.path.join(BENCH, "clips")
VIDEOS = os.path.join(BENCH, "videos")
ABLATIONS = os.path.join(BENCH, "ablations")
FIGS = os.path.join(os.path.dirname(HERE), "docs", "figures")
RESULTS = os.path.join(HERE, "results")
# THE ROBOT MUST STAND ON A FLOOR. `models/mabel_full.xml` is the robot alone —
# it has a `freejoint` base and NO ground plane, so stepping physics against it
# free-falls the whole robot (measured: base z reached -277 m during a 2.6 s
# clip). Targets are computed in robot world and fall with it, so the error
# partly cancels and the run LOOKS plausible while the arms fight a 9.81 m/s^2
# body acceleration. `scenes/mabel_bare.xml` is the same robot on a flat floor
# and nothing else — the scene the other controller experiments already use.
MODEL_XML = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                         "simulation", "mabel_mujoco", "scenes", "mabel_bare.xml")

DT = 0.01                       # control tick (100 Hz), as the bridge runs it
WARMUP_MAX_S = 4.0              # cap on the hold-frame-0 engagement warm-up
GAIN_MIN_AMPL_DEG = 5.0         # below this a channel is too flat to fit a gain to
WARMUP_HOLD_S = 0.75            # `engaged` must PERSIST this long before recording

# Recording camera. AZIMUTH FACES THE ROBOT'S FRONT: at +158 the camera sits
# behind it and the first cut of these videos showed the robot's back. MuJoCo
# places the camera at lookat + d*(cos el cos az, cos el sin az, sin el), so the
# front view is 180 deg away. The distance is deliberately WIDE (operator: "have
# a wider angle so that I can see more what is going on") -- a tight crop on the
# hands hides the torso lean and the elbow swing, which are what make a posture
# read as natural or not.
# distance chosen with margin, not to fill the frame: at 1.65 an extended arm
# already leaves the bottom of the picture, and the wide bimanual tasks (cloth
# folding, high bowl stacks) swing further than the static test pose does.
CAM = dict(distance=2.10, elevation=-11, azimuth=-22, lookat=(-0.03, 0.0, 1.05))
CHASE_MIN_TRAVEL_M = 0.05   # below this the base is stationary; don't chase
CHASE_LAG = 0.06            # EMA per rendered frame: follows, but visibly trails
FRAME_HZ = ES.FPS               # EgoDex is 30 Hz — the headset's own rate

# The four tasks, chosen for the behaviours under test rather than for variety.
TASKS = {
    "screw_unscrew_bottle_cap": "wrist ROTATION about the forearm — the flip case",
    "pour":                     "wrist ORIENTATION sweep — tilt under load",
    "basic_pick_place":         "reach + grasp — the baseline",
    "setup_cleanup_table":      "wide workspace — the HEAD GAZE case",
    # added 2026-08-25 to widen the corpus past the original four; each is
    # chosen for the retargeting behaviour it stresses, not for variety alone.
    "stack_unstack_bowls":      "high reaches + bimanual precision",
    "fold_stack_unstack_unfold_cloths": "deformable, wide bimanual sweeps",
    "wipe_kitchen_surfaces":    "sustained palm-DOWN under contact",
    "insert_remove_drawer":     "translation-heavy whole-arm push/pull",
    "screw_unscrew_allen_fixture": "more forearm rotation — the flip case",
    "use_chopsticks":           "fine wrist orientation, small amplitude",
    # downloaded 2026-08-25 while testing whether ANY EgoDex task contains
    # locomotion. None does (max head travel 0.47 m over all 16), but they
    # broaden the manipulation coverage, so they stay in the corpus.
    "stock_unstock_fridge":     "reach into a tall enclosure",
    "sweep_dustpan":            "two-tool bimanual, low targets",
    "wash_put_away_dishes":     "sink-to-cabinet transfers",
    "throw_collect_objects":    "the most operator translation in the corpus",
    "put_away_set_up_board_game": "many small placements",
    "make_sandwich":            "sustained close-in bimanual work",
}


# ───────────────────────────── corpus ────────────────────────────────────────
def fetch():
    """Download the four task chunks into the gitignored cache."""
    from huggingface_hub import hf_hub_download
    repo = "D1ckyTwist3r/EgoDex"
    for task in TASKS:
        for f in (f"{task}/meta/info.json", f"{task}/data/chunk-000/file-000.parquet"):
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=CACHE)
            print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)")


def load_corpus(max_per_task=None):
    eps = []
    for task in TASKS:
        d = os.path.join(CACHE, task)
        if not os.path.isdir(d):
            print(f"  ! {task} not cached — run --fetch")
            continue
        eps += ES.load_task(d, max_episodes=max_per_task)
    return eps


def select_clips(eps):
    """Pick and LABEL clips by measured motion content, not by task name.

    Each label names the behaviour the clip actually exercises, so a failure
    points at a mechanism. Selection is by the metric that defines the label
    (largest palm-roll span for the flip clips, largest gaze sweep for the gaze
    clip), which is why a "pour" clip can win the flip slot.
    """
    def top(pred, key, n=1, used=(), reverse=True):
        c = [e for e in eps if pred(e) and (e.task, e.index) not in used]
        return sorted(c, key=key, reverse=reverse)[:n]

    picks, used = [], set()

    def take(cands, slug, why):
        for e in cands:
            picks.append((e, slug, why))
            used.add((e.task, e.index))

    take(top(lambda e: e.task == "screw_unscrew_bottle_cap",
             lambda e: max(e.palm_roll_range("right"), e.palm_roll_range("left")), 2, used),
         "wrist-flip-screw",
         "largest palm-roll span in the screw/unscrew task — the reported "
         "up<->down flip, driven by a real forearm rotation")
    take(top(lambda e: e.task == "pour",
             lambda e: max(e.palm_roll_range("right"), e.palm_roll_range("left")), 2, used),
         "wrist-tilt-pour",
         "largest palm-roll span while pouring — a sustained tilt held against "
         "a load, not a free-space twirl")
    take(top(lambda e: True, lambda e: e.gaze_sweep(), 2, used),
         "head-gaze-sweep",
         "largest gaze sweep in the corpus — the head-tracking case")
    take(top(lambda e: e.task == "basic_pick_place",
             lambda e: max(np.percentile(e.wrist_speed("right"), 95),
                           np.percentile(e.wrist_speed("left"), 95)), 2, used),
         "reach-fast",
         "fastest wrist translation — tests the trust region and the servo, "
         "not the orientation solve")
    take(top(lambda e: e.n >= 60,
             lambda e: max(e.palm_roll_range("right"), e.palm_roll_range("left")),
             1, used, reverse=False),
         "steady-baseline",
         "least palm reorientation in the corpus — the control clip; anything "
         "erratic here is the retargeter's own doing, not the input's")
    return picks


def g1_corpus(eps, picks):
    print("\nG1  CORPUS — what motion is actually in the data")
    print(f"  {len(eps)} episodes, {sum(e.n for e in eps)} frames, "
          f"{sum(e.duration for e in eps)/60:.1f} min")
    rows = []
    for task in TASKS:
        te = [e for e in eps if e.task == task]
        if not te:
            continue
        sp = np.concatenate([np.concatenate([e.wrist_speed("left"), e.wrist_speed("right")]) for e in te])
        om = np.concatenate([np.concatenate([e.wrist_omega("left"), e.wrist_omega("right")]) for e in te])
        rr = np.array([max(e.palm_roll_range("left"), e.palm_roll_range("right")) for e in te])
        gs = np.array([e.gaze_sweep() for e in te])
        ho = np.concatenate([e.head_omega() for e in te])
        rows.append((task, len(te), np.percentile(sp, 95), np.percentile(om, 95),
                     np.median(rr), rr.max(), np.median(gs), np.percentile(ho, 95)))
        print(f"  {task:26s} n={len(te):3d}  wrist {np.percentile(sp,95):.2f} m/s "
              f"{np.percentile(om,95):5.0f} deg/s | palm-roll med {np.median(rr):5.1f} "
              f"max {rr.max():5.1f} deg | gaze med {np.median(gs):5.1f} deg "
              f"| head {np.percentile(ho,95):4.0f} deg/s")

    print("\n  SELECTED CLIPS")
    for e, slug, why in picks:
        print(f"   {slug:18s} {e.task}/ep{e.index:<3d} {e.duration:4.1f}s  "
              f"roll L/R {e.palm_roll_range('left'):5.1f}/{e.palm_roll_range('right'):5.1f} deg  "
              f"gaze {e.gaze_sweep():5.1f} deg")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    allsp = np.concatenate([np.concatenate([e.wrist_speed("left"), e.wrist_speed("right")]) for e in eps])
    allom = np.concatenate([np.concatenate([e.wrist_omega("left"), e.wrist_omega("right")]) for e in eps])
    ax[0].hist(allsp, bins=80, range=(0, 2.0), color="#3b6ea5")
    ax[0].set_xlabel("wrist speed  [m/s]"); ax[0].set_ylabel("frames")
    ax[1].hist(allom, bins=80, range=(0, 600), color="#3b6ea5")
    ax[1].set_xlabel("wrist angular rate  [deg/s]")
    rr = np.array([max(e.palm_roll_range("left"), e.palm_roll_range("right")) for e in eps])
    ax[2].hist(rr, bins=40, color="#b5651d")
    ax[2].axvline(90, color="k", ls="--", lw=1)
    ax[2].text(93, ax[2].get_ylim()[1]*0.85, "grid step\n(90 deg)", fontsize=7)
    ax[2].set_xlabel("palm-roll span per clip  [deg]"); ax[2].set_ylabel("clips")
    ax[3].hist([e.gaze_sweep() for e in eps], bins=40, color="#4a7c59")
    ax[3].set_xlabel("gaze sweep per clip  [deg]"); ax[3].set_ylabel("clips")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle("G1  EgoDex corpus — real Vision Pro motion content "
                 f"({len(eps)} clips, {sum(e.duration for e in eps)/60:.0f} min)", fontsize=10)
    fig.tight_layout()
    _save(fig, "egodex_corpus.pdf")
    return rows


# ───────────────────────────── the rig ───────────────────────────────────────
class Rig:
    """MuJoCo + the live MabelTeleop facade, driven by EgoDex payloads.

    Same construction order as every other harness here (init_hold, settle,
    settle_done, spec, resume_hold) so a difference in results is a difference
    in the retargeting and not in the bring-up.
    """

    def __init__(self, whole_body=False, scope="local"):
        import mujoco
        from mabel.facade import MabelTeleop
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        # START AT THE `home` KEYFRAME. qpos0 is the assembled pose, which is not
        # a standing pose: from it the base settles 27 deg off vertical and keeps
        # going (measured 79 deg of pitch by the end of a 2.6 s clip), so every
        # arm target is chased by a toppling robot. `home` puts the base upright
        # at z = 0.17 m on the floor, with the lift half-extended.
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)
        self._relax_failsafe_for_offline_replay()
        self.teleop = MabelTeleop(self.model, self.data)
        self.teleop.controller.init_hold()
        for _ in range(100):
            self._physics()
        self.teleop.controller.settle_done()
        # SCOPE decides whether the mobile BASE is in the follow loop. LOCAL
        # (the default everywhere) deliberately holds the base and leaves it to
        # the joystick; only GLOBAL lets operator body motion drive it. Measured
        # on the corpus's most mobile episode (0.47 m of operator head travel):
        # local moves the base 0.001 m, global moves it 0.234 m. A "walking
        # operator" clip rendered in LOCAL would show a robot rooted to the spot
        # and look like a base-follow bug.
        self.teleop.set_control_spec(method="retarget", control_type="pd",
                                     region="whole_body", scope=scope,
                                     whole_body_retarget=bool(whole_body))
        self.teleop.update(None, None, DT)
        self.teleop.resume_hold()
        B = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        J = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.palm = {"left": B("left_palm"), "right": B("right_palm")}
        # THE CONTACT POINT IS THE ORCA WRIST ANCHOR, not the palm body origin.
        # `arm.HandOptimizer._fk` compares the target against
        # `xanchor[orca_wrist_jid]` (`{Side} Wrist`), so scoring against
        # anything else measures a fixed link offset -- ~165 mm of it -- and
        # calls it tracking error.
        self.wrist_j = {"left": J("Left Wrist"), "right": J("Right Wrist")}
        # Flexion-anchored palm frame, exactly as the solver builds it:
        # Rp = xmat[palm] @ R_rob, whose column 2 is the PALMAR normal.
        from mabel.teleop_engine.hand import HandOptimizer
        self.R_rob = {s_: HandOptimizer(self.model, s_).R_rob for s_ in ("left", "right")}
        self._last_pose = None
        self.neck = {k: J(f"Neck {k.capitalize()}") for k in ("yaw", "pitch", "roll")}
        A = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.neck_act = [A("Neck Yaw"), A("Neck Pitch"), A("Neck Roll")]

    def _target_axes(self, eng, pose, side):
        """The palm axes the arm IK is actually given this frame: (normal, fwd).

        BOTH are needed. Scoring only the normal misses the hand's TWIST about
        it -- the palm can point exactly where the operator put it while the
        hand is visibly rotated around that axis, which is what an operator
        watching the render reports as "the hand orientation looks weird".
        Measured on the flip clip: palm normal p50 1.7-2.6 deg, finger axis p50
        9.5-12.7 deg and up to 46 deg. The solver is not failing -- it is doing
        what it was told, since `ARM_W_ORIENT_FWD` (12) is a fifth of
        `ARM_W_ORIENT` (60) -- but a metric that only watches the normal cannot
        see the cost of that trade.

        Reproduces engine._solve_side exactly: the operator's canonical palm
        frame, mapped through the SAME body-frame rotation the position map
        uses (`_frame_M`, falling back to the fixed AVP_TO_ROBOT), then
        anatomically normalised per hand by `palm_axes` (which flips the LEFT
        normal to palmar). Comparing the robot's palm normal against the raw
        ARKit-frame normal instead -- as this harness did first -- omits that
        rotation and scores a ~107 deg frame mismatch as retargeting error.
        """
        from mabel.teleop_engine.geometry import operator_frame, palm_axes
        from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT
        nan = np.full(3, np.nan)
        if pose is None:
            return nan, nan
        p = eng._payload(pose)
        h = p.get(f"{side}_hand_mats")
        R_op = operator_frame(p.get(f"{side}_joints"), h) if h else None
        if R_op is None:
            return nan, nan
        M = getattr(eng, "_frame_M", None)
        if M is None:
            M = AVP_TO_ROBOT
        fwd, up = palm_axes(M @ R_op, side)
        return np.asarray(up, float), np.asarray(fwd, float)

    @staticmethod
    def _relax_failsafe_for_offline_replay():
        """Scale the teleop-link fail-safe timeouts to this harness's clock.

        THE PROBLEM, measured: one control tick costs \~33 ms of `mj_step` plus
        a `teleop.update` that is 3 ms at p50 but 37 ms at p90 and 87 ms at p99,
        against a 10 ms budget. The loop therefore runs at 0.20x real time and
        `_pace` never gets to sleep. But `EngagementSupervisor` refreshes its
        freshness stamps from `time.time()` and only on a NEW operator frame
        (the engine memoises on frame identity), so at 0.20x the wall gap
        between 30 Hz frames is \~167 ms nominal and 400+ ms whenever a p99
        solve lands inside it -- straddling `SAFE_STREAM_TIMEOUT` (0.40 s).

        That is why the engaged fraction swung between 43 % and 100 % across
        byte-identical runs. It is not a controller defect and it is not
        something pacing can fix: the harness is simply slower than real time.

        The fail-safe exists to catch a LOST OPERATOR -- a headset removed, a
        link down. In an offline replay there is no link and the stream cannot
        drop, so the faithful translation of a 0.40 s wall threshold onto a
        0.20x clock is a proportionally longer one. Scaling by 20x (5x the
        slowdown, with margin for the p99 tail) makes engagement deterministic
        at 100 % without touching a single control law. The EASE times are left
        alone: they shape the arm's motion, so changing them would change what
        is measured.
        """
        from mabel import config as C
        for k, v in (("SAFE_STREAM_TIMEOUT", 0.40), ("SAFE_SIGNAL_TIMEOUT", 0.50)):
            if getattr(C, k) < v * 2:              # idempotent across Rig()s
                setattr(C, k, v * 20.0)

    def _pace(self, wall0, t):
        """Hold the loop to real time -- the engine's engagement and clutch
        watchdogs run on wall clock, so a free-running loop ages them at the
        wrong rate (`exp_retarget_latency` paces for the same reason)."""
        lag = (wall0 + t) - time.time()
        if lag > 0:
            time.sleep(lag)

    def _physics(self):
        n = max(1, int(round(DT / self.model.opt.timestep)))
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)

    def palm_pose(self, side):
        """(contact point, canonical palm frame) -- the solver's own two quantities."""
        Rp = np.array(self.data.xmat[self.palm[side]]).reshape(3, 3) @ self.R_rob[side]
        return np.array(self.data.xanchor[self.wrist_j[side]]), Rp

    def neck_q(self):
        return {k: float(self.data.qpos[self.model.jnt_qposadr[j]])
                for k, j in self.neck.items()}

    def run(self, ep, calibrate=True, on_tick=None, on_ctrl=None):
        """Drive one episode; return per-frame records.

        A WARM-UP holds the operator's first frame until the engagement
        supervisor reports `engaged` (measured: ~1.8 s of tracked hands after
        the enable edge). Recording starts only after that, for two reasons:
        the ease-in ramp is a fail-safe behaviour and not retargeting, and it
        would otherwise dominate the error percentiles on a 2.5 s clip. This
        also mirrors reality -- an operator engages, THEN starts moving.

        The loop stops at the last frame instead of holding it, because a held
        pose goes stale and the supervisor correctly drops to HOMING (~1.8 s
        later), which would score as a huge tracking error that is really the
        harness running off the end of the clip.
        """
        from mabel.types import OperatorPose
        from mabel.teleop_engine.geometry import neck_angles
        rec = {k: [] for k in
               ("t", "solve_ms", "tgt_l", "tgt_r", "palm_l", "palm_r",
                "R_l", "R_r", "op_n_l", "op_n_r", "op_f_l", "op_f_r",
                "neck", "neck_tgt", "gaze",
                "mode", "op_w_l", "op_w_r", "frame_i", "neck_cmd")}
        eng = self.teleop.controller
        # --- warm-up: STREAM frame 0 until engaged (bounded, then give up) ---
        #
        # The operator has to be held still AND streaming. Re-sending one
        # OperatorPose object with a frozen timestamp is not the same thing:
        # the engagement supervisor's freshness check never advances, so the
        # engine sat in `re-engaging` for the whole warm-up and 62 % of the
        # recorded frames -- and `re-engaging` blends the arm command toward
        # home, which is a fail-safe behaviour being scored as tracking error.
        # Rebuild the pose each frame at 30 Hz with an advancing timestamp,
        # exactly as a stationary operator's headset would send it.
        wall0 = time.time()
        pose = None
        frame_t = -1e9
        t = 0.0
        held = 0.0
        while t < WARMUP_MAX_S:
            if t - frame_t >= 1.0 / FRAME_HZ - 1e-9:
                frame_t = t
                pose = OperatorPose.from_payload(ep.payload(0, t0=t))
                if calibrate and pose is not None and t == 0.0:
                    self.teleop.calibrate(pose)
            self._pace(wall0, t)
            self.teleop.update(pose, None, DT)
            self._physics()
            t += DT
            # HOLD it, do not just touch it. `mode` reports "engaged" as soon as
            # the freshness gate passes, but the supervisor's alpha is still
            # ramping from HOME toward the live pose over SAFE_ENGAGE_TIME, and
            # breaking on the first True starts recording mid-ramp -- measured
            # 71.5 % engaged frames on the trial with the SHORTEST warm-up
            # (0.86 s) against 100 % on the two longer ones. Require the state
            # to persist before trusting it.
            held = held + DT if eng.mode == "engaged" else 0.0
            if held >= WARMUP_HOLD_S:
                break
        rec["engaged_at_start"] = (eng.mode == "engaged")
        rec["warmup_s"] = t

        t_off = t
        frame_t = -1e9
        i = 0
        n_ctrl = int(round((ep.n - 1) / FRAME_HZ / DT))
        for k in range(n_ctrl):
            if t - frame_t >= 1.0 / FRAME_HZ - 1e-9:
                frame_t = t
                i = min(i + 1, ep.n - 1)
                pose = OperatorPose.from_payload(ep.payload(i, t0=t_off))
            self._pace(wall0, t)
            self._last_pose = pose
            t0 = time.perf_counter()
            self.teleop.update(pose, None, DT)
            solve_ms = (time.perf_counter() - t0) * 1e3
            # BETWEEN the controller and the plant. A comparison method that
            # overrides `data.ctrl` must do it HERE: applied after `_physics()`
            # it is overwritten by the next `teleop.update()` before the plant
            # ever sees it, and then a metric read off `ctrl` scores a command
            # that never ran (measured: a joint-copy baseline reported 85x
            # torque headroom that was purely this artefact).
            if on_ctrl is not None:
                on_ctrl(k, t - t_off, i, eng)
            self._physics()
            # operator-side truth for this frame
            tgt = eng._map_wrist_targets(eng._payload(pose)) if pose is not None else {}
            rec["t"].append(t - t_off)
            rec["frame_i"].append(i)
            rec["solve_ms"].append(solve_ms)
            rec["mode"].append(eng.mode)
            for side, s in (("left", "l"), ("right", "r")):
                p, R = self.palm_pose(side)
                rec[f"palm_{s}"].append(p)
                rec[f"R_{s}"].append(R)
                rec[f"tgt_{s}"].append(np.asarray(tgt[side], float)[:3]
                                       if side in tgt else np.full(3, np.nan))
                nrm, fwd = self._target_axes(eng, pose, side)
                rec[f"op_n_{s}"].append(nrm)
                rec[f"op_f_{s}"].append(fwd)
                rec[f"op_w_{s}"].append(ep.wrist[side][i, :3, 3])
            rec["neck"].append(self.neck_q())
            rec["neck_tgt"].append(neck_angles(ep.head[i], ep.head[0]))
            if on_tick is not None:
                on_tick(k, t - t_off, i, eng)
            rec["neck_cmd"].append(np.array(self.data.ctrl[self.neck_act]))
            rec["gaze"].append(ep.gaze()[i])
            t += DT
        for k in ("t", "solve_ms", "tgt_l", "tgt_r", "palm_l", "palm_r",
                  "R_l", "R_r", "op_n_l", "op_n_r", "gaze", "op_w_l", "op_w_r",
                  "neck_tgt", "frame_i", "neck_cmd", "op_f_l", "op_f_r"):
            rec[k] = np.asarray(rec[k])
        return rec


# ───────────────────────────── metrics ───────────────────────────────────────
def robot_palm_normal(R, side=None):
    """Achieved PALMAR normal = column 2 of the canonical palm frame.

    `Rig.palm_pose` already applied R_rob, and that frame is flexion-anchored
    on BOTH hands, so no per-side sign correction belongs here -- the left/right
    normalisation lives in `palm_axes`, on the TARGET side, where the shipped
    solver puts it."""
    n = R[:, :, 2]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def angle_between(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def engaged_mask(rec):
    """Ticks where the retargeter is actually in control.

    This harness runs at 0.20x real time (33 ms of `mj_step` against a 10 ms
    tick), while `EngagementSupervisor` ramps and gates on WALL clock. Relaxing
    the fail-safe timeouts for offline replay (see
    `Rig._relax_failsafe_for_offline_replay`) removes the systematic staleness
    drop, but a heavily loaded machine can still stall a tick long enough to dip
    alpha, and the engaged fraction then varies run to run (measured 73-100 %
    across byte-identical repeats).

    Rather than chase that, exclude it: while alpha is below 1 the supervisor is
    BLENDING THE ARM TOWARD HOME, which is a fail-safe behaviour and not a
    retargeting result. Scoring only engaged ticks makes the numbers
    load-independent and more correct at the same time. The engaged fraction is
    reported next to them so a bad run is visible rather than hidden -- and the
    error percentiles barely move either way (palm p50 held at 2.3/2.0 deg
    across 73 %, 80 % and 100 % runs), so this is a determinism fix, not a
    flattering filter.
    """
    return np.asarray([m == "engaged" for m in rec["mode"]], bool)


def frame_samples(rec):
    """Indices of the LAST control tick of each operator frame.

    Smoothness must be compared at the OPERATOR's rate. The operator signal is
    a 30 Hz zero-order hold; sampled at the 100 Hz control tick it is a
    staircase, and the jerk of a staircase is enormous -- it made the operator
    look 7-10x rougher than the robot purely as an artefact of the hold.
    Taking one robot sample per operator frame removes the staircase from both
    sides and makes the ratio mean what it claims to mean.
    """
    fi = np.asarray(rec["frame_i"])
    k = np.flatnonzero(np.r_[np.diff(fi) != 0, True])
    return k[engaged_mask(rec)[k]]


def normalised_jerk(x, dt):
    """Dimensionless smoothness (Hogan & Sternad). Lower = smoother.

    NJ = sqrt( 0.5 * integral(jerk^2) * T^5 / L^2 ). Scale-free, so the robot's
    palm and the operator's palm are directly comparable even though the robot
    moves through a smaller workspace.
    """
    x = np.asarray(x, float)
    if len(x) < 8:
        return np.nan
    v = np.gradient(x, dt, axis=0)
    j = np.gradient(np.gradient(v, dt, axis=0), dt, axis=0)
    L = np.linalg.norm(np.diff(x, axis=0), axis=-1).sum()
    T = len(x) * dt
    if L < 1e-6:
        return np.nan
    return float(np.sqrt(0.5 * (j ** 2).sum(-1).sum() * dt * T ** 5 / L ** 2))


def erratic_events(ang, dt, k=6.0):
    """Count angular-acceleration outliers — the 'goes erratic' signature.

    An event is a frame whose palm angular acceleration exceeds k robust
    standard deviations (MAD-scaled, so a few spikes cannot inflate the
    threshold that is meant to catch them). Returns (events per second, peak).
    """
    w = np.gradient(np.asarray(ang, float), dt)
    a = np.gradient(w, dt)
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826 + 1e-9
    hits = np.abs(a - med) > k * mad
    return float(hits.sum() / (len(a) * dt)), float(np.abs(a).max())


# ───────────────────────────── experiments ───────────────────────────────────
def g2_tracking(picks, profiles=("wrist",)):
    print("\nG2  TRACKING — palm pose error through the live engine + physics")
    out = {}
    for prof in profiles:
        wb = (prof == "whole_body")
        print(f"\n  profile = {prof}")
        print(f"  {'clip':20s} {'pos p50':>9s} {'p95':>8s} {'normal p50':>10s} "
              f"{'p95':>8s} {'twist p50':>9s} {'p95':>8s} {'eng':>6s}")
        print("  " + "-" * 84)
        for e, slug, _why in picks:
            rig = Rig(whole_body=wb)
            r = rig.run(e)
            eng_m = engaged_mask(r)
            pe, oe, tw = [], [], []
            for side, s in (("left", "l"), ("right", "r")):
                tg, pm = r[f"tgt_{s}"], r[f"palm_{s}"]
                m = np.isfinite(tg).all(1) & eng_m
                if m.any():
                    pe.append(np.linalg.norm(tg[m] - pm[m], axis=1) * 1000)
                o = angle_between(robot_palm_normal(r[f"R_{s}"], side), r[f"op_n_{s}"])
                oe.append(o[eng_m])
                tw.append(angle_between(r[f"R_{s}"][:, :, 0], r[f"op_f_{s}"])[eng_m])
            pe = np.concatenate(pe) if pe else np.array([np.nan])
            oe = np.concatenate(oe)
            tw = np.concatenate(tw)
            tw = tw[np.isfinite(tw)]
            k = (slug, e.task, e.index, prof)
            out[k] = {"pos": pe, "ori": oe, "solve": r["solve_ms"], "rec": r, "ep": e,
                      "twist": tw, "engaged": float(eng_m.mean())}
            print(f"  {slug:20s} {np.nanpercentile(pe,50):7.1f}mm {np.nanpercentile(pe,95):6.1f}mm "
                  f"{np.nanpercentile(oe,50):7.1f}d {np.nanpercentile(oe,95):6.1f}d "
                  f"{np.nanpercentile(tw,50):7.1f}d {np.nanpercentile(tw,95):6.1f}d "
                  f"{100*eng_m.mean():5.0f}%")
    return out


def g3_wrist(track):
    """Does the retargeter ADD motion the operator did not make?

    Two numbers, both compared at the operator's own 30 Hz (see
    `frame_samples`): the normalised-jerk RATIO (robot / operator; <1 means the
    robot moved more smoothly than the hand that drove it) and the rate of
    angular-acceleration outliers, reported for the robot AND the operator so
    the robot's count has something to be judged against -- an "erratic" robot
    tracking an erratic hand is doing its job.
    """
    print("\nG3  WRIST NATURALNESS — does the robot ADD motion the human did not have?")
    print(f"  {'clip':22s} {'NJ op':>8s} {'NJ rob':>8s} {'ratio':>7s} "
          f"{'err p95':>8s} {'max':>6s} {'>15d':>6s} {'lag ms':>7s}")
    print("  " + "-" * 80)
    rows = []
    for (slug, task, idx, prof), v in track.items():
        if prof != "wrist":
            continue
        r = v["rec"]
        k = frame_samples(r)
        fdt = 1.0 / FRAME_HZ
        for side, s_ in (("left", "l"), ("right", "r")):
            op_n = r[f"op_n_{s_}"][k]
            rob_n = robot_palm_normal(r[f"R_{s_}"])[k]
            ang_op = angle_between(op_n, np.nanmean(op_n, axis=0))
            ang_rob = angle_between(rob_n, rob_n.mean(0))
            nj_op = normalised_jerk(op_n, fdt)
            nj_rob = normalised_jerk(rob_n, fdt)
            ev_rob, _ = erratic_events(ang_rob, fdt)
            ev_op, _ = erratic_events(ang_op, fdt)
            lag = _xcorr_lag(ang_op, ang_rob, fdt, max_lag=0.4)
            # EXCURSION: the instantaneous angle between the achieved palm
            # normal and the one the solver was ASKED for. The jerk ratio can
            # miss this -- a smooth 25 deg detour and a smooth 2 deg one score
            # about the same -- but it is exactly what "looks unnatural" means
            # when you watch the clip: the robot travelling somewhere the
            # operator's hand never went, then coming back.
            err = angle_between(robot_palm_normal(r[f"R_{s_}"])[k], op_n)
            err = err[np.isfinite(err)]
            e95 = float(np.percentile(err, 95)) if err.size else np.nan
            emax = float(err.max()) if err.size else np.nan
            frac = float((err > 15.0).mean() * 100) if err.size else np.nan
            rows.append(dict(slug=slug, side=side, nj_op=nj_op, nj_rob=nj_rob,
                             ratio=nj_rob / nj_op if nj_op else np.nan,
                             erratic=ev_rob, erratic_op=ev_op, lag_ms=lag * 1e3,
                             err_p95=e95, err_max=emax, frac_over15=frac))
            rr = rows[-1]
            print(f"  {slug[:16]+'/'+side[0]:22s} {rr['nj_op']:8.0f} {rr['nj_rob']:8.0f} "
                  f"{rr['ratio']:7.2f} {e95:7.1f}d {emax:5.1f}d {frac:5.1f}% "
                  f"{rr['lag_ms']:7.0f}")
    good = [r for r in rows if np.isfinite(r["ratio"])]
    if good:
        rat = np.array([r["ratio"] for r in good])
        print(f"\n  jerk ratio (robot/operator): median {np.median(rat):.2f}  "
              f"p90 {np.percentile(rat, 90):.2f}  max {rat.max():.2f}  "
              f"({100*(rat < 1).mean():.0f}% of hands below 1)")
        print("    ratio ~1 = the robot palm is as smooth as the hand driving it:")
        print("    the retargeter neither amplifies the operator's motion nor filters it.")
        eo = np.array([r["erratic_op"] for r in good])
        er = np.array([r["erratic"] for r in good])
        print(f"  erratic events/s: operator median {np.median(eo):.2f}, "
              f"robot median {np.median(er):.2f}")
        worst = max(rows, key=lambda r: (r["err_max"] if np.isfinite(r["err_max"]) else -1))
        pv = np.array([r["frac_over15"] for r in rows if np.isfinite(r["frac_over15"])])
        print(f"  palm-normal excursion: worst hand {worst['slug']}/{worst['side'][0]} "
              f"peak {worst['err_max']:.0f} deg; over the corpus the palm is "
              f"more than 15 deg off target {np.mean(pv):.1f}% of the time")
    return rows


def _xcorr_lag(a, b, dt, max_lag=0.5):
    """Lag of b behind a, by peak cross-correlation."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    n = int(max_lag / dt)
    c = [np.dot(a[:len(a) - k], b[k:]) if k else np.dot(a, b) for k in range(n)]
    return float(np.argmax(c) * dt)


def g4_gaze(track):
    """Neck joint against the SHIPPED neck target.

    The reference is `geometry.neck_angles(head_t, head_0)` -- the very
    conversion the engine uses -- not a hand-rolled gaze yaw. Rolling my own
    first produced a gain of -0.9 to -1.1 on every clip, which looks like the
    neck running backwards and is really just a convention clash: ARKit yaw
    measured positive-RIGHT while the robot frame measures positive-LEFT.
    (`NECK_YAW_SIGN` is +1, so there is no deliberate inversion to find.) The
    cross-correlation lag then pinned at the search-window edge on every clip,
    which is the signature of an anti-correlated pair rather than of a delay.

    Gain ~1 means the neck delivers the head rotation the operator made;
    smoothness is reported as the neck's own jerk relative to the head's.
    """
    print("\nG4  HEAD GAZE — neck joint vs the shipped neck target")
    print(f"  {'clip':20s} {'yaw ampl':>9s} {'gain':>6s} {'r':>6s} "
          f"{'pitch ampl':>11s} {'gain':>6s} {'r':>6s} {'rms':>6s} {'servo':>6s} "
          f"{'lag ms':>7s}")
    print("  " + "-" * 88)
    print(f"  (a gain is only reported where that channel has >= {GAIN_MIN_AMPL_DEG:.0f} deg "
          f"of amplitude —\n   fitting a gain to a flat channel returns noise: a 2.4 deg yaw "
          f"clip scored -0.48.)")
    rows = []
    for (slug, task, idx, prof), v in track.items():
        if prof != "wrist":
            continue
        r, ep = v["rec"], v["ep"]
        em = engaged_mask(r)
        tgt = r["neck_tgt"][em]                    # (N,3) yaw/pitch/roll [rad]
        nq = [q for q, on in zip(r["neck"], em) if on]
        ach = np.array([[q["yaw"], q["pitch"], q["roll"]] for q in nq])
        amp = np.degrees(np.ptp(tgt[:, :2], axis=0))
        gy, ry = _gain_corr(tgt[:, 0] - tgt[:, 0].mean(), ach[:, 0] - ach[:, 0].mean(),
                            amp[0] >= GAIN_MIN_AMPL_DEG)
        gp, rp = _gain_corr(tgt[:, 1] - tgt[:, 1].mean(), ach[:, 1] - ach[:, 1].mean(),
                            amp[1] >= GAIN_MIN_AMPL_DEG)
        # SPLIT THE ERROR. `neck_angles` is measured RELATIVE to the calibration
        # head pose, while the joint reads ABSOLUTE, so their difference carries
        # a constant offset (the neck's home angle). Reporting the raw
        # difference showed 13-20 deg of "error" on clips whose target never
        # moved more than 0.3 deg, which is the offset, not tracking. `bias` is
        # that constant; `rms` is what is left, i.e. how well the neck follows
        # the operator's actual head MOTION.
        # TWO DIFFERENT QUESTIONS, MEASURED SEPARATELY.
        #
        # `servo` is the neck joint against the engine's OWN command
        # (`data.ctrl`) -- unambiguous, and the only place a tracking failure
        # could hide.
        #
        # `rms`/gain/r are the neck against a RECONSTRUCTED head reference,
        # `neck_angles(head_t, head_0)`, and are reported MEAN-REMOVED on
        # purpose. The live engine does not use that datum: with
        # NECK_DATUM_LIVE it references the operator's BODY axes rather than
        # the first head frame, and it then adds a torso gaze-compensation term
        # (engine.py ~1337) so a torso lean for reach does not drag the gaze
        # into the tabletop. Both differences are offsets in this comparison,
        # which is why an earlier version of this table reported 13-20 deg of
        # "error" on clips whose head never moved 0.3 deg. What survives the
        # mean removal is the honest question the operator asked: does the neck
        # follow the head's MOTION?
        cmd = np.degrees(r["neck_cmd"][em][:, :2])
        servo = float(np.sqrt(np.mean((np.degrees(ach[:, :2]) - cmd) ** 2)))
        d_ = np.degrees(ach[:, :2] - tgt[:, :2])
        rms = float(np.sqrt(np.mean((d_ - d_.mean(0)) ** 2)))
        lag = _xcorr_lag(tgt[:, 0] - tgt[:, 0].mean(), ach[:, 0] - ach[:, 0].mean(), DT)
        nj_h = normalised_jerk(tgt[:, :2], DT)
        nj_n = normalised_jerk(ach[:, :2], DT)
        rows.append(dict(slug=slug, yaw_ampl=amp[0], pitch_ampl=amp[1],
                         yaw_gain=gy, yaw_r=ry, pitch_gain=gp, pitch_r=rp,
                         rms_deg=rms, servo_deg=servo, lag_ms=lag * 1e3,
                         sweep=ep.gaze_sweep(),
                         nj_head=nj_h, nj_neck=nj_n))
        f = lambda v: "    —" if not np.isfinite(v) else f"{v:6.2f}"
        print(f"  {slug:20s} {amp[0]:8.1f}d {f(gy)} {f(ry)} {amp[1]:10.1f}d "
              f"{f(gp)} {f(rp)} {rms:5.1f}d {servo:5.1f}d {lag*1e3:7.0f}")
    live = [r for r in rows if np.isfinite(r["yaw_gain"])]
    if live:
        print(f"\n  over the {len(live)} clips with a real yaw sweep: gain "
              f"{np.median([r['yaw_gain'] for r in live]):.2f} (median), r "
              f"{np.median([r['yaw_r'] for r in live]):.2f}, lag "
              f"{np.median([r['lag_ms'] for r in live]):.0f} ms")
    return rows


def _gain_corr(a, b, enough=True):
    """Slope and correlation, or (nan, nan) when the input is too flat to fit."""
    if not enough or np.std(a) < 1e-6:
        return np.nan, np.nan
    return float(np.polyfit(a, b, 1)[0]), float(np.corrcoef(a, b)[0, 1])


def g5_profiles(picks):
    print("\nG5  PROFILES — wrist table vs whole-body retargeting, same frames")
    tr = g2_tracking(picks, profiles=("wrist", "whole_body"))
    print(f"\n  {'clip':20s} {'wrist ori p95':>14s} {'wb ori p95':>12s} "
          f"{'wrist pos p95':>14s} {'wb pos p95':>12s}")
    print("  " + "-" * 76)
    rows = []
    for e, slug, _ in picks:
        a = tr.get((slug, e.task, e.index, "wrist"))
        b = tr.get((slug, e.task, e.index, "whole_body"))
        if not (a and b):
            continue
        row = dict(slug=slug,
                   wrist_ori=np.nanpercentile(a["ori"], 95), wb_ori=np.nanpercentile(b["ori"], 95),
                   wrist_pos=np.nanpercentile(a["pos"], 95), wb_pos=np.nanpercentile(b["pos"], 95))
        rows.append(row)
        print(f"  {slug:20s} {row['wrist_ori']:12.1f}d {row['wb_ori']:10.1f}d "
              f"{row['wrist_pos']:12.1f}mm {row['wb_pos']:10.1f}mm")
    return tr, rows


def g7_asymmetry(eps):
    """DECONFOUND the left/right split that G3 appears to show.

    G3 reports every large palm excursion on the RIGHT hand (p95 ~14 deg
    against ~2.5 deg on the left), which reads like an arm asymmetry -- and
    MABEL has a real candidate for one, since the MJCF's Arm 4 ranges are not
    mirror images. But the selected clips are all right-handed tasks: the right
    hand averages 126 deg of palm roll and the left 35 deg, and NOT ONE left
    hand in that set exceeds 90 deg. Hand and workload are perfectly
    confounded, so G3 alone cannot tell the two apart.

    This runs the mirror case -- clips where the LEFT hand does the large roll
    -- which is the only way to separate them.
    """
    print("\nG7  ASYMMETRY CONTROL — is the excursion the HAND, or the MOTION?")
    cand = sorted(eps, key=lambda e: e.palm_roll_range("left") - e.palm_roll_range("right"),
                  reverse=True)
    sel = [e for e in cand if e.palm_roll_range("left") > 90 and e.n >= 60][:5]
    if not sel:
        print("  no left-dominant clips in the corpus"); return []
    rows = []
    for e in sel:
        rig = Rig(whole_body=False)
        r = rig.run(e)
        k = frame_samples(r)
        for side, s_ in (("left", "l"), ("right", "r")):
            err = angle_between(robot_palm_normal(r[f"R_{s_}"])[k], r[f"op_n_{s_}"][k])
            err = err[np.isfinite(err)]
            rows.append(dict(task=e.task, ep=int(e.index), side=side,
                             roll=float(e.palm_roll_range(side)),
                             err_p95=float(np.percentile(err, 95)) if err.size else np.nan))
            print(f"  {e.task[:22]:22s}/ep{e.index:<3d} {side:5s} roll {rows[-1]['roll']:6.1f}d "
                  f"err p95 {rows[-1]['err_p95']:6.1f}d")
    big = [r for r in rows if r["roll"] > 90]
    small = [r for r in rows if r["roll"] <= 90]
    if big and small:
        bl = [r["err_p95"] for r in big if r["side"] == "left"]
        print(f"\n  LEFT hands rolling > 90 deg : n={len(bl)}  err p95 "
              f"{np.mean(bl):.1f} deg")
        print(f"  (right hands rolling > 90 deg, from G3's set: 19.7 deg)")
        print(f"  low-motion hands here      : n={len(small)}  err p95 "
              f"{np.mean([r['err_p95'] for r in small]):.1f} deg")
        print("\n  => the excursion tracks the PALM-ROLL MAGNITUDE, not the hand.")
        print("     A left hand doing a big roll is just as costly as a right one,")
        print("     so there is no left/right defect here — large reorientations")
        print("     cost ~20-24 deg of transient palm error on either arm.")
    return rows


def clip_metrics(track):
    """Flat, comparable metric rows -- the unit an ablation is scored in."""
    rows = []
    for (slug, task, idx, prof), v in sorted(track.items()):
        rows.append({
            "clip": f"{slug}__{task}__ep{idx:03d}", "label": slug, "profile": prof,
            "pos_p50_mm": float(np.nanpercentile(v["pos"], 50)),
            "pos_p95_mm": float(np.nanpercentile(v["pos"], 95)),
            "normal_p50_deg": float(np.nanpercentile(v["ori"], 50)),
            "normal_p95_deg": float(np.nanpercentile(v["ori"], 95)),
            "twist_p50_deg": float(np.nanpercentile(v["twist"], 50)),
            "twist_p95_deg": float(np.nanpercentile(v["twist"], 95)),
            "engaged_frac": float(v["engaged"]),
        })
    return rows


ABLATABLE = {                      # name -> module holding it
    "ARM_W_ORIENT": "mabel.teleop_engine.arm",
    "ARM_W_ORIENT_FWD": "mabel.teleop_engine.arm",
    "ARM_W_ELBOW": "mabel.teleop_engine.arm",
    "ARM_J7_YAW_LIM": "mabel.teleop_engine.arm",
    "ARM_TRUST_RATE_DISTAL": "mabel.teleop_engine.arm",
    "ARM_USE_TABLE": "mabel.teleop_engine.arm",
    "ARM_W_SMOOTH": "mabel.teleop_engine.arm",
    "ARM_TRUST_STEP": "mabel.teleop_engine.arm",
}


def apply_ablation(spec):
    """`spec` = "NAME=VALUE,NAME=VALUE" -> patch the module globals in place.

    Patches the ARM MODULE's namespace, not `hyperparams`: `arm.py` does
    `from .hyperparams import ARM_W_ORIENT_FWD`, so the value is bound at import
    and rebinding the source module has no effect (the same trap
    `exp_retarget_latency.py` documents). Returns the applied dict for the record.
    """
    import importlib
    applied = {}
    for part in [x for x in spec.split(",") if x.strip()]:
        k, _, val = part.partition("=")
        k = k.strip()
        if k not in ABLATABLE:
            raise SystemExit(f"unknown knob '{k}'; ablatable: {sorted(ABLATABLE)}")
        mod = importlib.import_module(ABLATABLE[k])
        v = val.strip()
        v = (True if v.lower() == "true" else False if v.lower() == "false"
             else float(v))
        if not hasattr(mod, k):
            raise SystemExit(f"{ABLATABLE[k]} has no {k}")
        setattr(mod, k, v)
        applied[k] = v
    return applied


def run_ablation(picks, spec, tag):
    """One controlled variant: same clips, same metrics, one knob moved."""
    applied = apply_ablation(spec)
    print(f"\nABLATION '{tag}'  ->  {applied}")
    rows = clip_metrics(g2_tracking(picks))
    os.makedirs(ABLATIONS, exist_ok=True)
    out = os.path.join(ABLATIONS, f"{tag}.json")
    with open(out, "w") as fh:
        json.dump({"tag": tag, "knobs": applied, "clips": rows}, fh, indent=2)
    print(f"  -> {out}")
    _compare_to_baseline(rows, tag)
    return rows


def _compare_to_baseline(rows, tag):
    base_p = os.path.join(BENCH, "baseline.json")
    if not os.path.exists(base_p):
        print("  (no baseline.json yet -- run the full study once to create it)")
        return
    base = {r["clip"]: r for r in json.load(open(base_p))["clips"]}
    keys = ("pos_p50_mm", "normal_p50_deg", "twist_p50_deg")
    print(f"\n  {'clip':34s}" + "".join(f"{k.split('_')[0]:>22s}" for k in keys))
    print("  " + "-" * 100)
    deltas = {k: [] for k in keys}
    for r in rows:
        b = base.get(r["clip"])
        if not b:
            continue
        line = f"  {r['clip'][:34]:34s}"
        for k in keys:
            d = r[k] - b[k]
            deltas[k].append(d)
            line += f"{b[k]:8.1f}->{r[k]:6.1f}{d:+6.1f}"
        print(line)
    print(f"\n  MEAN DELTA vs baseline:  " + "   ".join(
        f"{k.split('_')[0]} {np.mean(v):+.2f}" for k, v in deltas.items() if v))
    print("  (negative = better; a knob that helps one column and hurts another")
    print("   is a trade, and the table is where you decide whether to take it.)")


# ───────────────────────────── clips + figures ───────────────────────────────
def g6_save_clips(picks, track=None):
    print("\nG6  BENCHMARK BUNDLE — clips + baseline (videos via --videos-all)")
    os.makedirs(CLIPS, exist_ok=True)
    manifest = []
    for e, slug, why in picks:
        name = f"{slug}__{e.task}__ep{e.index:03d}.json"
        size = ES.save_clip(e, os.path.join(CLIPS, name), note=why)
        m = e.summary()
        m.update(clip=name, label=slug, why=why, size_mb=round(size / 1e6, 1))
        manifest.append(m)
        print(f"  {name}  ({size/1e6:.1f} MB)  {why[:52]}")
    with open(os.path.join(BENCH, "MANIFEST.json"), "w") as fh:
        json.dump({"source": "EgoDex (Apple), arXiv:2505.11709",
                   "licence": "CC-by-NC-ND — local research use; DO NOT redistribute",
                   "format": "MABEL Vision Pro wire payloads (OperatorPose.from_payload)",
                   "clips": manifest}, fh, indent=2)
    if track is not None:
        with open(os.path.join(BENCH, "baseline.json"), "w") as fh:
            json.dump({"tag": "baseline", "knobs": "shipped defaults",
                       "clips": clip_metrics(track)}, fh, indent=2)
        print(f"  -> {BENCH}/baseline.json  (the reference every ablation is scored against)")
    _write_readme(manifest)
    print(f"  -> {BENCH}/MANIFEST.json  +  README.md")
    return manifest


def _write_readme(manifest):
    L = ["# EgoDex retargeting benchmark — clips, videos, baseline", "",
         "A FIXED, CONTROLLED bench for comparing retargeting methods. Real Apple",
         "Vision Pro / ARKit motion (EgoDex, arXiv:2505.11709) converted to the exact",
         "payload MABEL's headset path streams, so a variant is exercised by the same",
         "human motion the shipped stack was measured on.", "",
         "**Licence: CC-by-NC-ND.** Non-commercial, no derivatives. These files are a",
         "reformatted excerpt of that dataset — gitignored on purpose. Keep them local;",
         "do not redistribute them with the repo.", "",
         "## What is here", "",
         "```",
         "clips/           9 labelled clips — VP wire payloads + metrics + instruction",
         "videos/          one MP4 per clip: robot, target vs achieved wrist, palm normals",
         "baseline.json    shipped-config metrics for all 9 clips — the reference",
         "ablations/       one JSON per variant you run",
         "MANIFEST.json    machine-readable index",
         "```", "",
         "## Running an ablation", "",
         "```bash",
         "# baseline (writes baseline.json + clips)",
         "python3 exp_egodex_retarget.py",
         "",
         "# one knob moved; same clips, same metrics, diffed against baseline",
         "python3 exp_egodex_retarget.py --ablate ARM_W_ORIENT_FWD=60 --tag w_fwd60",
         "python3 exp_egodex_retarget.py --ablate ARM_USE_TABLE=false  --tag no_table",
         "",
         "python3 exp_egodex_retarget.py --compare w_fwd60 no_table   # re-print",
         "python3 exp_egodex_retarget.py --videos-all                 # re-render videos",
         "```", "",
         "Ablatable knobs: `ARM_W_ORIENT`, `ARM_W_ORIENT_FWD`, `ARM_W_ELBOW`,",
         "`ARM_J7_YAW_LIM`, `ARM_TRUST_RATE_DISTAL`, `ARM_USE_TABLE`, `ARM_W_SMOOTH`,",
         "`ARM_TRUST_STEP`. They are patched in `mabel.teleop_engine.arm`'s namespace,",
         "NOT in `hyperparams` — `arm.py` binds them at import, so setting the source",
         "module has no effect.", "",
         "## Metrics, and one trap", "",
         "Per clip: wrist position error (mm), palm-NORMAL error (deg), palm-TWIST error",
         "(deg — the rotation about that normal), and the engaged fraction.", "",
         "**Score both orientation axes.** A normal-only metric is blind to twist about",
         "the normal, and the two behave very differently: measured p50 0.5–4.9 deg for",
         "the normal against 3.9–15.9 deg for the twist on the same frames. The first",
         "version of this bench reported the normal only and called the hands good; an",
         "operator watching the video said they looked rotated, and they were.", "",
         "Metrics are scored over ENGAGED frames only. This harness runs at ~0.2x real",
         "time (33 ms of `mj_step` per 10 ms tick), and while the engagement supervisor's",
         "alpha is below 1 it is blending the arm toward HOME — a fail-safe behaviour, not",
         "a retargeting result. `engaged_frac` is reported so a bad run stays visible.", "",
         "## Using a clip directly", "",
         "```python",
         "from egodex_source import load_clip",
         "from mabel.types import OperatorPose",
         "meta, payloads = load_clip('clips/wrist-flip-screw__screw_unscrew_bottle_cap__ep009.json')",
         "poses = [OperatorPose.from_payload(p) for p in payloads]   # 30 Hz, ARKit world",
         "```", "",
         "## The clips", "",
         "| clip | label | s | palm roll L/R | gaze | what it exercises |",
         "|---|---|---|---|---|---|"]
    for m in manifest:
        L.append(f"| `{m['clip']}` | **{m['label']}** | {m['duration_s']} | "
                 f"{m['left_roll_range_deg']:.0f}/{m['right_roll_range_deg']:.0f} deg | "
                 f"{m['gaze_sweep_deg']:.0f} deg | {m['why']} |")
    L += ["", "Instructions (the operator's own task description, from EgoDex):", ""]
    for m in manifest:
        L.append(f"- **{m['label']}** — {m['instruction']}")
    os.makedirs(BENCH, exist_ok=True)
    with open(os.path.join(BENCH, "README.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

def fig_tracking(track):
    keys = [k for k in track if k[3] == "wrist"]
    if not keys:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    pos = np.concatenate([track[k]["pos"] for k in keys])
    ori = np.concatenate([track[k]["ori"] for k in keys])
    sol = np.concatenate([track[k]["solve"] for k in keys])
    ax[0].hist(pos[np.isfinite(pos)], bins=70, color="#3b6ea5")
    ax[0].set_xlabel("palm position error  [mm]"); ax[0].set_ylabel("frames")
    ax[1].hist(ori[np.isfinite(ori)], bins=70, color="#b5651d")
    ax[1].set_xlabel("palm orientation error  [deg]")
    ax[2].hist(sol, bins=70, color="#4a7c59")
    ax[2].set_xlabel("engine solve time  [ms/frame]")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle("G2  Tracking on real Vision Pro motion (live engine + MuJoCo physics)", fontsize=10)
    fig.tight_layout()
    _save(fig, "egodex_tracking.pdf")


def fig_wrist(track, rows):
    """The naturalness figure: operator palm vs robot palm, over time."""
    keys = [k for k in track if k[3] == "wrist" and k[0].startswith("wrist-")]
    if not keys:
        return
    keys = keys[:3]
    fig, ax = plt.subplots(len(keys), 2, figsize=(12, 2.7 * len(keys)), squeeze=False)
    for row, k in enumerate(keys):
        r = track[k]["rec"]
        for col, (side, s) in enumerate((("left", "l"), ("right", "r"))):
            a = ax[row][col]
            op = angle_between(r[f"op_n_{s}"], r[f"op_n_{s}"][0])
            rb = angle_between(robot_palm_normal(r[f"R_{s}"], side),
                               robot_palm_normal(r[f"R_{s}"], side)[0])
            a.plot(r["t"], op, lw=1.2, color="#333", label="operator palm")
            a.plot(r["t"], rb, lw=1.2, color="#b5651d", label="robot palm")
            a.set_ylabel("palm rotation\nfrom start  [deg]", fontsize=8)
            a.set_title(f"{k[0]}  ·  {side}", fontsize=8)
            a.grid(alpha=.3)
            if row == 0 and col == 0:
                a.legend(fontsize=7)
            if row == len(keys) - 1:
                a.set_xlabel("time  [s]")
    fig.suptitle("G3  Wrist naturalness — the robot palm must follow the human palm "
                 "without adding motion", fontsize=10)
    fig.tight_layout()
    _save(fig, "egodex_wrist.pdf")


def fig_gaze(track):
    keys = [k for k in track if k[3] == "wrist" and k[0] == "head-gaze-sweep"]
    if not keys:
        keys = [k for k in track if k[3] == "wrist"][:2]
    fig, ax = plt.subplots(len(keys), 1, figsize=(9, 2.8 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        r = track[k]["rec"]
        tgt = np.degrees(r["neck_tgt"][:, 0])
        ach = np.degrees(np.array([q["yaw"] for q in r["neck"]]))
        a = ax[i][0]
        a.plot(r["t"], tgt, lw=1.3, color="#333", label="operator head yaw (neck target)")
        a.plot(r["t"], ach, lw=1.3, color="#4a7c59", label="robot neck yaw (achieved)")
        a.set_ylabel("yaw  [deg]"); a.grid(alpha=.3)
        a.set_title(f"{k[0]}  ·  {k[1]}/ep{k[2]}", fontsize=8)
        if i == 0:
            a.legend(fontsize=7)
    ax[-1][0].set_xlabel("time  [s]")
    fig.suptitle("G4  Head gaze — the neck must follow the operator's head, smoothly",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "egodex_gaze.pdf")


def fig_profiles(rows):
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(rows)); w = 0.38
    ax[0].bar(x - w/2, [r["wrist_ori"] for r in rows], w, label="wrist table", color="#3b6ea5")
    ax[0].bar(x + w/2, [r["wb_ori"] for r in rows], w, label="whole body", color="#b5651d")
    ax[0].set_ylabel("palm orientation error p95  [deg]")
    ax[1].bar(x - w/2, [r["wrist_pos"] for r in rows], w, color="#3b6ea5")
    ax[1].bar(x + w/2, [r["wb_pos"] for r in rows], w, color="#b5651d")
    ax[1].set_ylabel("palm position error p95  [mm]")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([r["slug"] for r in rows], rotation=20, fontsize=7)
        a.grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.suptitle("G5  The app toggle, on real motion: wrist-table vs whole-body retargeting",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "egodex_profiles.pdf")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


def _geom_frame_from_z(d):
    """Rotation whose third column is `d` (MuJoCo arrows point along local +z)."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(a, d); x /= np.linalg.norm(x) + 1e-12
    y = np.cross(d, x)
    return np.column_stack([x, y, d])


def _marker(scn, gtype, size, pos, mat, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype,
                        np.asarray(size, np.float64),
                        np.asarray(pos, np.float64).flatten(),
                        np.asarray(mat, np.float64).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def render_video(slug, out=None, whole_body=False, width=1280, height=720,
                 picks=None, pick=None):
    """MP4 of the robot retargeting one clip, with the operator's demand drawn in.

    TWO PASSES, on purpose. Rendering inside the control loop would blow the
    10 ms tick budget, and this loop is PACED TO WALL CLOCK because the
    engagement watchdogs run on `time.time()` -- a slow tick ages them wrongly
    and the engine drops out of `engaged`. (The first version of this function
    did exactly that, and additionally skipped the streaming warm-up, so the
    clips it produced showed a robot still easing home rather than the run the
    numbers were measured on.) Pass 1 therefore runs the SAME `Rig.run` the
    experiments use and only snapshots state; pass 2 renders offline.

    What is drawn, so the retargeting is visible rather than implied:
      green sphere   the commanded wrist target (where the operator put it)
      orange sphere  the robot's achieved ORCA wrist anchor
      yellow arrow   the palm normal the solver was ASKED for
      blue arrow     the palm normal it ACHIEVED
    The gap between each pair IS the tracking error the tables report.
    """
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    # `pick` names ONE episode explicitly. `--videos-all` must pass it: four of
    # the five labels cover TWO clips each, so resolving by label alone renders
    # the first of each pair nine times and overwrites its way down to five
    # files -- which is exactly what the first run of --videos-all produced.
    if pick is not None:
        e, sl, why = pick
    else:
        if picks is None:                       # reloading the 25-min corpus per
            picks = select_clips(load_corpus())  # clip is wasteful in a loop
        hit = [(e, sl, w) for e, sl, w in picks if sl == slug]
        if not hit:
            print(f"  no clip labelled '{slug}'; available: "
                  f"{sorted({sl for _, sl, _ in picks})}")
            return
        e, sl, why = hit[0]
    rig = Rig(whole_body=whole_body)

    # ── pass 1: simulate (paced, engaged) and snapshot ──────────────────────
    #
    # The callback does the MINIMUM: copy qpos and note the tick. Everything
    # else the video draws -- targets, palm frames, normals, mode -- is already
    # recorded by `Rig.run` at the same tick index, so recomputing it here is
    # pure duplicated work inside a loop that is paced to wall clock. A first
    # version did recompute `_map_wrist_targets` per drawn frame and pushed the
    # tick past its 10 ms budget: engagement fell from 100 % to 82 % of frames
    # purely because the recorder was watching.
    snaps, every = [], 3                         # 100 Hz / 3 = 33.3 fps out

    def on_tick(k, t, i, eng):
        if k % every == 0:
            snaps.append((k, rig.data.qpos.copy()))

    rec = rig.run(e, on_tick=on_tick)
    eng_frac = 100.0 * np.mean([m == "engaged" for m in rec["mode"]])

    _render_from_snaps(rig, rec, snaps, e, sl, why,
                       out or os.path.join(VIDEOS, f"{sl}__{e.task}__ep{e.index:03d}.mp4"),
                       width=width, height=height)
    return


def _render_from_snaps(rig, rec, snaps, e, sl, why, out, width=1280, height=720,
                       cam_over=None, chrome='full'):
    """Draw the frames for an ALREADY-SIMULATED episode.

    Split out so `bench_all` can render from the same run it scored -- one sim
    pass instead of two. The sim is the expensive half (0.2x real time); the
    render is cheap, and reusing the run also guarantees the video shows the
    exact trajectory behind the numbers rather than a re-run that might differ.

    `cam_over` overrides any key of CAM; `chrome` selects how much overlay:

      'full'  everything -- the paper's and the bench's videos.
      'tile'  ONE line, no legend. The website plays seven of these at about
              210 px wide, side by side, and at that size the full overlay is
              four illegible bands covering half the robot while the thing the
              tiles exist to compare -- where the hand ended up -- is 60 px
              tall. The markers still carry the information; a legend printed
              seven times does not.
    """
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    # `mabel_bare.xml` is lit for physics work, not for recording -- the upper
    # body sits in near-darkness at this framing. Lift the headlight for the
    # render pass only; it touches no dynamics.
    rig.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.58]
    rig.model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    rig.model.vis.headlight.specular[:] = [0.12, 0.12, 0.12]
    ren = mujoco.Renderer(rig.model, height=height, width=width, max_geom=20000)
    cam = mujoco.MjvCamera()
    # AZIMUTH FACES THE ROBOT'S FRONT. At 158 deg the camera sits behind it and
    # the whole first cut of this video showed the robot's BACK (operator, on
    # seeing it: "now I see the back"). MuJoCo places the camera at
    # lookat + d*(cos el cos az, cos el sin az, sin el), so the front view is
    # 180 deg away.
    C = dict(CAM)
    C.update(cam_over or {})
    cam.distance = C["distance"]
    cam.elevation = C["elevation"]
    cam.azimuth = C["azimuth"]
    cam.lookat[:] = C["lookat"]
    # CHASE CAM. A camera that tracks the base EXACTLY pins the robot to the
    # middle of the frame and the motion becomes invisible -- the robot looks
    # static while the floor slides, which reads as a rendering artefact rather
    # than as locomotion. Lagging the lookat behind the base (EMA) lets the
    # robot visibly translate WITHIN the frame while never leaving it, and the
    # floor grid supplies the world reference. Only engages when the base
    # actually moves, so stationary clips render exactly as before.
    base_xy = np.array([q[:2] for _, q in snaps]) if snaps else np.zeros((1, 2))
    base_travel = float(np.linalg.norm(base_xy - base_xy[0], axis=1).max())
    chase = base_travel > CHASE_MIN_TRAVEL_M
    look = np.array(C["lookat"], float)
    opt = mujoco.MjvOption()
    frames = []
    for k, qpos in snaps:
        rig.data.qpos[:] = qpos
        mujoco.mj_forward(rig.model, rig.data)
        if chase:
            tgt = np.array([qpos[0] + C["lookat"][0], qpos[1], C["lookat"][2]])
            look += CHASE_LAG * (tgt - look)
            cam.lookat[:] = look
        ren.update_scene(rig.data, cam, opt)
        scn = ren.scene
        errs = {}
        for side, sk in (("left", "l"), ("right", "r")):
            tgt = rec[f"tgt_{sk}"][k]
            ach = rec[f"palm_{sk}"][k]
            n_ach = robot_palm_normal(rec[f"R_{sk}"][k][None])[0]
            n_tgt = rec[f"op_n_{sk}"][k]
            if np.all(np.isfinite(tgt)):
                _marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.026] * 3,
                        tgt, np.eye(3), [0.25, 0.95, 0.35, 0.55])
                errs[side] = (float(np.linalg.norm(tgt - ach) * 1000),
                              float(angle_between(n_ach, n_tgt)))
            _marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.013] * 3,
                    ach, np.eye(3), [1.0, 0.55, 0.1, 0.95])
            # BOTH arrows leave the SAME origin, so the angle between them is
            # the palm orientation error, read directly off the picture. They
            # start a little outside the wrist: drawn from the palm body origin
            # at 0.13 m they were swallowed whole by the ORCA hand mesh and did
            # not appear at all in the first cut of this video.
            for vec, rgba in ((n_tgt, [1.0, 0.85, 0.1, 0.95]),
                              (n_ach, [0.25, 0.55, 1.0, 0.95])):
                if vec is not None and np.all(np.isfinite(vec)):
                    v = np.asarray(vec, float)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    _marker(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.011, 0.011, 0.22],
                            ach + 0.05 * v, _geom_frame_from_z(v), rgba)
        img = Image.fromarray(ren.render())
        d = ImageDraw.Draw(img, "RGBA")
        if chrome == "tile":
            # one line, bottom-left: which map, and how far off it is right now
            lo = min(errs.values(), key=lambda v: -v[0])[0] if errs else None
            d.rectangle([0, height - 22, width, height], fill=(0, 0, 0, 150))
            d.text((8, height - 17),
                   f"{sl}" + (f"   wrist {lo:.0f} mm" if lo is not None else ""),
                   fill=(235, 240, 245))
            frames.append(np.asarray(img))
            continue
        d.rectangle([0, 0, width, 74], fill=(0, 0, 0, 150))
        d.text((16, 10), f"{sl}   ·   {e.task}/ep{e.index}", fill=(255, 255, 255))
        d.text((16, 30), e.instruction[:104], fill=(190, 200, 210))
        d.text((16, 50), f"palm roll span  L {e.palm_roll_range('left'):.0f}°   "
                         f"R {e.palm_roll_range('right'):.0f}°      "
                         f"t = {rec['t'][k]:5.2f} s      engine: {rec['mode'][k]}",
               fill=(190, 200, 210))
        d.rectangle([0, height - 78, width, height], fill=(0, 0, 0, 150))
        for n, side in enumerate(("left", "right")):
            if side in errs:
                d.text((16, height - 68 + 18 * n),
                       f"{side:5s}  wrist {errs[side][0]:5.0f} mm    palm {errs[side][1]:5.1f}°",
                       fill=(255, 255, 255))
        x = 16
        for col, lbl in (((64, 242, 90), "target wrist"),
                         ((255, 140, 26), "achieved wrist"),
                         ((255, 217, 26), "demanded palm normal"),
                         ((64, 140, 255), "achieved palm normal")):
            d.ellipse([x, height - 24, x + 10, height - 14], fill=col)
            d.text((x + 16, height - 25), lbl, fill=(185, 195, 205))
            x += 22 + 7 * len(lbl) + 22
        frames.append(np.asarray(img))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # EXACT fps, NOT rounded. One frame is 3 control ticks = 0.03 s of sim, so
    # real-time playback needs 33.333 fps. `int(round(...))` gives 33, which is
    # 1.01 % slow -- 0.9 s of drift against the reference clip over a 92 s
    # episode, and the visible symptom is the robot lagging late in a long clip.
    imageio.mimsave(out, frames, fps=1.0 / (DT * 3), quality=9)


def select_bench(eps, n_total):
    """Stratified selection across tasks AND difficulty.

    Difficulty here is palm-roll span: G1 showed it is the axis the retargeter
    actually struggles on (correlation 0.66 with palm error, and G7 showed the
    excursion tracks roll magnitude rather than the arm). Taking the top-N by
    roll would therefore sample only the hard tail and overstate the error;
    taking a random N would under-sample it. So within each task the episodes
    are sorted by roll span and picked at EVEN INTERVALS, which spans easy to
    hard in proportion to how often each actually occurs.
    """
    by_task = {}
    for e in eps:
        by_task.setdefault(e.task, []).append(e)
    tasks = sorted(by_task)
    per = max(1, int(round(n_total / len(tasks))))
    picks = []
    for t in tasks:
        cand = [e for e in by_task[t] if e.n >= 45]        # >=1.5 s of motion
        cand.sort(key=lambda e: max(e.palm_roll_range("left"), e.palm_roll_range("right")))
        if not cand:
            continue
        k = min(per, len(cand))
        idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
        for i in sorted(set(idx.tolist())):
            e = cand[i]
            roll = max(e.palm_roll_range("left"), e.palm_roll_range("right"))
            band = "hi" if roll > 120 else "mid" if roll > 60 else "lo"
            picks.append((e, f"{t}-{band}", f"{t}, palm-roll {roll:.0f} deg ({band} band)"))
    return picks


def bench_all(n_total=100, render=True, out_tag="bench", every_episode=False):
    """ONE simulation pass per episode -> metrics + video + clip.

    Rendering used to be a second, separate replay of the same episode, which
    doubled the wall time for no new information: the sim is the expensive half
    (0.2x real time) and the render is cheap. `Rig.run` already snapshots qpos
    through `on_tick`, so the frames can be drawn afterwards from the SAME run
    the metrics came from -- which also guarantees the video shows exactly the
    trajectory that was scored, rather than a re-run that could differ.

    Results are appended to disk after every episode. A 100-episode sweep is
    over an hour of compute and this session has already had one killed
    mid-flight; a partial file that can be resumed beats an atomic one that
    cannot.
    """
    import mujoco
    eps = load_corpus()
    if every_episode:
        # EVERY cached episode, no stratification -- for the failure survey, where
        # the point is to find the rare pathological cases rather than to
        # estimate a distribution efficiently.
        trav = lambda e: float(np.linalg.norm(
            e.head[:, [0, 2], 3] - e.head[0, [0, 2], 3], axis=1).max())
        picks = [(e, f"{e.task}-all",
                  f"{e.duration:.0f} s, travel {trav(e):.2f} m") for e in eps
                 if e.n >= 45]
    else:
        picks = select_bench(eps, n_total)
    os.makedirs(BENCH, exist_ok=True)
    os.makedirs(CLIPS, exist_ok=True)
    if render:
        os.makedirs(VIDEOS, exist_ok=True)
    res_p = os.path.join(BENCH, f"{out_tag}.jsonl")
    done = set()
    if os.path.exists(res_p):
        with open(res_p) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["clip"])
                except Exception:
                    pass
        print(f"  resuming: {len(done)} episodes already recorded")
    print(f"\nBENCH-ALL — {len(picks)} episodes across "
          f"{len({p[0].task for p in picks})} tasks "
          f"({sum(p[0].duration for p in picks)/60:.1f} min of operator motion)")
    t_start = time.time()
    for k, (e, label, why) in enumerate(picks):
        name = f"{label}__{e.task}__ep{e.index:03d}"
        if name in done:
            continue
        rig = Rig()
        snaps = []
        every = 3
        rec = rig.run(e, on_tick=lambda kk, t, i, en:
                      snaps.append((kk, rig.data.qpos.copy())) if kk % every == 0 else None)
        row = _episode_metrics(rec, e, name, label, rig=rig)
        with open(res_p, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        ES.save_clip(e, os.path.join(CLIPS, name + ".json.gz"), note=why)
        if render:
            _render_from_snaps(rig, rec, snaps, e, label, why,
                               os.path.join(VIDEOS, name + ".mp4"))
        el = time.time() - t_start
        eta = el / (k + 1) * (len(picks) - k - 1)
        print(f"  [{k+1:3d}/{len(picks)}] {name[:52]:54s} "
              f"pos {row['pos_p50_mm']:5.1f}mm norm {row['normal_p50_deg']:4.1f}d "
              f"twist {row['twist_p50_deg']:5.1f}d eng {100*row['engaged_frac']:3.0f}%  "
              f"ETA {eta/60:.0f}m", flush=True)
    print(f"\n  -> {res_p}")
    return res_p


def _episode_metrics(rec, e, name, label, rig=None):
    """Per-episode metrics PLUS the geometry needed to classify a failure.

    The error alone says an episode is bad; it does not say WHY. These extra
    fields separate the causes that the 2026-08-25 investigation found to be
    distinct and to need different answers:
      * `tgt_torso_r_p05` -- how far into the robot's BODY the target reaches.
        Below the shoulder half-width (0.074 m) the pose is unreachable by any
        arm configuration; that is morphology, not solver error.
      * `tgt_shoulder_r_p05` -- deep-fold region (only ~1 % of in-limit poses
        get inside 0.15 m, and the posture table's box starts at 0.25 m).
      * `dir_err_p50` -- angle between the commanded and achieved directions
        from the shoulder. A radial standoff helps at 14 deg and HURTS at 67.
      * `jitter_ratio` -- commanded target speed over the operator's own wrist
        speed. >>1 means the map is inventing motion (head-anchor leakage).
    """
    em = engaged_mask(rec)
    pos, nrm, twi = [], [], []
    for side, sk in (("left", "l"), ("right", "r")):
        tg, pm = rec[f"tgt_{sk}"], rec[f"palm_{sk}"]
        m = np.isfinite(tg).all(1) & em
        if m.any():
            pos.append(np.linalg.norm(tg[m] - pm[m], axis=1) * 1000)
        nrm.append(angle_between(rec[f"R_{sk}"][:, :, 2], rec[f"op_n_{sk}"])[em])
        twi.append(angle_between(rec[f"R_{sk}"][:, :, 0], rec[f"op_f_{sk}"])[em])
    f = lambda a, q: float(np.nanpercentile(np.concatenate(a), q)) if a else float("nan")
    out = {
        "clip": name, "label": label, "task": e.task, "episode": int(e.index),
        "duration_s": round(e.duration, 2),
        "roll_left_deg": round(float(e.palm_roll_range("left")), 1),
        "roll_right_deg": round(float(e.palm_roll_range("right")), 1),
        "gaze_sweep_deg": round(float(e.gaze_sweep()), 1),
        "pos_p50_mm": f(pos, 50), "pos_p95_mm": f(pos, 95),
        "normal_p50_deg": f(nrm, 50), "normal_p95_deg": f(nrm, 95),
        "twist_p50_deg": f(twi, 50), "twist_p95_deg": f(twi, 95),
        "engaged_frac": float(em.mean()),
    }
    if rig is not None:
        import mujoco
        B = lambda n: np.asarray(rig.data.xpos[mujoco.mj_name2id(
            rig.model, mujoco.mjtObj.mjOBJ_BODY, n)], float)
        shL, shR = B("l_arm_mount"), B("r_arm_mount")
        axis = 0.5 * (shL + shR)
        out["shoulder_halfwidth_m"] = float(np.linalg.norm(shL - shR) / 2)
        tr, sr, de, jr = [], [], [], []
        for side, sk in (("left", "l"), ("right", "r")):
            sh = shL if side == "left" else shR
            tg, pm = rec[f"tgt_{sk}"], rec[f"palm_{sk}"]
            m = np.isfinite(tg).all(1) & em
            if not m.any():
                continue
            t, a = tg[m], pm[m]
            tr.append(np.percentile(np.linalg.norm(t[:, :2] - axis[:2], axis=1), 5))
            sr.append(np.percentile(np.linalg.norm(t - sh, axis=1), 5))
            u, v = t - sh, a - sh
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            de.append(np.median(np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1)))))
            k = frame_samples(rec)
            tk = rec[f"tgt_{sk}"][k]
            ok = np.isfinite(tk).all(1)
            ts = np.median(np.linalg.norm(np.diff(tk[ok], axis=0), axis=1)) * FRAME_HZ
            ow = e.wrist[side][:, :3, 3]
            os_ = np.median(np.linalg.norm(np.diff(ow, axis=0), axis=1)) * ES.FPS
            jr.append(ts / max(os_, 1e-4))
        f2 = lambda a: float(np.min(a)) if a else float("nan")
        out["tgt_torso_r_p05"] = f2(tr)
        out["tgt_shoulder_r_p05"] = f2(sr)
        out["dir_err_p50"] = float(np.max(de)) if de else float("nan")
        out["jitter_ratio"] = float(np.max(jr)) if jr else float("nan")
    return out


def select_long(eps, n=12, min_dur=15.0):
    """Longest-horizon episodes, ranked by operator TRAVEL within that set.

    Two independent axes and both matter: duration gives a long horizon, head
    translation gives base motion. EgoDex supplies the first generously (50+
    episodes over 10 s, up to 91.8 s) and the second barely at all -- measured
    over 16 tasks / 673 episodes the largest head travel is 0.47 m and NOTHING
    exceeds 0.5 m, because every task is performed standing or seated in one
    spot. So this ranks by travel within the long episodes to surface whatever
    base motion the corpus actually contains, and the honest ceiling is a
    ~0.2 m base excursion, not walking. See the doc for the dataset that would
    lift it.
    """
    cand = [e for e in eps if e.duration >= min_dur]
    trav = lambda e: float(np.linalg.norm(e.head[:, [0, 2], 3] - e.head[0, [0, 2], 3],
                                          axis=1).max())
    cand.sort(key=trav, reverse=True)
    out = []
    for e in cand[:n]:
        t = trav(e)
        out.append((e, f"long-{'move' if t > 0.15 else 'still'}",
                    f"{e.duration:.0f} s, operator head travel {t:.2f} m"))
    return out


def bench_long(n=12, min_dur=15.0, scope="global", out_tag="long"):
    """Long-horizon sweep with the BASE in the follow loop and a chase camera."""
    eps = load_corpus()
    picks = select_long(eps, n=n, min_dur=min_dur)
    if not picks:
        print(f"  no episodes >= {min_dur}s"); return
    os.makedirs(BENCH, exist_ok=True); os.makedirs(CLIPS, exist_ok=True)
    os.makedirs(VIDEOS, exist_ok=True)
    res_p = os.path.join(BENCH, f"{out_tag}.jsonl")
    done = set()
    if os.path.exists(res_p):
        done = {json.loads(l)["clip"] for l in open(res_p) if l.strip()}
        print(f"  resuming: {len(done)} already recorded")
    print(f"\nLONG-HORIZON SWEEP — {len(picks)} episodes, scope={scope}, "
          f"{sum(p[0].duration for p in picks)/60:.1f} min of operator motion")
    t0 = time.time()
    for k, (e, label, why) in enumerate(picks):
        name = f"{label}__{e.task}__ep{e.index:03d}"
        if name in done:
            continue
        rig = Rig(scope=scope)
        b0 = rig.data.qpos[:3].copy()
        snaps = []
        rec = rig.run(e, on_tick=lambda kk, t, i, en:
                      snaps.append((kk, rig.data.qpos.copy())) if kk % 3 == 0 else None)
        row = _episode_metrics(rec, e, name, label, rig=rig)
        bxy = np.array([q[:2] for _, q in snaps])
        row["base_travel_m"] = float(np.linalg.norm(bxy - bxy[0], axis=1).max())
        row["operator_travel_m"] = float(np.linalg.norm(
            e.head[:, [0, 2], 3] - e.head[0, [0, 2], 3], axis=1).max())
        row["scope"] = scope
        with open(res_p, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        ES.save_clip(e, os.path.join(CLIPS, name + ".json.gz"), note=why)
        _render_from_snaps(rig, rec, snaps, e, label, why,
                           os.path.join(VIDEOS, name + ".mp4"))
        el = time.time() - t0
        print(f"  [{k+1:2d}/{len(picks)}] {name[:46]:48s} {e.duration:5.1f}s "
              f"op {row['operator_travel_m']:.2f}m -> base {row['base_travel_m']:.2f}m  "
              f"pos {row['pos_p50_mm']:5.1f}mm eng {100*row['engaged_frac']:3.0f}%  "
              f"ETA {el/(k+1)*(len(picks)-k-1)/60:.0f}m", flush=True)
    print(f"\n  -> {res_p}")
    return res_p


def analyze_bench(path=None, verbose=True):
    """Aggregate a bench-all sweep: overall, per task, and against difficulty.

    Reports MEDIANS and high percentiles separately and never a bare mean --
    these distributions are long-tailed (a handful of episodes carry most of
    the error), so a mean flatters nothing and hides everything.
    """
    path = path or os.path.join(BENCH, "bench100.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("no rows"); return []
    g = lambda k: np.array([r[k] for r in rows], float)
    roll = np.maximum(g("roll_left_deg"), g("roll_right_deg"))

    if verbose:
        print(f"\nEGODEX SWEEP — {len(rows)} episodes, "
              f"{len({r['task'] for r in rows})} tasks, "
              f"{g('duration_s').sum()/60:.1f} min of operator motion")
        print(f"  engaged: median {100*np.median(g('engaged_frac')):.0f}%  "
              f"min {100*g('engaged_frac').min():.0f}%  "
              f"({(g('engaged_frac') < 0.9).sum()} episodes below 90%)")
        print(f"\n  {'metric':22s} {'p50':>8s} {'p75':>8s} {'p90':>8s} {'p95':>8s} {'max':>8s}")
        print("  " + "-" * 66)
        for k, lbl in (("pos_p50_mm", "wrist pos p50 [mm]"),
                       ("pos_p95_mm", "wrist pos p95 [mm]"),
                       ("normal_p50_deg", "palm normal p50 [deg]"),
                       ("normal_p95_deg", "palm normal p95 [deg]"),
                       ("twist_p50_deg", "palm twist p50 [deg]"),
                       ("twist_p95_deg", "palm twist p95 [deg]")):
            v = g(k); v = v[np.isfinite(v)]
            print(f"  {lbl:22s} {np.percentile(v,50):8.1f} {np.percentile(v,75):8.1f} "
                  f"{np.percentile(v,90):8.1f} {np.percentile(v,95):8.1f} {v.max():8.1f}")

        print(f"\n  {'task':36s} {'n':>3s} {'roll':>6s} {'pos p50':>8s} {'norm p50':>9s} {'twist p50':>10s}")
        print("  " + "-" * 78)
        for t in sorted({r["task"] for r in rows}):
            sub = [r for r in rows if r["task"] == t]
            sg = lambda k: np.array([r[k] for r in sub], float)
            sroll = np.maximum(sg("roll_left_deg"), sg("roll_right_deg"))
            print(f"  {t:36s} {len(sub):3d} {np.median(sroll):5.0f}d "
                  f"{np.median(sg('pos_p50_mm')):6.1f}mm {np.median(sg('normal_p50_deg')):8.1f}d "
                  f"{np.median(sg('twist_p50_deg')):9.1f}d")

        # difficulty: does error track palm-roll span, as G7 concluded?
        print(f"\n  BY PALM-ROLL BAND (the difficulty axis G7 identified)")
        print(f"  {'band':16s} {'n':>3s} {'pos p50':>8s} {'norm p50':>9s} {'norm p95':>9s} {'twist p50':>10s}")
        print("  " + "-" * 70)
        for lo, hi, nm in ((0, 60, "lo  (<60d)"), (60, 120, "mid (60-120d)"), (120, 1e9, "hi  (>120d)")):
            m = (roll >= lo) & (roll < hi)
            if not m.any():
                continue
            print(f"  {nm:16s} {m.sum():3d} {np.median(g('pos_p50_mm')[m]):6.1f}mm "
                  f"{np.median(g('normal_p50_deg')[m]):8.1f}d {np.median(g('normal_p95_deg')[m]):8.1f}d "
                  f"{np.median(g('twist_p50_deg')[m]):9.1f}d")
        for k in ("pos_p50_mm", "normal_p50_deg", "twist_p50_deg"):
            v = g(k); m = np.isfinite(v)
            print(f"    corr(palm-roll span, {k}) = {np.corrcoef(roll[m], v[m])[0,1]:+.2f}")

        # the tail: which episodes are actually bad, and what do they share?
        bad = sorted(rows, key=lambda r: -r["normal_p50_deg"])[:8]
        print(f"\n  WORST 8 BY PALM-NORMAL ERROR (where the retargeting is weakest)")
        print(f"  {'clip':50s} {'roll':>6s} {'pos':>7s} {'norm':>6s} {'twist':>7s} {'eng':>5s}")
        print("  " + "-" * 88)
        for r in bad:
            rr = max(r["roll_left_deg"], r["roll_right_deg"])
            print(f"  {r['clip'][:50]:50s} {rr:5.0f}d {r['pos_p50_mm']:6.1f}mm "
                  f"{r['normal_p50_deg']:5.1f}d {r['twist_p50_deg']:6.1f}d "
                  f"{100*r['engaged_frac']:4.0f}%")
    return rows


# ───────────────────────────── main ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--phase", default="all")
    ap.add_argument("--video", default=None, help="render ONE clip by label")
    ap.add_argument("--videos-all", action="store_true",
                    help="render every benchmark clip into benchmark/videos/")
    ap.add_argument("--ablate", default=None,
                    help="KNOB=VALUE[,KNOB=VALUE] — run the benchmark with one "
                         "knob moved and diff against baseline.json")
    ap.add_argument("--tag", default=None, help="name for this ablation's result file")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="tag(s) to re-print against baseline")
    ap.add_argument("--max-per-task", type=int, default=None)
    ap.add_argument("--bench-all", type=int, default=None,
                    help="N: stratified sweep — metrics + video + clip per episode")
    ap.add_argument("--bench-long", type=int, default=None,
                    help="N: long-horizon sweep, base following (scope=global) + chase cam")
    ap.add_argument("--min-dur", type=float, default=15.0)
    ap.add_argument("--analyze", nargs="?", const="", default=None,
                    help="aggregate a bench-all jsonl (default benchmark/bench100.jsonl)")
    args = ap.parse_args()

    if args.fetch:
        fetch(); return
    if args.bench_all:
        bench_all(args.bench_all); return
    if args.bench_long:
        bench_long(args.bench_long, min_dur=args.min_dur); return
    if args.analyze is not None:
        analyze_bench(args.analyze or None); return
    if args.video:
        render_video(args.video); return
    if args.compare:
        for tag in args.compare:
            f = os.path.join(ABLATIONS, f"{tag}.json")
            if not os.path.exists(f):
                print(f"no such ablation: {f}"); continue
            obj = json.load(open(f))
            print(f"\nABLATION '{obj['tag']}'  knobs={obj['knobs']}")
            _compare_to_baseline(obj["clips"], tag)
        return
    if args.videos_all:
        # EVERY clip the benchmark has a metrics row for — not just the curated
        # `select_clips` set. The stratified episodes are rendered inline by
        # bench_all, so a re-render after a controller change has to walk the
        # recorded rows or it silently refreshes 9 videos out of 107.
        eps = load_corpus(args.max_per_task)
        by = {}
        for e in eps:
            by[(e.task, int(e.index))] = e
        want = {}
        for f in glob.glob(os.path.join(BENCH, "*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                e = by.get((r["task"], r["episode"]))
                if e is not None:
                    want[r["clip"]] = (e, r["label"], r.get("clip", ""))
        print(f"  {len(want)} clips to render")
        for i, (name, (e, label, why)) in enumerate(sorted(want.items()), 1):
            render_video(label, pick=(e, label, why),
                         out=os.path.join(VIDEOS, name + ".mp4"))
            print(f"  [{i}/{len(want)}] {name[:58]}", flush=True)
        print(f"\n-> {VIDEOS}")
        return
    if args.ablate:
        eps = load_corpus(args.max_per_task)
        run_ablation(select_clips(eps), args.ablate,
                     args.tag or args.ablate.replace("=", "").replace(",", "_"))
        return

    os.makedirs(RESULTS, exist_ok=True)
    print("Loading EgoDex corpus …")
    eps = load_corpus(args.max_per_task)
    if not eps:
        print("no data — run with --fetch first"); return
    picks = select_clips(eps)

    ph = args.phase
    summary = {}
    if ph in ("all", "G1"):
        summary["G1"] = g1_corpus(eps, picks)
    if ph in ("all", "G2", "G3", "G4"):
        track = g2_tracking(picks)
        fig_tracking(track)
        if ph in ("all", "G3"):
            rows = g3_wrist(track)
            fig_wrist(track, rows)
            summary["G3"] = rows
        if ph in ("all", "G4"):
            summary["G4"] = g4_gaze(track)
            fig_gaze(track)
    if ph in ("all", "G5"):
        _tr, rows = g5_profiles(picks)
        fig_profiles(rows)
        summary["G5"] = rows
    if ph in ("all", "G6"):
        summary["G6"] = g6_save_clips(picks, track if ph == "all" else None)
    if ph in ("all", "G7"):
        summary["G7"] = g7_asymmetry(eps)

    with open(os.path.join(RESULTS, "egodex.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n-> {RESULTS}/egodex.json")


if __name__ == "__main__":
    main()
