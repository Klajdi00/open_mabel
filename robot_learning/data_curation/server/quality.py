"""
Temporal-integrity / sync quality for MABEL episodes (Phase 1 of the curation loop).

MABEL records ONE shared ~15 Hz clock: every stream is captured in a single loop
and index-aligned, so there is nothing to *re-synchronize* between clocks. What
IS real and learning-relevant:

  1. Action -> observation control lag. The commanded action leads the resulting
     motion by the control/actuation latency. We estimate it by cross-correlating
     the *motion energy* of a command stream against the resulting state stream:
       - d(action/ctrl)/dt   vs   obs/joint_vel     (arm/hand; the usable probe)
       - teleop/base_cmd      vs   obs/base_twist     (base; only when driven)
     A non-zero systematic lag means obs_t is paired with the wrong a_t at train
     time -- an invisible label-noise floor. It's correctable via clip.modOffset
     and gated at export.

  2. Timing gaps. Inter-frame Delta t on time/wall vs the episode's own median
     cadence -> stalls / dropped frames (real on hardware; sim is compute-jittery
     but not dropped, so we judge against the median, not a hardcoded fps).

Pure NumPy over the raw HDF5 (via h5pool); no model, no mujoco.
"""
import json
import os

import numpy as np

from . import h5pool

try:
    import cv2
except Exception:
    cv2 = None

# ── defect-detection thresholds ───────────────────────────────────────────────
_EPS_POS = 1e-9        # |dpos| ~0 = a REPEATED sample (freeze/dropout), not slow motion
_EPS_CMD = 1e-4        # |d action| above this = "commanded to move"
_MIN_FROZEN = 6        # frozen-run length (frames) to flag (0.4 s @ 15 Hz)
_EPS_TQ = 1e-4         # |torque| below this = "no torque reported"
_VEL_MOVE = 1e-3       # |vel| above this = "joint is moving"
_MIN_CAM_RUN = 5       # repeated/black camera frames to flag
_BLACK_LUMA = 12.0     # mean grey below this = blackout
_HELD_FRAC = 0.45      # >this fraction of held frames = low effective rate


def _runs(mask, min_len):
    """(start, end) half-open spans of consecutive True with length >= min_len."""
    out, T, i = [], len(mask), 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out

# probe pairs: (label, command_key, command_is_target_position, state_key)
_PROBES = [
    ("action->joint_vel", "action/ctrl", True, "obs/joint_vel"),
    ("base_cmd->base_twist", "teleop/base_cmd", False, "obs/base_twist"),
]
_MAX_LAG = 8            # search +/- this many frames
_MIN_CONF = 0.30        # normalized xcorr peak below this = "insufficient signal"
_LAG_TOL = 1            # |lag| <= this frame(s) is fine (MABEL control latency ~0-1)


def _fps(f):
    """Episode cadence (Hz) from time/sim, else time/wall, else 15."""
    for k in ("time/sim", "time/wall"):
        if k in f:
            t = np.asarray(f[k][:], dtype=np.float64)
            if t.size >= 2 and t[-1] > t[0]:
                return float((t.size - 1) / (t[-1] - t[0]))
    return 15.0


def _motion_energy(arr, is_target):
    """1-D per-frame motion energy: L2 norm across channels of the velocity. For a
    target-position stream we differentiate first (command velocity)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if is_target:
        a = np.vstack([a[1:] - a[:-1], np.zeros((1, a.shape[1]))])   # d/dt, edge-padded
    return np.linalg.norm(a, axis=1)                                  # (T,)


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    return (x - m) / s if s > 1e-9 else None                         # None = degenerate/flat


def _xcorr_lag(a, b, K):
    """Lag k (frames) maximizing normalized cross-correlation of command energy a
    against state energy b, where b[t] ~ a[t-k]  (k>0 => state lags command). Returns
    (k, peak_corr) or None if either signal is flat."""
    az, bz = _zscore(a), _zscore(b)
    if az is None or bz is None:
        return None
    T = len(az)
    best_k, best_c = 0, -2.0
    for k in range(-K, K + 1):
        if k >= 0:
            x, y = az[:T - k], bz[k:]            # a[t], b[t+k]
        else:
            x, y = az[-k:], bz[:T + k]           # a[t], b[t+k] with t>=-k
        if len(x) < 8:
            continue
        c = float(np.mean(x * y))
        if c > best_c:
            best_c, best_k = c, k
    return best_k, best_c


def _detect_defects(f, T, fps):
    """Per-stream failure detection (frozen/dead motors, torque-dead, whole-arm-down,
    camera freeze/blackout, low/changed rate). Returns a list of span dicts."""
    spans = []
    pos = np.asarray(f["obs/joint_pos"][:], float) if "obs/joint_pos" in f else None
    vel = np.asarray(f["obs/joint_vel"][:], float) if "obs/joint_vel" in f else None
    tq = np.asarray(f["obs/joint_torque"][:], float) if "obs/joint_torque" in f else None
    act = np.asarray(f["action/ctrl"][:], float) if "action/ctrl" in f else None

    covered = np.zeros((int(T), pos.shape[1]), bool) if pos is not None else None
    # A frozen motor is a COMMAND-RESPONSE MISMATCH: the command is changing but the
    # measured position isn't (a joint merely at rest has a flat command too, so it is
    # NOT flagged). Needs action/ctrl aligned 1:1 with joint_pos (both 59-actuator).
    if pos is not None and act is not None and act.shape == pos.shape:
        held = np.abs(np.diff(pos, axis=0)) < _EPS_POS                 # (T-1, J)
        cmd_move = np.abs(np.diff(act, axis=0)) > _EPS_CMD
        heldF = np.vstack([held, np.zeros((1, pos.shape[1]), bool)])   # (T, J) position repeated
        mism = np.vstack([held & cmd_move, np.zeros((1, pos.shape[1]), bool)])   # (T, J) commanded-but-frozen
        # a dead joint: commanded-but-frozen, OR frozen with torque telemetry exactly
        # 0 (power loss kills torque sensing even when the arm isn't being commanded).
        dead = mism.copy()
        if tq is not None and tq.shape == pos.shape:
            dead = mism | (heldF & (np.abs(tq) < _EPS_TQ))
        try:
            sj = json.loads(f.attrs["schema_json"]); names = sj["actuator_names"]
            groups = {g: [names.index(n) for n in mm if n in names]
                      for g, mm in sj.get("actuator_groups", {}).items()}
        except Exception:
            groups = {}
        # whole-arm-down first (power loss): most of an arm mismatched together
        for g in ("left_arm", "right_arm"):
            js = groups.get(g, [])
            if not js:
                continue
            for s, e in _runs(dead[:, js].mean(axis=1) > 0.6, _MIN_FROZEN):
                spans.append({"type": "arm_dead", "start": int(s), "end": int(e), "group": g,
                              "joints": js, "severity": 1.0, "label": f"{g} down {s}-{e}"})
                covered[s:e][:, js] = True
        # per-joint frozen (comms dropout) not already inside an arm_dead
        for j in range(pos.shape[1]):
            for s, e in _runs(dead[:, j] & ~covered[:, j], _MIN_FROZEN):
                spans.append({"type": "motor_dropout", "start": int(s), "end": int(e),
                              "joints": [j], "severity": 0.6, "label": f"motor {j} frozen {s}-{e}"})

    if tq is not None and vel is not None:                             # torque telemetry dead while moving
        reports = tq.std(axis=0) > _EPS_TQ * 10
        for j in range(tq.shape[1]):
            if not reports[j]:
                continue
            m = (np.abs(tq[:, j]) < _EPS_TQ) & (np.abs(vel[:, j]) > _VEL_MOVE)
            for s, e in _runs(m, _MIN_FROZEN):
                spans.append({"type": "torque_dead", "start": int(s), "end": int(e), "joints": [j],
                              "severity": 0.5, "label": f"motor {j} torque dead {s}-{e}"})

    # NaN corruption in any joint stream (bad sensor read / serialization glitch)
    for key, arr in (("obs/joint_pos", pos), ("obs/joint_vel", vel), ("obs/joint_torque", tq)):
        if arr is None:
            continue
        bad = np.isnan(arr).any(axis=1)
        for s, e in _runs(bad, 1):
            js = sorted(np.nonzero(np.isnan(arr[s:e]).any(axis=0))[0].tolist())
            spans.append({"type": "nan_corrupt", "start": int(s), "end": int(e), "joints": js,
                          "key": key, "severity": 1.0, "label": f"{key} NaN {s}-{e}"})

    spans += _detect_cameras(f, T)
    spans += _detect_rate(f, T, fps)
    return spans


def _detect_cameras(f, T):
    try:
        cams = [c["name"] for c in json.loads(f.attrs["schema_json"]).get("cameras", [])]
    except Exception:
        cams = []
    spans = []
    for cam in cams:
        rk = f"obs/cam/{cam}/rgb"
        if rk not in f:
            continue
        rgb = f[rk]
        blackouts = []
        if cv2 is not None:                                           # blackout via strided luma
            luma = np.full(T, np.nan)
            for i in range(0, T, 2):
                im = cv2.imdecode(np.frombuffer(np.asarray(rgb[i], np.uint8), np.uint8), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    luma[i] = float(im.mean())
            med = float(np.nanmedian(luma)) if np.isfinite(luma).any() else 0.0
            if med > 30.0:                                            # only cameras that are NORMALLY bright
                last, blk = med, np.zeros(T, bool)
                for i in range(T):
                    last = luma[i] if not np.isnan(luma[i]) else last
                    blk[i] = last < _BLACK_LUMA
                for s, e in _runs(blk, _MIN_CAM_RUN):
                    if s == 0:                                        # skip render/startup warmup at frame 0
                        continue
                    blackouts.append((s, e))
                    spans.append({"type": "camera_blackout", "start": int(s), "end": int(e), "cam": cam,
                                  "severity": 0.9, "label": f"{cam} blackout {s}-{e}"})
        h = [hash(np.asarray(rgb[i]).tobytes()) for i in range(T)]     # freeze: repeated raw JPEG bytes
        same = np.array([i > 0 and h[i] == h[i - 1] for i in range(T)])
        for s, e in _runs(same, _MIN_CAM_RUN):
            fs = int(s - 1)
            if any(bs <= fs and e <= be + 2 for bs, be in blackouts):  # a blackout already explains it
                continue
            spans.append({"type": "camera_freeze", "start": fs, "end": int(e), "cam": cam,
                          "severity": 0.7, "label": f"{cam} frozen {fs}-{e}"})
    return spans


def _detect_rate(f, T, fps):
    spans = []
    for key in ("obs/joint_pos", "obs/joint_vel", "obs/base_twist", "action/ctrl"):
        if key not in f:
            continue
        a = np.asarray(f[key][:], float)
        if a.ndim == 1:
            a = a[:, None]
        changed = np.abs(np.diff(a, axis=0)).sum(axis=1) > 1e-8        # (T-1,)
        if int(changed.sum()) < 3:                                     # naturally-static stream: not a rate defect
            continue
        held_frac = 1.0 - float(changed.mean())
        h = len(changed) // 2
        hf1, hf2 = 1.0 - float(changed[:h].mean()), 1.0 - float(changed[h:].mean())
        eff = float(changed.mean()) * fps
        if held_frac > _HELD_FRAC:
            spans.append({"type": "low_rate", "start": 0, "end": T, "key": key, "severity": 0.4,
                          "label": f"{key} ~{eff:.1f} Hz (held {held_frac*100:.0f}%)"})
        elif abs(hf2 - hf1) > 0.3:
            spans.append({"type": "rate_change", "start": int(h), "end": T, "key": key, "severity": 0.4,
                          "label": f"{key} rate change @ {h}"})
    return spans


_DRIFT_MIN_CONF = 0.6   # a drift window must be well-conditioned (a startup window
                        # near conf~0.44 is a spurious outlier, not a trend point)


def _theilsen_slope(ts, ls):
    """Median of pairwise slopes — robust to a lone outlier window (Theil–Sen), the
    RANSAC/Huber-lite the drift fit needs so one bad lag can't invent a trend."""
    slopes = [(ls[j] - ls[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts))
              if ts[j] - ts[i] > 1e-9]
    return float(np.median(slopes)) if slopes else 0.0


def _windowed_drift(a, b, K, nwin=5):
    """Clock-drift slope (frames per 1000 frames) + whether it is a REAL linear trend
    rather than ±1-frame lag-quantization jitter or a lone startup outlier. Robust
    (Theil–Sen) slope over WELL-CONFIDENT windows; 'significant' only with ≥4 such
    windows, a sub-frame median residual about the robust line, and ≥2 frames of
    predicted drift across the whole clip. A clean single-clock episode reads flat."""
    W = len(a); seg = max(W // nwin, K * 3); pts = []
    for s in range(0, W, seg):
        r = _xcorr_lag(a[s:s + seg], b[s:s + seg], K)
        if r and r[1] >= max(_MIN_CONF, _DRIFT_MIN_CONF):
            pts.append((s + seg / 2.0, r[0]))
    if len(pts) < 4:
        return 0.0, False
    ts = np.array([c for c, _ in pts], dtype=np.float64)
    ls = np.array([l for _, l in pts], dtype=np.float64)
    beta = _theilsen_slope(ts, ls)
    line = np.median(ls) + beta * (ts - np.median(ts))
    med_resid = float(np.median(np.abs(ls - line)))   # scatter about the robust line
    total_drift = abs(beta) * W                        # frames of drift across the clip
    significant = (med_resid <= 1.0) and (total_drift >= 2.0)
    return float(beta * 1000.0), bool(significant)


def analyze(path, max_lag=_MAX_LAG):
    """Full temporal-integrity + defect report for one episode (dict, JSON-safe)."""
    with h5pool.reading(path) as f:
        fps = _fps(f)
        T = 0
        probes = []
        for label, ck, is_tgt, sk in _PROBES:
            if ck not in f or sk not in f:
                continue
            cmd = np.asarray(f[ck][:], dtype=np.float64)
            st = np.asarray(f[sk][:], dtype=np.float64)
            T = max(T, cmd.shape[0])
            a = _motion_energy(cmd, is_tgt)
            b = _motion_energy(st, False)
            r = _xcorr_lag(a, b, max_lag)
            if r is None:
                probes.append({"probe": label, "usable": False, "reason": "flat signal"})
                continue
            lag, conf = r
            usable = conf >= _MIN_CONF
            drift, drift_sig = _windowed_drift(a, b, max_lag) if usable else (0.0, False)
            probes.append({
                "probe": label, "usable": bool(usable),
                "lag_frames": int(lag), "lag_ms": round(1000.0 * lag / fps, 1),
                "confidence": round(conf, 3),
                "flag": bool(usable and abs(lag) > _LAG_TOL),
                "drift_per_kframe": round(drift, 2) if drift_sig else 0.0,
                "drift_significant": bool(drift_sig),
            })
        if T == 0:
            T = (f["obs/joint_pos"].shape[0] if "obs/joint_pos" in f
                 else int(f.attrs.get("num_frames", 0)))
        # timing gaps from wall clock, judged against the episode's own median cadence
        gaps = _timing_gaps(f)
        defects = _detect_defects(f, T, fps)

    usable_probes = [p for p in probes if p.get("usable")]
    headline = max(usable_probes, key=lambda p: p["confidence"], default=None)
    systematic_lag = bool(headline and headline.get("flag"))

    # fold timing gaps + a systematic lag into the unified defect list
    for gf in gaps.get("gap_frames", []):
        defects.append({"type": "timing_gap", "start": int(gf), "end": int(min(T, gf + 1)),
                        "severity": 0.5, "label": f"timing stall @ {gf}"})
    if systematic_lag and headline:
        defects.append({"type": "lag", "start": 0, "end": int(T), "severity": 0.5,
                        "label": f"action↔obs lag {headline['lag_frames']}f"})
    # per-frame severity heat + an episode quality score (1 = perfect)
    sev = np.zeros(max(1, int(T)))
    for sp in defects:
        s, e = max(0, int(sp.get("start", 0))), min(int(T), int(sp.get("end", T)))
        if e > s:
            sev[s:e] = np.maximum(sev[s:e], float(sp.get("severity", 0.3)))
    quality_score = round(1.0 - float(sev.mean()), 3)
    frac_bad = round(float((sev > 0).mean()), 3)
    defects.sort(key=lambda d: (-float(d.get("severity", 0)), int(d.get("start", 0))))

    return {
        "fps": round(fps, 3), "T": int(T),
        "probes": probes,
        "headline": headline,                          # best-confidence usable probe
        "gaps": gaps,
        "defects": defects,                            # detected failure spans (severity-sorted)
        "quality_score": quality_score,                # 1.0 = clean; lower = more corrupted
        "frac_bad": frac_bad,                          # fraction of frames touched by a defect
        "severity": [round(float(x), 2) for x in sev],  # per-frame heat strip
        "flags": {
            "systematic_lag": systematic_lag,
            "timing_gaps": gaps["n_gaps"] > 0,
            "defects": len(defects) > 0,
            "clock_drift": bool(headline and headline.get("drift_significant")),
        },
        # what the "Apply correction" button should write (modOffset on the command
        # modality) to zero out the measured lag; null if nothing to fix.
        # modality IDs match materialize._modality_keys ("action"->action/ctrl,
        # "teleop"->teleop/*) so the client can write clip.modOffset directly.
        "recommended_offset": (
            {"modality": "action" if headline["probe"].startswith("action") else "teleop",
             "frames": int(headline["lag_frames"])}
            if systematic_lag else None),
    }


def _movavg(x, w):
    if w <= 1 or x.size == 0:
        return x
    k = np.ones(w) / w
    return np.convolve(x.astype(float), k, mode="same")


def frequency_tracks(path, window=8):
    """Per-channel effective-Hz tracks (cameras + actuator modules + action) for the
    frequency heat. Uses the stored per-stream profile (freq/<ch>/hz + nominal_fps)
    when present; otherwise estimates the effective rate from fresh-sample fraction."""
    with h5pool.reading(path) as f:
        fps = _fps(f)
        if "freq" in f and "meta_json" in f["freq"].attrs:            # authored multi-rate profile
            meta = json.loads(f["freq"].attrs["meta_json"])
            out, T = {}, 0
            for ch, m in meta.items():
                dk = f"freq/{ch}/hz"
                if dk not in f:
                    continue
                hz = np.asarray(f[dk][:], float); nom = float(m.get("nominal_fps") or fps) or fps
                T = max(T, hz.size)
                out[ch] = {"nominal_fps": round(nom, 1), "kind": m.get("kind", "proprio"),
                           "hz": [round(float(x), 1) for x in hz],
                           "ratio": [round(float(v), 3) for v in np.clip(hz / nom, 0, 1.2)]}
            return {"T": T, "fps": round(fps, 2), "authored": True, "channels": out}
        return _estimate_frequency(f, fps, window)


def _estimate_frequency(f, fps, window):
    """Effective Hz per channel from the fresh-sample fraction (legacy single-clock
    bags): a stream that repeats/holds values is running below the loop rate."""
    out, T = {}, 0
    try:
        sj = json.loads(f.attrs.get("schema_json", "{}")); names = sj.get("actuator_names", [])
        groups = {g: [names.index(n) for n in mm if n in names] for g, mm in sj.get("actuator_groups", {}).items()}
        cams = [c["name"] for c in sj.get("cameras", [])]
    except Exception:
        groups, cams = {}, []

    def track(fresh, nominal, kind):
        r = np.clip(_movavg(fresh.astype(float), window), 0, 1)
        return {"nominal_fps": round(nominal, 1), "kind": kind,
                "hz": [round(float(x * nominal), 1) for x in r], "ratio": [round(float(x), 3) for x in r]}

    for cam in cams:
        k = f"obs/cam/{cam}/rgb"
        if k not in f:
            continue
        n = f[k].shape[0]; T = max(T, n)
        h = [hash(np.asarray(f[k][i]).tobytes()) for i in range(n)]
        fresh = np.array([1] + [1 if h[i] != h[i - 1] else 0 for i in range(1, n)])
        out[cam] = track(fresh, fps, "camera")
    if "obs/joint_pos" in f:
        pos = np.asarray(f["obs/joint_pos"][:], float); T = max(T, pos.shape[0])
        d = np.abs(np.diff(pos, axis=0))
        for g, js in groups.items():
            if not js:
                continue
            fresh = np.concatenate([[1], (d[:, js].sum(axis=1) > 1e-9).astype(int)])
            out[g] = track(fresh, fps, "proprio")
    if "action/ctrl" in f:
        a = np.asarray(f["action/ctrl"][:], float)
        fresh = np.concatenate([[1], (np.abs(np.diff(a, axis=0)).sum(axis=1) > 1e-9).astype(int)])
        out["action"] = track(fresh, fps, "action")
    return {"T": T, "fps": round(fps, 2), "authored": False, "channels": out}


def _ee_speed(f, T):
    """End-effector speed = summed palm-position velocity (the motion the head camera
    actually sees), falling back to ‖joint_vel‖ if pose is absent."""
    e = np.zeros(T)
    got = False
    for pk in ("obs/pose/left_palm", "obs/pose/right_palm"):
        if pk in f:
            p = np.asarray(f[pk][:T, :3], float)
            e[1:] += np.linalg.norm(np.diff(p, axis=0), axis=1); got = True
    if not got and "obs/joint_vel" in f:
        return np.linalg.norm(np.asarray(f["obs/joint_vel"][:T], float), axis=1)
    return e


def _cam_motion(f, cam, T):
    """Per-frame camera motion energy = downscaled grayscale frame-difference. Spikes
    at motion onset (the shared physical observable for camera↔proprio sync)."""
    k = f"obs/cam/{cam}/rgb"
    if k not in f or cv2 is None:
        return None
    e = np.zeros(T); prev = None
    for i in range(min(T, f[k].shape[0])):
        im = cv2.imdecode(np.frombuffer(np.asarray(f[k][i], np.uint8), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        small = cv2.resize(im, (80, 45)).astype(np.float32)
        if prev is not None:
            e[i] = float(np.abs(small - prev).mean())
        prev = small
    return e


def align_to_pixels(path, cam=None, max_lag=_MAX_LAG, nwin=6):
    """ROBUST lag between the camera (pixels) and proprioception, for the
    'Align proprio+control to pixels' button. Cross-correlates camera motion energy
    vs ‖joint_vel‖ over MULTIPLE windows, takes the median lag (drops low-confidence
    windows), and fits a drift slope. Returns a modOffset to apply to proprio+action."""
    with h5pool.reading(path) as f:
        fps = _fps(f)
        try:
            cams = [c["name"] for c in json.loads(f.attrs.get("schema_json", "{}")).get("cameras", []) if c.get("rgb")]
        except Exception:
            cams = []
        cam = cam or next((c for c in ("zed_left", "body_front_cam", "body_cam") if c in cams), cams[0] if cams else None)
        if cam is None or "obs/joint_vel" not in f:
            return {"usable": False, "reason": "no camera / proprio", "cam": cam}
        T = int(f["obs/joint_vel"].shape[0])
        ce = _cam_motion(f, cam, T)
        pe = _ee_speed(f, T)                             # end-effector speed = what the head cam sees
    if ce is None:
        return {"usable": False, "reason": "camera decode unavailable", "cam": cam}

    g = _xcorr_lag(ce, pe, max_lag)                       # global fallback
    if g is None:
        return {"usable": False, "reason": "flat signal", "cam": cam}
    glag, gconf = g
    ests, W, seg = [], len(ce), max(len(ce) // nwin, max_lag * 3)
    for s in range(0, W, seg):                            # per-window estimates
        r = _xcorr_lag(ce[s:s + seg], pe[s:s + seg], max_lag)
        if r and r[1] >= _MIN_CONF:
            ests.append((s + seg / 2.0, r[0], r[1]))
    lags = [l for _, l, _ in ests]
    lag = int(round(float(np.median(lags)))) if len(ests) >= 3 else glag   # median needs ≥3 windows; else global
    conf = float(np.median([c for *_, c in ests])) if ests else gconf
    lag_std = float(np.std(lags)) if len(lags) >= 2 else 0.0
    drift = 0.0
    if len(ests) >= 3:                                    # lag(t)=a+b·t → b = drift
        drift = float(np.polyfit([c for c, _, _ in ests], [l for _, l, _ in ests], 1)[0])
    # ROBUST/fail-safe: only claim an alignment when ≥3 windows independently AGREE
    # (low lag spread) at good confidence. Weakly-coupled data → not usable (honest).
    agree = len(ests) >= 3 and lag_std <= 1.5
    usable = bool(agree and conf >= _MIN_CONF)
    return {"usable": usable, "cam": cam,
            "lag_frames": int(lag), "lag_ms": round(1000.0 * lag / fps, 1),
            "confidence": round(conf, 3), "n_windows": len(ests), "lag_std": round(lag_std, 2),
            "agree": bool(agree), "drift_per_kframe": round(drift * 1000, 2), "global_lag": int(glag),
            "reason": None if usable else "camera↔proprio coupling too weak / windows disagree",
            "recommended_offset": ({"modalities": ["proprio", "action"], "frames": int(lag)} if usable else None)}


_CORPUS_CACHE = {}


def corpus_quality(dataset):
    """Whole-dataset quality: run analyze() over every episode, rank worst-first,
    and summarise a joint dataset score. Memoized on (file, mtime) so re-opening is
    instant."""
    from . import catalog
    eps = catalog.list_episodes(dataset)
    key = tuple((e["file"], _safe_mtime(catalog.episode_path(dataset, e["file"]))) for e in eps)
    cached = _CORPUS_CACHE.get(dataset)
    if cached and cached[0] == key:
        return cached[1]

    out = []
    total_frames, total_secs, size_mb, meta = 0, 0.0, 0.0, None
    for e in eps:
        path = catalog.episode_path(dataset, e["file"])
        r = analyze(path)
        tc = {}
        for d in r["defects"]:
            tc[d["type"]] = tc.get(d["type"], 0) + 1
        out.append({"file": e["file"], "quality": r["quality_score"], "frac_bad": r["frac_bad"],
                    "n_defects": len(r["defects"]), "types": tc, "T": r["T"], "flags": r["flags"]})
        total_frames += r["T"]; total_secs += r["T"] / max(1.0, r["fps"]); size_mb += _safe_size(path) / 1e6
        if meta is None:
            meta = _episode_meta(path)
    out.sort(key=lambda x: x["quality"])                    # worst first
    scores = [o["quality"] for o in out] or [1.0]
    res = {
        "dataset": dataset, "n_episodes": len(out),
        "dataset_score": round(sum(scores) / len(scores), 3),   # mean episode quality
        "min_quality": round(min(scores), 3),
        "n_clean": sum(1 for o in out if o["n_defects"] == 0),
        "n_flagged": sum(1 for o in out if o["n_defects"] > 0),
        "overview": {
            "total_frames": total_frames, "hours": round(total_secs / 3600.0, 3),
            "size_mb": round(size_mb, 1),
            "task": (meta or {}).get("task", "—"), "modalities": (meta or {}).get("modalities", []),
        },
        "episodes": out,
    }
    _CORPUS_CACHE[dataset] = (key, res)
    return res


def _safe_mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


def _safe_size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _episode_meta(path):
    """Task + a concise modality summary for the dataset overview."""
    with h5pool.reading(path) as f:
        task = str(f.attrs.get("task", "—"))
        try:
            sj = json.loads(f.attrs.get("schema_json", "{}"))
        except Exception:
            sj = {}
    cams = sj.get("cameras", []); depth = sum(1 for c in cams if c.get("depth"))
    mods = []
    if cams:
        mods.append(f"{len(cams)} cameras ({depth} depth)")
    if sj.get("lidar"):
        mods.append(f"lidar {sj['lidar'].get('n_rays', '')}".strip())
    nu = sj.get("nu")
    if nu:
        mods.append(f"proprio {nu} DOF")
        mods.append(f"action {nu}")
    mods += ["teleop", "pose / tf"]
    return {"task": task, "modalities": mods}


def _timing_gaps(f):
    if "time/wall" not in f:
        return {"n_gaps": 0, "gap_frames": [], "median_dt": None, "fps_wall": None}
    tw = np.asarray(f["time/wall"][:], dtype=np.float64)
    if tw.size < 3:
        return {"n_gaps": 0, "gap_frames": [], "median_dt": None, "fps_wall": None}
    dt = np.diff(tw)
    med = float(np.median(dt))
    thresh = 1.5 * med
    gap_idx = (np.nonzero(dt > thresh)[0] + 1).tolist()      # frame where the gap lands
    return {
        "n_gaps": len(gap_idx),
        "gap_frames": gap_idx[:500],
        "median_dt": round(med, 4),
        "fps_wall": round(1.0 / med, 3) if med > 1e-9 else None,
        "max_dt": round(float(dt.max()), 4),
    }
