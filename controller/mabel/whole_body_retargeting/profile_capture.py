"""Offline screening of explicitly labeled, fully extended arm captures.

Fit one shoulder/wrist sphere per arm from *hold medians*, not thousands of
correlated tracking frames. A separate, unused hold checks the resulting reach.
Four wrist positions can determine a sphere exactly, so a zero fit residual is
not evidence of accuracy. Conditioning and a 5 mm per-hold uncertainty floor
are checked even for noise-free input. The linearized uncertainty is a screening
model, not a validated ARKit accuracy guarantee.

Input is the raw Mac lab {rt,p} or NOVA {t,cid,msg} recording dialect. All windows
are explicitly labeled seconds after the first recorded teleop frame. Nothing
here connects to a robot, loads a saved profile, or activates/saves a profile.
The returned candidate is for review only. Wrist positions alone cannot prove
that elbows were straight or shoulders stayed fixed; operator confirmation and
independent validation are required in addition to these numerical checks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .calibration import OperatorProfile, body_frame


@dataclass(frozen=True)
class CaptureLimits:
    min_samples: int = 20
    min_span_s: float = 0.6
    min_valid_fraction: float = 0.9
    max_gap_s: float = 0.15
    max_wrist_spread_m: float = 0.012
    max_head_motion_m: float = 0.015
    max_head_rotation_deg: float = 5.0
    hold_uncertainty_floor_m: float = 0.005
    max_condition: float = 30.0
    max_center_95_m: float = 0.030
    max_radius_95_m: float = 0.025
    max_fit_error_m: float = 0.010
    max_validation_error_m: float = 0.015
    min_validation_separation_deg: float = 10.0
    max_forearm_alignment_deg: float = 15.0


def _matrix(value):
    if isinstance(value, dict):
        value = value.get("matrix")
    try:
        a = np.asarray(value, float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    r = a[:3, :3]
    if (not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-4)
            or np.linalg.norm(r.T @ r - np.eye(3)) > 0.05 or np.linalg.det(r) < 0.95):
        return None
    return a


def _hand_points(hand):
    if not isinstance(hand, dict) or hand.get("isTracked") is not True:
        return None, None
    anchor = _matrix(hand.get("anchorTransform"))
    if anchor is None:
        return None, None
    joints = {j.get("joint"): j for j in hand.get("joints", []) if isinstance(j, dict)}
    points = []
    for name in ("wrist", "forearm_arm"):
        j = joints.get(name, {})
        local = _matrix(j.get("localTransform")) if j.get("isTracked") is True else None
        points.append((anchor @ local)[:3, 3] if local is not None else None)
    return tuple(points)


def _frames(rows, source):
    frames = []
    for row in rows:
        if "msg" in row:
            msg = row["msg"]
            if msg.get("type") != "teleop_frame":
                continue
            p, rt, cid = msg.get("payload", {}), row.get("t"), str(row.get("cid", "unknown"))
        elif "p" in row:
            p, rt, cid = row["p"], row.get("rt", row.get("t")), "retarget_lab"
        else:
            continue
        if not isinstance(p, dict) or not isinstance(rt, (int, float)) or not np.isfinite(rt):
            continue
        frames.append({"p": p, "rt": float(rt), "source": cid})
    sources = sorted({f["source"] for f in frames})
    if source is None:
        if len(sources) != 1:
            raise ValueError(f"select one capture source; available sources: {sources}")
        source = sources[0]
    frames = [f for f in frames if f["source"] == source]
    if not frames:
        raise ValueError(f"no teleop frames from {source!r}")
    origin = frames[0]["rt"]
    last_stamp, seen_seq = -np.inf, set()
    for f in frames:
        p = f.pop("p")
        f["t"] = f["rt"] - origin
        stamp = p.get("timestamp", f["rt"])
        seq = p.get("sequence")
        valid_stamp = isinstance(stamp, (int, float)) and np.isfinite(stamp) and stamp > last_stamp
        f["clock"] = float(stamp) if valid_stamp else 0.0
        f["error"] = None
        if not valid_stamp or (seq is not None and seq in seen_seq):
            f["error"] = "duplicate_or_reversed_frame"
        if valid_stamp:
            last_stamp = float(stamp)
        if seq is not None:
            seen_seq.add(seq)
        h = p.get("head") or {}
        h = h if isinstance(h, dict) else {}
        head_enabled = p.get("headEnabled", h.get("enabled", True)) is not False
        f["head"] = _matrix(h.get("transform")) if head_enabled and h.get("trackingState") == "normal" else None
        for side, key in (("l", "leftHand"), ("r", "rightHand")):
            f[side], f[side + "_elbow"] = _hand_points(p.get(key))
        if f["head"] is None:
            f["error"] = f["error"] or "head_not_tracked"
        elif f["l"] is None or f["r"] is None:
            f["error"] = f["error"] or "both_wrists_not_tracked"
    return frames, source, origin


def fit_wrist_sphere(points, uncertainty_floor_m=0.005):
    """Geometric least-squares sphere and linearized per-hold uncertainty.

    points must contain one independent hold median per row. Repeating frames
    from a hold must not increase the apparent information in this fit.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError("at least four finite 3D hold positions are required")
    mean = p.mean(axis=0)
    a = 2 * (p - mean)
    b = np.sum(p * p, axis=1)
    if np.linalg.matrix_rank(a, tol=1e-8) != 3:
        raise ValueError("hold positions are coplanar; shoulder/radius are not identifiable")
    center = np.linalg.lstsq(a, b - b.mean(), rcond=None)[0]
    radius = float(np.linalg.norm(p - center, axis=1).mean())
    for _ in range(15):
        distances = np.linalg.norm(p - center, axis=1)
        if np.any(distances < 1e-8):
            raise ValueError("invalid sphere fit")
        j = np.column_stack([(center - p) / distances[:, None], -np.ones(len(p))])
        delta = np.linalg.lstsq(j, -(distances - radius), rcond=None)[0]
        center += delta[:3]
        radius += float(delta[3])
        if np.linalg.norm(delta) < 1e-10:
            break
    distances = np.linalg.norm(p - center, axis=1)
    j = np.column_stack([(center - p) / distances[:, None], -np.ones(len(p))])
    singular = np.linalg.svd(j, compute_uv=False)
    if singular[-1] < 1e-8 or radius <= 0:
        raise ValueError("sphere fit is singular")
    residuals = distances - radius
    sigma = max(float(uncertainty_floor_m),
                float(np.linalg.norm(residuals) / np.sqrt(max(1, len(p) - 4))))
    covariance = sigma ** 2 * np.linalg.inv(j.T @ j)
    uncertainty = 1.96 * np.sqrt(np.maximum(0, np.diag(covariance)))
    return {"center_body_m": center.tolist(), "radius_m": radius,
            "condition": float(singular[0] / singular[-1]),
            "fit_residuals_m": residuals.tolist(), "uncertainty_floor_m": sigma,
            "linearized_center_95_m": uncertainty[:3].tolist(),
            "linearized_radius_95_m": float(uncertainty[3])}


def _measure_holds(rows, holds, source, limits):
    """Shared tracking and stationary-hold screening; no anthropometry inferred."""
    frames, source, origin = _frames(rows, source)
    errors = []
    report = {"status": "rejected", "source": source, "time_origin": origin,
              "limits": asdict(limits), "errors": errors, "holds": {}, "arms": {},
              "candidate_profile": None,
              "warning": "Wrist-only fitting cannot prove straight elbows or fixed shoulders. "
                         "This report does not activate a profile."}
    names, previous_end = set(), -np.inf
    selected = []
    for hold in sorted(holds, key=lambda h: h["start_s"]):
        name, lo, hi = str(hold["name"]), float(hold["start_s"]), float(hold["end_s"])
        use = hold.get("use", "fit")
        if (name in names or not np.isfinite([lo, hi]).all() or hi <= lo or lo < previous_end
                or use not in ("fit", "validate")):
            raise ValueError("hold names must be unique and windows finite, disjoint, with fit/validate use")
        names.add(name)
        previous_end = hi
        all_rows = [f for f in frames if lo <= f["t"] < hi]
        valid = [f for f in all_rows if f["error"] is None]
        detail = {"use": use, "total_frames": len(all_rows), "valid_frames": len(valid),
                  "rejected_frames": dict(Counter(f["error"] for f in all_rows if f["error"]))}
        report["holds"][name] = detail
        if (len(valid) < limits.min_samples or not all_rows
                or len(valid) / len(all_rows) < limits.min_valid_fraction):
            errors.append(f"{name}: insufficient fresh, fully tracked head and both wrists")
            continue
        clock = np.array([f["clock"] for f in valid])
        detail["span_s"], detail["max_gap_s"] = float(clock[-1] - clock[0]), float(np.max(np.diff(clock)))
        if detail["span_s"] < limits.min_span_s or detail["max_gap_s"] > limits.max_gap_s:
            errors.append(f"{name}: tracking duration or continuity is insufficient")
        selected.append((name, use, valid))
    if not selected:
        return report, selected, None
    reference = selected[0][2][0]["head"]
    if all(f["head"].tobytes() == reference.tobytes() for _, _, fs in selected for f in fs):
        errors.append("head transform never updated across holds; enable Head tracking and recapture")
    _, basis = body_frame(reference)
    head_positions = np.array([f["head"][:3, 3] for _, _, fs in selected for f in fs])
    head_center = np.median(head_positions, axis=0)
    head_motion = float(np.max(np.linalg.norm(head_positions - head_center, axis=1)))
    head_angles = [np.degrees(np.arccos(np.clip((np.trace(reference[:3, :3].T @ f["head"][:3, :3]) - 1) / 2, -1, 1)))
                   for _, _, fs in selected for f in fs]
    report["head_motion_m"], report["head_rotation_deg"] = head_motion, float(max(head_angles))
    if head_motion > limits.max_head_motion_m or max(head_angles) > limits.max_head_rotation_deg:
        errors.append("head moved between holds; keep torso and head stationary")
    for name, _, fs in selected:
        for side in ("l", "r"):
            pts = np.array([basis.T @ (f[side] - f["head"][:3, 3]) for f in fs])
            median = np.median(pts, axis=0)
            spread = float(np.percentile(np.linalg.norm(pts - median, axis=1), 95))
            report["holds"][name][side] = {"wrist_body_m": median.tolist(), "spread95_m": spread}
            if spread > limits.max_wrist_spread_m:
                errors.append(f"{name}/{side}: wrist was not held still")
            elbows = [basis.T @ (f[side + "_elbow"] - f["head"][:3, 3])
                      for f in fs if f[side + "_elbow"] is not None]
            if elbows:
                elbow = np.median(elbows, axis=0)
                report["holds"][name][side].update(
                    elbow_body_m=elbow.tolist(), elbow_valid_fraction=len(elbows)/len(fs),
                    elbow_spread95_m=float(np.percentile(np.linalg.norm(elbows-elbow, axis=1), 95)))
    return report, selected, basis


def assess_operator_capture(rows, holds, *, user="operator", source=None,
                            operator_confirmed_straight_arms=False, limits=None):
    """Screen four or more sphere fit holds plus unused straight validation.

    holds: [{name,start_s,end_s,use:'fit'|'validate'}]. Windows must be disjoint.
    The validation positions are never used to fit the shoulder or reach.
    """
    limits = limits or CaptureLimits()
    report, selected, basis = _measure_holds(rows, holds, source, limits)
    errors = report["errors"]
    report["operator_confirmed_straight_arms"] = operator_confirmed_straight_arms is True
    if operator_confirmed_straight_arms is not True:
        errors.append("operator has not confirmed that all labeled arms were fully straight")
    fit = [h for h in selected if h[1] == "fit"]
    validation = [h for h in selected if h[1] == "validate"]
    if len(fit) < 4:
        errors.append("at least four valid fit holds are required")
    if not validation:
        errors.append("a separate fully straight validation hold is required")
    if len(fit) < 4 or not validation:
        return report
    centers, lengths = {}, []
    for side in ("l", "r"):
        try:
            sphere = fit_wrist_sphere([report["holds"][name][side]["wrist_body_m"] for name, _, _ in fit],
                                      limits.hold_uncertainty_floor_m)
        except ValueError as exc:
            errors.append(f"{side}: {exc}")
            continue
        report["arms"][side] = sphere
        center, radius = np.array(sphere["center_body_m"]), sphere["radius_m"]
        centers[side], lengths = center, lengths + [radius]
        if (sphere["condition"] > limits.max_condition
                or max(sphere["linearized_center_95_m"]) > limits.max_center_95_m
                or sphere["linearized_radius_95_m"] > limits.max_radius_95_m):
            errors.append(f"{side}: shoulder/reach are too uncertain; capture a wider spatial arc")
        if max(abs(x) for x in sphere["fit_residuals_m"]) > limits.max_fit_error_m:
            errors.append(f"{side}: fit holds do not share a consistent shoulder and reach")
        if (not 0.30 <= radius <= 0.90 or not -0.45 <= center[1] <= -0.05
                or not -0.35 <= center[2] <= 0.15 or not 0.08 <= abs(center[0]) <= 0.40
                or (side == "l" and center[0] >= 0) or (side == "r" and center[0] <= 0)):
            errors.append(f"{side}: fitted anatomy is implausible")
        checks, alignment = [], []
        fit_vectors = np.array([report["holds"][name][side]["wrist_body_m"] for name, _, _ in fit]) - center
        fit_vectors /= np.linalg.norm(fit_vectors, axis=1)[:, None]
        for name, use, fs in selected:
            if use == "validate":
                wrist = np.array(report["holds"][name][side]["wrist_body_m"])
                reach = float(np.linalg.norm(wrist - center))
                separation = float(np.degrees(np.arccos(np.clip(np.max(fit_vectors @ ((wrist - center) / reach)), -1, 1))))
                checks.append({"hold": name, "reach_m": reach, "rho": reach / radius,
                               "error_m": reach - radius, "nearest_fit_angle_deg": separation})
                if abs(reach - radius) > limits.max_validation_error_m:
                    errors.append(f"{name}/{side}: independent straight reach does not match")
            for f in fs:
                if f[side + "_elbow"] is None:
                    continue
                forearm = basis.T @ (f[side] - f[side + "_elbow"])
                reach = basis.T @ (f[side] - f["head"][:3, 3]) - center
                norms = np.linalg.norm(forearm) * np.linalg.norm(reach)
                if norms > 1e-5:
                    alignment.append(float(np.degrees(np.arccos(np.clip(forearm @ reach / norms, -1, 1)))))
        sphere["validation"] = checks
        if not any(c["nearest_fit_angle_deg"] >= limits.min_validation_separation_deg for c in checks):
            errors.append(f"{side}: validation needs an unused arm direction, not a repeat fit pose")
        sphere["forearm_alignment95_deg"] = float(np.percentile(alignment, 95)) if alignment else None
        if alignment and sphere["forearm_alignment95_deg"] > limits.max_forearm_alignment_deg:
            errors.append(f"{side}: forearm cue is inconsistent with a fully straight arm")
    if len(centers) == 2:
        width = float(centers["r"][0] - centers["l"][0])
        if not 0.24 <= width <= 0.65 or abs(lengths[0] - lengths[1]) > 0.08:
            errors.append("left/right shoulder width or reach measurements are inconsistent")
    if not errors and len(centers) == 2:
        profile = OperatorProfile(user=user, created=report["time_origin"], arm_length_lr=tuple(lengths),
                                  arm_length_m=float(np.mean(lengths)), reach_m=max(lengths),
                                  shoulder_width_m=width,
                                  neck_height_m=float(-np.mean([c[1] for c in centers.values()])),
                                  shoulder_offset_body={s: c.tolist() for s, c in centers.items()},
                                  quality={"method": "labeled_straight_arm_spheres", "source": report["source"],
                                           "fit_holds": [h[0] for h in fit],
                                           "validation_holds": [h[0] for h in validation],
                                           "uncertainty": {s: {k: v for k, v in report["arms"][s].items()
                                                                if k in ("condition", "linearized_center_95_m", "linearized_radius_95_m")}
                                                           for s in ("l", "r")},
                                           "review_required": True})
        report["candidate_profile"] = asdict(profile)
        report["status"] = "candidate_for_review"
    return report


def assess_neutral_forward_capture(rows, holds, *, neutral, forward, source=None,
                                   operator_confirmed_poses=False, limits=None):
    """Report-only reconstruction from neutral elbow-90 and straight forward.

    Neutral: upper arms vertical, forearms horizontal and parallel forward.
    Forward: both arms fully straight, horizontal and parallel forward.
    At least one *unused* fully straight diagonal hold must be labeled validate.
    The anatomical assumptions cannot be proven by these sparse observations.
    Consequently this mode NEVER exports an OperatorProfile, even when every
    consistency check passes. Review measurements and the validation first.
    """
    limits = limits or CaptureLimits()
    report, selected, basis = _measure_holds(rows, holds, source, limits)
    errors = report["errors"]
    report.update(method="neutral_forward_with_unused_validation",
                  operator_confirmed_poses=operator_confirmed_poses is True, review_measurements=None,
                  warning="Neutral upper-arm verticality and anatomical elbow accuracy are assumptions. "
                          "No profile is exported or activated by this mode.")
    if operator_confirmed_poses is not True:
        errors.append("operator has not confirmed the labeled neutral, straight-forward and validation poses")
    fit_names = {h[0] for h in selected if h[1] == "fit"}
    validation = [h for h in selected if h[1] == "validate"]
    if neutral == forward or fit_names != {neutral, forward}:
        errors.append("label exactly one neutral and one forward fit hold")
    if not validation:
        errors.append("a separate fully straight diagonal validation hold is required")
    if neutral not in fit_names or forward not in fit_names or not validation:
        return report

    def angle(a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.degrees(np.arccos(np.clip(a @ b / norm, -1, 1)))) if norm > 1e-8 else 180.0

    centers, lengths = {}, []
    for side in ("l", "r"):
        n, f = report["holds"][neutral][side], report["holds"][forward][side]
        if any(report["holds"][name][side].get("elbow_valid_fraction", 0) < limits.min_valid_fraction
               for name, _, _ in selected):
            errors.append(f"{side}: tracked forearm_arm elbow proxies are required in every hold")
            continue
        if any(report["holds"][name][side]["elbow_spread95_m"] > limits.max_wrist_spread_m
               for name, _, _ in selected):
            errors.append(f"{side}: elbow proxy moved during a hold")
        wn, en = np.array(n["wrist_body_m"]), np.array(n["elbow_body_m"])
        wf, ef = np.array(f["wrist_body_m"]), np.array(f["elbow_body_m"])
        center = np.array([(en[0] + wf[0]) / 2, wf[1], en[2]])
        length = float(np.linalg.norm(wf - center))
        upper = float(center[1] - en[1])
        forearm = float(np.linalg.norm(wn - en))
        lengths.append(length)
        centers[side] = center
        arm = {"center_body_m": center.tolist(), "reach_m": length,
               "neutral_upper_length_m": upper, "neutral_forearm_length_m": forearm,
               "segment_sum_error_m": length - upper - forearm,
               "neutral_forearm_forward_deg": angle(wn-en, np.array([0, 0, 1.])),
               "forward_straightness_deg": angle(ef-center, wf-ef),
               "lateral_pose_error_m": float(wf[0] - en[0]), "validation": []}
        report["arms"][side] = arm
        if (abs(wf[0] - en[0]) > .025 or arm["neutral_forearm_forward_deg"] > 15
                or arm["forward_straightness_deg"] > 10 or abs(ef[1] - wf[1]) > .02):
            errors.append(f"{side}: labeled neutral/forward pose geometry does not match the instructions")
        if (not .15 <= upper <= .45 or not .15 <= forearm <= .40 or not .30 <= length <= .90
                or abs(arm["segment_sum_error_m"]) > .020
                or abs(np.linalg.norm(wf-ef)-forearm) > .020):
            errors.append(f"{side}: upper/forearm lengths and full reach are inconsistent")
        if (not -.45 <= center[1] <= -.05 or not -.35 <= center[2] <= .15
                or not .08 <= abs(center[0]) <= .40
                or (side == "l" and center[0] >= 0) or (side == "r" and center[0] <= 0)):
            errors.append(f"{side}: reconstructed shoulder is implausible")
        unused_direction = False
        for name, _, _ in validation:
            v = report["holds"][name][side]
            wv, ev = np.array(v["wrist_body_m"]), np.array(v["elbow_body_m"])
            reach = float(np.linalg.norm(wv-center))
            separation = angle(wv-center, wf-center)
            straightness = angle(ev-center, wv-ev)
            residual = reach-length
            arm["validation"].append({"hold": name, "reach_m": reach, "rho": reach/length,
                                      "error_m": residual, "forward_separation_deg": separation,
                                      "straightness_deg": straightness,
                                      "forward_offset_sensitivity": 1 - (wv-center)[2]/max(reach, 1e-8)})
            unused_direction |= separation >= max(20, limits.min_validation_separation_deg)
            if abs(residual) > limits.max_validation_error_m or straightness > 10:
                errors.append(f"{name}/{side}: unused straight diagonal does not validate the estimated shoulder/reach")
        if not unused_direction:
            errors.append(f"{side}: unused diagonal must differ by at least 20 degrees from forward")
    if len(centers) == 2:
        width = float(centers["r"][0]-centers["l"][0])
        if not .24 <= width <= .65 or abs(lengths[0]-lengths[1]) > .08:
            errors.append("left/right reconstructed shoulders or reach are inconsistent")
        if not errors:
            report["status"] = "consistent_under_pose_assumptions"
            report["review_measurements"] = {"arm_length_lr": lengths, "shoulder_width_m": width,
                                               "neck_height_m": float(-np.mean([c[1] for c in centers.values()])),
                                               "shoulder_offset_body": {s: c.tolist() for s, c in centers.items()}}
    return report
