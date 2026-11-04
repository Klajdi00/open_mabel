#pragma once

// ActuatorLimits — the per-actuator scale/range block published in RobotInfo.
//
// Extracted from DmActuator (where it lived as DmActuator::Limits) so that the
// generic Robot/host-link layer can talk about an actuator's declared ranges
// WITHOUT naming a transport. DmActuator keeps `using Limits = ActuatorLimits;`
// so every existing reference still compiles.
//
// ── What these numbers MEAN (and what they do not) ─────────────────────────────
// For a DAMIAO DM-J in MIT mode these are NOT a safety clamp — they are the
// fixed-point full-scale ranges that p/v/t are encoded against on BOTH sides of
// the CAN wire. A disagreement silently rescales every command and every reading
// with no error anywhere. Hence the standing rule: the MIT ranges live in the
// motor and nothing else may declare them (the upstream design note actuator-limits).
//
// For an actuator whose protocol has NO fixed-point MIT encoding — a DYNAMIXEL
// speaks absolute register units, not a fraction of a full-scale range — p_max /
// v_max / t_max are MEANINGLESS, and are published as 0.0. That reuses the
// convention this firmware already had for an offline actuator: 0 is unambiguous
// "unknown / not applicable" and can never be mistaken for a measurement.
// A DYNAMIXEL joint's travel limits are owned by the calibration store (`pos_limits`), which
// is where the arms' limits already live — this struct is deliberately not a
// second home for them.
//
// kp_max / kd_max are the exception: they are NOT motor registers. For MIT they
// are protocol constants (0..500, 0..5). For a DYNAMIXEL they are the real range
// of the Position P/D Gain registers (0..16383, unitless) — see DxlActuator.h,
// which explains why kp/kd are passed through in RAW REGISTER UNITS rather than
// being zeroed like the genuinely-unsupported fields.
struct ActuatorLimits {
  float p_max  = 12.5f;
  float v_max  = 45.0f;
  float t_max  = 54.0f;
  float kp_max = 500.0f;
  float kd_max = 5.0f;
};
