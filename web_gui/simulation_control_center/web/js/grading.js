// Pure tracking-error + temperature grading — shared by the joint panel and
// the tests. 0 = green (tracking), 1 = yellow (lagging), 2 = red (far behind /
// overheating). Thresholds are per-mode: position joints get radians, drive
// wheels rad/s, etc.
export const STATUS_COL = ['#3ecf8e', '#E7C24B', '#D9463A'];

// per-mode absolute-error thresholds [warn, bad]
export const ERR_THRESH = {
  pos: [0.05, 0.18],          // rad (or m for the lift)
  vel: [0.3, 1.0],            // rad/s
  torque: [0.5, 1.5],         // Nm-ish (effort units)
};

export function trackStatus(cur, tgt, mode = 'pos'){
  if (cur == null || tgt == null) return null;      // no data → ungraded
  const [warn, bad] = ERR_THRESH[mode] || ERR_THRESH.pos;
  const e = Math.abs(cur - tgt);
  return e >= bad ? 2 : e >= warn ? 1 : 0;
}

export function tempStatus(t, warn = 55, hot = 70){
  if (t == null) return null;
  return t >= hot ? 2 : t >= warn ? 1 : 0;
}

// display normalization: scale a value into [-1, 1] for the bar height
export function normVal(v, mode, joint){
  if (v == null) return 0;
  const span = mode === 'pos' ? (joint?.span || Math.PI)
    : mode === 'vel' ? 3.0 : 5.0;
  return Math.max(-1, Math.min(1, v / span));
}
