// Client-side mapping from quality.json `defects` to per-joint status (for the
// joint-state window) and per-modality clip-local spans (for the timeline).
// Colour policy: RED = data missing / out-of-sync (serious); YELLOW = degraded
// (warning); GREEN/none = ok.

const ERR = new Set(['arm_dead', 'motor_dropout', 'camera_blackout', 'timing_gap', 'lag', 'nan_corrupt']);
export function defectClass(type){ return ERR.has(type) ? 'err' : 'warn'; }

// three colour-graded categories for the timeline:
//   dropped  = data missing  (deep red)      : motor_dropout, arm_dead, camera_blackout
//   unsynced = temporal / out-of-sync (magenta-red) : lag, timing_gap
//   degraded = present but poor (yellow)      : torque_dead, camera_freeze, low_rate, rate_change
const _DROPPED = new Set(['motor_dropout', 'arm_dead', 'camera_blackout', 'nan_corrupt']);
const _UNSYNCED = new Set(['lag', 'timing_gap']);
export function defectCategory(type){
  return _DROPPED.has(type) ? 'dropped' : _UNSYNCED.has(type) ? 'unsynced' : 'degraded';
}
const _CATWORD = { dropped: 'DROPPED', unsynced: 'UNSYNCED', degraded: 'low-rate' };
export function categoryWord(cat){ return _CATWORD[cat] || cat; }

// concise, human-readable one-liner for the Issues log
function _stream(k){ return (k || '').replace('obs/', '').replace('action/', '').replace('teleop/', ''); }
const _MSG = {
  arm_dead: (d) => `${(d.group || 'arm').replace('_', ' ')} disconnected`,
  motor_dropout: (d) => `motor${(d.joints || []).length > 1 ? 's' : ''} ${(d.joints || []).join(', ')} dropped`,
  torque_dead: (d) => `torque dead · ${d.group ? d.group.replace('_', ' ') : (d.joints || []).join(', ')}`,
  camera_blackout: (d) => `${d.cam} — footage missing`,
  camera_freeze: (d) => `${d.cam} — frozen (low fps)`,
  low_rate: (d) => `${_stream(d.key)} — low frequency`,
  rate_change: (d) => `${_stream(d.key)} — rate drop`,
  lag: (d) => `${_stream(d.key)} — out of sync`,
  timing_gap: () => `timing stall (out of sync)`,
  nan_corrupt: (d) => `${_stream(d.key)} — corrupt (NaN)`,
};
export function defectMessage(d){ const fn = _MSG[d.type]; return fn ? fn(d) : (d.label || d.type); }

// Joint-window health: GREEN(0)=good, YELLOW(1)=low-frequency/warning,
// RED(2)=dropping data / disconnection. Only an actual per-joint dropout or a
// dead arm is red; low-rate / rate-change / lag / timing stalls / torque-loss are
// warnings (yellow).
const _JOINT_RED = new Set(['motor_dropout', 'arm_dead', 'nan_corrupt']);   // data missing / corrupt
export function jointStatusAt(quality, f, n){
  const st = new Uint8Array(n);
  if (!quality || !quality.defects) return st;
  for (const d of quality.defects){
    if (f < d.start || f >= d.end) continue;
    if (d.joints && d.joints.length){                        // per-joint defect
      const sev = _JOINT_RED.has(d.type) ? 2 : 1;            // dropout/arm-dead=red, torque-dead=yellow
      for (const j of d.joints) if (j < n) st[j] = Math.max(st[j], sev);
    } else if (d.type === 'timing_gap' ||
               (['low_rate', 'rate_change', 'lag'].includes(d.type) && (d.key || '').startsWith('obs/joint'))){
      for (let j = 0; j < n; j++) st[j] = Math.max(st[j], 1);   // low-freq / sync warning = yellow (all joints)
    }
  }
  return st;
}

// worst per-joint status over the WHOLE episode (window-level indicator)
export function jointStatusAny(quality, n){
  const st = new Uint8Array(n);
  if (!quality || !quality.defects) return st;
  for (const d of quality.defects){
    const sev = ERR.has(d.type) ? 2 : 1;
    if (d.joints && d.joints.length){
      for (const j of d.joints) if (j < n) st[j] = Math.max(st[j], sev);
    } else if (d.type === 'timing_gap' || (d.key || '').startsWith('obs/joint')){
      for (let j = 0; j < n; j++) st[j] = Math.max(st[j], sev);
    }
  }
  return st;
}

// per-camera status at a frame: 0 ok, 1 warn (frozen / low-freq), 2 error (footage missing)
export function camStatusAt(quality, cam, f){
  if (!quality || !quality.defects) return 0;
  let s = 0;
  for (const d of quality.defects){
    if (f < d.start || f >= d.end) continue;
    if (d.type === 'timing_gap'){ s = Math.max(s, 1); continue; }   // out-of-sync → warn
    if (d.cam !== cam) continue;
    s = Math.max(s, d.type === 'camera_blackout' ? 2 : 1);          // blackout=missing(red), freeze=degraded(yellow)
  }
  return s;
}

// clip-local coloured spans for one timeline modality lane (optionally a single DOF)
export function modalityDefects(quality, modId, cam, inFrame, len, joint){
  if (!quality || !quality.defects) return [];
  const out = [];
  for (const d of quality.defects){
    if (!_belongs(d, modId, cam, joint)) continue;
    const lo = Math.max(0, d.start - inFrame), hi = Math.min(len, d.end - inFrame);
    if (hi > lo) out.push({ lo, hi, cat: defectCategory(d.type), label: d.label || d.type });
  }
  return out;
}

// clip-local coloured spans for a proprio/action lane, optionally filtered to a set
// of joint indices (a module or a single DOF). kind: 'proprio' | 'action'.
export function laneDefects(quality, kind, jointSet, inFrame, len){
  if (!quality || !quality.defects) return [];
  const out = [];
  for (const d of quality.defects){
    let belongs = false;
    if (d.type === 'timing_gap') belongs = true;                     // out-of-sync everywhere
    else if (kind === 'proprio'){
      if (['motor_dropout', 'arm_dead', 'torque_dead', 'nan_corrupt'].includes(d.type))
        belongs = !jointSet || (d.joints || []).some(j => jointSet.has(j));
      else if (['low_rate', 'rate_change', 'lag'].includes(d.type))
        belongs = (d.key || '').startsWith('obs/joint');             // joint-wide
    } else if (kind === 'action'){
      if (['low_rate', 'rate_change', 'lag'].includes(d.type)) belongs = (d.key || '') === 'action/ctrl';
    }
    if (!belongs) continue;
    const lo = Math.max(0, d.start - inFrame), hi = Math.min(len, d.end - inFrame);
    if (hi > lo) out.push({ lo, hi, cat: defectCategory(d.type), label: d.label || d.type });
  }
  return out;
}

function _belongs(d, modId, cam, joint){
  if (d.type === 'timing_gap') return true;              // affects every modality lane
  if (modId === 'proprio'){
    if (['motor_dropout', 'arm_dead', 'torque_dead', 'nan_corrupt'].includes(d.type))
      return joint == null || (d.joints || []).includes(joint);
    if (['low_rate', 'rate_change', 'lag'].includes(d.type)) return (d.key || '').startsWith('obs/joint');
    return false;
  }
  if (modId === 'action') return ['low_rate', 'rate_change', 'lag'].includes(d.type) && (d.key || '') === 'action/ctrl';
  if (modId && modId.startsWith('cam:')) return ['camera_freeze', 'camera_blackout'].includes(d.type) && d.cam === cam;
  if (modId === 'lidar') return (d.key || '') === 'obs/lidar';
  return false;
}
