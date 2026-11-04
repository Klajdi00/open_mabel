// Setpoint sequence model — the cinema-robot timeline's data layer, kept pure
// for tests: keyframes {t, pose{joint:rad}, label} on a time axis, linear
// pose interpolation between neighbours, and an executor step function.
export class SetpointSeq {
  constructor(){
    this.keys = [];                 // sorted by t
    this.onChange = null;
  }
  add(t, pose, label = ''){
    const k = { t: Math.max(0, t), pose: { ...pose }, label };
    this.keys.push(k);
    this.keys.sort((a, b) => a.t - b.t);
    this._emit();
    return k;
  }
  remove(k){
    const i = this.keys.indexOf(k);
    if (i >= 0){ this.keys.splice(i, 1); this._emit(); }
  }
  move(k, t){
    k.t = Math.max(0, t);
    this.keys.sort((a, b) => a.t - b.t);
    this._emit();
  }
  duration(){ return this.keys.length ? this.keys[this.keys.length - 1].t : 0; }
  _emit(){ this.onChange && this.onChange(); }

  // pose at time t: hold before first / after last, lerp between neighbours
  poseAt(t){
    const K = this.keys;
    if (!K.length) return null;
    if (t <= K[0].t) return { ...K[0].pose };
    if (t >= K[K.length - 1].t) return { ...K[K.length - 1].pose };
    let i = 0;
    while (i < K.length - 1 && K[i + 1].t < t) i++;
    const a = K[i], b = K[i + 1];
    const u = (t - a.t) / Math.max(1e-6, b.t - a.t);
    return lerpPose(a.pose, b.pose, u);
  }

  toJSON(){ return { keys: this.keys.map(k => ({ t: k.t, pose: k.pose, label: k.label })) }; }
  static fromJSON(j){
    const s = new SetpointSeq();
    for (const k of j?.keys || []) s.keys.push({ t: k.t, pose: { ...k.pose }, label: k.label || '' });
    s.keys.sort((a, b) => a.t - b.t);
    return s;
  }
}

export function lerpPose(a, b, u){
  const out = {};
  const names = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const n of names){
    const va = a[n], vb = b[n];
    out[n] = va == null ? vb : vb == null ? va : va + (vb - va) * u;
  }
  return out;
}
