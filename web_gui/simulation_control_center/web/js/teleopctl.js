// Teleop input layer: on-screen virtual joysticks + WASD/arrow keyboard, fed
// through a 30 Hz pump into the bridge. Semantics match the iOS cockpit:
//   left stick  TRANSLATE  (ly = forward, lx = strafe)
//   right stick ROTATE·LIFT (rx = yaw, ry = lift rate)
// In manipulate mode the same inputs become wrist-target nudges instead.
// Keyboard: WASD drive, QE turn, RF lift, arrows nudge the active wrist,
// [ ] switch active side. Sends stop instantly when disengaged / soft / estop.
import * as bus from './bus.js';

export class Joystick {
  constructor(el, onChange){
    this.el = el;
    this.knob = el.querySelector('.stick-knob');
    this.v = { x: 0, y: 0 };
    this.active = false;              // pointer is holding the stick
    this.onChange = onChange;
    el.addEventListener('pointerdown', (e) => {
      try { el.setPointerCapture(e.pointerId); } catch { /* capture optional */ }
      this.active = true;
      this._track(e);
      const move = (ev) => { if (ev.pointerId === e.pointerId) this._track(ev); };
      // Release MUST always zero the command (iOS-cockpit semantics). The
      // handlers live on WINDOW — a pointerup outside the element, a lost
      // capture (Safari), a cancelled touch, or a window switch can otherwise
      // leave the stick latched at full deflection with the pump streaming
      // that speed forever (observed live).
      const up = (ev) => {
        if (ev.pointerId != null && ev.pointerId !== e.pointerId
            && ev.type !== 'blur') return;
        try { el.releasePointerCapture(e.pointerId); } catch { /* already lost */ }
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        window.removeEventListener('pointercancel', up);
        window.removeEventListener('blur', up);
        el.removeEventListener('lostpointercapture', up);
        this.active = false;
        this._set(0, 0);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
      window.addEventListener('pointercancel', up);
      window.addEventListener('blur', up);
      el.addEventListener('lostpointercapture', up);
    });
  }
  // keyboard visualization: move the knob to reflect key-driven axes without
  // fighting an actual pointer drag
  setVisual(x, y){
    if (this.active) return;
    const r = this.el.getBoundingClientRect();
    const rad = r.width / 2 - 8;
    this.knob.style.transform = `translate(${x * rad}px, ${y * rad}px)`;
  }
  _track(e){
    const r = this.el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const rad = r.width / 2 - 8;
    let dx = (e.clientX - cx) / rad, dy = (e.clientY - cy) / rad;
    const m = Math.hypot(dx, dy);
    if (m > 1){ dx /= m; dy /= m; }
    this._set(dx, dy);
  }
  _set(x, y){
    this.v = { x, y };
    const r = this.el.getBoundingClientRect();
    const rad = r.width / 2 - 8;
    this.knob.style.transform = `translate(${x * rad}px, ${y * rad}px)`;
    this.onChange && this.onChange(this.v);
  }
}

// WASD = LEFT stick (translate) · ARROWS = RIGHT stick (rotate/lift in drive,
// wrist nudges in manipulate) · QE/RF stay as extra yaw/lift keys
const KEYS = {
  w: 'fwd', s: 'back', a: 'left', d: 'right', q: 'ccw', e: 'cw',
  r: 'liftup', f: 'liftdown',
  ArrowUp: 'aup', ArrowDown: 'adown', ArrowLeft: 'aleft', ArrowRight: 'aright',
  PageUp: 'wup', PageDown: 'wdown',
};

export class TeleopCtl {
  constructor(link, ctl, viewport){
    this.link = link;
    this.ctl = ctl;                       // shared control-state object
    this.viewport = viewport;
    this.stick = { lx: 0, ly: 0, rx: 0, ry: 0 };
    this.keys = new Set();
    this.activeSide = 'right';
    this._wireKeys();
    this._timer = setInterval(() => this._pump(), 33);
  }

  attachSticks(elL, elR, udL, udR){
    this.jl = new Joystick(elL, (v) => { this.stick.lx = v.x; this.stick.ly = -v.y; });
    this.jr = new Joystick(elR, (v) => { this.stick.rx = v.x; this.stick.ry = -v.y; });
    // hold-buttons: per-arm wrist height in manipulate mode
    this.ud = { left: 0, right: 0 };
    const wire = (el, side) => {
      if (!el) return;
      el.querySelectorAll('button').forEach((b) => {
        const v = +b.dataset.ud;
        b.addEventListener('pointerdown', (e) => {
          b.setPointerCapture(e.pointerId);
          this.ud[side] = v;
        });
        const stop = () => { this.ud[side] = 0; };
        b.addEventListener('pointerup', stop);
        b.addEventListener('pointercancel', stop);
      });
    };
    wire(udL, 'left'); wire(udR, 'right');
  }

  _wireKeys(){
    window.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;   // cmd+W etc: keyup never fires
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.isContentEditable) return;
      // [ ] switch the active side — the module rail follows (app.js), so the
      // keyboard and the rail can never disagree about which wrist is live
      if (e.key === '[' || e.key === ']'){
        this.activeSide = e.key === '[' ? 'left' : 'right';
        bus.emit('sideSwitch', this.activeSide);
        bus.emit('ctl');
        return;
      }
      const k = KEYS[e.key] || KEYS[e.key.toLowerCase?.()];
      if (k){ this.keys.add(k); e.preventDefault(); }
    });
    window.addEventListener('keyup', (e) => {
      const k = KEYS[e.key] || KEYS[e.key.toLowerCase?.()];
      if (k) this.keys.delete(k);
    });
    const clearAll = () => {
      this.keys.clear();
      // zero any latched stick too — nothing may keep driving unattended
      this.stick = { lx: 0, ly: 0, rx: 0, ry: 0 };
      this.jl?._set?.(0, 0); this.jr?._set?.(0, 0);
    };
    window.addEventListener('blur', clearAll);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState !== 'visible') clearAll();
    });
  }

  // effective drive command combining sticks + keys, in [-1, 1]
  // (WASD → left stick; ARROWS → right stick: ←→ yaw, ↑↓ lift)
  driveVec(){
    const k = this.keys;
    const key = (a, b) => (k.has(a) ? 1 : 0) - (k.has(b) ? 1 : 0);
    return {
      ly: clamp(this.stick.ly + key('fwd', 'back')),
      lx: clamp(this.stick.lx + key('right', 'left')),
      rx: clamp(this.stick.rx + key('cw', 'ccw') + key('aright', 'aleft')),
      ry: clamp(this.stick.ry + key('liftup', 'liftdown') + key('aup', 'adown')),
    };
  }
  // KEYBOARD wrist nudge (manipulate mode) in [-1, 1] per axis — arrows move
  // the [ ]-selected wrist in the plane, PgUp/PgDn its height
  wristVec(){
    const k = this.keys;
    const key = (a, b) => (k.has(a) ? 1 : 0) - (k.has(b) ? 1 : 0);
    return {
      x: clamp(key('aright', 'aleft')),
      y: clamp(key('aup', 'adown')),
      z: clamp(key('wup', 'wdown')),
    };
  }

  _pump(){
    const c = this.ctl;
    // watchdog: a stick nobody is holding MUST read zero — self-heal any
    // latch (lost pointerup) before it becomes a runaway drive command
    if (this.jl && !this.jl.active && (this.stick.lx || this.stick.ly)){
      this.stick.lx = this.stick.ly = 0; this.jl._set?.(0, 0);
    }
    if (this.jr && !this.jr.active && (this.stick.rx || this.stick.ry)){
      this.stick.rx = this.stick.ry = 0; this.jr._set?.(0, 0);
    }
    if (c.webcam) return;   // webcam body-teleop owns the wire (bodyteleop.js)
    if (!c.teleop || c.estop || c.feel === 'soft'){
      if (this._wasDriving){ this.link.frame({ lx: 0, ly: 0, rx: 0, ry: 0 }); this._wasDriving = false; }
      return;
    }
    if (c.mode === 'drive'){
      const d = this.driveVec();
      // wire signs (teleop-app.js): forward⇒ly+, right strafe⇒lx−,
      // stick-right⇒rx− (CW yaw = turn right) — all three axes negate the
      // raw stick x so the robot goes the way you push
      this.link.frame({ lx: -d.lx, ly: d.ly, rx: -d.rx, ry: d.ry });
      // knobs follow the keyboard so the viewer shows what's commanded
      this.jl?.setVisual(d.lx, -d.ly);
      this.jr?.setVisual(d.rx, -d.ry);
      this._wasDriving = true;
    } else {
      // manipulate: LEFT stick + its ▲▼ = left wrist, RIGHT = right wrist
      // (sticks give planar XY; the flanking buttons give height)
      const sides = {
        left: { x: this.stick.lx, y: this.stick.ly, z: this.ud?.left || 0 },
        right: { x: this.stick.rx, y: this.stick.ry, z: this.ud?.right || 0 },
      };
      // keyboard arrows still nudge the [ ]-selected side
      const kv = this.wristVec();
      if (Math.abs(kv.x) + Math.abs(kv.y) + Math.abs(kv.z) > 0.05){
        const s = sides[this.activeSide];
        s.x = clamp(s.x + kv.x); s.y = clamp(s.y + kv.y); s.z = clamp(s.z + kv.z);
      }
      for (const side of ['left', 'right']){
        const v = sides[side];
        if (Math.abs(v.x) + Math.abs(v.y) + Math.abs(v.z) > 0.05)
          this.viewport && this.nudgeWrist(side, v);
      }
      this.jl?.setVisual(sides.left.x, -sides.left.y);
      this.jr?.setVisual(sides.right.x, -sides.right.y);
      // whole-body region: the viewport's base follower creeps toward
      // out-of-reach wrist targets (client analog of the server reach-assist);
      // otherwise keep the base still
      this.viewport?.refreshBaseAssist?.();   // live re-eval: stops in deadband
      const assist = this.viewport?.baseAssist;
      if (assist && c.region === 'whole_body'){
        this.link.frame({ lx: -clamp(assist.lx), ly: clamp(assist.ly), rx: 0, ry: 0 });
        this._wasDriving = true;
      } else if (this._wasDriving){ this.link.frame({ lx: 0, ly: 0, rx: 0, ry: 0 }); this._wasDriving = false; }
    }
  }

  nudgeWrist(side, v){
    const vp = this.viewport;
    if (!vp?.wrist?.[side]?.grp.visible) return;
    const s = 0.006;                                     // m per tick at full deflection
    const g = vp.wrist[side].grp;
    // camera-relative planar nudge: x → screen right, y → screen forward, z → up
    const cam = vp.stage.camera;
    const fwd = new (g.position.constructor)();          // THREE.Vector3 without import
    cam.getWorldDirection(fwd);
    fwd.y = 0; fwd.normalize();
    const rgt = fwd.clone().cross(cam.up).normalize();
    g.position.addScaledVector(rgt, v.x * s);
    g.position.addScaledVector(fwd, v.y * s);
    g.position.y += v.z * s;
    vp._solveArm(side);
  }

  destroy(){ clearInterval(this._timer); }
}

const clamp = (v) => Math.max(-1, Math.min(1, v));
