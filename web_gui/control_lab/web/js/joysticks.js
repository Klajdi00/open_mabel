/* Dual virtual joystick — vendored from the studio cockpit (js/teleopctl.js):
   pointer-tracked knob normalized to the unit disc, with the robust release
   path (window-level up/move + blur/pointercancel force (0,0) so a lost
   pointerup can never latch a runaway drive). Left stick = translate
   (ly forward, lx strafe); right stick = yaw (rx) + lift (ry). */

export class Joystick {
  constructor(el, onChange) {
    this.el = el; this.knob = el.querySelector('.stick-knob');
    this.onChange = onChange; this.x = 0; this.y = 0; this._id = null;
    el.addEventListener('pointerdown', e => {
      this._id = e.pointerId; el.setPointerCapture(e.pointerId);
      this._track(e);
    });
    const up = e => { if (this._id != null) this._set(0, 0); this._id = null; };
    window.addEventListener('pointermove', e => {
      if (this._id === e.pointerId) this._track(e);
    });
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    window.addEventListener('blur', () => { this._set(0, 0); this._id = null; });
    el.addEventListener('lostpointercapture', up);
  }
  _track(e) {
    const r = this.el.getBoundingClientRect();
    let x = (e.clientX - r.left - r.width / 2) / (r.width / 2);
    let y = (e.clientY - r.top - r.height / 2) / (r.height / 2);
    const m = Math.hypot(x, y);
    if (m > 1) { x /= m; y /= m; }
    this._set(x, -y);                      // up = +y
  }
  _set(x, y) {
    this.x = x; this.y = y;
    this.knob.style.transform =
      `translate(calc(-50% + ${x * 36}px), calc(-50% + ${-y * 36}px))`;
    this.onChange && this.onChange(x, y);
  }
}

export class Sticks {
  /* pumps {lx, ly, rx, ry} to `send` at 30 Hz while any axis is non-zero,
     plus one zero frame on release (the cockpit watchdog pattern). */
  constructor(elL, elR, send) {
    this.v = { lx: 0, ly: 0, rx: 0, ry: 0 };
    this.left = new Joystick(elL, (x, y) => { this.v.lx = x; this.v.ly = y; });
    this.right = new Joystick(elR, (x, y) => { this.v.rx = x; this.v.ry = y; });
    this._wasZero = true;
    setInterval(() => {
      const z = !this.v.lx && !this.v.ly && !this.v.rx && !this.v.ry;
      if (!z || !this._wasZero) send({ ...this.v });
      this._wasZero = z;
    }, 33);
  }
}
