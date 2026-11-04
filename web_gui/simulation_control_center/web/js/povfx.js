// PovFx — the 1st↔3rd person camera choreography. One instance owns the fpv
// overlay fade + the viewport camera flight; the live UI drives it with a
// wall-clock rAF loop (interval fallback for throttled tabs), and the e2e
// recorder drives the SAME applyStep() deterministically frame by frame.
import { DURATION_MS, povSchedule, povPose } from './transition.js';

export class PovFx {
  constructor(viewport, fpvEl, vpEl, opts = {}){
    this.vp = viewport;
    this.fpv = fpvEl;           // .fpv overlay (img wrapper)
    this.vpEl = vpEl;           // .viewport3d host (gets .pip-3d in 1st person)
    this.onLayers = opts.onLayers || (() => {});   // show cloud+frustum layers
    this.onDone = opts.onDone || (() => {});
    this.running = false;
    this.pov = 'third';
    this.savedOrbit = null;
  }

  // prepare endpoint poses + UI for a direction; returns the flight endpoints
  begin(dir){
    const vp = this.vp;
    this.onLayers();                              // pointcloud + head-cam projection on
    if (dir === 'to1st'){
      this.savedOrbit = vp.orbitPose();
      this.A = this.savedOrbit;
      this.B = vp.headCamPose();
    } else {
      // leaving 1st person: canvas comes back full-bleed at the head pose
      this.vpEl.classList.remove('pip-3d');
      window.dispatchEvent(new Event('studio-resize'));   // re-aspect the buffer
      this.A = vp.headCamPose();
      this.B = this.savedOrbit || vp.orbitPose();
      vp.applyPov(this.A);
    }
    this.fpv.classList.remove('hidden');
    this.fpv.style.opacity = dir === 'to1st' ? 0 : 1;
    vp.setControlsEnabled(false);
    this.dir = dir;
  }

  // u ∈ [0,1] raw wall-clock progress → camera pose + fpv alpha
  applyStep(u){
    const { f, a } = povSchedule(this.dir, u);
    this.vp.applyPov(povPose(this.A, this.B, f));
    this.fpv.style.opacity = a;
    return { f, a };
  }

  finish(){
    const vp = this.vp;
    if (this.dir === 'to1st'){
      this.pov = 'first';
      this.fpv.style.opacity = 1;
      // steady 1st person: fpv full-screen, canvas becomes the PiP model view
      this.vpEl.classList.add('pip-3d');
      window.dispatchEvent(new Event('studio-resize'));   // re-aspect for 240×170
      if (this.savedOrbit){
        vp.applyPov(this.savedOrbit);
        vp.stage.controls.target.copy(this.savedOrbit.target);
      }
      vp.setControlsEnabled(true);
    } else {
      this.pov = 'third';
      this.fpv.style.opacity = 0;
      this.fpv.classList.add('hidden');
      if (this.savedOrbit) vp.stage.controls.target.copy(this.savedOrbit.target);
      else vp.recenter();
      vp.setControlsEnabled(true);
    }
    this.running = false;
    this.onDone(this.pov);
  }

  // live animated run (2 s, wall-clock — duration holds even at low fps)
  run(dir){
    if (this.running || (dir === 'to1st' ? 'first' : 'third') === this.pov)
      return Promise.resolve(this.pov);
    this.running = true;
    this.begin(dir);
    const t0 = performance.now();
    return new Promise((resolve) => {
      const tick = () => {
        if (!this.running) return;
        const u = (performance.now() - t0) / DURATION_MS;
        if (u >= 1){
          clearInterval(iv);
          this.applyStep(1);
          this.finish();
          resolve(this.pov);
          return;
        }
        this.applyStep(u);
        requestAnimationFrame(tick);
      };
      // interval floor so the move completes even in a throttled/hidden tab
      const iv = setInterval(() => {
        const u = (performance.now() - t0) / DURATION_MS;
        if (u >= 1){
          clearInterval(iv);
          if (this.running){ this.applyStep(1); this.finish(); resolve(this.pov); }
        } else if (this.running) this.applyStep(u);
      }, 66);
      requestAnimationFrame(tick);
    });
  }
}
