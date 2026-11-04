// camview.js — ONE camera surface, fast path first.
//
// Every camera in the studio (strip tiles, add-window tiles, the first-person
// FPV, the 3-D projection textures) goes through CamView so they all get the
// same transport policy:
//
//   H.264 / WebCodecs -> <canvas>      (the fast path: inter-frame, ~3-4 Mbps @720p,
//                                       no jitter buffer -> decode + one frame)
//   MJPEG <img>                        (fallback ONLY: no WebCodecs, no /h264 route
//                                       on the camera server, or a stalled stream)
//
// This mirrors what the iOS app and the website teleop page already do
// (website/assets/teleop-app.js:_syncH264). The studio used to be the last client
// on plain MJPEG <img> tags, which is why its video was far slower than theirs.
//
// IMPORTANT — we do NOT run both at once. The MJPEG stream is the bandwidth hog we
// are trying to escape (intra-only: a whole JPEG every frame, ~30-60 Mbps at 720p,
// and it is what saturates a weak Wi-Fi link). So when WebCodecs is available we open the
// H.264 socket ONLY, and attach the MJPEG <img> lazily if H.264 has not produced a
// frame within GRACE_MS. Not opening it also dodges the Safari trap that a native
// MJPEG <img> can never be reliably aborted once started.
import { camUrl, camStreamProxied, camH264Url } from './config.js';
import { H264Stream, supported as h264Supported } from './h264.js';

const GRACE_MS = 1800;      // no H.264 frame by now -> consider the MJPEG fallback
const GRACE_EXTENDS = 2;    // ...but keep waiting while BYTES are still arriving
const STALE_MS = 2500;      // H.264 went quiet this long -> fall back again
const WATCH_MS = 1000;

export class CamView {
  /**
   * @param {HTMLImageElement} img   the existing MJPEG <img> (kept as fallback)
   * @param {string}  name           stream name, e.g. 'main' / 'wrist_left'
   * @param {object}  [opts]
   * @param {HTMLCanvasElement} [opts.canvas]  reuse a canvas instead of inserting one
   * @param {(live:boolean)=>void} [opts.onLive]  pixels arriving (either transport)
   * @param {boolean} [opts.proxied]  force the MJPEG fallback through our own
   *        same-origin /api/cam proxy. REQUIRED for anything textured into WebGL
   *        (the 3-D camera projections): a direct :8080 <img> is cross-origin and
   *        would taint the canvas. The H.264 canvas is never tainted — its pixels
   *        come from WebSocket bytes, not from a cross-origin image — so the fast
   *        path needs no proxy either way.
   */
  constructor(img, name, opts = {}){
    this.img = img;
    this.name = name;
    this.stopped = false;
    this._proxied = !!opts.proxied;
    this._onLive = opts.onLive || (() => {});
    this._wasLive = null;
    this._h264 = null;
    this._mjpegOn = false;
    this._everLive = false;
    this._graceTimer = null;
    this._watchTimer = null;

    if (h264Supported){
      this.canvas = opts.canvas || this._mkCanvas(img);
      this.canvas.style.display = 'none';
    } else {
      this.canvas = null;
    }
    // MJPEG-fallback liveness (the H.264 path reports via H264Stream.onLive)
    img.addEventListener('load', () => { if (this._mjpegOn) this._emit(true); });
    img.addEventListener('error', () => { if (this._mjpegOn) this._emit(false); });
    this.connect();
  }

  _emit(live){
    if (live === this._wasLive) return;
    this._wasLive = live;
    this._onLive(live);
  }

  // A canvas layered exactly over the <img>: same box, same object-fit (canvas is
  // a replaced element, so object-fit applies to it too), so swapping between the
  // two is visually seamless. .cam-tile / .fpv are already position:relative.
  _mkCanvas(img){
    const c = document.createElement('canvas');
    c.className = 'cam-h264';
    c.style.objectFit = getComputedStyle(img).objectFit || 'cover';
    img.insertAdjacentElement('afterend', c);
    return c;
  }

  /** (Re)connect the stream — also the scene-switch / camera-restart recovery,
   *  and how a stop()ped view (e.g. the FPV when hidden) is brought back up. */
  connect(){
    this.stopped = false;
    this._teardownStreams();

    if (h264Supported){
      this._h264 = new H264Stream({
        wsUrl: camH264Url(this.name),
        canvas: this.canvas,
        onLive: (live) => this._show(live),
        // no /h264 on this server -> stop waiting out the grace window
        onDead: () => { if (!this.stopped) this._mjpeg(true); },
      });
      this._h264.start();
      this._graceLeft = GRACE_EXTENDS;
      this._graceTimer = setTimeout(() => this._graceCheck(), GRACE_MS);
      this._watchTimer = setInterval(() => this._watch(), WATCH_MS);
    } else {
      this._mjpeg(true);         // no WebCodecs: MJPEG is all we have
    }
  }

  // Grace expired without a decoded frame. Only pay for MJPEG if H.264 is truly
  // not delivering: if BYTES are arriving the socket is healthy and the decoder is
  // merely still settling (it waits for the first IDR to configure, and may burn a
  // keyframe if it has to step down a decode backend — see h264.js HW_LADDER), so
  // wait rather than open a second, far heavier stream we would drop a moment later.
  _graceCheck(){
    if (this.stopped || !this._h264 || this._h264.live) return;
    if (this._h264.bytesIn > 0 && this._graceLeft > 0){
      this._graceLeft--;
      this._graceTimer = setTimeout(() => this._graceCheck(), GRACE_MS);
      return;
    }
    this._mjpeg(true);            // socket dead / no /h264 route -> fall back
  }

  // H.264 stalled (camera restart, encoder death, socket drop) -> reveal MJPEG;
  // recovered -> hide it again. This watchdog owns ONLY the post-startup case:
  // before the first frame the grace path above decides, or this would attach
  // MJPEG at t=1s while H.264 is still legitimately coming up.
  _watch(){
    if (this.stopped || !this._h264 || !this._everLive) return;
    const fresh = this._h264.live && this._h264.isFresh(STALE_MS);
    this._show(fresh);
    if (!fresh) this._mjpeg(true);
  }

  _show(live){
    if (!this.canvas) return;
    this.canvas.style.display = live ? '' : 'none';
    this.img.style.display = live ? 'none' : '';
    if (live){
      this._everLive = true;
      clearTimeout(this._graceTimer);  // beat the clock: no MJPEG needed at all
      this._mjpeg(false);              // fast path is up: drop the MJPEG bandwidth
      this._emit(true);
    }
  }

  _mjpeg(on){
    if (on === this._mjpegOn || this.stopped) return;
    this._mjpegOn = on;
    if (on) this._mjpegAttach();
    else {
      clearTimeout(this._mjpegRetry); this._mjpegRetry = null;
      this.img.onerror = null; this.img.onload = null;
      this.img.removeAttribute('src');
    }
  }

  /** Attach the MJPEG fallback, and KEEP TRYING while it fails.
   *
   *  COLD START: run.sh opens the browser as soon as the STUDIO answers, which
   *  is before the camera server is up. An MJPEG <img> never recovers a
   *  connection on its own — one error and the tile is black for the life of the
   *  page. Setting `src` once was therefore a permanent failure whenever the
   *  page won the race, which is most launches. Retry with backoff (0.5 s → 8 s)
   *  until it paints; `stop()` and a switch back to H.264 cancel it, so a tile
   *  nobody is looking at costs nothing. */
  _mjpegAttach(){
    if (this.stopped || !this._mjpegOn) return;
    const u = this._proxied ? camStreamProxied(this.name) : camUrl(this.name);
    this.img.onerror = () => {
      if (this.stopped || !this._mjpegOn) return;
      this._mjpegBackoff = Math.min((this._mjpegBackoff || 500) * 1.6, 8000);
      clearTimeout(this._mjpegRetry);
      this._mjpegRetry = setTimeout(() => this._mjpegAttach(), this._mjpegBackoff);
    };
    this.img.onload = () => { this._mjpegBackoff = 500; };
    // cache-buster: a kept-over src can be a dead socket from before a restart
    this.img.src = u + (u.includes('?') ? '&' : '?') + 'r=' + Date.now();
  }

  /** Switch which camera this surface shows. */
  setName(name){
    if (name === this.name) return;
    this.name = name;
    this.connect();
  }

  /** The element currently carrying live pixels — for WebGL texture uploads. */
  surface(){
    const liveH264 = this._h264 && this._h264.live && this._h264.isFresh(STALE_MS);
    return liveH264 ? this.canvas : this.img;
  }

  /** True once the surface has pixels worth uploading. */
  ready(){
    const s = this.surface();
    return s === this.canvas ? this._h264.frames > 0
                             : !!(this.img.complete && this.img.naturalWidth);
  }

  live(){ return !!(this._h264 && this._h264.live); }

  _teardownStreams(){
    if (this._h264){ this._h264.stop(); this._h264 = null; }
    clearTimeout(this._graceTimer); this._graceTimer = null;
    clearInterval(this._watchTimer); this._watchTimer = null;
    this._mjpegOn = false;
    this._everLive = false;      // a reconnect is a cold start: grace path again
    this.img.removeAttribute('src');
  }

  stop(){
    this.stopped = true;
    clearTimeout(this._mjpegRetry); this._mjpegRetry = null;
    this._teardownStreams();
    if (this.canvas) this.canvas.style.display = 'none';
  }
}
