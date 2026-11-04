// h264.js — low-latency H.264 in the browser via WebCodecs.
//
// ES-module port of website/assets/h264-video.js (the teleop web app's decoder),
// kept byte-compatible with it: the robot's camera server encodes each camera ONCE
// with hardware NVENC and pushes the Annex-B access units over a WebSocket at
//   /camera/<name>/h264      (one binary message == one access unit)
// see hardware_bridge/mabel_hw/cameras/video_server.py:_h264_ws_handler.
//
// Why this beats the MJPEG <img> the studio used to use everywhere: MJPEG is
// intra-only, so every frame ships a whole JPEG (~30-60 Mbps at 720p) and the
// browser decodes+reflows an <img> per frame. H.264 is inter-frame (~3-4 Mbps at
// the same resolution) and WebCodecs has NO MSE jitter buffer, so glass-to-glass
// is decode + one frame. On a weak Wi-Fi / Tailscale link the bandwidth cut is the whole ballgame.
//
// The MJPEG <img> stays underneath as the automatic fallback (browser without
// WebCodecs, camera server without the /h264 route, or a stalled stream) — see
// camview.js, which owns that switchover.

export const supported = typeof VideoDecoder === 'function' &&
                         typeof EncodedVideoChunk === 'function';

// Which decode backend actually works HERE — sticky across every camera on the
// page and across reloads, so that if hardware decode ever does fail (see
// HW_LADDER) the other camera tiles don't each rediscover it, burning a keyframe
// and a brief MJPEG fallback apiece. Normally stays 0 (hardware).
const HW_KEY = 'mabel.h264.hw';
let hwStart = 0;
try { hwStart = parseInt(sessionStorage.getItem(HW_KEY), 10) || 0; } catch { /* no DOM */ }
function rememberHw(idx){
  hwStart = idx;
  try { sessionStorage.setItem(HW_KEY, String(idx)); } catch { /* no DOM */ }
}

const nalType = (nalu) => nalu[0] & 0x1f;

// Annex-B buffer -> array of NAL units (start codes stripped).
function splitNALUs(buf){
  const out = [];
  const n = buf.length;
  const startLen = (i) => {
    if (i + 3 < n && buf[i] === 0 && buf[i+1] === 0 && buf[i+2] === 0 && buf[i+3] === 1) return 4;
    if (i + 2 < n && buf[i] === 0 && buf[i+1] === 0 && buf[i+2] === 1) return 3;
    return 0;
  };
  let i = 0;
  while (i < n && startLen(i) === 0) i++;
  while (i < n){
    const sl = startLen(i); if (sl === 0) break;
    const start = i + sl;
    let j = start;
    while (j < n && startLen(j) === 0) j++;
    if (start < j) out.push(buf.subarray(start, j));
    i = j;
  }
  return out;
}

// "avc1.PPCCLL" from an SPS NAL (profile_idc, constraint flags, level_idc).
function codecString(sps){
  const hex = (b) => b.toString(16).padStart(2, '0');
  return 'avc1.' + hex(sps[1]) + hex(sps[2]) + hex(sps[3]);
}

// AVCDecoderConfigurationRecord (avcC) from SPS+PPS — the decoder `description`.
function buildAvcC(sps, pps){
  const b = new Uint8Array(11 + sps.length + pps.length);
  let o = 0;
  b[o++] = 1;                            // configurationVersion
  b[o++] = sps[1];                       // AVCProfileIndication
  b[o++] = sps[2];                       // profile_compatibility
  b[o++] = sps[3];                       // AVCLevelIndication
  b[o++] = 0xff;                         // reserved | lengthSizeMinusOne = 3
  b[o++] = 0xe1;                         // reserved | numOfSPS = 1
  b[o++] = (sps.length >> 8) & 0xff; b[o++] = sps.length & 0xff;
  b.set(sps, o); o += sps.length;
  b[o++] = 1;                            // numOfPPS = 1
  b[o++] = (pps.length >> 8) & 0xff; b[o++] = pps.length & 0xff;
  b.set(pps, o);
  return b;
}

const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

export class H264Stream {
  /**
   * @param {number} [maxQueue=2]  DROP-DON'T-QUEUE depth. WebCodecs never drops on
   *   its own: if decode is slower than the stream (software decode, a busy page),
   *   decodeQueueSize grows without bound and the picture falls further and further
   *   behind real time — video that is smooth but SECONDS late, which is exactly
   *   what "laggy" feels like. Past this depth we stop feeding and resync at the
   *   next IDR, trading a brief skip for bounded latency (what the iOS decoder does).
   *   Pass Infinity to queue everything (used by tools/exp_video_latency.py as the
   *   A/B baseline).
   */
  constructor({ wsUrl, canvas, onLive, onDead, maxQueue = 2 } = {}){
    this.wsUrl = wsUrl;
    this.canvas = canvas || null;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.onLive = onLive || (() => {});
    this.onDead = onDead || (() => {});   // no /h264 route on this server
    this.maxQueue = maxQueue;
    this.dropped = 0;
    this.queuePeak = 0;
    this._resync = false;        // dropped a delta -> the GOP is broken until the next IDR
    this._arrival = new Map();   // chunk ts -> when its bytes reached us (latency probe)
    this._lat = [];              // recent arrival->painted latencies, ms
    this.ws = null;
    this.decoder = null;
    this.sps = null; this.pps = null;
    this.configured = false;
    this.ts = 0;              // monotonic µs stamp for EncodedVideoChunk
    this.live = false;
    this.lastFrame = 0;
    this.frames = 0;
    this.bytesIn = 0;      // >0 means the socket IS delivering (decoder may still be settling)
    this._stopped = false;
    this._backoff = 1000;
    this._hwIdx = hwStart;          // start on the backend already known to work
  }

  // Decode-backend ladder. We ASK for hardware (measured 3607 fps decode at 720p
  // vs 1102 fps software on this Mac — both far above real time), but we must not
  // DEPEND on it: isConfigSupported() can report a config supported and then fail
  // every decode() on it, which would leave the studio black. So if the very first
  // frames error out, step down to software and reconfigure at the next IDR rather
  // than falling all the way back to MJPEG. Never steps down once frames are
  // flowing — a later error is a damaged GOP, not a broken backend.
  static get HW_LADDER(){ return ['prefer-hardware', 'prefer-software']; }

  static get supported(){ return supported; }

  start(){
    if (!supported || !this.wsUrl){ this.onLive(false); return; }
    this._stopped = false;
    this._open();
  }

  _open(){
    if (this._stopped) return;
    let ws;
    try { ws = new WebSocket(this.wsUrl); } catch { return; }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onmessage = (ev) => {
      this._backoff = 1000;
      this.bytesIn += ev.data.byteLength || 0;
      this._onAU(new Uint8Array(ev.data));
    };
    // Reconnect with backoff (1→10 s). A camera server without the /h264 route
    // just keeps failing here while the MJPEG fallback shows — back off instead
    // of hammering a 404 every second.
    ws.onclose = () => {
      if (this._stopped) return;
      if (this.live){ this.live = false; this.onLive(false); }
      // Closed having never delivered a single byte => there is no /h264 route
      // here (the sim's camera server answers 503). Say so ONCE so the viewer can
      // reveal MJPEG immediately instead of staring at black for the full grace
      // window on every camera, every load. We still back off and retry, so a
      // camera server that gains the route later is picked up.
      if (this.bytesIn === 0 && !this._deadRoute){
        this._deadRoute = true;
        try { this.onDead && this.onDead(); } catch {}
      }
      this._backoff = Math.min(this._backoff * 1.7, 10000);
      setTimeout(() => this._open(), this._backoff);
    };
    ws.onerror = () => { try { ws.close(); } catch {} };
  }

  _ensureDecoder(){
    if (this.decoder && this.decoder.state !== 'closed') return;
    this.decoder = new VideoDecoder({
      output: (frame) => {
        const now = performance.now();
        this.lastFrame = now;
        this.frames++;
        const arrived = this._arrival.get(frame.timestamp);
        if (arrived !== undefined){
          this._arrival.delete(frame.timestamp);
          this._lat.push(now - arrived);           // bytes-in -> pixels-on-canvas
          if (this._lat.length > 300) this._lat.shift();
        }
        if (!this.live){ this.live = true; this.onLive(true); }
        if (this.ctx){
          if (this.canvas.width !== frame.displayWidth) this.canvas.width = frame.displayWidth;
          if (this.canvas.height !== frame.displayHeight) this.canvas.height = frame.displayHeight;
          this.ctx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
        }
        frame.close();       // MUST close or WebCodecs stalls on backpressure
      },
      error: (e) => this._onDecoderError(e),
    });
  }

  // A decode error BEFORE we ever produced a frame means this backend cannot play
  // the stream at all -> step down the ladder. An error after frames have flowed is
  // just a damaged GOP (packet loss): keep the backend and resync at the next IDR.
  _onDecoderError(e){
    this.configured = false;
    const canStepDown = this.frames === 0 &&
                        this._hwIdx < H264Stream.HW_LADDER.length - 1;
    if (canStepDown){
      this._hwIdx++;
      rememberHw(this._hwIdx);     // spare every other camera the same discovery
      console.info('H264: falling back to',
                   H264Stream.HW_LADDER[this._hwIdx], '—', e.message || e);
    } else {
      console.warn('H264 decode error', e);
    }
    try { this.decoder?.close(); } catch {}
    this.decoder = null;
    // Drop the cached parameter sets so the next IDR (which carries SPS+PPS
    // in-band, per the encoder contract) re-triggers _configure() on the new
    // backend instead of being treated as "parameters unchanged".
    this.sps = this.pps = null;
  }

  _configure(){
    if (!this.sps || !this.pps) return;
    this._ensureDecoder();
    try {
      this.decoder.configure({
        codec: codecString(this.sps),
        description: buildAvcC(this.sps, this.pps),
        optimizeForLatency: true,
        hardwareAcceleration: H264Stream.HW_LADDER[this._hwIdx],
      });
      this.configured = true;
    } catch (e){
      console.warn('VideoDecoder.configure failed', e);
      this.configured = false;
    }
  }

  _onAU(au){
    const nalus = splitNALUs(au);
    let isKey = false;
    const vcl = [];
    let paramChanged = false;
    for (const nalu of nalus){
      const t = nalType(nalu);
      if (t === 7){ if (!this.sps || !eq(this.sps, nalu)){ this.sps = nalu.slice(); paramChanged = true; } }
      else if (t === 8){ if (!this.pps || !eq(this.pps, nalu)){ this.pps = nalu.slice(); paramChanged = true; } }
      else if (t === 5){ vcl.push(nalu); isKey = true; }
      else if (t === 1){ vcl.push(nalu); }
    }
    if (paramChanged) this._configure();
    if (!this.configured || vcl.length === 0) return;

    // ── DROP-DON'T-QUEUE ──────────────────────────────────────────────────────
    // The decoder is the only place latency can accumulate silently: WebCodecs
    // queues whatever we hand it and never sheds load, so a decoder that runs
    // slower than the stream (software decode, a busy page, a 4-tile studio)
    // falls further behind every frame — smooth video, seconds late. Bound it:
    // once the queue is deeper than maxQueue, stop feeding and resync at the
    // next IDR. A dropped delta breaks the GOP, so we must skip to a keyframe
    // rather than feed a reference-less frame.
    const depth = this.decoder.decodeQueueSize;
    if (depth > this.queuePeak) this.queuePeak = depth;
    if (this._resync && !isKey){ this.dropped++; return; }
    if (!isKey && depth >= this.maxQueue){
      this._resync = true;                 // behind: skip ahead to the next IDR
      this.dropped++;
      return;
    }
    if (isKey) this._resync = false;

    // VCL NALUs -> AVCC (4-byte length-prefixed), concatenated = one frame.
    let total = 0;
    for (const v of vcl) total += 4 + v.length;
    const data = new Uint8Array(total);
    let o = 0;
    for (const v of vcl){
      data[o++] = (v.length >>> 24) & 0xff; data[o++] = (v.length >>> 16) & 0xff;
      data[o++] = (v.length >>> 8) & 0xff;  data[o++] = v.length & 0xff;
      data.set(v, o); o += v.length;
    }
    try {
      // frames the decoder never emits (error, reconfigure) would leak entries
      if (this._arrival.size > 240) this._arrival.clear();
      this._arrival.set(this.ts, performance.now());
      this.decoder.decode(new EncodedVideoChunk({
        type: isKey ? 'key' : 'delta',
        timestamp: this.ts,
        data,
      }));
      this.ts += 33333;      // ~30 fps in µs (timing only; display is immediate)
    } catch { /* mid-GOP join: wait for the next IDR */ }
  }

  /** Live pipeline health — bytes-on-the-wire to pixels-on-the-canvas. */
  stats(){
    const l = this._lat.slice().sort((a, b) => a - b);
    const pct = (p) => (l.length ? l[Math.min(l.length - 1,
      Math.floor(p * l.length))] : 0);
    return {
      frames: this.frames,
      dropped: this.dropped,
      queue: this.decoder ? this.decoder.decodeQueueSize : 0,
      queuePeak: this.queuePeak,
      backend: H264Stream.HW_LADDER[this._hwIdx],
      latP50: pct(0.5),
      latP95: pct(0.95),
      latMax: l.length ? l[l.length - 1] : 0,
    };
  }

  /** True if a decoded frame arrived within `ms` — the MJPEG-fallback signal. */
  isFresh(ms = 2500){ return (performance.now() - this.lastFrame) < ms; }

  stop(){
    this._stopped = true;
    if (this.ws){ try { this.ws.close(); } catch {} this.ws = null; }
    if (this.decoder && this.decoder.state !== 'closed'){ try { this.decoder.close(); } catch {} }
    this.decoder = null;
    this.configured = false;
    this.sps = this.pps = null;
    if (this.live){ this.live = false; this.onLive(false); }
  }
}

// pure helpers exported for the node unit tests
export const _internals = { splitNALUs, codecString, buildAvcC };
