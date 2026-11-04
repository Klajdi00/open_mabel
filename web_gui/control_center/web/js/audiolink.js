// AudioLink — client of the robot's :9610 voice plane (the ONE duplex plane).
//
//   push-to-talk:  mic → AudioWorklet → 16 kHz mono PCM16 → AUDIO 0x07 → robot
//   listen:        robot mic → AUDIO 0x07 → AudioContext → your speakers
//
// Everything here exists because the browser and the robot disagree about audio
// in three specific ways, and each one is silent if you get it wrong:
//
//  1. RATE. getUserMedia gives you whatever the OS wants (48 kHz on every Mac,
//     44.1 kHz on some Windows boxes) as Float32. The robot's bar is 16 kHz mono
//     S16_LE on the wire. We downsample HERE — the server REFUSES a frame whose
//     rate it didn't advertise rather than resample it (a 48 kHz frame played at
//     16 kHz is an octave and a half down, out of a real speaker in a room with
//     people). The `hello` frame carries the format; we adopt it, never assume it.
//  2. THREADS. Capture must run in an AudioWorklet, not a ScriptProcessorNode:
//     ScriptProcessor runs on the main thread, so a 3D viewport repaint becomes an
//     audible dropout. The worklet is inlined as a Blob so this stays one file with
//     no build step (the studio ships raw ES modules).
//  3. SEQUENCE, NOT SNAPSHOT. Playback schedules each frame at an explicit time on
//     a running clock. The obvious `ctx.currentTime` per frame drifts and clicks;
//     we keep a playhead and only resync when it falls behind, which is the same
//     resync-don't-reorder rule the server hub uses (audio_streams.py).
import * as bus from './bus.js';
import { audioUrl } from './config.js';

const AUDIO_TAG = 0x07;
const HDR = 16;                       // <BBHIII — see server/docs/data_streams.md

// The capture worklet: hand the main thread raw Float32 blocks and nothing else.
// Resampling lives on the main thread so this stays trivial (it runs in the audio
// render quantum — anything expensive here is a dropout).
const WORKLET_SRC = `
class MabelCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));   // copy: the buffer is recycled
    return true;
  }
}
registerProcessor('mabel-capture', MabelCapture);
`;

function encodeAudio(pcm16, seq, rate, channels) {
  const buf = new ArrayBuffer(HDR + pcm16.byteLength);
  const v = new DataView(buf);
  v.setUint8(0, AUDIO_TAG);
  v.setUint8(1, 0);                             // codec 0 = PCM16
  v.setUint16(2, channels, true);
  v.setUint32(4, seq >>> 0, true);
  v.setUint32(8, rate, true);
  v.setUint32(12, pcm16.byteLength, true);
  new Uint8Array(buf, HDR).set(new Uint8Array(pcm16.buffer, pcm16.byteOffset,
                                              pcm16.byteLength));
  return buf;
}

function decodeAudio(buf) {
  if (!buf || buf.byteLength < HDR) return null;
  const v = new DataView(buf);
  if (v.getUint8(0) !== AUDIO_TAG) return null;
  const codec = v.getUint8(1);
  if (codec !== 0) return null;                 // Opus reserved — never play it as PCM
  const n = v.getUint32(12, true);
  if (buf.byteLength < HDR + n) return null;    // truncated: never play a partial frame
  return {
    channels: v.getUint16(2, true),
    seq: v.getUint32(4, true),
    rate: v.getUint32(8, true),
    pcm: new Int16Array(buf.slice(HDR, HDR + n)),
  };
}

export class AudioLink {
  constructor() {
    this.ws = null;
    this.fmt = null;              // adopted from the server's hello
    this.talking = false;
    this.listening = false;
    this.seq = 0;
    this.lastSeq = null;
    this.gaps = 0;
    this.micLevel = 0;            // 0..1, what WE are sending
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._playhead = 0;
  }

  // ── connection ────────────────────────────────────────────────────────────
  connect() {
    const url = audioUrl();
    if (!url) {                   // same-origin (https) mode: no voice proxy
      bus.emit('audio:state', {
        available: false,
        reason: 'voice needs a direct connection to the robot host (:9610) — open '
                + 'the studio over http on the LAN / Tailscale address',
      });
      return;
    }
    if (this.ws) return;
    let ws;
    try { ws = new WebSocket(url); } catch { return; }
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        let m; try { m = JSON.parse(ev.data); } catch { return; }
        if (m.type === 'hello') {
          // Adopt the server's format. Hardcoding 16 kHz here would "work" right
          // up until the robot's bar changed and every voice went chipmunk.
          this.fmt = { rate: m.rate, channels: m.channels, frameMs: m.frameMs };
          bus.emit('audio:state', { available: true, ...this.fmt });
        } else if (m.type === 'notice') {
          bus.emit('audio:state', { available: false, reason: m.reason });
        }
        return;
      }
      this._onFrame(ev.data);
    };
    ws.onclose = () => {
      this.ws = null; this.fmt = null;
      this.stopTalking(); this.stopListening();
      bus.emit('audio:state', { available: false, reason: 'disconnected' });
    };
    ws.onerror = () => { /* onclose follows; the UI reads state from there */ };
    this.ws = ws;
  }

  disconnect() {
    this.stopTalking(); this.stopListening();
    this.ws?.close(); this.ws = null;
  }

  // ── listen: robot mic → your speakers ─────────────────────────────────────
  async startListening() {
    if (this.listening) return;
    this._ctx = this._ctx || new (window.AudioContext || window.webkitAudioContext)();
    await this._ctx.resume();     // a user gesture is required; the button is one
    this._playhead = 0;
    this.listening = true;
    bus.emit('audio:listening', true);
  }

  stopListening() {
    this.listening = false;
    bus.emit('audio:listening', false);
  }

  _onFrame(buf) {
    const f = decodeAudio(buf);
    if (!f) return;
    // seq is the ONLY gap detector: audio is a sequence, so a dropped frame is an
    // audible click, not a stale render we can ignore. Count them — a climbing gap
    // count is the difference between "the network is fine" and "it isn't".
    if (this.lastSeq !== null && f.seq !== ((this.lastSeq + 1) >>> 0)) this.gaps++;
    this.lastSeq = f.seq;
    let sum = 0;
    for (let i = 0; i < f.pcm.length; i++) sum += f.pcm[i] * f.pcm[i];
    bus.emit('audio:level', {
      robot: Math.min(1, Math.sqrt(sum / Math.max(f.pcm.length, 1)) / 32768),
      gaps: this.gaps,
    });
    if (!this.listening || !this._ctx) return;

    const ab = this._ctx.createBuffer(f.channels, f.pcm.length / f.channels, f.rate);
    for (let c = 0; c < f.channels; c++) {
      const out = ab.getChannelData(c);
      for (let i = 0; i < out.length; i++) out[i] = f.pcm[i * f.channels + c] / 32768;
    }
    const src = this._ctx.createBufferSource();
    src.buffer = ab;
    src.connect(this._ctx.destination);
    // Schedule on a PLAYHEAD, not ctx.currentTime: starting every frame at "now"
    // leaves a gap the size of the scheduling jitter between each one — a 50 Hz
    // click track. Resync only when we fall behind (the same drop-the-backlog
    // rule the server hub uses), so one late frame doesn't compound forever.
    const now = this._ctx.currentTime;
    if (this._playhead < now) this._playhead = now + 0.02;
    src.start(this._playhead);
    this._playhead += ab.duration;
  }

  // ── talk: your mic → the robot's speaker ──────────────────────────────────
  async startTalking() {
    if (this.talking || !this.ws || !this.fmt) return;
    this._ctx = this._ctx || new (window.AudioContext || window.webkitAudioContext)();
    await this._ctx.resume();
    // Let the browser do the hard DSP: this bar is a speakerphone in the same room
    // as its own speaker, so without echoCancellation the operator hears themselves
    // a beat later and the robot's mic feeds back into the loop.
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
    const blob = new Blob([WORKLET_SRC], { type: 'application/javascript' });
    await this._ctx.audioWorklet.addModule(URL.createObjectURL(blob));
    const src = this._ctx.createMediaStreamSource(this._stream);
    this._node = new AudioWorkletNode(this._ctx, 'mabel-capture');
    this._node.port.onmessage = (e) => this._onCapture(e.data);
    src.connect(this._node);
    // A worklet with no downstream is culled by some engines; a zero-gain sink
    // keeps the graph alive without putting the operator's own voice on their
    // speakers (which would be a feedback loop with the robot's bar).
    const mute = this._ctx.createGain();
    mute.gain.value = 0;
    this._node.connect(mute).connect(this._ctx.destination);
    this._pending = [];
    this.talking = true;
    bus.emit('audio:talking', true);
  }

  stopTalking() {
    this.talking = false;
    try { this._node?.disconnect(); } catch { /* already gone */ }
    this._stream?.getTracks().forEach((t) => t.stop());
    this._node = null; this._stream = null; this._pending = [];
    bus.emit('audio:talking', false);
  }

  _onCapture(block) {
    if (!this.talking || !this.fmt || this.ws?.readyState !== 1) return;
    const inRate = this._ctx.sampleRate;                 // 48000 on most machines
    const want = this.fmt.rate;                          // 16000 on the wire
    // Linear decimation. Not a windowed-sinc: voice band is well under 8 kHz and
    // the browser's own capture chain already low-passes, so the aliasing this
    // could cause lives above the band we send. Cheap and audibly fine.
    const ratio = inRate / want;
    const n = Math.floor(block.length / ratio);
    const pcm = new Int16Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, block[Math.floor(i * ratio)]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      sum += s * s;
    }
    this.micLevel = Math.sqrt(sum / Math.max(n, 1));
    bus.emit('audio:level', { self: this.micLevel });

    // Re-chunk to the wire's frame size. The worklet hands us 128-sample quanta;
    // sending each as its own record would be 400 tiny frames/s of 88% header.
    const per = Math.round(want * this.fmt.frameMs / 1000);
    this._pending.push(...pcm);
    while (this._pending.length >= per) {
      const frame = Int16Array.from(this._pending.splice(0, per));
      this.ws.send(encodeAudio(frame, this.seq, want, this.fmt.channels));
      this.seq = (this.seq + 1) >>> 0;
    }
  }
}

export const audioLink = new AudioLink();
