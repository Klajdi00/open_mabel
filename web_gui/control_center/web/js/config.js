// Connection config: the Control Center backend serves this page from the
// robot host, so every plane (teleop 9090 / map 9092 / video 8080) defaults to
// the page's own hostname. ?host= overrides for a split/remote deployment and
// ?key= carries the optional access code (MABEL_TOKEN on the robot host).
// node-safe (unit tests import this module outside a browser)
const loc = typeof location !== 'undefined'
  ? location : { search: '', hostname: '', host: '', protocol: 'http:' };
const params = new URLSearchParams(loc.search);

let cfg = {
  robotHost: '', teleopPort: 9090, mapPort: 9092, videoPort: 8080,
  audioPort: 9610,
  tokenGated: false, tempWarn: 55, tempHot: 70, telemetrySource: '?',
  sim: false, ros: false, scene: null, manageSim: false,
};

export async function loadConfig(){
  try { cfg = { ...cfg, ...(await fetch('/api/config').then(r => r.json())) }; }
  catch { /* backend down: keep same-host defaults so viewing still works */ }
  return cfg;
}

export const conf = () => cfg;
export const host = () => params.get('host') || cfg.robotHost || loc.hostname;
export const token = () => params.get('key') || '';

const q = (extra = '') => {
  const parts = [extra, token() ? 'key=' + encodeURIComponent(token()) : '']
    .filter(Boolean);
  return parts.length ? '?' + parts.join('&') : '';
};
const wsProto = () => (loc.protocol === 'https:' ? 'wss' : 'ws');

// Per-tab session id: rides the teleop URL as ?sid= (the bridge exposes it in
// the /monitor roster) so the network panel can recognize THIS page's own
// connection and label it "Current window".
export const sessionId = (() => {
  try {
    let s = sessionStorage.getItem('mabel.sid');
    if (!s){
      s = Math.random().toString(36).slice(2, 8);
      sessionStorage.setItem('mabel.sid', s);
    }
    return s;
  } catch { return Math.random().toString(36).slice(2, 8); }
})();

// SAME-ORIGIN MODE — the page is served over https (e.g. behind `tailscale
// serve` or your own reverse proxy), where the browser blocks plain ws:// to
// the robot's other ports. Teleop, monitor and video then ride the studio
// backend's own same-origin proxies (/ws/teleop, /ws/monitor, /api/cam,
// /ws/cam). Force with ?proxy=1/0.
export const isProxied = () => {
  const p = params.get('proxy');
  if (p != null) return p === '1';
  return loc.protocol === 'https:' && !params.get('host');
};

const teleopQ = () => q(`client=web&sid=${sessionId}`);
export const teleopUrl = () => isProxied()
  ? `${wsProto()}://${loc.host}/ws/teleop${teleopQ()}`
  : `${wsProto()}://${host()}:${cfg.teleopPort}/teleop${teleopQ()}`;
export const monitorUrl = () => isProxied()
  ? `${wsProto()}://${loc.host}/ws/monitor${q()}`
  : `${wsProto()}://${host()}:${cfg.teleopPort}/monitor${q()}`;
// mapPort 0 = a backend-served same-origin map plane at /ws/map (the sim twin
// uses this; the real robot keeps the bridge's :9092 plane).
export const mapUrl = () => !cfg.mapPort
  ? `${wsProto()}://${loc.host}/ws/map`
  : `${wsProto()}://${host()}:${cfg.mapPort}/map${q()}`;
// Voice: the dedicated :9610 duplex plane, dialled direct. It has no
// same-origin proxy, so in same-origin mode we return null and the UI says why.
export const audioUrl = () => isProxied()
  ? null
  : `${wsProto()}://${host()}:${cfg.audioPort}/audio${q()}`;
export const camUrl = (name) => isProxied()
  ? `/api/cam/${name}/stream.mjpg`             // same-origin proxy
  : `http://${host()}:${cfg.videoPort}/camera/${name}/stream.mjpg`;
// same-origin proxy of the same stream — REQUIRED for WebGL textures (the 3D
// camera projections); direct :8080 images are cross-origin and would taint
export const camStreamProxied = (name) => `/api/cam/${name}/stream.mjpg`;

// H.264/WebCodecs socket — the FAST video path (see js/h264.js, js/camview.js).
// The camera server encodes each camera once with hardware NVENC and serves the
// Annex-B access units at /camera/<name>/h264 (video_server.py). Same route the
// website teleop page and the iOS app consume; MJPEG above is now only a fallback.
//   direct       -> the camera server (ws :8080)
//   same-origin  -> OUR /ws/cam proxy (mirrors the MJPEG proxy above)
export const camH264Url = (name) => isProxied()
  ? `${wsProto()}://${loc.host}/ws/cam/${name}/h264`
  : `ws://${host()}:${cfg.videoPort}/camera/${name}/h264`;
export const telemetryUrl = () => `${wsProto()}://${loc.host}/ws/telemetry`;

export const CAMERAS = ['main', 'front', 'wrist_left', 'wrist_right'];
