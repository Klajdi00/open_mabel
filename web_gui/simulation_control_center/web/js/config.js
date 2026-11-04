// Connection config: the Control Center backend serves this page from the
// robot host, so every plane (teleop 9090 / map 9092 / video 8080) defaults to
// the page's own hostname. ?host= overrides for a split/remote deployment and
// ?key= carries the optional access code (MABEL_TOKEN on the robot host).
// node-safe (unit tests import this module outside a browser)
const loc = typeof location !== 'undefined'
  ? location : { search: '', hostname: '', host: '', protocol: 'http:' };
const params = new URLSearchParams(loc.search);

let cfg = {
  robotHost: '', teleopPort: 9090, mapPort: 0, videoPort: 8080,
  tokenGated: false, tempWarn: 55, tempHot: 70, telemetrySource: '?',
  sim: true, ros: false, scene: 'home', manageSim: false,
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
// the sim's other ports. Teleop, monitor and video then ride the studio
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
// mapPort 0 = the backend's own MuJoCo-mirror map plane at /ws/map — always
// same-origin.
export const mapUrl = () => !cfg.mapPort
  ? `${wsProto()}://${loc.host}/ws/map`
  : `${wsProto()}://${host()}:${cfg.mapPort}/map${q()}`;
export const camUrl = (name) => isProxied()
  ? `/api/cam/${name}/stream.mjpg`             // same-origin proxy
  : `http://${host()}:${cfg.videoPort}/camera/${name}/stream.mjpg`;
// same-origin proxy of the same stream — REQUIRED for WebGL textures (the 3D
// camera projections); direct :8080 images are cross-origin and would taint
export const camStreamProxied = (name) => `/api/cam/${name}/stream.mjpg`;

// H.264/WebCodecs fast path (js/camview.js prefers this and keeps MJPEG only as
// a fallback); same-origin mode tunnels through our own /ws/cam proxy.
// If the sim camera server does not serve this route the socket simply carries
// no bytes, and CamView falls back to MJPEG after its grace window — so this is
// safe to export even where H.264 is unavailable.
export const camH264Url = (name) => isProxied()
  ? `${wsProto()}://${loc.host}/ws/cam/${name}/h264`
  : `${wsProto()}://${host()}:${cfg.videoPort}/camera/${name}/h264`;

// Cancel an MJPEG <img> stream and blank the element. Pointing it at a 1×1
// transparent GIF does both: the browser aborts the in-flight multipart
// download and paints the pixel over the last frame. The two obvious
// alternatives are traps — `img.src = ''` re-resolves to the document URL (the
// browser refetches the whole page), and removeAttribute('src') leaves WebKit's
// multipart body streaming, so every camera swap / scene switch would leak a
// live pump. naturalWidth stays 1, which the FPV watchdog already reads as
// "nothing painted".
const BLANK_PIXEL =
  'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
export function stopStream(img){
  if (!img) return;
  img.src = BLANK_PIXEL;
}
export const telemetryUrl = () => `${wsProto()}://${loc.host}/ws/telemetry`;

// SAME roster as the real robot (sim_cameras.py serves the identical paths,
// including the front base cam) — sim/real UI parity by construction.
export const CAMERAS = ['main', 'front', 'wrist_left', 'wrist_right'];
