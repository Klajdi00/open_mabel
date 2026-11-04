// Permanent camera strip above the joint-state band: ALWAYS shows the four
// robot views — head (main), front base, wrist left, wrist right — as live
// tiles (H.264/WebCodecs, MJPEG fallback — see camview.js). Each tile has an
// RGB/D chip that switches to the matching depth stream when the server exposes
// one (auto-probed; disabled otherwise).
import * as bus from './bus.js';
import { CamView } from './camview.js';

// head uses the HONEST stream (head_left): the 'main' slot silently falls
// back to the front camera when the ZED is absent, which would mislabel the
// RealSense as the head cam
const VIEWS = [
  { name: 'head_left', label: 'head' },
  { name: 'front', label: 'front base' },
  { name: 'wrist_left', label: 'wrist L' },
  { name: 'wrist_right', label: 'wrist R' },
];

export class CamStrip {
  constructor(host){
    this.host = host;
    host.classList.add('cam-strip');
    this._tiles = {};
    this._views = {};
    for (const v of VIEWS) this._tile(v);
    // hardware-truth overlay: mark tiles whose camera is actually absent
    const refresh = async () => {
      try {
        const live = await fetch('/api/cameras/live').then(r => r.json());
        this.live = live;
        for (const { name } of VIEWS)
          this._tiles[name]?.classList.toggle('dead', live[name] === false);
        bus.emit('camerasLive', live);
      } catch { /* backend offline */ }
    };
    refresh();
    this._liveTimer = setInterval(refresh, 20000);
  }

  _tile({ name, label }){
    const tile = document.createElement('div');
    tile.className = 'cam-tile strip';
    tile.innerHTML = `<img alt="${label}">
      <span class="cam-tag">${label}</span>
      <button class="cam-dchip" title="probing depth stream…" disabled>D</button>
      <span class="cam-missing">stream offline</span>`;
    const img = tile.querySelector('img');
    const dchip = tile.querySelector('.cam-dchip');
    const state = { depth: false, hasDepth: false };

    // CamView owns the transport (H.264 -> canvas, MJPEG <img> fallback) and the
    // reconnect: a scene switch restarts the camera server and neither an MJPEG
    // <img> nor a dropped WS recovers on its own.
    const view = new CamView(img, name, {
      onLive: (live) => tile.classList.toggle('dead', !live),
    });
    bus.on('streamsRefresh', () => view.connect());
    this._views[name] = view;

    // Depth availability probe. MUST be the one-shot /frame.jpg, never the MJPEG
    // stream: pointing an <img> at stream.mjpg opens a full multipart stream that
    // we then cannot reliably close (removeAttribute('src') does not abort an
    // MJPEG <img> in WebKit), so the old probe could leave FOUR hidden depth
    // streams running for the life of the page, competing for bandwidth with the
    // video the operator is actually watching. /frame.jpg returns one JPEG (or
    // 404) and closes.
    fetch(`/api/cam/${name}_depth/frame.jpg`, { method: 'GET' })
      .then((r) => {
        if (!r.ok) throw new Error('no depth');
        state.hasDepth = true;
        dchip.disabled = false;
        dchip.title = 'toggle RGB / depth';
      })
      .catch(() => { dchip.title = 'no depth stream'; });

    dchip.onclick = () => {
      if (!state.hasDepth) return;
      state.depth = !state.depth;
      dchip.classList.toggle('on', state.depth);
      // depth has no H.264 encoder — CamView falls back to MJPEG for it on its own
      view.setName(state.depth ? `${name}_depth` : name);
    };
    this.host.appendChild(tile);
    this._tiles[name] = tile;
  }
}
