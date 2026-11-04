// Selectable camera tiles, frame-accurate. We drive everything from one global
// frame, so each tile is an <img> whose src = the stored JPEG blob for the
// current source frame (crisp + perfectly in lockstep with the 3D + plots; the
// browser caches blobs so playback is smooth on localhost).
//
// The VideoWall owns a DETACHED grid element (this.el). The floating "Cameras"
// panel hosts it (appends it into its body); closing the panel just detaches the
// grid, so the active-camera selection and tiles persist across open/close.
import { api } from './api.js';
import { camStatusAt } from './defects.js';

export class VideoWall {
  constructor(){
    this.el = document.createElement('div'); this.el.className = 'vw-grid';
    this.active = new Set();          // camera names shown
    this.tiles = new Map();           // name -> {tile, img}
    this.available = [];              // [{name,...}] for current episode
    this._empty();
  }

  _empty(){
    this.el.innerHTML = '<div class="empty">select cameras in the Topics panel →</div>';
  }

  setAvailable(cams){ this.available = cams; }

  toggle(name, on){
    if (on) this.active.add(name); else this.active.delete(name);
    this._rebuild();
  }
  isActive(name){ return this.active.has(name); }

  _rebuild(){
    this.el.innerHTML = '';
    this.tiles.clear();
    if (!this.active.size){ this._empty(); return; }
    for (const name of this.available.map(c => c.name)){
      if (!this.active.has(name)) continue;
      const tile = document.createElement('div'); tile.className = 'cam-tile';
      const img = document.createElement('img'); img.decoding = 'async';
      const tag = document.createElement('span'); tag.className = 'cam-tag'; tag.textContent = name;
      const x = document.createElement('button'); x.className = 'closec'; x.textContent = '✕';
      const miss = document.createElement('div'); miss.className = 'cam-missing'; miss.textContent = '⚠ camera footage missing';
      tile.append(img, tag, x, miss); this.el.appendChild(tile);
      x.onclick = () => { this.toggle(name, false); document.dispatchEvent(new CustomEvent('cam-untoggle', { detail: name })); };
      this.tiles.set(name, { tile, img, miss });
    }
  }

  // per-frame camera health colours the tile: yellow border = frozen/low-freq,
  // red border + "footage missing" = dropped/blacked-out frame. Fed via `.quality`.
  update(ds, ep, sourceFrame){
    if (!this.tiles.size) return;
    for (const [name, { tile, img, miss }] of this.tiles){
      const st = camStatusAt(this.quality, name, sourceFrame);   // 0 ok / 1 warn / 2 missing
      tile.dataset.status = st;
      if (miss) miss.style.display = st === 2 ? 'flex' : 'none';
      img.src = api.camFrameUrl(ds, ep, name, sourceFrame);
    }
  }
}
