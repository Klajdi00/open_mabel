// Left library: datasets -> episode cards (thumbnail + task + frame count).
// Click (or the +) appends the whole episode to the timeline as a clip.
// Each card carries a CURATION mark (see status.js): green ✓ = saved, amber ◐ =
// opened-but-not-saved (where you left off); the dataset header shows saved/total.
import { api } from './api.js';
import { get as statusGet, datasetProgress, onChange as onStatusChange } from './status.js';

export class Browser {
  constructor(el, onAdd, onLoadBatch){
    this.el = el; this.onAdd = onAdd; this.onLoadBatch = onLoadBatch;
    // live-update the marks (no refetch) whenever curation status changes
    onStatusChange(() => this._applyMarks());
  }

  async load(){
    let cat;
    try { cat = await api.catalog(); }
    catch (e){ this.el.innerHTML = `<div class="empty">server offline</div>`; return; }
    this.el.innerHTML = '';
    if (!cat.datasets.length){
      this.el.innerHTML = `<div class="empty" style="padding:20px;color:var(--ash-dim);font-family:var(--mono);font-size:11px">
        No datasets found.<br><br>Click <b>Generate Data</b> ↗ or drop episode folders into<br>
        <code>data_collection/datasets/</code><br><br>
        <button id="reloadLib" class="gbtn" style="margin-top:6px">↻ Reload</button></div>`;
      this.el.querySelector('#reloadLib').onclick = () => this.load();
      return;
    }
    for (const ds of cat.datasets){
      const wrap = document.createElement('div'); wrap.className = 'lib-ds';
      const h = document.createElement('div'); h.className = 'lib-ds-h';
      const prog = datasetProgress(ds.name, ds.episodes);
      const eps = ds.episodes.filter(e => !e.error);
      const todo = eps.filter(e => statusGet(ds.name, e.file) !== 'saved');
      h.innerHTML = `<span>${ds.name}</span>
        <span class="lib-ds-prog${prog.saved === prog.total ? ' done' : ''}" data-ds="${ds.name}"
          title="curated (saved) episodes">${prog.saved}/${prog.total}</span>`;
      wrap.appendChild(h);
      // batch-load row: load ALL episodes of this dataset into the timeline as
      // independent clips (no re-importing one at a time), or just the ones not
      // yet saved (resume where you left off).
      const load = document.createElement('div'); load.className = 'lib-ds-load';
      load.innerHTML = `<button class="lib-load all" title="load every episode into the timeline">⤓ Load all ${eps.length}</button>`
        + (todo.length && todo.length < eps.length
            ? `<button class="lib-load todo" title="load only episodes not yet saved">⤓ Unedited ${todo.length}</button>` : '');
      load.querySelector('.all').onclick = (e) => { e.stopPropagation(); this.onLoadBatch?.(ds.name, eps); };
      const tb = load.querySelector('.todo'); if (tb) tb.onclick = (e) => { e.stopPropagation(); this.onLoadBatch?.(ds.name, todo); };
      wrap.appendChild(load);
      for (const ep of ds.episodes){
        if (ep.error) continue;
        const card = document.createElement('div'); card.className = 'ep-card';
        card.dataset.ds = ds.name; card.dataset.ep = ep.file;
        const cam = (ep.cameras && ep.cameras[0]) || null;
        const thumb = cam
          ? `<img class="thumb" src="${api.camFrameUrl(ds.name, ep.file, cam, 0)}" loading="lazy">`
          : `<div class="thumb"></div>`;
        card.innerHTML = `${thumb}
          <div class="meta"><div class="task">${ep.task || ep.file}</div>
          <div class="nums">${ep.num_frames}f · ${ep.duration_s}s · ${ep.fps}Hz</div></div>
          <div class="ep-mark" title=""></div>
          <div class="add" title="add to timeline">+</div>`;
        const add = () => this.onAdd({ ds: ds.name, ep: ep.file, epLabel: ep.task || ep.file,
          num_frames: ep.num_frames, fps: ep.fps, cameras: ep.cameras });
        card.querySelector('.add').onclick = (e) => { e.stopPropagation(); add(); };
        card.onclick = add;
        card.draggable = true;
        card.ondragstart = (e) => e.dataTransfer.setData('text/plain', JSON.stringify({ ds: ds.name, ep: ep.file }));
        wrap.appendChild(card);
      }
      this.el.appendChild(wrap);
    }
    this._applyMarks();
  }

  // Paint each card's curation mark from status.js (called on render + on change,
  // in place so we never refetch the catalog just to flip a checkmark).
  _applyMarks(){
    for (const card of this.el.querySelectorAll('.ep-card')){
      const st = statusGet(card.dataset.ds, card.dataset.ep);
      const mark = card.querySelector('.ep-mark');
      card.classList.toggle('is-saved', st === 'saved');
      card.classList.toggle('is-opened', st === 'opened');
      if (mark){
        mark.className = 'ep-mark' + (st ? ' ' + st : '');
        mark.textContent = st === 'saved' ? '✓' : st === 'opened' ? '◐' : '';
        mark.title = st === 'saved' ? 'saved (curated)'
                   : st === 'opened' ? 'opened — not saved yet (where you left off)' : '';
      }
    }
    // refresh each dataset header's saved/total rollup
    for (const badge of this.el.querySelectorAll('.lib-ds-prog')){
      const dsName = badge.dataset.ds;
      const eps = [...this.el.querySelectorAll(`.ep-card[data-ds="${CSS.escape(dsName)}"]`)]
        .map(c => ({ file: c.dataset.ep }));
      const saved = eps.filter(e => statusGet(dsName, e.file) === 'saved').length;
      badge.textContent = `${saved}/${eps.length}`;
      badge.classList.toggle('done', saved === eps.length && eps.length > 0);
    }
  }
}
