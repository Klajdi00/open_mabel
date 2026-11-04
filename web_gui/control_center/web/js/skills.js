// Skills: composed choreographies saved like video clips + the robot's full
// ability roster.
//   SkillStore  — pure CRUD over injected storage (node-testable): a skill is
//                 a serialized timeline {choreo, keys, annos} + metadata.
//   SkillsPanel — right-dock "Skills" tab: iOS-app-style roster of everything
//                 the robot can do (registry actions grouped by family +
//                 composed skills); click = insert at the playhead.
//   LibraryPanel— left-rail "Library" view: composed skills with insert /
//                 rename / export / delete, JSON import, compose-from-timeline.
import * as bus from './bus.js';
import { clipDur, defaults } from './motions.js';

const LS_SKILLS = 'mabel.cc.skills.v1';

export class SkillStore {
  constructor(storage){
    this.storage = storage || (typeof localStorage !== 'undefined' ? localStorage : null);
    this.skills = this._load();
    this.onChange = null;
  }
  _load(){
    try { return JSON.parse(this.storage?.getItem(LS_SKILLS) || '[]') || []; }
    catch { return []; }
  }
  _save(){
    this.storage?.setItem(LS_SKILLS, JSON.stringify(this.skills));
    this.onChange && this.onChange();
  }
  compose(title, timeline){
    const dur = timeline.duration();
    const s = {
      id: 'sk' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
      title: String(title || 'untitled skill'),
      emoji: '🎬',
      dur: +dur.toFixed(2),
      createdAt: new Date().toISOString(),
      blocks: timeline.track.blocks.length,
      keyframes: timeline.seq.keys.length,
      choreo: timeline.track.toJSON(),
      keys: timeline.seq.toJSON(),
      annos: timeline.annos.toJSON(),
    };
    this.skills.unshift(s);
    this._save();
    return s;
  }
  rename(id, title){
    const s = this.byId(id);
    if (s){ s.title = String(title); this._save(); }
  }
  remove(id){
    const i = this.skills.findIndex((s) => s.id === id);
    if (i >= 0){ this.skills.splice(i, 1); this._save(); }
  }
  byId(id){ return this.skills.find((s) => s.id === id) || null; }
  exportJSON(id){ return JSON.stringify(this.byId(id), null, 1); }
  importJSON(text){
    const j = JSON.parse(text);
    const list = Array.isArray(j) ? j : [j];
    const added = [];
    for (const s of list){
      if (!s || (!s.choreo && !s.keys)) continue;        // not a skill file
      const copy = { ...s, id: 'sk' + Date.now().toString(36) +
        Math.random().toString(36).slice(2, 5) };
      this.skills.unshift(copy);
      added.push(copy);
    }
    if (added.length) this._save();
    return added;
  }
  // insert a skill's contents into the live timeline at time t (flattened,
  // like pasting a nested sequence in an NLE)
  insertInto(id, timeline, t){
    const s = this.byId(id);
    if (!s) return 0;
    let n = 0;
    n += timeline.track.paste(s.choreo, t).length;
    for (const k of s.keys?.keys || []){
      timeline.seq.add((+k.t || 0) + t, { ...k.pose }, k.label || '');
      n++;
    }
    for (const a of s.annos?.notes || []){
      timeline.annos.add((+a.t0 || 0) + t, +a.dur || 1, a.text);
      n++;
    }
    return n;
  }
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ── right dock: everything the robot can do (iOS skills view) ───────────────
export class SkillsPanel {
  constructor(el, store, timeline){
    this.el = el;
    this.store = store;
    this.timeline = timeline;
    store.onChange = () => { this.render(); bus.emit('skillsChanged'); };
    this.render();
  }
  render(){
    const reg = this.timeline.registry;
    const fam = [
      ['MY SKILLS · composed', null],
      ['TASK POLICIES', 'policy'],
      ['ANIMATIONS · DANCES', 'animation'],
      ['NAVIGATION', 'nav'],
      ['POSES', 'pose'],
    ];
    let h = '<div class="sk-list">';
    for (const [label, cat] of fam){
      h += `<div class="np-sec">${label}</div>`;
      if (!cat){
        if (!this.store.skills.length)
          h += `<div class="np-empty">compose one from the timeline (⊕ Skill)</div>`;
        for (const s of this.store.skills){
          h += `<div class="sk-card mine" data-skill="${s.id}" title="insert at the playhead">
            <span class="sk-emoji">${esc(s.emoji)}</span>
            <span class="sk-name">${esc(s.title)}</span>
            <span class="sk-meta mono">${s.dur.toFixed(1)}s · ${s.blocks} blk</span>
          </div>`;
        }
        continue;
      }
      for (const [key, c] of Object.entries(reg)){
        if (c.category !== cat) continue;
        const soon = c.status !== 'available';
        const d = soon ? null : clipDur(c, defaults(c));
        h += `<div class="sk-card ${soon ? 'soon' : ''}" data-action="${soon ? '' : key}"
            title="${soon ? esc(c.note || 'coming soon') : 'insert at the playhead'}">
          <span class="sk-emoji">${esc(c.emoji || '🤖')}</span>
          <span class="sk-name">${esc(c.title)}</span>
          ${soon ? '<span class="am-soon">coming soon</span>'
                 : `<span class="sk-meta mono">${d.toFixed(1)}s</span>`}
        </div>`;
      }
    }
    h += '</div>';
    this.el.innerHTML = h;
    this.el.querySelectorAll('[data-action]').forEach((c) => {
      const key = c.dataset.action;
      if (key) c.onclick = () => this.timeline._addAction(key);
    });
    this.el.querySelectorAll('[data-skill]').forEach((c) => {
      c.onclick = () => {
        const n = this.store.insertInto(c.dataset.skill, this.timeline, this.timeline.t);
        bus.emit('toast', { msg: `skill inserted (${n} items)` });
      };
    });
  }
}

// ── left rail: the skill library (compose / import / export / insert) ───────
export class LibraryPanel {
  constructor(el, store, timeline){
    this.el = el;
    this.store = store;
    this.timeline = timeline;
    bus.on('skillsChanged', () => this.render());
    this.render();
  }
  render(){
    let h = `<div class="lib-actions">
      <button class="gbtn" id="libCompose" title="Save the whole timeline as a reusable skill">⊕ Compose from timeline</button>
      <label class="gbtn lib-import" title="Import skill JSON files">📥 Import
        <input type="file" id="libImport" accept=".json,application/json" multiple hidden>
      </label>
    </div>
    <div class="np-sec">Composed Skills · ${this.store.skills.length}</div>`;
    if (!this.store.skills.length)
      h += `<div class="np-empty">nothing yet — arrange the timeline, then ⊕ Compose</div>`;
    for (const s of this.store.skills){
      h += `<div class="sk-card mine lib" data-id="${s.id}">
        <span class="sk-emoji">${esc(s.emoji)}</span>
        <div class="sk-mid">
          <span class="sk-name">${esc(s.title)}</span>
          <span class="sk-meta mono">${s.dur.toFixed(1)}s · ${s.blocks} blocks · ${s.keyframes} keys</span>
        </div>
        <span class="sk-btns">
          <button data-op="insert" title="Insert at the playhead">⤵</button>
          <button data-op="rename" title="Rename">✎</button>
          <button data-op="export" title="Export JSON">⇪</button>
          <button data-op="del" title="Delete">⌫</button>
        </span>
      </div>`;
    }
    this.el.innerHTML = h;
    this.el.querySelector('#libCompose').onclick = () => {
      if (!this.timeline.track.blocks.length && !this.timeline.seq.keys.length){
        bus.emit('toast', { msg: 'timeline is empty — nothing to compose', err: true });
        return;
      }
      const title = prompt('skill name', 'my skill');
      if (title) {
        this.store.compose(title, this.timeline);
        bus.emit('toast', { msg: `skill saved: ${title}` });
      }
    };
    this.el.querySelector('#libImport').onchange = async (e) => {
      let n = 0;
      for (const f of e.target.files) {
        try { n += this.store.importJSON(await f.text()).length; }
        catch { bus.emit('toast', { msg: `bad skill file: ${f.name}`, err: true }); }
      }
      if (n) bus.emit('toast', { msg: `imported ${n} skill(s)` });
      e.target.value = '';
    };
    this.el.querySelectorAll('.sk-card.lib').forEach((card) => {
      const id = card.dataset.id;
      card.querySelectorAll('button').forEach((b) => {
        b.onclick = (e) => {
          e.stopPropagation();
          const op = b.dataset.op;
          if (op === 'insert'){
            const n = this.store.insertInto(id, this.timeline, this.timeline.t);
            bus.emit('toast', { msg: `skill inserted (${n} items)` });
          } else if (op === 'rename'){
            const t = prompt('skill name', this.store.byId(id)?.title || '');
            if (t) this.store.rename(id, t);
          } else if (op === 'export'){
            const s = this.store.byId(id);
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([this.store.exportJSON(id)],
              { type: 'application/json' }));
            a.download = `${(s?.title || 'skill').replace(/\W+/g, '_')}.skill.json`;
            a.click();
            URL.revokeObjectURL(a.href);
          } else if (op === 'del'){
            if (confirm('delete this skill?')) this.store.remove(id);
          }
        };
      });
    });
  }
}
