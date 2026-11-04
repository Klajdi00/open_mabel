// Persistent per-episode CURATION status (localStorage). Two states, so the
// operator can see at a glance what's done and where they left off:
//   'saved'  — a curated version of this episode was properly saved  → green ✓
//   'opened' — inspected/edited in the timeline but NOT saved yet     → amber ◐
// 'saved' is sticky: re-opening a saved episode never downgrades it.
const KEY = 'mabel.dc.status.v1';
const listeners = new Set();
let _map = load();

function load(){ try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
function persist(){ try { localStorage.setItem(KEY, JSON.stringify(_map)); } catch {} emit(); }
function emit(){ for (const f of listeners) { try { f(); } catch (e){ console.error('status listener', e); } } }
const k = (ds, ep) => ds + '::' + ep;

export function get(ds, ep){ return _map[k(ds, ep)] || null; }

export function markOpened(ds, ep){
  const key = k(ds, ep);
  if (_map[key] === 'saved' || _map[key] === 'opened') return;   // don't downgrade / no-op
  _map[key] = 'opened'; persist();
}
export function markSaved(ds, ep){
  const key = k(ds, ep);
  if (_map[key] === 'saved') return;
  _map[key] = 'saved'; persist();
}
export function onChange(fn){ listeners.add(fn); return () => listeners.delete(fn); }

// Dataset-level rollup over a catalog episode list ([{file},…]).
export function datasetProgress(dsName, episodes){
  let saved = 0, opened = 0;
  for (const ep of (episodes || [])){
    const s = _map[k(dsName, ep.file)];
    if (s === 'saved') saved++; else if (s === 'opened') opened++;
  }
  return { total: (episodes || []).length, saved, opened };
}
