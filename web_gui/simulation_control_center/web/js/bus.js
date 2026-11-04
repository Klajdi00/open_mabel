// Tiny pub/sub. Panels subscribe to 'frame', 'sequenceChanged', 'topics',
// 'episodeLoaded', 'selection'.
const map = new Map();
export function on(ev, fn){ (map.get(ev) || map.set(ev, new Set()).get(ev)).add(fn); return () => off(ev, fn); }
export function off(ev, fn){ map.get(ev)?.delete(fn); }
export function emit(ev, ...a){ map.get(ev)?.forEach(fn => { try { fn(...a); } catch(e){ console.error(e); } }); }
