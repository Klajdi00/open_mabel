// Per-episode cache: meta + the dense numeric bundle decoded into Float32 views,
// plus the actuator->rig map. Arrays are tiny so we fetch the whole episode once;
// scrubbing then never touches the network. LRU-evicts to a few episodes.
import { api } from './api.js';
import { buildActMap } from './names.js';

const LRU = 4;
const cache = new Map();              // key "ds/ep" -> entry
const inflight = new Map();

const key = (ds, ep) => `${ds}::${ep}`;

function decodeBundle(buf){
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + hlen)));
  const dataStart = 4 + hlen;
  let base = buf.buffer, off = buf.byteOffset + dataStart;
  if (off % 4 !== 0){                       // realign by copying (server pads, but be safe)
    const copy = buf.slice(dataStart); base = copy.buffer; off = copy.byteOffset;
  }
  const f32 = new Float32Array(base, off, (buf.byteLength - dataStart) >> 2);
  const topics = {};
  for (const t of header.topics){
    topics[t.name] = { dim: t.dim, data: f32.subarray(t.offset, t.offset + t.count) };
  }
  return { T: header.T, fps: header.fps, topics };
}

export async function ensure(ds, ep){
  const k = key(ds, ep);
  if (cache.has(k)){ const e = cache.get(k); cache.delete(k); cache.set(k, e); return e; }
  if (inflight.has(k)) return inflight.get(k);
  const p = (async () => {
    const [meta, arraysBuf, cams] = await Promise.all([
      api.meta(ds, ep), api.arrays(ds, ep), api.cameras(ds, ep),
    ]);
    const bundle = decodeBundle(arraysBuf);
    const entry = {
      ds, ep, meta, cams, bundle,
      actMap: buildActMap(meta.actuator_names || []),
      T: bundle.T || meta.num_frames, fps: bundle.fps || meta.fps || 15,
    };
    cache.set(k, entry);
    while (cache.size > LRU){ const fk = cache.keys().next().value; cache.delete(fk); }
    inflight.delete(k);
    return entry;
  })();
  inflight.set(k, p);
  return p;
}

export function peek(ds, ep){ return cache.get(key(ds, ep)); }

// Frame accessor: returns a subarray view of `topic` at frame f, or null.
export function frameVec(entry, topic, f){
  const t = entry.bundle.topics[topic];
  if (!t) return null;
  const T = entry.T;
  const i = Math.max(0, Math.min(f, T - 1));
  return t.data.subarray(i * t.dim, i * t.dim + t.dim);
}

export function topicDim(entry, topic){ return entry.bundle.topics[topic]?.dim || 0; }
export function hasTopic(entry, topic){ return !!entry.bundle.topics[topic]; }
