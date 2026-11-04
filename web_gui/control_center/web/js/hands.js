// hands.js — which ORCA hands are MOUNTED, shown on the 3D view and followed live.
//
// MABEL's ORCA v1 and NOVA's ORCA v2 hands swap between the two robots (NOVA servos/orca_hands.py). The
// server says which is mounted per side (/api/hands — the HAL's own rule: the hand whose adapter is
// plugged in) and which generation is the robot's own (`native`), and serves the rig with those hands
// under the usual /website-assets/mabel_rig.glb. This module (identical in both robots' Control Centers):
//   · puts a chip on the viewer — "HANDS  L ORCA v1 · R ORCA v2" — a SWAPPED-IN hand in amber, the
//     robot's own in grey, so a borrowed hand is never mistaken for the robot's own;
//   · when a side's generation CHANGES (a hand swapped, seen on two polls in a row) the drawn rig is stale:
//     reload to redraw — at once while the robot is limp, behind a button while it is energised.
import * as bus from './bus.js';

const POLL_MS = 3000;
let shown = null, pending = null, energised = false, native = 'v2';

const chip = document.createElement('div');
chip.className = 'vp-hands mono';
chip.id = 'vpHands';
chip.title = 'ORCA hands mounted, per side (amber = swapped in from the other robot, grey = this robot\u2019s own). The 3D view draws these.';

function render(h){
  const side = (s, lab) => {
    const g = h[s]?.generation || '?';
    const here = h[s]?.present !== false;
    const swapped = g !== native;
    return `<span class="vh-side ${swapped ? 'swapped' : 'own'}${here ? '' : ' absent'}" title="${swapped ? 'swapped in from the other robot' : 'this robot\u2019s own hand'}${here ? '' : ' · adapter not plugged in — last seen'}">`
         + `${lab}&nbsp;ORCA&nbsp;${g}</span>`;
  };
  chip.innerHTML = `<span class="vh-lab">HANDS</span>${side('left', 'L')}<span class="vh-sep">·</span>${side('right', 'R')}`;
}

const key = (h) => `${h.left?.generation}-${h.right?.generation}`;

async function poll(){
  let r;
  try { r = await fetch('/api/hands', { cache: 'no-store' }).then((x) => x.ok ? x.json() : null); }
  catch { r = null; }
  if (!r?.hands) return;
  native = r.native || native;
  const k = key(r.hands);
  if (shown === null){ shown = k; render(r.hands); return; }
  render(r.hands);
  if (k === shown){ pending = null; return; }
  if (pending !== k){ pending = k; return; }          // must hold for two polls: never redraw on a blip
  shown = k; pending = null;
  if (!energised){
    bus.emit('toast', { msg: `hands changed (L ${r.hands.left.generation} · R ${r.hands.right.generation}) — redrawing the robot` });
    setTimeout(() => location.reload(), 900);
  } else {
    chip.classList.add('stale');
    chip.insertAdjacentHTML('beforeend', '<button class="vh-reload" title="Reload to draw the hands now mounted">redraw</button>');
    chip.querySelector('.vh-reload').onclick = () => location.reload();
    bus.emit('toast', { msg: 'hands changed — press "redraw" on the HANDS chip once the robot is safe', err: true });
  }
}

export function initHands(host){
  host.appendChild(chip);
  bus.on('robotState', (s) => {
    const t = Object.values(s?.torque || {}).filter((v) => v != null);
    energised = !s?.estop && t.length > 0 && t.some(Boolean);
  });
  poll();
  setInterval(poll, POLL_MS);
}
