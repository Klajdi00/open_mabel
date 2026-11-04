// Draggable panel dividers that drive the grid CSS variables. Sizes persist to
// localStorage so the layout survives reloads.
const ROOT = document.documentElement.style;
const LS = 'mabel.studio.layout';

const VARS = {
  browser: { v: '--browser-w', axis: 'x', min: 180, max: 460, dir: 1, unit: 'px' },
  dock: { v: '--dock-w', axis: 'x', min: 200, max: 520, dir: -1, unit: 'px' },
  // joint-state graph height: measured up from the timeline splitter so it stays
  // independent of the timeline height
  plots: { v: '--plots-h', axis: 'y', min: 70, max: 420, dir: 1, unit: 'px', anchor: 'tls' },
  timeline: { v: '--tl-h', axis: 'y', min: 120, max: 520, dir: -1, unit: 'px' },
};

function load(){
  try { const s = JSON.parse(localStorage.getItem(LS) || '{}');
    for (const k in s) ROOT.setProperty(VARS[k]?.v || k, s[k]); } catch {}
}
function persist(){
  const s = {};
  for (const k in VARS) s[k] = getComputedStyle(document.documentElement).getPropertyValue(VARS[k].v).trim();
  localStorage.setItem(LS, JSON.stringify(s));
}

export function initSplitters(){
  load();
  document.querySelectorAll('[data-split]').forEach((el) => {
    const cfg = VARS[el.dataset.split]; if (!cfg) return;
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault(); el.setPointerCapture(e.pointerId);
      const move = (ev) => {
        let val;
        if (cfg.anchor === 'tls'){          // height measured up from the timeline splitter
          const tls = document.querySelector('[data-split="timeline"]');
          const bottom = tls ? tls.getBoundingClientRect().top : window.innerHeight;
          val = bottom - ev.clientY;
        } else if (cfg.unit === 'px'){
          if (cfg.axis === 'x'){
            val = cfg.dir === 1 ? ev.clientX
              : window.innerWidth - ev.clientX;
          } else {
            val = window.innerHeight - ev.clientY;
          }
        } else { // % of the center column height for the stage split
          const center = document.querySelector('.center').getBoundingClientRect();
          val = ((ev.clientY - center.top) / center.height) * 100;
        }
        val = Math.max(cfg.min, Math.min(cfg.max, val));
        ROOT.setProperty(cfg.v, val + cfg.unit);
        window.dispatchEvent(new Event('studio-resize'));
      };
      const up = (ev) => { el.releasePointerCapture(ev.pointerId);
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up); persist(); };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
  });
}
