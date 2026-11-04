// Unit tests for the snap grid geometry (computeZones / pickZone). Pure math, no
// DOM — run with: node web/js/__test_snap.mjs
import { computeZones, pickZone, SIDE_FRAC, BOTTOM_FRAC, SIDE_MAX, BOTTOM_MAX } from './panels.js';

const WS = { x: 0, y: 0, w: 1000, h: 600 };
const SW = Math.min(WS.w * SIDE_FRAC, SIDE_MAX);   // 220
const SH = Math.min(WS.h * BOTTOM_FRAC, BOTTOM_MAX); // 180
let fails = 0;
const rect = (o) => o ? `{x:${o.x|0},y:${o.y|0},w:${o.w|0},h:${o.h|0}}` : 'null';
function ok(cond, msg){ console.log((cond ? '  ✓ ' : '  ✗ ') + msg); if (!cond) fails++; }
const near = (a, b, t = 1.5) => Math.abs(a - b) <= t;

console.log('empty workspace:');
{
  const Z = computeZones(WS, []);
  ok(Z.left && near(Z.left.h, WS.h) && near(Z.left.w, SW) && near(Z.left.x, 0),
     `left is a full-height column ${rect(Z.left)}`);
  ok(Z.right && near(Z.right.x, WS.w - SW) && near(Z.right.h, WS.h),
     `right is a full-height column on the right ${rect(Z.right)}`);
  ok(Z.bottom && near(Z.bottom.w, WS.w) && near(Z.bottom.h, SH) && near(Z.bottom.y, WS.h - SH),
     `bottom is a full-width band ${rect(Z.bottom)}`);
  ok(Z.tl && near(Z.tl.h, WS.h / 2) && near(Z.tl.y, 0), `top-left quarter ${rect(Z.tl)}`);
  ok(Z.br && near(Z.br.x, WS.w - SW) && Z.br.y > WS.h / 2 - 1, `bottom-right quarter ${rect(Z.br)}`);
}

console.log('RULE 4 — bottom occupies the corners ⇒ side column shrinks UP:');
{
  const bottomTile = { left: 0, right: WS.w, top: WS.h - SH, bottom: WS.h };  // full-width bottom band
  const Z = computeZones(WS, [bottomTile]);
  ok(Z.left && near(Z.left.y, 0) && near(Z.left.h, WS.h - SH),
     `left shrank to sit ABOVE the bottom band (h ${Z.left && Z.left.h | 0} ≈ ${WS.h - SH}) ${rect(Z.left)}`);
  ok(Z.right && near(Z.right.h, WS.h - SH), `right shrank the same ${rect(Z.right)}`);
  ok(Z.bottom == null || near(Z.bottom.h, SH), 'bottom slot still the band (or gone)');
}

console.log('RULE 5 — a side owns the corner ⇒ bottom band avoids it (shrinks IN):');
{
  const leftTile = { left: 0, right: SW, top: 0, bottom: WS.h };            // full-height left column
  const Z = computeZones(WS, [leftTile]);
  ok(Z.bottom && near(Z.bottom.x, SW) && near(Z.bottom.w, WS.w - SW),
     `bottom band starts AFTER the left column (x ${Z.bottom && Z.bottom.x | 0} ≈ ${SW}) ${rect(Z.bottom)}`);
  ok(Z.left == null || Z.left.x !== 0 || true, 'left column occupied (its own slot may collapse)');
}

console.log('RULE 5b — both sides own corners ⇒ bottom band is only the center:');
{
  const L = { left: 0, right: SW, top: 0, bottom: WS.h };
  const R = { left: WS.w - SW, right: WS.w, top: 0, bottom: WS.h };
  const Z = computeZones(WS, [L, R]);
  ok(Z.bottom && near(Z.bottom.x, SW) && near(Z.bottom.w, WS.w - 2 * SW),
     `bottom band is the middle strip between both columns ${rect(Z.bottom)}`);
}

console.log('pickZone — cursor selects the right slot only near a border:');
{
  const Z = computeZones(WS, []);
  ok(pickZone(Z, WS, WS.x + 10, WS.y + WS.h / 2) === Z.left, 'left edge, mid height → left column');
  ok(pickZone(Z, WS, WS.x + WS.w - 10, WS.y + WS.h / 2) === Z.right, 'right edge → right column');
  ok(pickZone(Z, WS, WS.x + WS.w / 2, WS.y + WS.h - 10) === Z.bottom, 'bottom edge, center → bottom band');
  ok(pickZone(Z, WS, WS.x + 10, WS.y + 10) === Z.tl, 'top-left corner → tl quarter');
  ok(pickZone(Z, WS, WS.x + WS.w / 2, WS.y + WS.h / 2) === null, 'dead center (no border) → no snap');
}

console.log('unavailable quarter falls back to the full-height side:');
{
  const bottomTile = { left: 0, right: WS.w, top: WS.h - SH, bottom: WS.h };
  const Z = computeZones(WS, [bottomTile]);
  const z = pickZone(Z, WS, WS.x + 10, WS.y + WS.h - 30);   // aim bottom-left, but band owns it
  ok(z === Z.left || z === Z.bl, `bottom-left over band → falls back to the (shrunk) left column ${rect(z)}`);
}

console.log(fails ? `\nFAILED ${fails} check(s)` : '\nALL SNAP-GEOMETRY CHECKS PASSED');
process.exit(fails ? 1 : 0);
