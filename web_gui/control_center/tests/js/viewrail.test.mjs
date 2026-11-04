// The viewer rail (web/js/viewrail.js) is the ONLY way to reach a scene layer
// now that the top chip strip is gone. So the invariant that matters is
// reachability: every layer app.js defaults MUST have a rail entry, and every
// rail entry MUST name a real layer. A drift either way is a control the
// operator cannot see or a button that toggles nothing.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RAIL_GROUPS, RAIL_ITEMS, ICONS, railItemById, railLayerKeys,
         railMenuKeys, railFor, railFits, railCols, railWidth, RAIL_MAX_COLS,
         RAIL_COL_W } from '../../web/js/viewrail.js';

const APP = readFileSync(new URL('../../web/js/app.js', import.meta.url), 'utf8');
const CSS = readFileSync(new URL('../../web/css/control.css', import.meta.url), 'utf8');

/** app.js's LAYER_DEFAULTS, read from the source (the rail's counterpart). */
const layerDefaults = () => {
  const m = APP.match(/const LAYER_DEFAULTS = \{([\s\S]*?)\};/);
  assert.ok(m, 'app.js must still declare LAYER_DEFAULTS');
  const out = {};
  for (const [, k, v] of m[1].matchAll(/(\w+):\s*(true|false)/g)) out[k] = v === 'true';
  return out;
};

/** The MENU_ITEMS keys app.js can build a flyout for. */
const menuKeys = () => {
  const m = APP.match(/const MENU_ITEMS = \{([\s\S]*?)\n  \};/);
  assert.ok(m, 'app.js must still declare MENU_ITEMS');
  return [...m[1].matchAll(/^\s{4}(\w+):\s*\(\)/gm)].map((x) => x[1]);
};

test('every layer app.js defaults is reachable from the rail', () => {
  const defaults = Object.keys(layerDefaults());
  const rail = railLayerKeys();
  for (const k of defaults)
    assert.ok(rail.includes(k), `layer "${k}" has no rail entry — unreachable`);
});

test('every rail layer entry names a real layer', () => {
  const defaults = layerDefaults();
  for (const k of railLayerKeys())
    assert.ok(k in defaults, `rail toggles "${k}", which app.js never declares`);
});

test('every expandable entry names a menu app.js can build', () => {
  const known = menuKeys();
  for (const k of railMenuKeys())
    assert.ok(known.includes(k), `rail expands "${k}", which MENU_ITEMS lacks`);
});

test('every entry has an icon — a rail button with no glyph is invisible', () => {
  for (const it of RAIL_ITEMS)
    assert.ok(ICONS[it.glyph], `"${it.id}" wants glyph "${it.glyph}", which ICONS lacks`);
});

test('ids are unique (the rail keys its buttons by id)', () => {
  const ids = RAIL_ITEMS.map((i) => i.id);
  assert.equal(new Set(ids).size, ids.length, 'duplicate rail id');
});

test('an entry is a layer, a camera action, or a hold control — exactly one', () => {
  for (const it of RAIL_ITEMS){
    // `hold` is push-to-talk: it drives no scene layer and fires no verb, it
    // opens the mic only while pressed.
    assert.ok(it.layer || it.action || it.hold, `"${it.id}" does nothing`);
    assert.ok(!(it.layer && it.action), `"${it.id}" is both a layer and an action`);
    assert.ok(!(it.hold && (it.layer || it.action)),
      `"${it.id}" is a hold control AND something else`);
  }
});

test('camera verbs come first, scene layers after', () => {
  // The rail reads top-to-bottom as "where do I look" then "what is drawn".
  assert.ok(RAIL_GROUPS[0].every((i) => i.action), 'group 1 must be camera verbs');
  assert.ok(RAIL_GROUPS[1].every((i) => i.layer), 'group 2 must be scene layers');
});

test('only a toggling action can light up', () => {
  // Center and Fit are one-shot verbs: an `on` state on them would lie. The two
  // stateful actions are Follow (camera) and Listen (are we hearing the robot).
  const stateful = RAIL_ITEMS.filter((i) => i.action && i.toggle).map((i) => i.id);
  assert.deepEqual(stateful.sort(), ['follow', 'listen']);
});

test('railItemById finds entries and refuses unknowns', () => {
  assert.equal(railItemById('cloud').label, 'Point Cloud');
  assert.equal(railItemById('nope'), null);
});

// ── the collapsed rail (landscape phone) ──────────────────────────────────
// A 15-button column does not fit a ~180px-tall 3-D view: it wrapped into
// three columns and landed on the left joystick. Collapsed, it must still
// reach every layer — through Overlays — or the phone loses controls the desk
// has.
test('collapsed, the rail keeps the camera verbs, Overlays, and voice', () => {
  const flat = railFor(true).flat();
  assert.deepEqual(flat.map((i) => i.id),
    ['recenter', 'fit', 'follow', 'overlays', 'speak', 'listen']);
  // Overlays is still the door to every scene layer; it just is not last now.
  assert.ok(flat.some((i) => i.menu === 'overlays'));
});

test('collapsed, NO layer button survives — Overlays is the only door', () => {
  // A stray layer button here is what would reach the joystick again.
  assert.equal(railFor(true).flat().filter((i) => i.layer).length, 0);
});

test('expanded, the rail is the full list and has no Overlays', () => {
  const flat = railFor(false).flat();
  assert.equal(flat.filter((i) => i.id === 'overlays').length, 0);
  assert.deepEqual(flat, RAIL_ITEMS);
});

test('the Overlays glyph exists — the collapsed rail is otherwise blank', () => {
  assert.ok(ICONS[railFor(true).flat().at(-1).glyph]);
});

test('a desk gets ONE clean column of the full list', () => {
  // 1920x1080 leaves the view ~580px once the bands and timeline take their
  // share (the AUDIO group added two rows, so the single-column height rose a
  // little) — still a single column, which is the shape the operator asked for.
  for (const h of [580, 760]){
    assert.equal(railCols(h), 1, `${h}px should be one column`);
    assert.ok(railFits(h));
  }
  // And the old ~516px desk now WRAPS to a second column rather than collapsing
  // — still within the 2-column budget, the tested middle regime.
  assert.ok(railFits(516) && railCols(516) === 2, '516px wraps, does not collapse');
});

test('a shorter desk WRAPS rather than collapsing', () => {
  // 1600x1000 leaves ~436px. Too short for one column, but nowhere near phone
  // territory — it must keep every button, in two columns, not hide them
  // behind Overlays.
  assert.equal(railCols(436), 2);
  assert.ok(railFits(436));
});

test('a landscape phone and a short laptop collapse', () => {
  // 844x414 phone → ~180px of view; 1400x700 laptop → ~140px. Wrapping these
  // would need 3-4 columns — a wall of buttons across the robot, reaching the
  // joysticks. Collapse instead.
  assert.ok(railCols(180) > RAIL_MAX_COLS);
  assert.ok(!railFits(180));
  assert.ok(!railFits(140));
});

test('the stick inset always covers the rail it must clear', () => {
  // The joysticks are placed at calc(var(--rail-w) + 14px). If railWidth ever
  // under-reported, the rail would sit on the stick again — the exact bug this
  // arithmetic replaced.
  for (const h of [140, 180, 436, 516, 760]){
    const cols = railFits(h) ? railCols(h) : 1;   // collapsed = one column
    assert.ok(railWidth(cols) >= RAIL_COL_W,
      `railWidth must cover at least one column at ${h}px`);
    assert.equal(railWidth(cols), cols * RAIL_COL_W);
  }
});

test('the collapsed rail is styled — .compact must exist in the CSS', () => {
  assert.ok(CSS.includes('.vp-rail.compact'),
    'js adds .compact to the rail; control.css must style it');
});

test('nav goal stays reachable — the floor tap needs its layer on', () => {
  // Tapping the floor only places a goal while the navgoal layer is live
  // (viewport3d gates on `this.layers.navgoal`), so the layer must both exist
  // in the rail and default ON, or tap-to-nav silently does nothing.
  assert.ok(railLayerKeys().includes('navgoal'));
  assert.equal(layerDefaults().navgoal, true);
});
