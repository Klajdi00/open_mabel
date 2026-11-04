// The robot's MODULE vocabulary — the web twin of the iOS app's `ModuleFocus`
// (apps/ios/…/Visualization/LiveMapView.swift). One list, same order,
// same labels, so the phone and the studio speak about the robot identically.
//
// A module is the unit the operator THINKS in ("I'm working the right arm").
// It is the single selector behind the whole manipulate surface:
//   · which sliders the Joint-space panel shows (a hand is just a module —
//     there is no separate hand panel any more),
//   · which wrist the Task-space sticks / arrow keys drive,
//   · where the 3-D camera flies when you pick it.
//
// Pure data + name math — no DOM, no THREE (node-testable: tests/js/).

// `focus` = the rig node the camera frames when the module is selected.
// `glyph` names an SVG in ICONS below; `mirror` flips it (one arm/hand drawing
// serves both sides, exactly like SF Symbols' L/R badge on iOS).
export const MODULES = [
  { id: 'arm_left',   label: 'L Arm',  side: 'left',  glyph: 'arm',   mirror: true,
    focus: 'left_arm_4',   dist: 1.4 },
  { id: 'arm_right',  label: 'R Arm',  side: 'right', glyph: 'arm',   mirror: false,
    focus: 'right_arm_4',  dist: 1.4 },
  { id: 'neck',       label: 'Neck',   side: null,    glyph: 'neck',  mirror: false,
    focus: 'neck_pitch',   dist: 0.9 },
  { id: 'hand_left',  label: 'L Hand', side: 'left',  glyph: 'hand',  mirror: true,
    focus: 'left_wrist',   dist: 0.55 },
  { id: 'hand_right', label: 'R Hand', side: 'right', glyph: 'hand',  mirror: false,
    focus: 'right_wrist',  dist: 0.55 },
  { id: 'lift',       label: 'Lift',   side: null,    glyph: 'lift',  mirror: false,
    focus: 'lift_upper',   dist: 2.0 },
  { id: 'torso',      label: 'Torso',  side: null,    glyph: 'torso', mirror: false,
    focus: 'torso',        dist: 1.8 },
];

export const MODULE_IDS = MODULES.map(m => m.id);
export const moduleById = (id) => MODULES.find(m => m.id === id) || null;

// Which rig joints belong to a module. The `lift` module owns BOTH prismatic
// stages — they are mechanically synced on the robot, so the panel drives them
// from one height setpoint (the iOS "Sync Lift" equality).
const HAND_RE = /_(thumb|index|middle|ring|pinky)_(abd|mcp|pip|dip)$/;
const hand = (side) => (n) =>
  n === `${side}_wrist` || (n.startsWith(`${side}_`) && HAND_RE.test(n));

const OWNS = {
  arm_left:   (n) => /^left_arm_\d$/.test(n),
  arm_right:  (n) => /^right_arm_\d$/.test(n),
  neck:       (n) => n.startsWith('neck_'),
  hand_left:  hand('left'),
  hand_right: hand('right'),
  lift:       (n) => n === 'lift_lower' || n === 'lift_upper',
  torso:      (n) => n === 'torso',
};

/** Joint names of `id`, in manifest order. */
export const moduleJoints = (id, names = []) => {
  const owns = OWNS[id];
  return owns ? names.filter(owns) : [];
};

/** The module a joint belongs to (null for the swerve base / unknowns). */
export const moduleOf = (name) => {
  for (const m of MODULES) if (OWNS[m.id](name)) return m.id;
  return null;
};

/** Hand modules render the ORCA dexterity controls instead of raw sliders. */
export const isHand = (id) => id === 'hand_left' || id === 'hand_right';

// ── glyphs ─────────────────────────────────────────────────────────────────
// Geometric line icons on one 24×24 grid with one stroke weight, so the rail
// reads as a single family (the SF-Symbols discipline, hand-rolled).
export const ICONS = {
  // shoulder → elbow → wrist
  arm:   '<circle cx="6" cy="5.6" r="2.2"/><path d="M6 7.8v5.4l6.2 4"/>' +
         '<circle cx="13.6" cy="18" r="1.8"/>',
  // head on a neck, over shoulders
  neck:  '<circle cx="12" cy="6.2" r="3.2"/><path d="M12 9.4v3.4"/>' +
         '<path d="M6.8 17c1.6-2.4 8.8-2.4 10.4 0"/>',
  // three fingers + thumb over a palm
  hand:  '<path d="M9.4 12.4V6.1a1.3 1.3 0 0 1 2.6 0v5.6"/>' +
         '<path d="M12 11.7V5.3a1.3 1.3 0 0 1 2.6 0v6.4"/>' +
         '<path d="M14.6 11.7V6.6a1.3 1.3 0 0 1 2.6 0v6.9"/>' +
         '<path d="M9.4 12.4V9.9a1.3 1.3 0 0 0-2.6 0v4.6c0 3.1 2.2 5.5 5.3 5.5h1.6' +
         'c2.7 0 4.9-2.2 4.9-4.9v-1.6"/>',
  // the column: one axis, two directions
  lift:  '<path d="M12 3.8v16.4"/><path d="M8.9 7 12 3.8 15.1 7"/>' +
         '<path d="M8.9 17 12 20.2 15.1 17"/>',
  // torso pitching about its hinge
  torso: '<path d="M8.6 19.4 7.2 9.6a4.9 4.9 0 0 1 9.6 0l-1.4 9.8"/>' +
         '<path d="M4.6 7.4A8.6 8.6 0 0 1 8 4.2"/>',
};

/** Inline SVG for a module chip (mirrored for the left-side twins). */
export const iconSvg = (m) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"` +
  ` stroke-linecap="round" stroke-linejoin="round"` +
  `${m.mirror ? ' transform="scale(-1,1)" style="transform-origin:center"' : ''}>` +
  `${ICONS[m.glyph]}</svg>`;
