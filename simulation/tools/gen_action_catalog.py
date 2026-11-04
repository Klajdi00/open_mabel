#!/usr/bin/env python3
"""
gen_action_catalog.py — the timeline action catalogue, MOTION_CLIPS -> Swift.

The web control studio's `web_gui/control_center/web/js/motions.js` is the source
of truth for what MABEL can be asked to do on a timeline: title, emoji, duration,
which lane it occupies, its category, whether it can be held past its duration,
and its typed parameters. The iOS and Vision Pro apps need the same list.

Hand-typing 81 clips into Swift twice is exactly the drift this repo keeps paying
for elsewhere (see the model pipeline in the root README), so the Swift table is GENERATED.
Re-run this whenever motions.js gains or loses a clip:

    python3 simulation/tools/gen_action_catalog.py

The METADATA is read by running the registry under node, not by pattern-matching
the file. That used to be a regex over `\n  key: {`, which silently missed every
clip the registry BUILDS rather than spells out -- the twenty static poses are
`...poses({...})` now, and a regex cannot see a spread, a computed duration or a
title held in a table. A generator that quietly ports 61 of 81 clips is worse
than one that fails, so it asks the module.

What is NOT ported here: the `joints(t)` / `nav(t)` sampling closures. Those are
motion MATH; `gen_motion_clips.py` bakes them into a sampled table (also by
running the registry) so there is still no second implementation of them.

PER ROBOT (2026-10-25). NOVA runs its OWN copy of the registry
(~/Desktop/NOVA/control_center/web/js/motions.js — tip-over-safe clip revisions,
EXEC_RATE 50, NOVA's neck range), exactly as `gen_motion_clips.py --robot nova`
bakes `nova_motion_clips.json` from it. So the catalogue is generated from BOTH
copies: `bundled` (MABEL) and `bundledNova` (NOVA), and `ActionCatalog` picks
the one for the connected robot the same way `MotionClips` picks its table
(`RobotRig.isNova`). When the two registries agree on every field this script
ports (today they do — NOVA only re-authored clip MATH), `bundledNova` is
emitted as an alias of `bundled` rather than a second 95-entry copy, and the
generated file says it was checked.

    python3 simulation/tools/gen_action_catalog.py           # regenerate
    python3 simulation/tools/gen_action_catalog.py --check   # CI: fail on drift
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "web_gui", "control_center", "web", "js", "motions.js")
# NOVA's own copy of the registry (same repo layout on the NOVA side).
NOVA_SRC = os.path.expanduser("~/Desktop/NOVA/control_center/web/js/motions.js")
OUTS = [
    os.path.join(REPO, "apps", "ios", "MABEL Mobile App", "Models",
                 "ActionCatalog+Generated.swift"),
    os.path.join(REPO, "apps", "visionos", "MabelTeleop", "Models",
                 "ActionCatalog+Generated.swift"),
]


DUMP = r"""
import { MOTION_CLIPS, clipDur, defaults } from './motions.js';
const out = [];
for (const [key, c] of Object.entries(MOTION_CLIPS)) {
  const p = defaults(c);
  out.push({
    key, title: c.title || '', emoji: c.emoji || '',
    dur: Math.round(clipDur(c, p) * 1e4) / 1e4,
    lane: c.lane || 'manip', category: c.category || 'animation',
    status: c.status || 'available', hold: !!c.hold,
    goal: c.kind === 'goal',
    // An animation whose motion is a JS closure is flagged so the app can say
    // where its math came from. Poses, nav primitives and the blend are
    // reproducible from primitives, so they are not.
    clientMath: !!c.joints && (c.category === 'animation' || c.category === 'policy'),
    // A `hidden` param is one the studio never prompts for (rock_paper_scissors'
    // `seed` rolls a random matchup per block) — the app plays the baked sample,
    // so it is not offered as a slider either. A function-valued default (that
    // same roll) has no single value to port.
    params: Object.entries(c.params || {}).filter(([, v]) => !v.hidden).map(([id, v]) =>
      ({ id, label: v.label, def: typeof v.def === 'function' ? v.min : v.def,
         min: v.min, max: v.max })),
    location: c.location || null,
  });
}
process.stdout.write(JSON.stringify(out));
"""


def parse(src=None):
    """Run the registry under node and read its metadata back."""
    js_dir = os.path.dirname(src or SRC)
    fd, path = tempfile.mkstemp(suffix=".mjs", dir=js_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(DUMP)
        raw = subprocess.run(["node", os.path.basename(path)], cwd=js_dir,
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return json.loads(raw)


def swift_str(v):
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_table(L, name, clips):
    """One `static let <name>: [ActionSpec] = [...]` table."""
    L.append(f"    static let {name}: [ActionSpec] = [")
    for c in clips:
        ps = ", ".join(
            "ActionParam(id: {}, label: {}, def: {}, min: {}, max: {})".format(
                swift_str(p["id"]), swift_str(p["label"]), p["def"], p["min"], p["max"])
            for p in c["params"])
        loc = ("nil" if not c["location"] else
               "ActionLocation(x: {}, y: {}, label: {})".format(
                   c["location"]["x"], c["location"]["y"],
                   swift_str(c["location"]["label"])))
        L.append("        ActionSpec(")
        L.append(f"            id: {swift_str(c['key'])},")
        L.append(f"            title: {swift_str(c['title'])},")
        L.append(f"            emoji: {swift_str(c['emoji'])},")
        L.append(f"            duration: {c['dur']},")
        L.append(f"            lane: .{ {'manip': 'manipulation', 'nav': 'navigation', 'both': 'both'}[c['lane']] },")
        L.append(f"            category: .{c['category']},")
        L.append(f"            available: {str(c['status'] == 'available').lower()},")
        L.append(f"            holdable: {str(c['hold']).lower()},")
        L.append(f"            isGoal: {str(c['goal']).lower()},")
        L.append(f"            clientMath: {str(c['clientMath']).lower()},")
        L.append(f"            location: {loc},")
        L.append(f"            params: [{ps}]),")
    L.append("    ]")


def emit(clips, src_rel, nova=None, nova_rel=None):
    L = []
    L.append("//")
    L.append("//  ActionCatalog+Generated.swift")
    L.append("//")
    L.append("//  GENERATED — do not edit. Source of truth:")
    L.append(f"//    {src_rel}  (MABEL → `bundled`)")
    if nova_rel:
        L.append(f"//    {nova_rel}  (NOVA → `bundledNova`)")
    L.append("//  Regenerate: python3 simulation/tools/gen_action_catalog.py  (--check: CI drift test)")
    L.append("//")
    L.append("//  The timeline's action registry, identical to the one the web control")
    L.append("//  studio composes with. Motion MATH is not ported (see the generator's")
    L.append("//  docstring): clips flagged `clientMath` are authored as JS closures in")
    L.append("//  the studio, so the app plays them through the shared primitives and")
    L.append("//  says so rather than quietly diverging.")
    L.append("//")
    L.append("")
    L.append("import Foundation")
    L.append("")
    L.append("extension ActionCatalog {")
    L.append("    /// Every action MABEL's studio offers, in registry order.")
    emit_table(L, "bundled", clips)
    if nova is not None:
        L.append("")
        if nova == clips:
            L.append(f"    /// NOVA's studio ({nova_rel}). Its registry agrees with MABEL's on every")
            L.append("    /// field ported here (id, title, emoji, duration, lane, category, status,")
            L.append("    /// hold, params) — checked by the generator — so it is the same table.")
            L.append("    static var bundledNova: [ActionSpec] { bundled }")
        else:
            L.append(f"    /// Every action NOVA's studio offers ({nova_rel}), in registry order.")
            emit_table(L, "bundledNova", nova)
    else:
        L.append("    /// NOVA's registry was not found when this was generated — fall back to MABEL's.")
        L.append("    static var bundledNova: [ActionSpec] { bundled }")
    L.append("}")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed Swift tables are stale")
    args = ap.parse_args()
    clips = parse(SRC)
    nova = parse(NOVA_SRC) if os.path.exists(NOVA_SRC) else None
    src_rel = os.path.relpath(SRC, REPO)
    nova_rel = "NOVA/control_center/web/js/motions.js" if nova is not None else None
    text = emit(clips, src_rel, nova, nova_rel)
    outs = [o for o in OUTS if os.path.isdir(os.path.dirname(o))]
    if args.check:
        stale = [o for o in outs if not os.path.exists(o) or open(o).read() != text]
        for o in stale:
            print(f"STALE: {os.path.relpath(o, REPO)} differs from motions.js — re-run without --check")
        if not stale:
            print(f"ok: {len(outs)} catalogues match the registries")
        return 1 if stale else 0
    for out in outs:
        open(out, "w").write(text)
        print(f"  wrote {os.path.relpath(out, REPO)}")
    for who, cl in (("MABEL", clips), ("NOVA", nova)):
        if cl is None:
            print(f"{who}: registry not found (catalogue falls back to MABEL's)")
            continue
        cats = {}
        for c in cl:
            cats[c["category"]] = cats.get(c["category"], 0) + 1
        avail = sum(1 for c in cl if c["status"] == "available")
        print(f"{who}: {len(cl)} actions ({avail} available) {cats}")
    if nova is not None:
        print("NOVA catalogue == MABEL catalogue (aliased)" if nova == clips
              else "NOVA catalogue DIFFERS — emitted as its own table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
