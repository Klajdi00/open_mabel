"""Cross-checks against the LIVE robot stack's contracts — if the bridge or
map producer changes its wire protocol, these fail before a user does.

1. Every message type the web client sends must exist in the bridge's
   wire_protocol.py enum.
2. The map decoder's record tags/layouts must match map_stream's producer spec.
3. The hand model's ranges/poses must match the ORCA firmware configs.
"""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(HERE, "..", "web", "js")


def _find_repo():
    """Walk up to the checkout root (the dir that holds server/wire_protocol.py).

    This used to be a hard-coded `HERE/../..`, which stopped resolving when the
    studio moved under web_control_gui/ — so EVERY cross-stack contract check
    below died in FileNotFoundError and the whole suite went quietly green-ish.
    These are the tests that are supposed to catch a client drifting off the
    bridge's wire contract, so a silent path break disarms the alarm."""
    d = HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "server", "wire_protocol.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (server/wire_protocol.py) not found above " + HERE)


REPO = _find_repo()


def _read(*p):
    with open(os.path.join(*p)) as f:
        return f.read()


def _bridge_types():
    src = _read(REPO, "server", "wire_protocol.py")
    tree = ast.parse(src)
    vals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                vals.add(node.value.value)
    return vals


def test_client_sends_only_known_message_types():
    wire = _bridge_types()
    link = _read(JS, "link.js")
    sent = set(re.findall(r"send\('([a-z_]+)'", link))
    sent |= set(re.findall(r"this\.send\('([a-z_]+)'", link))
    app = _read(JS, "app.js")
    sent |= set(re.findall(r"link\.send\('([a-z_]+)'", app))
    unknown = sent - wire
    assert not unknown, f"client sends types the bridge doesn't know: {unknown}"
    # and the essentials are covered
    for must in ("teleop_frame", "joint_command", "control_mode", "estop",
                 "nav_goal", "record", "data_collect", "ping"):
        assert must in wire


def test_control_mode_options_match_bridge():
    src = _read(REPO, "server", "wire_protocol.py")
    for region in ("arm", "upper_body", "whole_body"):
        assert f'"{region}"' in src or f"'{region}'" in src
    for method in ("impedance", "compliance"):
        assert method in src


def test_control_mode_sends_each_value_in_the_right_slot():
    """`method` and `controlType` are DIFFERENT vocabularies, and swapping them
    fails silently: the bridge just never matches the spec, the torque engine
    never engages, and joint-space teleop moves nothing on the real robot while
    `robot_state` keeps streaming (so the joints still read live). We shipped
    exactly that bug — the client sent method='impedance', controlType='joint'.

    method      = TeleopMethod  (navigation | joint | wrist | retarget)
    controlType = ControlType   (impedance | compliance)
    """
    wire = _read(REPO, "server", "wire_protocol.py")
    methods = set(re.findall(r'^\s+[A-Z_]+\s*=\s*"(\w+)"', wire, re.M))
    # the two enums, straight from the bridge
    teleop_methods = {"navigation", "joint", "wrist", "retarget"} & methods
    control_types = {"impedance", "compliance"} & methods
    assert teleop_methods == {"navigation", "joint", "wrist", "retarget"}
    assert control_types == {"impedance", "compliance"}
    # the two vocabularies must stay disjoint, or "the right slot" is meaningless
    assert not (teleop_methods & control_types)

    app = _read(JS, "app.js")
    call = re.search(r"link\.controlMode\(([^;]*?)\)\s*;", app, re.S)
    assert call, "app.js no longer calls link.controlMode(...)"
    method_arg, type_arg = [a.strip() for a in call.group(1).split(",")[:2]]

    # Resolve each argument to the set of string literals it can evaluate to,
    # then check that set lands in the vocabulary its SLOT expects. Comparison
    # operands are what we TEST on (ctl.space === 'drive'), not what we SEND, so
    # strip them first — otherwise the guard trips over its own ternaries.
    def literals(expr):
        return set(re.findall(r"['\"](\w+)['\"]",
                              re.sub(r"[=!]==?\s*['\"]\w+['\"]", "", expr)))

    body = app[app.index("const applyControlSpec"):call.start()]

    def resolve(arg):
        if lits := literals(arg):
            return lits
        decl = re.search(rf"\b(?:const|let)\s+{re.escape(arg)}\s*=([^;]*);", body, re.S)
        assert decl, f"cannot resolve controlMode argument {arg!r}"
        return literals(decl.group(1))

    sent_methods, sent_types = resolve(method_arg), resolve(type_arg)
    assert sent_methods, "the method argument resolves to no string literal"
    assert sent_types, "the controlType argument resolves to no string literal"
    assert sent_methods <= teleop_methods, (
        f"app.js sends {sorted(sent_methods)} as `method`, but a TeleopMethod is "
        f"one of {sorted(teleop_methods)} — a ControlType here never matches the "
        f"facade's _WBC_METHODS and the torque engine silently never engages")
    assert sent_types <= control_types, (
        f"app.js sends {sorted(sent_types)} as `controlType`, but a ControlType is "
        f"one of {sorted(control_types)} — Soft/compliance never takes effect")


def test_map_decoder_matches_producer_spec():
    """maplink.js must decode exactly the record layout server/docs/data_streams.md
    (and map_stream.py) emit: tags 0x01..0x05, 16-byte vertices, 32-byte grid
    header."""
    md = _read(REPO, "server", "docs", "data_streams.md")
    js = _read(JS, "maplink.js")
    for tag in ("0x01", "0x02", "0x03", "0x04", "0x05"):
        assert tag in md and tag in js, tag
    # producer constants still present in map_stream.py
    py = _read(REPO, "server", "map_stream.py")
    for name in ("0x01", "0x02", "0x03", "0x04", "0x05"):
        assert name in py or str(int(name, 16)) in py


def test_hand_ranges_match_firmware_yaml():
    import yaml
    # the ORCA driver moved into the installable HAL (mabel_hw); hands/orcahand/
    # is now an empty shim, so this used to die in FileNotFoundError
    with open(os.path.join(REPO, "hardware_bridge", "mabel_hw", "drivers",
                           "orcahand", "configs", "left_hand.yaml")) as f:
        cfg = yaml.safe_load(f)
    js = _read(JS, "handmodel.js")
    # ORCA → URDF name mapping (hands_subsystem._THUMB_TO_URDF)
    o2u = {"thumb_cmc": "thumb_mcp", "thumb_mp": "thumb_dip",
           "thumb_ip": "thumb_pip"}
    joints = cfg.get("joints") or cfg
    checked = 0
    for oname, spec in joints.items():
        if not isinstance(spec, dict) or "angle_min" not in spec:
            continue
        uname = o2u.get(oname, oname)
        m = re.search(rf"\['{uname}',\s*(-?[\d.]+),\s*(-?[\d.]+)\]", js)
        assert m, f"{uname} missing from handmodel.js"
        lo, hi = float(m.group(1)), float(m.group(2))
        assert abs(lo - spec["angle_min"]) < 1e-3, uname
        assert abs(hi - spec["angle_max"]) < 1e-3, uname
        checked += 1
    assert checked == 17


def test_hand_poses_match_firmware():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orca_poses", os.path.join(REPO, "hardware_bridge", "mabel_hw",
                                   "drivers", "orcahand", "poses.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    poses = getattr(mod, "POSES", None) or {
        k: v for k, v in vars(mod).items()
        if isinstance(v, dict) and "index_pip" in v}
    js = _read(JS, "handmodel.js")
    for name in ("flat", "fist", "pinch", "point", "peace", "ok",
                 "thumbs_up", "rock", "spider"):
        assert name in js, f"pose {name} missing from web hand model"
        assert any(name in k for k in poses) or name in poses


# ── 4. the :9610 voice plane: the JS codec vs the Python producer ─────────────

def _audio_js():
    """The two pure codec functions out of audiolink.js (the module itself imports
    browser-only deps, so we lift the functions rather than import it)."""
    src = _read(JS, "audiolink.js")
    return src[src.index("const AUDIO_TAG"):src.index("export class AudioLink")]


def test_audio_record_tag_matches_the_producer():
    """0x07 is claimed in map_stream.py; a client on a stale tag would render
    voice frames as a pointcloud (they share the plane's discriminator byte)."""
    js = _audio_js()
    assert "const AUDIO_TAG = 0x07" in js
    prod = _read(os.path.join(REPO, "server"), "map_stream.py")
    assert "_AUDIO_TYPE = 0x07" in prod


def test_audio_header_size_matches_the_producer():
    js = _audio_js()
    assert "const HDR = 16" in js, "JS header size drifted from <BBHIII (16 B)"


def test_js_encoded_audio_frame_decodes_in_python():
    """THE cross-language contract: a frame the browser sends must be byte-identical
    to what the robot's own encoder would produce. A mismatch here is the failure
    that is completely silent until someone hears chipmunks (or nothing)."""
    node = _node_bin()
    if node is None:
        import pytest
        pytest.skip("node not available (conda mabel env provides it)")

    import json
    import subprocess
    import sys

    script = _audio_js() + """
const pcm = new Int16Array(320);
for (let i = 0; i < 320; i++) pcm[i] = (i * 97) % 32767 - 16000;
const buf = encodeAudio(pcm, 7, 16000, 1);
const back = decodeAudio(buf);
console.log(JSON.stringify({
  hex: Buffer.from(buf).toString('hex'),
  roundTrip: back.seq === 7 && back.rate === 16000 && back.channels === 1
             && back.pcm.length === 320 && back.pcm.every((v, i) => v === pcm[i]),
}));
"""
    out = subprocess.run([node, "--input-type=module", "-e", script],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["roundTrip"], "the JS codec does not round-trip its own frame"

    sys.path.insert(0, os.path.join(REPO, "server"))
    from map_stream import decode_audio, encode_audio          # noqa: E402

    frame = bytes.fromhex(got["hex"])
    dec = decode_audio(frame)
    assert dec is not None, "the robot REJECTS the browser's frame"
    pcm, seq, rate, ch, codec = dec
    assert (seq, rate, ch, codec) == (7, 16000, 1, 0)
    assert encode_audio(pcm, seq=7, rate=16000, channels=1) == frame, \
        "the JS and Python encoders disagree byte-for-byte"


def _node_bin():
    import shutil
    for c in ("node",
              os.path.expanduser("~/miniconda3/envs/mabel/bin/node")):
        p = shutil.which(c) or (c if os.path.isfile(c) else None)
        if p:
            return p
    return None


def test_audio_url_dials_the_dedicated_port():
    """Voice is the dedicated :9610 duplex plane, dialled direct (LAN / Tailscale)."""
    cfg = _read(JS, "config.js")
    assert "audioPort: 9610" in cfg
    m = re.search(r"export const audioUrl = \(\) =>(.{0,260})", cfg, re.S)
    assert m, "audioUrl missing"
    body = m.group(1)
    assert "cfg.audioPort" in body, "the direct route must use the dedicated port"
    assert "/audio" in body
