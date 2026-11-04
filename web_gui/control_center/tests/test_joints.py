"""Joint registry: the 60-joint / 59-DOF contract every consumer relies on."""
from control_center.server import joints


def test_counts():
    assert len(joints.JOINT_NAMES) == 60
    assert joints.DOF_COUNT == 59
    assert len(set(joints.JOINT_NAMES)) == 60          # no duplicates


def test_module_partition():
    seen = []
    for key, label, names in joints.MODULES:
        assert names, key
        seen.extend(names)
    assert seen == joints.JOINT_NAMES                  # ordered, exact partition


def test_expected_module_sizes():
    sizes = {k: len(n) for k, _, n in joints.MODULES}
    assert sizes == {"base": 6, "lift_torso": 3, "neck": 3,
                     "left_arm": 7, "right_arm": 7,
                     "left_hand": 17, "right_hand": 17}


def test_group_lookup():
    assert joints.JOINT_GROUP["fl_steer"] == "base"
    assert joints.JOINT_GROUP["lift_upper"] == "lift_torso"
    assert joints.JOINT_GROUP["left_thumb_dip"] == "left_hand"
    assert joints.JOINT_GROUP["right_arm_7"] == "right_arm"


def test_matches_web_manifest():
    """The registry must agree with web_gui/shared_assets/mabel_joints.json —
    the file the 3D rig articulates from."""
    import json
    import os
    # repo root is three levels up (this lives at <repo>/web_gui/control_center/tests)
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "web_gui", "shared_assets", "mabel_joints.json")
    with open(p) as f:
        manifest = json.load(f)
    manifest_names = {j["name"] for j in manifest["joints"]}
    ours = set(joints.JOINT_NAMES)
    assert ours <= manifest_names, ours - manifest_names


def test_schema_shape():
    s = joints.schema()
    assert s["dof"] == 59
    assert len(s["names"]) == 60
    assert [m["key"] for m in s["modules"]] == [k for k, _, _ in joints.MODULES]
