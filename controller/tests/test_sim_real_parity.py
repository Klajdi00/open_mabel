"""ONE controller and ONE retargeter for both the sim and the real robot.

Operator directive, restated 2026-07-31: "ensure that the same controller and
re-targeter should work in both sim and real robot."

This has drifted before — twice this month the physical robot was found running
a different revision from the sim, and the class of bug that produces (a fix
that "does nothing" because the robot never had it) costs a whole debugging
session every time. The architecture is right: ONE `TeleopEngine` inside ONE
`MabelTeleop` facade, with the PLANT as the only fork. These tests pin that so
it cannot quietly stop being true.

What is deliberately NOT tested here: that a given HOST has pulled a given
commit. No test can enforce that — it is a deployment fact, checked by
`git log` on thor. What IS enforceable is that there is only one implementation
to deploy, and that the sim/real switch cannot reach into the control path.
"""
import ast
import inspect
import io
import pathlib
import tokenize

import pytest

import mabel
from mabel.facade import MabelTeleop
from mabel.teleop_engine import TeleopEngine

_REPO = pathlib.Path(__file__).resolve().parents[2]


def test_facade_builds_exactly_one_engine_regardless_of_target():
    """The facade takes (model, data) and nothing else — there is no sim/real
    switch it could branch on, which is what makes 'the same retargeter' a
    structural fact rather than a convention."""
    sig = inspect.signature(MabelTeleop.__init__)
    params = set(sig.parameters) - {"self"}
    forbidden = {p for p in params
                 if any(w in p.lower() for w in ("sim", "real", "target", "dispatch"))}
    assert not forbidden, (
        f"the facade gained a sim/real parameter {forbidden} — the control stack "
        "must not know which plant it is driving")


def test_engine_has_no_sim_or_real_branch():
    """No dispatch-target branching anywhere in the engine or the facade."""
    # TOKENIZE rather than grep the raw text: these modules DISCUSS the real
    # robot at length in their docstrings (correctly — that is where the "the
    # mirror is fed by real encoders" reasoning belongs), and a naive text
    # search flags all of it. Only NAME/STRING tokens in executable positions
    # count, so prose stays free and a genuine branch is still caught.
    needles = ("MABEL_DISPATCH_TARGET", "real:ros", "sim:bridge", "sim:mujoco")
    offenders = []
    for mod in (_REPO / "controller" / "mabel" / "teleop_engine",
                _REPO / "controller" / "mabel" / "facade.py"):
        files = sorted(mod.rglob("*.py")) if mod.is_dir() else [mod]
        for f in files:
            src = f.read_text()
            docstrings = set()
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    d = ast.get_docstring(node, clean=False)
                    if d:
                        docstrings.add(d)
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.STRING:
                    body = tok.string.strip("\'\"")
                    if any(body.strip() == d.strip() for d in docstrings):
                        continue          # a docstring: prose, not a branch
                if any(n in tok.string for n in needles):
                    hit = next(n for n in needles if n in tok.string)
                    offenders.append(f"{f.name}:{tok.start[0]} {hit}")
    assert not offenders, (
        "the retargeting stack branches on the dispatch target: " + "; ".join(offenders))


def test_the_only_fork_is_the_plant():
    """`make_plant` is the ONE place the bridge chooses sim vs real."""
    src = (_REPO / "server" / "plant.py").read_text()
    assert "def make_plant" in src
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "make_plant")
    # It dispatches on the target string and returns plant objects — that is the
    # seam, and it lives in server/, never in controller/.
    args = {a.arg for a in fn.args.args}
    assert any("target" in a for a in args), (
        f"make_plant no longer takes a dispatch target: {args}")


@pytest.mark.parametrize("scope", ["local", "global"])
def test_scope_is_plant_independent(scope, tmp_path):
    """The LOCAL/GLOBAL scope is engine state, so it behaves identically no
    matter which plant is underneath — the sim and the real robot cannot have
    different ideas about whether the base is in the follow loop."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(
        str(_REPO / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    t = MabelTeleop(m, d)
    t.set_control_spec(method="retarget", scope=scope)
    assert t.controller.retarget_scope == scope
    assert t.controller.wbc.base_in_loop == (scope == "global")


def test_engine_module_is_the_single_shared_implementation():
    """Both paths import the SAME module object; the old scripts location is a
    re-export shim, not a second copy (that duplication is what the 2026-07-22
    move to controller/mabel/teleop_engine existed to end)."""
    assert TeleopEngine.__module__.startswith("mabel.teleop_engine")
    shim = _REPO / "simulation" / "mabel_mujoco" / "scripts" / "teleop" / "__init__.py"
    if shim.exists():
        src = shim.read_text()
        assert "from mabel.teleop_engine import" in src, (
            "the scripts/teleop shim stopped re-exporting and may be a second copy")
        assert "class TeleopEngine" not in src, "the shim defines its own engine"


def test_hyperparams_are_not_forked_by_environment_target():
    """Tuning constants must not silently differ between sim and real. Env
    OVERRIDES are fine (they are explicit and per-run); a built-in branch on the
    dispatch target is not."""
    src = (_REPO / "controller" / "mabel" / "teleop_engine" / "hyperparams.py").read_text()
    for needle in ("MABEL_DISPATCH_TARGET", "real:ros", "sim:bridge"):
        assert needle not in src, f"hyperparams branch on {needle}"
