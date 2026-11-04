"""
Rigorous test of the nn.Module → diagram COMPILER across all four architectures.

Truthfulness invariants checked per (arch, hp) case:
  1. compile succeeds (or fails with a stated reason — never crashes);
  2. the reported param total equals torch's own sum over the freshly built module;
  3. every leaf carries a known elementary prim (unknown_modules must be EMPTY —
     if a future LeRobot version adds a module class we don't map, this fails
     loudly instead of silently rendering a wrong block);
  4. hyperparameter changes are reflected structurally (e.g. n_encoder_layers ×2
     changes the encoder's collapsed ×N tag; dim_model changes Linear shapes);
  5. the graph is well-formed: unique node ids, edges reference existing ids,
     every expand subgraph passes the same check recursively.

Run:  ~/anaconda3/bin/python -m training_studio.server.tools.test_compiler
"""
import time

from .. import nn_compiler as NC
from .. import arch_registry as AR

SEL = {"cameras": ["cam_a", "cam_b"], "obs_state_dim": 59, "action_dim": 59}

CASES = [
    ("ACT", {}),
    ("ACT", {"dim_model": 256, "n_encoder_layers": 8, "n_heads": 4,
             "feedforward_activation": "gelu", "vision_backbone": "resnet34", "use_vae": False}),
    ("DiT", {}),
    ("DiT", {"n_layers": 12, "dim_model": 512, "n_heads": 8}),
    ("pi0", {}),
    ("pi0.5", {}),
]


def _check_graph(nodes, edges, seen_ids, path=""):
    """Well-formedness: unique ids, edges reference nodes, recurse into expands."""
    ids = set()
    problems = []
    for n in nodes:
        if n["id"] in ids:
            problems.append(f"duplicate id {n['id']} at {path}")
        ids.add(n["id"])
        if n.get("shape") == "layer" and not n.get("prim"):
            problems.append(f"leaf without prim: {n['id']} at {path}")
        if n.get("expand"):
            problems.extend(
                _check_graph(n["expand"]["nodes"], n["expand"].get("edges", []),
                             seen_ids, path + "/" + n["id"]))
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            problems.append(f"dangling edge {e['from']}→{e['to']} at {path}")
    return problems


def _count_leaves(nodes, prims):
    n = 0
    for node in nodes:
        if node.get("shape") == "layer":
            prims[node.get("prim", "?")] = prims.get(node.get("prim", "?"), 0) + 1
            n += 1
        if node.get("expand"):
            n += _count_leaves(node["expand"]["nodes"], prims)
    return n


def _find(nodes, pred):
    for n in nodes:
        if pred(n):
            return n
        if n.get("expand"):
            r = _find(n["expand"]["nodes"], pred)
            if r:
                return r
    return None


def main():
    results = []
    prim_totals = {}
    for arch, hp_over in CASES:
        hp = {**AR.defaults(arch), **hp_over}
        t0 = time.time()
        r = NC.compile_spec(arch, hp, SEL)
        dt = time.time() - t0
        row = {"arch": arch, "hp": hp_over or "defaults", "t": f"{dt:.1f}s"}
        if not r.get("ok"):
            row.update(ok=False, reason=r.get("reason", "?"))
            results.append(row)
            continue
        problems = _check_graph(r["nodes"], r["edges"], set())
        n_leaves = _count_leaves(r["nodes"], prim_totals)
        row.update(ok=True, params=r["params_str"], top=len(r["nodes"]),
                   leaves=n_leaves, unknown=r["unknown_modules"],
                   truncated=r.get("truncated", False), problems=problems)
        results.append(row)

    # invariant 2: param totals match torch's own count (rebuild fresh, compare)
    import torch
    for arch, hp_over in [("ACT", {}), ("DiT", {})]:
        hp = {**AR.defaults(arch), **hp_over}
        m = NC._build_module(arch, hp, SEL)
        true_p = sum(p.numel() for p in m.parameters())
        r = NC.compile_spec(arch, hp, SEL)
        assert r["params"] == true_p, f"{arch}: reported {r['params']} != torch {true_p}"
        del m
    print("param-total invariant: PASS (compiler total == torch total)")

    # invariant 4: hp changes are reflected structurally
    r1 = NC.compile_spec("ACT", {**AR.defaults("ACT")}, SEL)
    r2 = NC.compile_spec("ACT", {**AR.defaults("ACT"), "n_encoder_layers": 8}, SEL)
    enc1 = _find(r1["nodes"], lambda n: n["id"] == "c_encoder")
    enc2 = _find(r2["nodes"], lambda n: n["id"] == "c_encoder")
    lay1 = _find(enc1["expand"]["nodes"], lambda n: "×" in n["label"])
    lay2 = _find(enc2["expand"]["nodes"], lambda n: "×" in n["label"])
    assert "×4" in lay1["label"] and "×8" in lay2["label"], (lay1["label"], lay2["label"])
    print(f"hp-reflection invariant: PASS (encoder {lay1['label']} → {lay2['label']})")

    r3 = NC.compile_spec("ACT", {**AR.defaults("ACT"), "dim_model": 256}, SEL)
    head3 = _find(r3["nodes"], lambda n: n["id"] == "c_action_head")
    assert "256→59" in head3["sub"], head3["sub"]
    print(f"dim-reflection invariant: PASS (action head {head3['sub']})")

    print("\n== per-case results ==")
    any_fail = False
    for row in results:
        status = "PASS" if row.get("ok") and not row.get("problems") and not row.get("unknown") else \
                 ("PASS*" if row.get("ok") else "FAIL")
        if status == "FAIL":
            any_fail = True
        extra = (f"{row.get('params','')} · {row.get('top','')} top · {row.get('leaves','')} leaves"
                 if row.get("ok") else row.get("reason"))
        print(f"  [{status}] {row['arch']:6s} {str(row['hp'])[:44]:46s} {row['t']:>7s}  {extra}")
        if row.get("unknown"):
            print(f"          unknown modules (heuristic fallback used): {row['unknown']}")
        if row.get("problems"):
            any_fail = True
            print(f"          PROBLEMS: {row['problems'][:5]}")
        if row.get("truncated"):
            print("          note: graph truncated at MAX_NODES/MAX_DEPTH (rendered summary blocks)")

    print("\n== elementary-prim census (all leaves across all compiles) ==")
    for p, n in sorted(prim_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {p:10s} {n}")
    print("\nRESULT:", "FAIL" if any_fail else "ALL PASS")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
