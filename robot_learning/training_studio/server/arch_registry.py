"""
Architecture registry — maps the 4 studio architectures (ACT, DiT, pi0, pi0.5)
to their real LeRobot policy types, a per-arch hyperparameter field schema (with
defaults/ranges/help + the real config key each maps to), validators that catch
mistakes before a long run, a policy-block builder for the train config, and the
config-accurate ``arch_spec`` graph that drives the animated architecture
diagram (blocks, token counts, tensor shapes on every wire, decode paradigm).

Field defaults are the real LeRobot config defaults verified against v0.4.4.
"""
import math

# ── field helpers ─────────────────────────────────────────────────────────────
def _f(name, label, typ, default, key=None, help="", **kw):
    d = {"name": name, "label": label, "type": typ, "default": default,
         "key": key or name, "help": help}
    d.update(kw)
    return d


# Shared training args (top-level TrainPipelineConfig).
TRAIN_FIELDS = [
    _f("batch_size", "Batch size", "int", 8, min=1, max=512, help="Samples per optimizer step."),
    _f("steps", "Training steps", "int", 5000, min=1, max=2_000_000,
       help="Total optimizer updates (offline)."),
    _f("log_freq", "Log every", "int", 10, min=1, max=10000, help="Steps between metric logs (low = more live feedback)."),
    _f("save_freq", "Checkpoint every", "int", 1000, min=1, max=100000, help="Steps between checkpoints."),
    _f("num_workers", "Data workers", "int", 0, min=0, max=32,
       help="0 = start instantly + use the in-process frame cache (best for small datasets). Raise for very large datasets."),
    _f("seed", "Seed", "int", 1000, min=0, max=2**31 - 1),
    _f("device", "Device", "select", "auto", options=["auto", "cuda", "mps", "cpu"],
       help="auto = cuda→mps→cpu."),
]

# Per-architecture hyperparameter schemas. `key` is the real LeRobot config field.
ARCHES = {
    "ACT": {
        "policy_type": "act", "native": True,
        "label": "ACT — Action Chunking Transformer",
        "decode": "transformer_decoder",
        "paper": "Zhao et al. 2023 (ALOHA)",
        "fields": [
            # NOTE: LeRobot's ACT only supports n_obs_steps=1 (single observation
            # step), so it is fixed at 1 and not exposed as an editable field.
            _f("chunk_size", "Chunk size", "int", 100, min=1, max=1000,
               help="Action horizon predicted per forward pass."),
            _f("n_action_steps", "Action steps", "int", 100, min=1, max=1000,
               help="Steps executed per inference (≤ chunk_size)."),
            _f("dim_model", "Model dim", "int", 512, min=64, max=2048, step=64),
            _f("n_heads", "Attention heads", "int", 8, min=1, max=32),
            _f("dim_feedforward", "FFN dim", "int", 3200, min=128, max=8192, step=128),
            _f("n_encoder_layers", "Encoder layers", "int", 4, min=1, max=24),
            _f("n_decoder_layers", "Decoder layers", "int", 1, min=1, max=24),
            _f("use_vae", "Use VAE (CVAE)", "bool", True),
            _f("latent_dim", "Latent dim", "int", 32, min=1, max=512),
            _f("kl_weight", "KL weight", "float", 10.0, min=0.0, max=1000.0),
            _f("dropout", "Dropout", "float", 0.1, min=0.0, max=0.9, step=0.05),
            _f("feedforward_activation", "FFN activation", "select", "relu",
               options=["relu", "gelu", "glu"],
               help="Activation inside every transformer feed-forward block (real LeRobot ACT field)."),
            _f("pre_norm", "Pre-norm", "bool", False,
               help="LayerNorm BEFORE (True) vs AFTER (False) each attention/FFN sublayer."),
            _f("n_vae_encoder_layers", "VAE encoder layers", "int", 4, min=1, max=12,
               help="Depth of the CVAE style encoder."),
            _f("vision_backbone", "Vision backbone", "select", "resnet18",
               options=["resnet18", "resnet34", "resnet50"]),
            _f("optimizer_lr", "Learning rate", "float", 1e-5, min=1e-7, max=1e-2),
        ],
    },
    "DiT": {
        "policy_type": "dit", "native": False,
        "label": "DiT — Diffusion Transformer policy",
        "decode": "diffusion",
        "paper": "Peebles & Xie 2023 (DiT) · Chi et al. 2023 (Diffusion Policy)",
        "fields": [
            _f("n_obs_steps", "Obs steps", "int", 2, min=1, max=10),
            _f("horizon", "Horizon", "int", 16, min=1, max=512,
               help="Action-chunk length denoised."),
            _f("n_action_steps", "Action steps", "int", 8, min=1, max=512),
            _f("dim_model", "Model dim", "int", 384, min=64, max=1536, step=64),
            _f("n_heads", "Attention heads", "int", 6, min=1, max=24),
            _f("n_layers", "DiT blocks", "int", 6, min=1, max=24,
               help="Transformer denoiser depth."),
            _f("num_train_timesteps", "Train timesteps", "int", 100, min=10, max=1000),
            _f("num_inference_steps", "Inference steps", "int", 10, min=1, max=1000,
               help="Denoising steps at rollout."),
            _f("beta_schedule", "Beta schedule", "select", "squaredcos_cap_v2",
               options=["squaredcos_cap_v2", "linear", "scaled_linear"]),
            _f("vision_backbone", "Vision backbone", "select", "resnet18",
               options=["resnet18", "resnet34"]),
            _f("optimizer_lr", "Learning rate", "float", 1e-4, min=1e-7, max=1e-2),
        ],
    },
    "pi0": {
        "policy_type": "pi0", "native": True,
        "label": "π₀ — Vision-Language-Action (flow matching)",
        "decode": "flow_matching",
        "paper": "Black et al. 2024 (π₀)",
        "fields": [
            _f("chunk_size", "Chunk size", "int", 50, min=1, max=200),
            _f("n_action_steps", "Action steps", "int", 50, min=1, max=200),
            _f("num_inference_steps", "Flow steps", "int", 10, min=1, max=100,
               help="Euler integration steps of the flow (t:0→1)."),
            _f("max_state_dim", "Max state dim (pad)", "int", 32, min=1, max=256),
            _f("max_action_dim", "Max action dim (pad)", "int", 32, min=1, max=256,
               help="VLA pads actions to this; must be ≥ selected action dim."),
            _f("dtype", "Dtype", "select", "bfloat16", options=["bfloat16", "float32", "float16"]),
            _f("freeze_vision_encoder", "Freeze vision", "bool", True),
            _f("train_expert_only", "Train action expert only", "bool", False),
            _f("gradient_checkpointing", "Grad checkpointing", "bool", False),
            _f("pretrained_path", "Pretrained path", "text", "",
               help="HF repo/path of π₀ base weights (blank = from scratch, huge)."),
            _f("optimizer_lr", "Learning rate", "float", 2.5e-5, min=1e-7, max=1e-2),
        ],
    },
    "pi0.5": {
        "policy_type": "pi05", "native": True,
        "label": "π₀.₅ — VLA with knowledge co-training",
        "decode": "flow_matching",
        "paper": "Physical Intelligence 2025 (π₀.₅)",
        "fields": [
            _f("chunk_size", "Chunk size", "int", 50, min=1, max=200),
            _f("n_action_steps", "Action steps", "int", 50, min=1, max=200),
            _f("num_inference_steps", "Flow steps", "int", 10, min=1, max=100),
            _f("max_state_dim", "Max state dim (pad)", "int", 32, min=1, max=256),
            _f("max_action_dim", "Max action dim (pad)", "int", 32, min=1, max=256,
               help="Must be ≥ selected action dim."),
            _f("tokenizer_max_length", "Tokenizer max len", "int", 200, min=8, max=512,
               help="π₀.₅ uses a longer prompt (state folded in) than π₀ (48)."),
            _f("dtype", "Dtype", "select", "bfloat16", options=["bfloat16", "float32", "float16"]),
            _f("freeze_vision_encoder", "Freeze vision", "bool", True),
            _f("train_expert_only", "Train action expert only", "bool", False),
            _f("gradient_checkpointing", "Grad checkpointing", "bool", False),
            _f("pretrained_path", "Pretrained path", "text", "",
               help="HF repo/path of π₀.₅ base weights."),
            _f("optimizer_lr", "Learning rate", "float", 2.5e-5, min=1e-7, max=1e-2),
        ],
    },
}

ARCH_ORDER = ["ACT", "DiT", "pi0", "pi0.5"]

# Resnet stage counts + downsample for the diagram (real torchvision structure).
_BACKBONE = {"resnet18": {"stages": 4, "blocks": [2, 2, 2, 2], "down": 32, "final": 512},
             "resnet34": {"stages": 4, "blocks": [3, 4, 6, 3], "down": 32, "final": 512},
             "resnet50": {"stages": 4, "blocks": [3, 4, 6, 3], "down": 32, "final": 2048}}


def defaults(arch):
    d = {f["name"]: f["default"] for f in ARCHES[arch]["fields"]}
    d.update({f["name"]: f["default"] for f in TRAIN_FIELDS})
    return d


def schema():
    """Full arch registry for the UI (arches + shared fields)."""
    return {"order": ARCH_ORDER, "train_fields": TRAIN_FIELDS,
            "arches": {a: {k: v for k, v in ARCHES[a].items()} for a in ARCH_ORDER}}


# ── validators ────────────────────────────────────────────────────────────────
def validate(arch, hp, sel):
    out = []

    def err(m): out.append({"level": "error", "msg": m})
    def warn(m): out.append({"level": "warn", "msg": m})

    A = ARCHES.get(arch)
    if not A:
        return [{"level": "error", "msg": f"unknown architecture {arch}"}]
    adim = sel.get("action_dim", 59)

    if arch == "ACT":
        if hp.get("n_action_steps", 1) > hp.get("chunk_size", 1):
            err("n_action_steps must be ≤ chunk_size")
        if hp.get("dim_model", 512) % max(1, hp.get("n_heads", 8)) != 0:
            err("dim_model must be divisible by n_heads")
    if arch == "DiT":
        if hp.get("dim_model", 384) % max(1, hp.get("n_heads", 6)) != 0:
            err("dim_model must be divisible by n_heads")
        if hp.get("n_action_steps", 1) > hp.get("horizon", 1):
            err("n_action_steps must be ≤ horizon")
        if hp.get("num_inference_steps", 1) > hp.get("num_train_timesteps", 100):
            warn("num_inference_steps > num_train_timesteps is unusual")
    if arch in ("pi0", "pi0.5"):
        pad = hp.get("max_action_dim", 32)
        if adim > pad:
            err(f"selected action dim {adim} exceeds max_action_dim {pad} — "
                f"raise Max action dim to ≥ {adim} (or narrow the module scope)")
        if hp.get("max_state_dim", 32) < sel.get("obs_state_dim", adim):
            err(f"max_state_dim must be ≥ selected state dim {sel.get('obs_state_dim')}")
        if not hp.get("pretrained_path"):
            warn("no pretrained_path: training a VLA from scratch is very expensive")
        if hp.get("n_action_steps", 1) > hp.get("chunk_size", 1):
            err("n_action_steps must be ≤ chunk_size")

    if len(sel.get("cameras", [])) == 0:
        err("select at least one camera")
    return out


# ── policy block for train_config.json ────────────────────────────────────────
def policy_block(arch, hp, sel, device="cuda"):
    A = ARCHES[arch]
    block = {"type": A["policy_type"], "device": device}
    for f in A["fields"]:
        v = hp.get(f["name"], f["default"])
        if f["type"] == "text" and (v is None or v == ""):
            continue  # omit empty optional strings (e.g. pretrained_path)
        block[f["key"]] = v
    # VLA: ensure pad dims cover the selected action/state dim
    if arch in ("pi0", "pi0.5"):
        adim = sel.get("action_dim", 59)
        sdim = sel.get("obs_state_dim", adim)
        block["max_action_dim"] = max(block.get("max_action_dim", 32), adim)
        block["max_state_dim"] = max(block.get("max_state_dim", 32), sdim)
    return block


# ── drill-down subgraphs (Simulink-style "look inside a block") ───────────────
def _layer_cols(nodes, edges):
    """Assign left→right columns to a subgraph by topological longest-path, so
    drill-down views (ResNet/transformer internals) flow left to right."""
    from collections import deque
    ids = {n["id"] for n in nodes}
    indeg = {n["id"]: 0 for n in nodes}
    outs = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["from"] in ids and e["to"] in ids:
            indeg[e["to"]] += 1
            outs[e["from"]].append(e["to"])
    col = {i: 0 for i in ids}
    deg = dict(indeg)
    q = deque([i for i in indeg if indeg[i] == 0])
    seen = set(q)
    while q:
        i = q.popleft()
        for t in outs[i]:
            col[t] = max(col[t], col[i] + 1)
            deg[t] -= 1
            if deg[t] <= 0 and t not in seen:
                seen.add(t)
                q.append(t)
    lanes = {}
    for n in nodes:
        c = col[n["id"]]
        n["col"] = c
        n["lane"] = lanes.get(c, 0)
        lanes[c] = lanes.get(c, 0) + 1


def _sub(nodes, edges, title):
    _layer_cols(nodes, edges)
    return {"nodes": nodes, "edges": edges, "title": title}


def _expand_resnet(back, d):
    chs = [64, 128, 256, 512] if back["final"] == 512 else [256, 512, 1024, 2048]
    # stem = Conv7×7/2 → BN → ReLU → MaxPool/2 (elementary layers)
    n = [_L("stem", "conv", "Conv 7×7 /2", "64ch"),
         _L("bn0", "norm", "BatchNorm", "64ch"),
         _L("r0", "act", "ReLU", ""),
         _L("pool", "pool", "MaxPool /2", "3×3")]
    e = [{"from": "stem", "to": "bn0", "kind": "data"}, {"from": "bn0", "to": "r0", "kind": "data"},
         {"from": "r0", "to": "pool", "kind": "data", "tensor": "64×135×240"}]
    prev = "pool"
    for i, (nb, ch) in enumerate(zip(back["blocks"], chs), 1):
        nid = f"layer{i}"          # a stage of `nb` residual BasicBlocks
        n.append({"id": nid, "shape": "round", "modality": "vision", "label": f"layer{i}",
                  "sub": f"{nb}× BasicBlock · {ch}ch", "expand": _expand_resblock(ch)})
        e.append({"from": prev, "to": nid, "kind": "data", "tensor": f"{ch}ch"})
        prev = nid
    n.append(_L("proj", "conv", "Conv 1×1 → d", f"{back['final']}→{d}"))
    e.append({"from": prev, "to": "proj", "kind": "data", "tensor": f"{back['final']}×H×W"})
    return _sub(n, e, "ResNet backbone (inside)")


# ── elementary NN layers — the ATOMS of every architecture ────────────────────
# A `prim` tag ("conv"|"linear"|"norm"|"act"|"matmul"|"softmax"|"pool"|
# "residual"|"embed"|"dropout") drives one consistent glyph + colour in the
# diagram, and marks the node as a leaf (no further drill-down). Every composite
# block (ResNet/BasicBlock, MHA, MLP, …) is built purely out of these.
_PRIM_MOD = {"conv": "vision", "pool": "vision", "linear": "proprio", "norm": "norm",
             "act": "act", "matmul": "attention", "softmax": "attention",
             "residual": "trunk", "embed": "trunk", "dropout": "trunk"}


def _L(nid, prim, label, sub=""):
    """One elementary layer node."""
    return {"id": nid, "shape": "layer", "prim": prim,
            "modality": _PRIM_MOD.get(prim, "trunk"), "label": label, "sub": sub}


def _expand_resblock(ch):
    # BasicBlock = Conv3×3 → BN → ReLU → Conv3×3 → BN → (⊕ identity) → ReLU,
    # each an elementary layer (the smallest building block).
    n = [_L("c1", "conv", "Conv 3×3", f"{ch}ch, pad 1"),
         _L("b1", "norm", "BatchNorm", f"{ch}ch"),
         _L("r1", "act", "ReLU", ""),
         _L("c2", "conv", "Conv 3×3", f"{ch}ch, pad 1"),
         _L("b2", "norm", "BatchNorm", f"{ch}ch"),
         _L("add", "residual", "⊕ add", "identity skip"),
         _L("r2", "act", "ReLU", "")]
    e = [{"from": "c1", "to": "b1", "kind": "data"}, {"from": "b1", "to": "r1", "kind": "data"},
         {"from": "r1", "to": "c2", "kind": "data"}, {"from": "c2", "to": "b2", "kind": "data"},
         {"from": "b2", "to": "add", "kind": "data"}, {"from": "add", "to": "r2", "kind": "data"},
         {"from": "c1", "to": "add", "kind": "residual", "tensor": "identity"}]
    return _sub(n, e, "BasicBlock (residual)")


def _expand_attention(d, heads, cross=False):
    # Multi-head attention, decomposed to elementary layers. Q from the block's
    # input; K,V from encoder memory (cross) or the same input (self). Scores =
    # QKᵀ/√dₖ → softmax → weighted sum ·V → output projection.
    dk = max(1, d // max(1, heads))
    src = "encoder memory" if cross else "input x"
    n = [_L("in_q", "embed", "queries", "Q source: " + ("action queries" if cross else "x")),
         _L("in_kv", "embed", "K,V source", src),
         _L("wq", "linear", "W_Q", f"d→{d}"),
         _L("wk", "linear", "W_K", f"d→{d}"),
         _L("wv", "linear", "W_V", f"d→{d}"),
         _L("qk", "matmul", "Q·Kᵀ / √dₖ", f"{heads}h · dₖ={dk}"),
         _L("sm", "softmax", "Softmax", "attention weights"),
         _L("av", "matmul", "weights · V", "context"),
         _L("wo", "linear", "W_O", f"{d}→d")]
    e = [{"from": "in_q", "to": "wq", "kind": "data"},
         {"from": "in_kv", "to": "wk", "kind": "data"},
         {"from": "in_kv", "to": "wv", "kind": "data"},
         {"from": "wq", "to": "qk", "kind": "data", "tensor": "Q"},
         {"from": "wk", "to": "qk", "kind": "data", "tensor": "Kᵀ"},
         {"from": "qk", "to": "sm", "kind": "data"},
         {"from": "sm", "to": "av", "kind": "data", "tensor": "A"},
         {"from": "wv", "to": "av", "kind": "data", "tensor": "V"},
         {"from": "av", "to": "wo", "kind": "data"}]
    return _sub(n, e, ("Cross" if cross else "Self") + "-attention (inside)")


# activation blocks carry an `edit` so a click opens the picker → real LeRobot field
_ACT_EDIT = {"hp": "feedforward_activation", "kind": "select",
             "options": ["relu", "gelu", "glu"], "label": "FFN activation"}


def _expand_ffn(d, ff, act="relu", editable=False):
    # Position-wise feed-forward: Linear d→ff, <activation>, Linear ff→d. When the
    # arch exposes it (ACT), the activation node is CLICK-EDITABLE → the real
    # LeRobot `feedforward_activation`; otherwise it's shown but fixed.
    g = _L("g", "act", act.upper(), "click to change" if editable else "")
    if editable:
        g["edit"] = _ACT_EDIT
    n = [_L("l1", "linear", "Linear", f"{d}→{ff}"), g, _L("l2", "linear", "Linear", f"{ff}→{d}")]
    e = [{"from": "l1", "to": "g", "kind": "data"}, {"from": "g", "to": "l2", "kind": "data"}]
    return _sub(n, e, "Feed-forward MLP (inside)")


_PRENORM_EDIT = {"hp": "pre_norm", "kind": "bool", "label": "Pre-norm (LN placement)"}


def _expand_transformer(layers, d, heads, ff, decoder=False, act="relu", pre=False, editable=False):
    # One transformer block: LN → (self-)attention → ⊕ → [LN → cross-attention →
    # ⊕] → LN → FFN → ⊕. When `editable` (ACT), LayerNorm nodes toggle pre/post
    # norm (→ LeRobot `pre_norm`) and the FFN activation is a picker.
    lnsub = ("pre-norm" if pre else "post-norm") if editable else ""
    def ln(nid):
        node = _L(nid, "norm", "LayerNorm", lnsub)
        if editable:
            node["edit"] = _PRENORM_EDIT
        return node
    n = [ln("ln1"),
         {"id": "mha", "shape": "round", "modality": "attention", "label": "Self-Attention",
          "sub": f"{heads} heads · d{d}", "expand": _expand_attention(d, heads, cross=False)},
         _L("add1", "residual", "⊕ add", "residual")]
    e = [{"from": "ln1", "to": "mha", "kind": "data"}, {"from": "mha", "to": "add1", "kind": "data"},
         {"from": "ln1", "to": "add1", "kind": "residual", "tensor": "x"}]
    prev = "add1"
    if decoder:
        # DETR/ACT-style: action queries self-attend, THEN cross-attend to the
        # encoder memory (queries = Q, memory = K,V).
        n += [ln("ln2"),
              {"id": "xa", "shape": "round", "modality": "attention", "label": "Cross-Attention",
               "sub": "Q: queries · K,V: memory", "expand": _expand_attention(d, heads, cross=True)},
              _L("add2", "residual", "⊕ add", "residual"),
              {"id": "mem_in", "shape": "tokencol", "modality": "trunk", "label": "encoder memory",
               "sub": "K,V", "tokens": 6}]
        e += [{"from": prev, "to": "ln2", "kind": "data"}, {"from": "ln2", "to": "xa", "kind": "data"},
              {"from": "xa", "to": "add2", "kind": "data"}, {"from": "ln2", "to": "add2", "kind": "residual"},
              {"from": "mem_in", "to": "xa", "kind": "cross_attn", "tensor": "memory"}]
        prev = "add2"
    n += [ln("ln3"),
          {"id": "mlp", "shape": "round", "modality": "proprio", "label": "Feed-forward MLP",
           "sub": f"{d}→{ff}→{d} · {act}", "expand": _expand_ffn(d, ff, act, editable)},
          _L("add3", "residual", "⊕ add", "residual")]
    e += [{"from": prev, "to": "ln3", "kind": "data"}, {"from": "ln3", "to": "mlp", "kind": "data"},
          {"from": "mlp", "to": "add3", "kind": "data"}, {"from": "ln3", "to": "add3", "kind": "residual"}]
    kind = "decoder" if decoder else "encoder"
    return _sub(n, e, f"{layers}× Transformer {kind} block (inside)")


def _expand_dit_block(layers, d, heads, ff):
    # DiT adaLN-Zero block: the conditioning VECTOR c (obs⊕timestep) → an MLP that
    # regresses per-block modulation (γ scale, β shift, α gate). The NOISED-ACTION
    # tokens x are the sequence; c only modulates their LayerNorms + gates the
    # residual — it is NEVER concatenated into the token sequence.
    n = [{"id": "cin", "shape": "rect", "modality": "trunk", "label": "Conditioning c", "sub": "obs⊕t vector"},
         _L("adamlp", "linear", "MLP", "c → γ,β,α (×2)"),
         _L("ln1", "norm", "LayerNorm", ""),
         _L("mod1", "matmul", "Scale, Shift", "γ₁, β₁"),
         {"id": "mha", "shape": "round", "modality": "attention", "label": "Multi-Head Self-Attention",
          "sub": f"{heads}h · over action tokens", "expand": _expand_attention(d, heads, cross=False)},
         _L("g1", "matmul", "Scale", "α₁"),
         _L("add1", "residual", "⊕ add", "residual"),
         _L("ln2", "norm", "LayerNorm", ""),
         _L("mod2", "matmul", "Scale, Shift", "γ₂, β₂"),
         {"id": "mlp", "shape": "round", "modality": "proprio", "label": "Pointwise Feed-forward",
          "sub": f"{d}→{ff}→{d} · GELU", "expand": _expand_ffn(d, ff, "gelu")},
         _L("g2", "matmul", "Scale", "α₂"),
         _L("add2", "residual", "⊕ add", "residual")]
    e = [{"from": "cin", "to": "adamlp", "kind": "data", "tensor": "c"},
         {"from": "ln1", "to": "mod1", "kind": "data", "tensor": "x (action tokens)"},
         {"from": "adamlp", "to": "mod1", "kind": "adaln", "tensor": "γ₁,β₁"},
         {"from": "mod1", "to": "mha", "kind": "data"},
         {"from": "mha", "to": "g1", "kind": "data"},
         {"from": "adamlp", "to": "g1", "kind": "adaln", "tensor": "α₁"},
         {"from": "g1", "to": "add1", "kind": "data"},
         {"from": "ln1", "to": "add1", "kind": "residual", "tensor": "x"},
         {"from": "add1", "to": "ln2", "kind": "data"},
         {"from": "ln2", "to": "mod2", "kind": "data"},
         {"from": "adamlp", "to": "mod2", "kind": "adaln", "tensor": "γ₂,β₂"},
         {"from": "mod2", "to": "mlp", "kind": "data"},
         {"from": "mlp", "to": "g2", "kind": "data"},
         {"from": "adamlp", "to": "g2", "kind": "adaln", "tensor": "α₂"},
         {"from": "g2", "to": "add2", "kind": "data"},
         {"from": "add1", "to": "add2", "kind": "residual"}]
    return _sub(n, e, f"{layers}× DiT block · adaLN-Zero (inside)")


def _expand_final(d, adim):
    # DiT FinalLayer: adaLN (SiLU→Linear → shift,scale) modulates a LayerNorm,
    # then a Linear projects d → action_dim (the predicted noise ε).
    n = [{"id": "cin", "shape": "rect", "modality": "trunk", "label": "cond c", "sub": "vector"},
         _L("ada", "linear", "adaLN MLP", "c → shift,scale"),
         _L("ln", "norm", "LayerNorm", ""),
         _L("mod", "matmul", "· γ + β", "modulate"),
         _L("proj", "linear", "Linear", f"{d}→{adim}")]
    e = [{"from": "cin", "to": "ada", "kind": "data"},
         {"from": "ln", "to": "mod", "kind": "data", "tensor": "x"},
         {"from": "ada", "to": "mod", "kind": "adaln", "tensor": "γ,β"},
         {"from": "mod", "to": "proj", "kind": "data"}]
    return _sub(n, e, "Final layer (adaLN → ε)")


def _expand_mlp(din, d):
    # state encoder: Linear → ReLU → Linear (elementary layers)
    n = [_L("l1", "linear", "Linear", f"{din}→128"),
         _L("a1", "act", "ReLU", ""),
         _L("l2", "linear", "Linear", f"128→{d}")]
    e = [{"from": "l1", "to": "a1", "kind": "data"}, {"from": "a1", "to": "l2", "kind": "data"}]
    return _sub(n, e, "MLP (inside)")


def _expand_cvae(sdim, latent, d, vae_layers=4, act="relu", pre=False):
    n = [{"id": "enc", "shape": "round", "modality": "latent", "label": "Transformer enc",
          "sub": f"style encoder · ×{vae_layers}",
          "expand": _expand_transformer(vae_layers, d, 8, 4 * d, act=act, pre=pre, editable=True)},
         _L("mu", "linear", "Linear → μ, logσ", f"→ 2×{latent}"),
         {"id": "z", "shape": "rect", "modality": "latent", "label": "reparam z", "sub": f"μ+σ·ε · {latent}-D"},
         _L("pz", "linear", "Linear", f"{latent}→{d}")]
    e = [{"from": "enc", "to": "mu", "kind": "data"}, {"from": "mu", "to": "z", "kind": "data"},
         {"from": "z", "to": "pz", "kind": "data"}]
    return _sub(n, e, "CVAE encoder (inside)")


# ── arch_spec: the config-accurate, editable, hierarchical diagram graph ──────
def arch_spec(arch, hp, sel, introspection=None):
    """Rich modality-decomposed graph the frontend renders in the reference
    (IRIS / pi0.5) style: shapes per block type, tokens as columns of pills, the
    token-stacking sequence, cross-attention, the decode head, plus per-block
    ``expand`` subgraphs for drill-down and ``editHp`` for add/remove-layer."""
    A = ARCHES[arch]
    cams = sel.get("cameras", [])
    adim = sel.get("action_dim", 59)
    sdim = sel.get("obs_state_dim", adim)
    img = [270, 480]
    N, E = [], []

    def node(nid, shape, modality, label, sub="", col=0, lane=0, **kw):
        n = {"id": nid, "shape": shape, "modality": modality, "label": label,
             "sub": sub, "col": col, "lane": lane}
        n.update(kw)
        N.append(n)
        return nid

    def edge(a, b, tensor="", kind="data"):
        E.append({"from": a, "to": b, "tensor": tensor, "kind": kind})

    back = _BACKBONE.get(hp.get("vision_backbone", "resnet18"), _BACKBONE["resnet18"])
    fmap = [max(1, math.ceil(img[0] / back["down"])), max(1, math.ceil(img[1] / back["down"]))]
    tpc = fmap[0] * fmap[1]
    d = hp.get("dim_model", 512 if arch == "ACT" else (384 if arch == "DiT" else 1024))

    # activation + norm placement are real, click-editable LeRobot ACT fields
    av = hp.get("feedforward_activation", "relu")
    pn = bool(hp.get("pre_norm", False))
    bb_edit = {"hp": "vision_backbone", "kind": "select",
               "options": ["resnet18", "resnet34", "resnet50"], "label": "Vision backbone (CNN)"}

    if arch == "ACT":
        n_cam = max(1, len(cams))
        # inputs (col 0) + encoders (1) + proj (2) + tokens (3)
        for i, c in enumerate(cams):
            # camera inputs are ALWAYS deletable (drop a video stream freely)
            node(f"in_{c}", "image", "vision", c, f"RGB {img[1]}×{img[0]}", col=0, lane=i,
                 camera=c, deletable=True, protect=None)
            node(f"cnn_{c}", "trapezoid", "vision", "ResNet",
                 f"{hp.get('vision_backbone','resnet18')} · {'+'.join(map(str, back['blocks']))}",
                 col=1, lane=i, protect="vision encoder", expand=_expand_resnet(back, d),
                 edit=bb_edit)
            node(f"tok_{c}", "tokencol", "vision", "patch tokens",
                 f"{fmap[0]}×{fmap[1]} = {tpc}", col=2, lane=i, tokens=tpc,
                 protect="produced by the backbone")
            edge(f"in_{c}", f"cnn_{c}", f"3×{img[0]}×{img[1]}")
            edge(f"cnn_{c}", f"tok_{c}", f"{back['final']}×{fmap[0]}×{fmap[1]}")
        # proprio lane
        pl = n_cam
        node("in_prop", "text", "proprio", "Proprioception", f"joint state · {sdim}-D",
             col=0, lane=pl, protect="proprioception is required")
        node("mlp", "round", "proprio", "MLP", f"{sdim}→128→{d}", col=1, lane=pl,
             protect="state encoder", expand=_expand_mlp(sdim, d))
        node("tok_prop", "tokencol", "proprio", "state token", "1 token", col=2, lane=pl, tokens=1,
             protect="the state token")
        edge("in_prop", "mlp", f"{sdim}")
        edge("mlp", "tok_prop", f"{d}")
        # VAE latent lane (optional)
        seq_edges = [f"tok_{c}" for c in cams] + ["tok_prop"]
        if hp.get("use_vae", True):
            vl = n_cam + 1
            node("in_vae", "text", "latent", "Target actions", "q_{t+1:t+H} (train)",
                 col=0, lane=vl, deletable=True, editHp={"use_vae": False},
                 protect="disable via Use VAE")
            node("cvae", "round", "latent", "CVAE encoder", f"→ latent {hp.get('latent_dim',32)}",
                 col=1, lane=vl,
                 expand=_expand_cvae(adim, hp.get("latent_dim", 32), d,
                                     hp.get("n_vae_encoder_layers", 4), av, pn),
                 deletable=True, editHp={"use_vae": False})
            node("tok_z", "tokencol", "latent", "latent z", "1 token", col=2, lane=vl, tokens=1)
            edge("in_vae", "cvae", f"{adim}·H")
            edge("cvae", "tok_z", f"{hp.get('latent_dim',32)}→{d}")
            seq_edges.append("tok_z")
        # fused sequence (col 3) — show the EXACT concatenation order so it's
        # unambiguous how patch/state/latent tokens stack into one sequence.
        n_seq = n_cam * tpc + 1 + (1 if hp.get("use_vae", True) else 0)
        groups = [{"mod": "vision", "n": n_cam * tpc, "label": f"{n_cam}×{tpc} patch"},
                  {"mod": "proprio", "n": 1, "label": "state"}]
        if hp.get("use_vae", True):
            groups.append({"mod": "latent", "n": 1, "label": "z"})
        node("seq", "tokencol", "trunk", "Token sequence", f"concat · +PosEmb · {n_seq} tokens",
             col=3, lane=0, tokens=min(n_seq, 24), seqlen=n_seq, tall=True, groups=groups,
             protect="the fused input sequence (self-attention runs over all these tokens)")
        for t in seq_edges:
            edge(t, "seq", "", "stack")
        # encoder (col 4)
        node("enc", "tall", "trunk", "Transformer Encoder",
             f"×{hp.get('n_encoder_layers',4)} · d{d} · {hp.get('n_heads',8)}h",
             col=4, lane=0, layers=hp.get("n_encoder_layers", 4), tokens=min(n_seq, 24),
             editHp={"n_encoder_layers": 1}, protect="the encoder",
             expand=_expand_transformer(hp.get("n_encoder_layers", 4), d, hp.get("n_heads", 8),
                                        hp.get("dim_feedforward", 3200), act=av, pre=pn, editable=True))
        edge("seq", "enc", f"{n_seq}×{d}")
        # memory (col 5)
        node("mem", "tokencol", "trunk", "Encoder memory", f"{n_seq} tokens", col=5, lane=0,
             tokens=min(n_seq, 24), protect="encoder output")
        edge("enc", "mem", f"{n_seq}×{d}")
        # decoder (col 6) + action queries (lane below)
        ch = hp.get("chunk_size", 100)
        node("dec", "tall", "trunk", "Transformer Decoder",
             f"×{hp.get('n_decoder_layers',1)} · {ch} queries", col=6, lane=0,
             layers=hp.get("n_decoder_layers", 1), tokens=min(ch, 20),
             editHp={"n_decoder_layers": 1}, protect="the decoder",
             expand=_expand_transformer(hp.get("n_decoder_layers", 1), d, hp.get("n_heads", 8),
                                        hp.get("dim_feedforward", 3200), decoder=True,
                                        act=av, pre=pn, editable=True))
        node("queries", "tokencol", "action", "Action queries", "+ PosEmb", col=6, lane=1,
             tokens=min(ch, 20), protect="learnt action queries")
        edge("mem", "dec", "memory (K,V)", "cross_attn")
        edge("queries", "dec", f"{ch}×{d}")
        # action embeddings → head → actions (7,8,9)
        node("aemb", "tokencol", "action", "Action embeddings", f"{ch} tokens", col=7, lane=0,
             tokens=min(ch, 20), protect="decoder output")
        node("head", "rect", "action", "Linear", f"{d}→{adim}", col=8, lane=0, protect="action head")
        node("out", "actions", "action", "Actions", f"{ch}×{adim} · q_{{t+1:t+H}}", col=9, lane=0,
             tokens=min(adim, 12), protect="policy output")
        edge("dec", "aemb", f"{ch}×{d}")
        edge("aemb", "head", f"{ch}×{d}")
        edge("head", "out", f"{ch}×{adim}")

    elif arch == "DiT":
        n_cam = max(1, len(cams))
        hz = hp.get("horizon", 16)
        n_layers = hp.get("n_layers", 6)
        heads = hp.get("n_heads", 6)
        # ── observation → ONE conditioning VECTOR (adaLN-Zero: obs is average-pooled
        #    to a vector that MODULATES the DiT blocks — it is NOT turned into tokens
        #    and NOT concatenated with the noised actions) ──
        conds = []
        for i, c in enumerate(cams):
            node(f"in_{c}", "image", "vision", c, f"RGB {img[1]}×{img[0]}", col=0, lane=i,
                 camera=c, deletable=True, protect=None)
            node(f"cnn_{c}", "trapezoid", "vision", "ResNet",
                 f"{hp.get('vision_backbone','resnet18')}", col=1, lane=i,
                 protect="vision encoder", expand=_expand_resnet(back, d),
                 edit={"hp": "vision_backbone", "kind": "select",
                       "options": ["resnet18", "resnet34"], "label": "Vision backbone (CNN)"})
            node(f"pool_{c}", "rect", "vision", "avg-pool", f"→ {back['final']}-D", col=2, lane=i,
                 protect="spatial pool → feature vector")
            edge(f"in_{c}", f"cnn_{c}", f"3×{img[0]}×{img[1]}")
            edge(f"cnn_{c}", f"pool_{c}", "feat map")
            edge(f"pool_{c}", "cond", f"{back['final']}", "stack")
            conds.append(f"pool_{c}")
        node("in_prop", "text", "proprio", "Proprioception", f"{sdim}-D", col=0, lane=n_cam,
             protect="required")
        edge("in_prop", "cond", f"{sdim}", "stack")
        # timestep t → sinusoidal embedding → part of the conditioning vector
        node("t_in", "rect", "noise", "timestep t", "diffusion step", col=1, lane=n_cam + 1,
             protect="which denoising step")
        node("t_emb", "round", "noise", "Time embed", "sinusoid → MLP", col=2, lane=n_cam + 1,
             protect="timestep embedding")
        edge("t_in", "t_emb", "t")
        edge("t_emb", "cond", "t-emb", "stack")
        # the single conditioning VECTOR c = time_emb(d) + Linear(concat[pooled-obs, state]).
        # global_cond_dim mirrors the DiT plugin: n_cam·backbone_dim + state_dim.
        cond_in = n_cam * back["final"] + sdim
        node("cond", "round", "trunk", "Conditioning c", f"obs {cond_in}-D ⊕ t → {d}-D", col=3, lane=1,
             protect="adaLN conditioning — a VECTOR, not tokens", cond_in=cond_in, cond_out=d,
             expand=_expand_mlp(cond_in, d))
        # the NOISED ACTION CHUNK (horizon × action_dim) is projected + position-
        # embedded → THIS is where the token sequence comes from (input_proj + pos_emb).
        node("noise", "tokencol", "noise", "Noised actions", f"x_t · {hz}×{adim}", col=3, lane=3,
             tokens=min(hz, 16), seqlen=hz, protect="the noised action chunk")
        node("aproj", "rect", "noise", "Linear + PosEmb", f"{adim}→{d} · +pos", col=4, lane=3,
             protect="input_proj: makes the action tokens")
        node("atok", "tokencol", "noise", "action tokens", f"{hz}×{d}", col=5, lane=3,
             tokens=min(hz, 16), seqlen=hz, protect="the DiT token sequence")
        edge("noise", "aproj", f"{hz}×{adim}")
        edge("aproj", "atok", f"{hz}×{d}")
        node("dit", "tall", "dit", "DiT blocks",
             f"×{n_layers} · self-attn+MLP · adaLN-Zero · {heads}h", col=6, lane=2,
             layers=n_layers, tokens=min(hz, 16), editHp={"n_layers": 1},
             protect="the DiT trunk (transformer)",
             expand=_expand_dit_block(n_layers, d, heads, 4 * d))
        edge("atok", "dit", f"{hz}×{d} tokens")                 # the token sequence (data path)
        edge("cond", "dit", f"adaLN γ,β,α · {d}-D", "adaln")    # modulation, NOT a token merge
        node("final", "round", "dit", "Final layer", f"adaLN + Linear {d}→{adim}", col=7, lane=2,
             protect="final adaLN projection → predicted noise ε",
             expand=_expand_final(d, adim))
        edge("dit", "final", f"{hz}×{d}")
        edge("cond", "final", "adaLN", "adaln")
        node("loop", "round", "diffusion", "Denoising loop",
             f"{hp.get('num_inference_steps',10)} steps · x_T→x_0", col=8, lane=2, loop=True,
             viz="diffusion", steps=hp.get("num_inference_steps", 10), protect="sampling loop")
        edge("final", "loop", "ε̂", "loop")
        node("out", "actions", "action", "Action chunk", f"{hz}×{adim}", col=9, lane=2,
             tokens=min(adim, 12), protect="policy output")
        edge("loop", "out", f"{hz}×{adim}")

    else:  # pi0 / pi0.5 — PaliGemma VLM + Gemma action expert + flow matching.
        # π₀.₅ differs from π₀ by: knowledge insulation (the VLM is co-trained on
        # DISCRETE FAST action tokens, gradients insulated from the flow expert),
        # the robot state is folded into the LANGUAGE PROMPT (discretised) rather
        # than a separate continuous token, and a longer tokenizer (200 vs 48).
        p05 = (arch == "pi0.5")
        n_cam = max(1, len(cams))
        tok_len = hp.get("tokenizer_max_length", 200 if p05 else 48)
        for i, c in enumerate(cams):
            node(f"in_{c}", "image", "vision", c, f"RGB {img[1]}×{img[0]}", col=0, lane=i,
                 camera=c, deletable=True, protect=None)
            node(f"vit_{c}", "trapezoid", "vision", "SigLIP ViT", "patch embed · 256 tok", col=1, lane=i,
                 tokens=256, protect="vision encoder",
                 expand=_expand_transformer(12, d, 8, 4 * d))
            edge(f"in_{c}", f"vit_{c}", f"3×{img[0]}×{img[1]}")
        # language prompt (π₀.₅ discretises the STATE into the prompt too)
        node("in_text", "text", "text", "Task prompt",
             (f"language + state ≤{tok_len} tok" if p05 else f"language ≤{tok_len} tok"),
             col=0, lane=n_cam, deletable=True, protect=None)
        node("prefix", "tall", "vla_backbone", "PaliGemma backbone",
             f"{hp.get('paligemma_variant','gemma_2b')} · image+text → prefix",
             col=2, lane=0, tokens=16, protect="the VLA backbone",
             expand=_expand_transformer(18, d, 8, 4 * d))
        for c in cams:
            edge(f"vit_{c}", "prefix", "256×d", "stack")
        edge("in_text", "prefix", "text tok", "stack")

        ch = hp.get("chunk_size", 50)
        pad = max(hp.get("max_action_dim", 32), adim)
        # STATE path differs: π₀ = separate continuous state token; π₀.₅ = in prompt
        if not p05:
            node("in_prop", "text", "proprio", "Proprioception", f"{sdim}-D", col=0, lane=n_cam + 1,
                 protect="required")
            node("state_tok", "tokencol", "proprio", "state token", "1 · continuous", col=2, lane=2, tokens=1)
            edge("in_prop", "state_tok", f"{sdim}")
        else:
            node("in_prop", "text", "proprio", "Proprioception", f"{sdim}-D → discretised", col=0, lane=n_cam + 1,
                 protect="π₀.₅ tokenises state into the prompt")
            edge("in_prop", "in_text", "state tokens", "stack")

        node("expert", "tall", "flow", "Action expert",
             f"{hp.get('action_expert_variant','gemma_300m')} · chunk {ch} · pad→{pad}", col=3, lane=0,
             tokens=min(ch, 16), protect="the action expert",
             expand=_expand_transformer(6, d, 8, 4 * d))
        node("noise", "tokencol", "noise", "noised actions", f"x_τ ~ 𝒩 · {ch}", col=3, lane=2, tokens=min(ch, 12),
             protect="flow prior")
        edge("prefix", "expert", "prefix (K,V)", "cross_attn")
        if not p05:
            edge("state_tok", "expert", "state", "stack")
        edge("noise", "expert", f"{ch}×{pad}")
        # π₀.₅ knowledge-insulation co-training head: the VLM also predicts DISCRETE
        # FAST action tokens (auxiliary), with gradients insulated from the expert.
        if p05:
            node("fast", "round", "text", "FAST token head",
                 "discrete action tokens · co-train", col=3, lane=4, protect="knowledge insulation")
            edge("prefix", "fast", "knowledge insulation", "adaln")
            node("fast_out", "tokencol", "text", "discrete actions", "aux loss", col=4, lane=4, tokens=6)
            edge("fast", "fast_out", "FAST")
        node("flow", "round", "flow", "Flow matching", f"{hp.get('num_inference_steps',10)} Euler · τ:0→1",
             col=4, lane=0, loop=True, viz="flow", steps=hp.get("num_inference_steps", 10),
             protect="flow integration")
        edge("expert", "flow", "vₜ", "loop")
        node("out", "actions", "action", "Actions", f"{ch}×{adim}", col=5, lane=0,
             tokens=min(adim, 12), protect="policy output")
        edge("flow", "out", f"{ch}×{adim}")

    return {
        "arch": arch, "policy_type": A["policy_type"], "label": A["label"],
        "decode": A["decode"], "paper": A["paper"],
        "inputs": {"cameras": cams, "proprio_dim": sdim, "image_shape": img,
                   "state_indices": sel.get("state_indices", []),
                   "joint_names": sel.get("joint_names", [])},
        "output": {"action_dim": adim, "action_indices": sel.get("action_indices", [])},
        "groups": _module_groups(),   # {name,label,indices} → per-module line colors
        "nodes": N, "edges": E,
        "introspection": introspection,
    }


# Global joint-index ranges per MABEL module, so the diagram's proprio-in and
# action-out line graphs can be colour-graded by module (arm / hand / base / …).
_GROUP_LABELS = {
    "base": "Base", "lift_torso": "Lift + torso", "neck": "Neck",
    "left_arm": "Left arm", "left_hand": "Left hand",
    "right_arm": "Right arm", "right_hand": "Right hand",
}
def _module_groups():
    try:
        from . import embodiment as EMB
        gi = EMB.group_indices()
        return [{"name": g, "label": _GROUP_LABELS.get(g, g), "indices": gi[g]}
                for g in EMB.GROUP_ORDER if gi.get(g)]
    except Exception:
        return []
