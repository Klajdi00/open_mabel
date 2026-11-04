"""
Architecture COMPILER — the ground truth for the diagram.

Instead of trusting the hand-authored ``arch_spec`` graph, this module instantiates
the REAL LeRobot policy ``nn.Module`` for a given (arch, hyperparameters, feature
selection) and walks its module tree, mapping every ``nn.Module`` onto a shared
vocabulary of ELEMENTARY BUILDING BLOCKS. The result is a nested graph in the exact
node/edge format the viewer renders — so the picture is compiled from the object
that trains, and cannot silently drift.

Elementary building blocks (the atoms — a module maps to one of these, or is a
container built out of them):

    PARAMETRIC (carry weights)
      linear   nn.Linear / *Linear            y = Wx + b
      conv     nn.Conv{1,2,3}d                convolution
      convT    nn.ConvTranspose*d             transposed / up-conv
      norm     LayerNorm/BatchNorm/GroupNorm/RMSNorm   normalisation
      embed    nn.Embedding / learned pos-emb lookup / position table

    NON-PARAMETRIC (shape / nonlinearity / routing)
      act      ReLU/GELU/SiLU/Tanh/Sigmoid/GLU/Mish    activation
      softmax  nn.Softmax                     attention / class weights
      pool     Max/Avg/AdaptiveAvgPool        spatial downsample
      dropout  nn.Dropout*                    regularisation (identity at eval)
      matmul   bmm / einsum (Q·Kᵀ, ·V)         batched matrix multiply
      residual add / skip (⊕)                 residual connection
      concat   cat / stack                    join token / feature streams
      reshape  flatten / view / patchify      tensor reshaping
      scale    FiLM / adaLN (γ·x+β)           feature-wise modulation
      posemb   sinusoidal position embedding  add position information

    SEMI-COMPOSITE, kept atomic unless expanded
      attention  nn.MultiheadAttention        (Q,K,V proj + matmul + softmax + O)

Containers (nn.Sequential / nn.ModuleList / any module WITH children) are drawn as
a nested block whose ``expand`` holds its children chained in registration order.
Repeated identical children (a ModuleList of N transformer layers) collapse to one
block tagged ``×N``.
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── the elementary vocabulary ─────────────────────────────────────────────────
# nn.Module class name → (elementary prim, friendly label). This is the exhaustive
# torch.nn (+ torchvision + LeRobot custom) coverage; anything NOT here falls to
# the NAME-HEURISTIC below, and anything the heuristic can't classify is reported
# in ``unknown_modules`` so coverage gaps are visible instead of silent.
_LEAF = {
    # linear / fully-connected
    "Linear": ("linear", "Linear"), "LazyLinear": ("linear", "Linear"),
    "NonDynamicallyQuantizableLinear": ("linear", "Linear"), "Bilinear": ("linear", "Bilinear"),
    # convolutions
    "Conv1d": ("conv", "Conv1d"), "Conv2d": ("conv", "Conv2d"), "Conv3d": ("conv", "Conv3d"),
    "LazyConv2d": ("conv", "Conv2d"),
    "ConvTranspose1d": ("convT", "ConvT1d"), "ConvTranspose2d": ("convT", "ConvT2d"),
    "ConvTranspose3d": ("convT", "ConvT3d"),
    # normalisation
    "LayerNorm": ("norm", "LayerNorm"), "RMSNorm": ("norm", "RMSNorm"),
    "BatchNorm1d": ("norm", "BatchNorm"), "BatchNorm2d": ("norm", "BatchNorm"),
    "BatchNorm3d": ("norm", "BatchNorm"), "SyncBatchNorm": ("norm", "SyncBN"),
    "GroupNorm": ("norm", "GroupNorm"), "InstanceNorm1d": ("norm", "InstanceNorm"),
    "InstanceNorm2d": ("norm", "InstanceNorm"), "LocalResponseNorm": ("norm", "LRN"),
    "FrozenBatchNorm2d": ("norm", "FrozenBN"),
    # activations
    "ReLU": ("act", "ReLU"), "ReLU6": ("act", "ReLU6"), "GELU": ("act", "GELU"),
    "SiLU": ("act", "SiLU"), "Tanh": ("act", "Tanh"), "Sigmoid": ("act", "Sigmoid"),
    "GLU": ("act", "GLU"), "Mish": ("act", "Mish"), "ELU": ("act", "ELU"),
    "LeakyReLU": ("act", "LeakyReLU"), "PReLU": ("act", "PReLU"),
    "Hardswish": ("act", "Hardswish"), "Hardsigmoid": ("act", "Hardsigmoid"),
    "Hardtanh": ("act", "Hardtanh"), "Softplus": ("act", "Softplus"),
    "Softsign": ("act", "Softsign"), "SELU": ("act", "SELU"), "CELU": ("act", "CELU"),
    "Softmax": ("softmax", "Softmax"), "LogSoftmax": ("softmax", "LogSoftmax"),
    "Softmax2d": ("softmax", "Softmax2d"),
    # pooling
    "MaxPool1d": ("pool", "MaxPool"), "MaxPool2d": ("pool", "MaxPool"), "MaxPool3d": ("pool", "MaxPool"),
    "AvgPool1d": ("pool", "AvgPool"), "AvgPool2d": ("pool", "AvgPool"), "AvgPool3d": ("pool", "AvgPool"),
    "AdaptiveAvgPool1d": ("pool", "AdaptiveAvgPool"), "AdaptiveAvgPool2d": ("pool", "AdaptiveAvgPool"),
    "AdaptiveAvgPool3d": ("pool", "AdaptiveAvgPool"), "AdaptiveMaxPool2d": ("pool", "AdaptiveMaxPool"),
    # regularisation
    "Dropout": ("dropout", "Dropout"), "Dropout1d": ("dropout", "Dropout"),
    "Dropout2d": ("dropout", "Dropout"), "Dropout3d": ("dropout", "Dropout"),
    "AlphaDropout": ("dropout", "AlphaDropout"), "DropPath": ("dropout", "DropPath"),
    # embeddings / position
    "Embedding": ("embed", "Embedding"), "EmbeddingBag": ("embed", "EmbeddingBag"),
    "SinusoidalPosEmb": ("posemb", "Sinusoidal PosEmb"),
    "ACTSinusoidalPositionEmbedding2d": ("posemb", "Sinusoidal PosEmb 2D"),
    # recurrent
    "LSTM": ("recurrent", "LSTM"), "GRU": ("recurrent", "GRU"), "RNN": ("recurrent", "RNN"),
    "LSTMCell": ("recurrent", "LSTMCell"), "GRUCell": ("recurrent", "GRUCell"),
    # shape / routing
    "Identity": ("reshape", "Identity (no-op)"), "Flatten": ("reshape", "Flatten"),
    "Unflatten": ("reshape", "Unflatten"), "PixelShuffle": ("reshape", "PixelShuffle"),
    "Upsample": ("reshape", "Upsample"), "UpsamplingBilinear2d": ("reshape", "Upsample"),
    "UpsamplingNearest2d": ("reshape", "Upsample"),
    "ZeroPad2d": ("reshape", "ZeroPad"), "ConstantPad1d": ("reshape", "Pad"),
    "ConstantPad2d": ("reshape", "Pad"), "ReflectionPad2d": ("reshape", "Pad"),
}
# containers whose children are a repeated stack (collapse to ×N)
_STACK = {"ModuleList", "Sequential"}

# name-based heuristic for CUSTOM leaf modules not in the table — ordered by
# specificity (checked as case-insensitive substrings of the class name).
_NAME_HINTS = [
    ("posemb", "posemb"), ("positionembedding", "posemb"), ("positionalencoding", "posemb"),
    ("posencoding", "posemb"), ("rotary", "posemb"),
    ("attention", "attention"), ("attn", "attention"),
    ("norm", "norm"), ("dropout", "dropout"), ("droppath", "dropout"),
    ("softmax", "softmax"), ("pool", "pool"), ("conv", "conv"),
    ("linear", "linear"), ("dense", "linear"), ("mlp", "linear"), ("proj", "linear"),
    ("embed", "embed"), ("activation", "act"), ("gelu", "act"), ("relu", "act"), ("silu", "act"),
    ("lstm", "recurrent"), ("gru", "recurrent"),
    ("film", "scale"), ("modulat", "scale"), ("adaln", "scale"),
    ("flatten", "reshape"), ("patchify", "reshape"), ("rearrange", "reshape"),
]

MAX_NODES = 2500     # hard cap so a giant model can't hang the UI
MAX_DEPTH = 10       # deepest nesting rendered (PaliGemma q/k/v proj ≈ depth 8)


def _params(m):
    try:
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return 0


def _fmt_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _leaf_sub(cls, m):
    """A short shape descriptor for an elementary layer — the numbers the manual
    connector + auto-adapter use, so keep the `in→out` convention consistent."""
    try:
        if hasattr(m, "in_features") and hasattr(m, "out_features"):
            return f"{m.in_features}→{m.out_features}"
        if hasattr(m, "in_channels") and hasattr(m, "out_channels"):
            k = getattr(m, "kernel_size", None)
            k = k[0] if isinstance(k, tuple) else k
            return f"{m.in_channels}→{m.out_channels}" + (f" · {k}×{k}" if k else "")
        if cls in ("LayerNorm", "RMSNorm"):
            ns = getattr(m, "normalized_shape", None)
            return f"{ns[0] if ns else ''}"
        if hasattr(m, "num_features"):
            return f"{m.num_features}ch"
        if hasattr(m, "num_channels"):
            return f"{m.num_channels}ch"
        if cls in ("Embedding", "EmbeddingBag"):
            return f"{m.num_embeddings}×{m.embedding_dim}"
        if cls in ("LSTM", "GRU", "RNN"):
            return f"{m.input_size}→{m.hidden_size}" + (f" ×{m.num_layers}" if m.num_layers > 1 else "")
        if hasattr(m, "p") and isinstance(getattr(m, "p"), float):
            return f"p={m.p}"
        if hasattr(m, "kernel_size"):
            k = m.kernel_size
            k = k[0] if isinstance(k, tuple) else k
            return f"{k}×{k}"
    except Exception:
        pass
    return ""


class _Walk:
    """Per-compile bookkeeping: node budget + unknown-module report."""

    def __init__(self):
        self.count = 0
        self.unknown = set()
        self.truncated = False


def _heuristic_prim(cls):
    low = cls.lower()
    for sub, prim in _NAME_HINTS:
        if sub in low:
            return prim
    return None


def _leaf_node(nid, module, cls, walk, name=None):
    """An elementary (atom) node — known table, name heuristic, or reported unknown.
    Keeps the module's registered NAME as the label (so fifteen Linears don't all
    read "Linear"); the layer type + dims live in the sub line."""
    if cls in _LEAF:
        prim, plabel = _LEAF[cls]
    else:
        prim = _heuristic_prim(cls)
        if prim is None:
            walk.unknown.add(cls)
            prim, plabel = "reshape", cls
        else:
            plabel = cls
    sub = _leaf_sub(cls, module)
    p = _params(module)
    if p >= 1000:                       # param count on heavy leaves
        sub = (sub + " · " if sub else "") + _fmt_params(p)
    label = plabel
    if name and not name.isdigit():     # a meaningful registered name → use it
        label = name.replace("_", " ")
        sub = (plabel + (" · " + sub if sub else ""))
    return {"id": nid, "shape": "layer", "prim": prim, "label": label, "sub": sub}


def _mha_node(nid, m, name, walk):
    """nn.MultiheadAttention → an attention block whose expand is the TRUE internal
    computation (packed W_QKV projection → Q·Kᵀ/√dₖ → softmax → ·V → W_O). The
    q/k/v weights are packed nn.Parameters (not child modules), so we synthesise
    the elementary subgraph from the module's real dims."""
    d, h = m.embed_dim, m.num_heads
    dk = max(1, d // h)
    walk.count += 8
    inner = [
        {"id": f"{nid}_wq", "shape": "layer", "prim": "linear", "label": "W_Q", "sub": f"{d}→{d}"},
        {"id": f"{nid}_wk", "shape": "layer", "prim": "linear", "label": "W_K", "sub": f"{d}→{d}"},
        {"id": f"{nid}_wv", "shape": "layer", "prim": "linear", "label": "W_V", "sub": f"{d}→{d}"},
        {"id": f"{nid}_qk", "shape": "layer", "prim": "matmul", "label": "Q·Kᵀ / √dₖ", "sub": f"{h}h · dₖ={dk}"},
        {"id": f"{nid}_sm", "shape": "layer", "prim": "softmax", "label": "Softmax", "sub": "attention weights"},
        {"id": f"{nid}_av", "shape": "layer", "prim": "matmul", "label": "weights · V", "sub": "context"},
        {"id": f"{nid}_wo", "shape": "layer", "prim": "linear", "label": "W_O", "sub": f"{d}→{d}"},
    ]
    edges = [
        {"from": f"{nid}_wq", "to": f"{nid}_qk", "kind": "data", "tensor": "Q"},
        {"from": f"{nid}_wk", "to": f"{nid}_qk", "kind": "data", "tensor": "Kᵀ"},
        {"from": f"{nid}_qk", "to": f"{nid}_sm", "kind": "data"},
        {"from": f"{nid}_sm", "to": f"{nid}_av", "kind": "data", "tensor": "A"},
        {"from": f"{nid}_wv", "to": f"{nid}_av", "kind": "data", "tensor": "V"},
        {"from": f"{nid}_av", "to": f"{nid}_wo", "kind": "data"},
    ]
    _layer_cols(inner, edges)
    return {"id": nid, "shape": "round", "modality": "attention",
            "label": name or "Multi-Head Attention", "sub": f"{h}h · d{d} · {_fmt_params(_params(m))}",
            "expand": {"nodes": inner, "edges": edges, "title": f"{name or 'MHA'} (compiled)"}}


def _node(nid, module, name, walk, depth=0):
    """Turn one nn.Module into a diagram node (leaf or nested container)."""
    cls = type(module).__name__
    walk.count += 1
    if cls == "MultiheadAttention":
        return _mha_node(nid, module, name, walk)
    children = list(module.named_children())
    label = name or cls

    if not children or cls in _LEAF:      # elementary leaf (atoms never expand)
        return _leaf_node(nid, module, cls, walk, name)

    # depth / node-budget guard: render as a summarised block, don't recurse
    if depth >= MAX_DEPTH or walk.count > MAX_NODES:
        walk.truncated = True
        return {"id": nid, "shape": "round", "modality": _MOD.get(cls, "trunk"),
                "label": label, "sub": f"{cls} · {_fmt_params(_params(module))} params (collapsed)"}

    # container → build the inner subgraph from its children (registration order)
    inner_nodes, inner_edges = [], []
    kids = children
    stacked = cls in _STACK and len(kids) > 1 and _all_same(kids)
    if stacked:
        kids = kids[:1]     # collapse the repeated stack to one representative
    prev = None
    for cname, child in kids:
        cid = f"{nid}_{cname}"
        try:
            sub = _node(cid, child, cname, walk, depth + 1)
        except Exception as e:            # never let one child kill the whole compile
            sub = {"id": cid, "shape": "layer", "prim": "reshape",
                   "label": type(child).__name__, "sub": f"walk error: {e}"}
            walk.unknown.add(type(child).__name__)
        inner_nodes.append(sub)
        if prev is not None:
            inner_edges.append({"from": prev, "to": cid, "kind": "data"})
        prev = cid
    _layer_cols(inner_nodes, inner_edges)
    n_rep = len(children) if stacked else 0
    return {"id": nid, "shape": "round", "modality": _MOD.get(cls, "trunk"),
            "label": label + (f" ×{n_rep}" if n_rep else ""),
            "sub": f"{cls} · {_fmt_params(_params(module))} params",
            "expand": {"nodes": inner_nodes, "edges": inner_edges, "title": f"{label} (compiled)"}}


def _all_same(kids):
    t = type(kids[0][1]).__name__
    return all(type(c).__name__ == t for _, c in kids)


# colour hint for known container classes (falls back to trunk)
_MOD = {
    "IntermediateLayerGetter": "vision", "ResNet": "vision", "BasicBlock": "vision",
    "Bottleneck": "vision", "DiffusionRgbEncoder": "vision", "SpatialSoftmax": "vision",
    "Conv2dNormActivation": "vision",
    "ACTEncoder": "trunk", "ACTDecoder": "trunk", "ACTEncoderLayer": "trunk",
    "ACTDecoderLayer": "trunk", "TransformerEncoder": "trunk", "TransformerDecoder": "trunk",
    "TransformerEncoderLayer": "trunk", "TransformerDecoderLayer": "trunk",
    "DiTDenoiser": "dit", "DiTBlock": "dit", "FinalLayer": "dit",
    "DiffusionConditionalUnet1d": "diffusion",
    "Sequential": "trunk", "ModuleList": "trunk",
}


def _layer_cols(nodes, edges):
    """Left→right columns by longest-path (mirror of arch_registry._layer_cols)."""
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


# ── build the real nn.Module for an arch (headless, no pretrained weights) ──────
def _build_module(arch, hp, sel):
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    cams = sel.get("cameras") or ["cam"]
    sdim = int(sel.get("obs_state_dim", 14))
    adim = int(sel.get("action_dim", sdim))
    img = (3, 240, 320)

    def _feats(cfg):
        cfg.input_features = {f"observation.images.{c}": PolicyFeature(FeatureType.VISUAL, img) for c in cams}
        cfg.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (sdim,))
        cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (adim,))}
        cfg.normalization_mapping = {k: NormalizationMode.MEAN_STD for k in ("VISUAL", "STATE", "ACTION")}
        return cfg

    if arch == "ACT":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACT
        cfg = _feats(ACTConfig(
            chunk_size=hp.get("chunk_size", 100), n_action_steps=hp.get("n_action_steps", 100),
            dim_model=hp.get("dim_model", 512), n_heads=hp.get("n_heads", 8),
            dim_feedforward=hp.get("dim_feedforward", 3200),
            n_encoder_layers=hp.get("n_encoder_layers", 4), n_decoder_layers=hp.get("n_decoder_layers", 1),
            use_vae=hp.get("use_vae", True), latent_dim=hp.get("latent_dim", 32),
            vision_backbone=hp.get("vision_backbone", "resnet18"),
            # None → no ImageNet download, and the weights enum default is resnet18-
            # specific (KeyError on resnet34/50). Structure is identical either way.
            pretrained_backbone_weights=None,
            feedforward_activation=hp.get("feedforward_activation", "relu"),
            pre_norm=hp.get("pre_norm", False), dropout=hp.get("dropout", 0.1)))
        return ACT(cfg)
    if arch == "DiT":
        from .lerobot_shim.dit_policy import DiTConfig
        from .lerobot_shim.dit_policy.modeling_dit import DiTDiffusionModel
        cfg = _feats(DiTConfig(
            horizon=hp.get("horizon", 16), n_action_steps=hp.get("n_action_steps", 8),
            n_obs_steps=hp.get("n_obs_steps", 2), dim_model=hp.get("dim_model", 384),
            n_heads=hp.get("n_heads", 6), n_layers=hp.get("n_layers", 6),
            vision_backbone=hp.get("vision_backbone", "resnet18")))
        return DiTDiffusionModel(cfg)
    if arch in ("pi0", "pi0.5"):
        # The 3B-param VLA builds on META tensors (accelerate.init_empty_weights):
        # full module STRUCTURE, zero memory. Params still report numel, so counts
        # and the layer tree are exact — only the weights are placeholders.
        from accelerate import init_empty_weights
        if arch == "pi0":
            from lerobot.policies.pi0.configuration_pi0 import PI0Config as Cfg
            from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch as Net
        else:
            from lerobot.policies.pi05.configuration_pi05 import PI05Config as Cfg
            from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch as Net
        cfg = _feats(Cfg(
            chunk_size=hp.get("chunk_size", 50), n_action_steps=hp.get("n_action_steps", 50),
            max_state_dim=max(hp.get("max_state_dim", 32), sdim),
            max_action_dim=max(hp.get("max_action_dim", 32), adim),
            num_inference_steps=hp.get("num_inference_steps", 10)))
        with init_empty_weights(include_buffers=False):
            # LeRobot's pi0/pi05 __init__ ends with a check for its PATCHED
            # transformers fork ("transformers_replace") and raises on stock
            # transformers — AFTER the full module tree is already registered.
            # Build via __new__ + __init__ and tolerate exactly that error: the
            # structure is complete and exact; only training would need the fork.
            net = Net.__new__(Net)
            try:
                net.__init__(cfg)
            except ValueError as e:
                if "transformer version" not in str(e):
                    raise
            return net
    raise NotImplementedError(f"{arch}: no headless builder registered")


# ── top-level DATAFLOW wiring per architecture (read from each forward(); the
#    module tree alone gives structure but not the graph — this restores it).
#    Pairs reference the REGISTERED child names; missing names are dropped, and
#    an arch without a table falls back to a registration-order chain. ──────────
_TOP_WIRING = {
    "ACT": [
        # CVAE style encoder: [CLS, state, actions] → vae_encoder → μ,logσ → z proj
        ("vae_encoder_cls_embed", "vae_encoder"),
        ("vae_encoder_robot_state_input_proj", "vae_encoder"),
        ("vae_encoder_action_input_proj", "vae_encoder"),
        ("vae_encoder", "vae_encoder_latent_output_proj"),
        ("vae_encoder_latent_output_proj", "encoder_latent_input_proj"),
        # vision: backbone → 1×1 proj (+2D sinusoidal pos) → encoder
        ("backbone", "encoder_cam_feat_pos_embed"),
        ("backbone", "encoder_img_feat_input_proj"),
        ("encoder_cam_feat_pos_embed", "encoder"),
        ("encoder_img_feat_input_proj", "encoder"),
        # state + latent + 1D pos tokens → encoder
        ("encoder_robot_state_input_proj", "encoder"),
        ("encoder_latent_input_proj", "encoder"),
        ("encoder_1d_feature_pos_embed", "encoder"),
        # trunk: encoder → decoder (+ decoder pos) → action head
        ("encoder", "decoder"),
        ("decoder_pos_embed", "decoder"),
        ("decoder", "action_head"),
    ],
    "DiT": [("rgb_encoder", "unet")],
    "pi0": [
        ("state_proj", "paligemma_with_expert"),
        ("action_in_proj", "action_time_mlp_in"),
        ("action_time_mlp_in", "action_time_mlp_out"),
        ("action_time_mlp_out", "paligemma_with_expert"),
        ("paligemma_with_expert", "action_out_proj"),
    ],
    "pi0.5": [
        ("state_proj", "paligemma_with_expert"),
        ("action_in_proj", "action_time_mlp_in"),
        ("action_time_mlp_in", "action_time_mlp_out"),
        ("action_time_mlp_out", "paligemma_with_expert"),
        ("paligemma_with_expert", "action_out_proj"),
    ],
}

# top-level style vocabulary: registered-name pattern → designed look (matches the
# hand-authored spec's shapes/colours so compiled + designed views read the same)
_TOP_STYLE = [
    ("backbone", {"modality": "vision", "shape": "trapezoid"}),
    ("rgb_encoder", {"modality": "vision", "shape": "trapezoid"}),
    ("vision", {"modality": "vision", "shape": "trapezoid"}),
    ("vae_encoder", {"modality": "latent", "shape": "tall"}),
    ("vae", {"modality": "latent"}),
    ("paligemma", {"modality": "vla_backbone", "shape": "tall"}),
    ("unet", {"modality": "dit", "shape": "tall"}),
    ("denoiser", {"modality": "dit", "shape": "tall"}),
    ("encoder", {"modality": "trunk", "shape": "tall"}),
    ("decoder", {"modality": "trunk", "shape": "tall"}),
    ("action", {"modality": "action"}),
    ("state", {"modality": "proprio"}),
    ("latent", {"modality": "latent"}),
    ("pos_embed", {"modality": "posemb"}),
    ("time", {"modality": "noise"}),
]


def _style_top(node, cname):
    """Apply the designed-look style to a top-level compiled node by its name."""
    low = cname.lower()
    for pat, style in _TOP_STYLE:
        if pat in low:
            node["modality"] = style.get("modality", node.get("modality", "trunk"))
            # only containers change shape (leaves keep the elementary glyph)
            if "shape" in style and node.get("expand"):
                node["shape"] = style["shape"]
            return
    return


_CACHE = {}          # (arch, hp-signature, sel-signature) → compiled result
_CACHE_MAX = 8


def _cache_key(arch, hp, sel):
    import json
    h = {k: v for k, v in sorted(hp.items()) if not isinstance(v, (dict, list))}
    s = {"cams": len(sel.get("cameras") or []), "sdim": sel.get("obs_state_dim"),
         "adim": sel.get("action_dim")}
    return json.dumps([arch, h, s], sort_keys=True, default=str)


def compile_spec(arch, hp, sel):
    """Compile the real policy module → a diagram graph. Returns
    {ok, nodes, edges, params, unknown_modules, note} or {ok:False, reason}."""
    key = _cache_key(arch, hp, sel)
    if key in _CACHE:
        return _CACHE[key]
    try:
        model = _build_module(arch, hp, sel)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        walk = _Walk()
        top = list(model.named_children())
        nodes, edges = [], []
        names = set()
        for cname, child in top:
            cid = f"c_{cname}"
            sub = _node(cid, child, cname, walk)
            _style_top(sub, cname)          # designed-look shapes/colours by name
            nodes.append(sub)
            names.add(cname)
        # REAL dataflow wiring (from the arch's forward()); chain fallback otherwise
        wiring = [(a, b) for a, b in _TOP_WIRING.get(arch, []) if a in names and b in names]
        if wiring:
            edges = [{"from": f"c_{a}", "to": f"c_{b}", "kind": "data"} for a, b in wiring]
        else:
            prev = None
            for cname, _ in top:
                cid = f"c_{cname}"
                if prev is not None:
                    edges.append({"from": prev, "to": cid, "kind": "data"})
                prev = cid
        _layer_cols(nodes, edges)
        out = {"ok": True, "arch": arch, "nodes": nodes, "edges": edges,
               "params": _params(model), "params_str": _fmt_params(_params(model)),
               "n_modules": walk.count, "truncated": walk.truncated,
               "unknown_modules": sorted(walk.unknown),
               "note": "compiled from the real nn.Module (registration order; residual wiring inside blocks)"
                       + (" · META weights (structure exact, no allocation)" if arch.startswith("pi0") else "")}
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
        del model
        return out
    except Exception as e:
        import traceback
        return {"ok": False, "reason": f"walk failed: {e}", "trace": traceback.format_exc()[-500:]}
