// Architecture diagram — a clean, interactive, Simulink-style editor.
//
// Renders a modality-decomposed graph (reference IRIS / pi0.5 style): light
// pastel colors per modality, distinct SHAPES per block type (trapezoid CNN,
// tall transformer, rounded MLP, token columns of tiny pills, image inputs,
// action pills), tensor shapes on wires, cross-attention, and tokens streaming
// DOWN into the transformer. Blocks are draggable (wires stay connected), the
// canvas pans + zooms, blocks can be deleted (protected ones prompt), layers can
// be added/removed, and composite blocks (ResNet, transformer) open inside with
// a breadcrumb — just like Simulink subsystems.

const MOD = {
  vision:       { f: '#AFD3EC', d: '#5E86A6', ink: '#1e3040', name: 'vision' },
  proprio:      { f: '#BEE3A6', d: '#7CB05E', ink: '#213316', name: 'proprioception' },
  latent:       { f: '#F4B9B2', d: '#D98177', ink: '#3f1d1a', name: 'VAE / latent' },
  text:         { f: '#F3C8DB', d: '#D68BAE', ink: '#3f1d2e', name: 'text' },
  trunk:        { f: '#DAC5EE', d: '#A97FCB', ink: '#2c1f3c', name: 'transformer' },
  attention:    { f: '#ECCDF2', d: '#C08FD0', ink: '#381f40', name: 'attention' },
  action:       { f: '#BEE3A6', d: '#7CB05E', ink: '#213316', name: 'action' },
  noise:        { f: '#E4E1D6', d: '#B0AC9E', ink: '#33302a', name: 'noise' },
  dit:          { f: '#CDBBEE', d: '#9578C8', ink: '#2c1f3c', name: 'DiT block' },
  diffusion:    { f: '#C3CAEE', d: '#8E97D0', ink: '#1f2340', name: 'diffusion' },
  flow:         { f: '#ADE3D3', d: '#5FB39C', ink: '#153630', name: 'flow matching' },
  vla_backbone: { f: '#EDDCAB', d: '#C9A85B', ink: '#3a3016', name: 'VLA backbone' },
  audio:        { f: '#C9E4CA', d: '#79B37F', ink: '#1e3320', name: 'audio' },
  goal:         { f: '#F0D0A8', d: '#D0A566', ink: '#3a2c14', name: 'goal cond' },
};
// INPUT modality blocks (sources) — the realistic conditioning inputs for
// ACT / DiT / VLA policies. OUTPUT blocks (sinks) — what a policy can emit.
const INPUT_BLOCKS = [
  { id: 'in_cam', label: 'Camera / video', modality: 'vision', shape: 'image', io: 'in', sub: 'RGB stream' },
  { id: 'in_goalimg', label: 'Goal image', modality: 'goal', shape: 'image', io: 'in', sub: 'goal-conditioned' },
  { id: 'in_lang', label: 'Task language', modality: 'text', io: 'in', sub: 'instruction (VLA)' },
  { id: 'in_goallang', label: 'Goal language', modality: 'goal', io: 'in', sub: 'goal-conditioned' },
  { id: 'in_prop', label: 'Proprioception', modality: 'proprio', io: 'in', sub: 'joint state', prompt: 'joints' },
  { id: 'in_goalprop', label: 'Goal state', modality: 'goal', io: 'in', sub: 'goal joint pose', prompt: 'joints' },
  { id: 'in_depth', label: 'Depth / point cloud', modality: 'vision', shape: 'image', io: 'in', sub: '3D sensing' },
  { id: 'in_audio', label: 'Audio', modality: 'audio', io: 'in', sub: 'mic stream' },
];
const OUTPUT_BLOCKS = [
  { id: 'out_action', label: 'Action chunk', modality: 'action', shape: 'actions', io: 'out', sub: 'joint targets' },
  { id: 'out_lang', label: 'Language output', modality: 'text', io: 'out', sub: 'reasoning / text (VLA)', accepts: ['trunk', 'vla_backbone', 'text'] },
  { id: 'out_value', label: 'Value / reward', modality: 'trunk', io: 'out', sub: 'critic head', accepts: ['trunk', 'proprio', 'action'] },
  { id: 'out_gripper', label: 'Discrete / gripper', modality: 'action', io: 'out', sub: 'discrete action', accepts: ['trunk', 'action', 'proprio'] },
];
// PointNet encoder internals — TRUTHFUL to Qi et al. 2017 (Fig. 2): input
// T-Net(3×3) → shared MLP(64,64) → feature T-Net(64×64) → shared MLP(64,128,1024)
// → max-pool over the n points (the symmetric function) → global 1024-D feature.
// Shared MLPs are per-point Conv1d(k=1)+BN+ReLU exactly as in the reference code.
function pointnetTnet(k) {
  return { title: `T-Net (${k}×${k} transform)`, nodes: [
    { id: 't1', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: `${k}→64 · BN · ReLU`, col: 0 },
    { id: 't2', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '64→128 · BN · ReLU', col: 1 },
    { id: 't3', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '128→1024 · BN · ReLU', col: 2 },
    { id: 'tp', shape: 'layer', prim: 'pool', modality: 'vision', label: 'MaxPool', sub: 'over n points', col: 3 },
    { id: 'f1', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'Linear', sub: '1024→512 · ReLU', col: 4 },
    { id: 'f2', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'Linear', sub: '256 → matrix', col: 5 },
    { id: 'mm', shape: 'layer', prim: 'matmul', modality: 'attention', label: '× transform', sub: `apply ${k}×${k}`, col: 6 }],
    edges: [{ from: 't1', to: 't2', kind: 'data' }, { from: 't2', to: 't3', kind: 'data' },
            { from: 't3', to: 'tp', kind: 'data' }, { from: 'tp', to: 'f1', kind: 'data' },
            { from: 'f1', to: 'f2', kind: 'data' }, { from: 'f2', to: 'mm', kind: 'data', tensor: `${k}×${k}` }] };
}
function pointnetExpand(d) {
  return { title: 'PointNet encoder (Qi et al. 2017)', nodes: [
    { id: 'pin', shape: 'rect', modality: 'vision', label: 'points', sub: 'n×3', col: 0 },
    { id: 'tn1', shape: 'round', modality: 'vision', label: 'Input T-Net', sub: '3×3 transform', col: 1, expand: pointnetTnet(3) },
    { id: 'mlp1', shape: 'round', modality: 'vision', label: 'shared MLP (64,64)', sub: 'Conv1d k=1 · BN · ReLU', col: 2,
      expand: { title: 'shared MLP (64,64)', nodes: [
        { id: 'c1', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '3→64', col: 0 },
        { id: 'b1', shape: 'layer', prim: 'norm', modality: 'norm', label: 'BatchNorm', sub: '64', col: 1 },
        { id: 'r1', shape: 'layer', prim: 'act', modality: 'act', label: 'ReLU', col: 2 },
        { id: 'c2', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '64→64', col: 3 },
        { id: 'b2', shape: 'layer', prim: 'norm', modality: 'norm', label: 'BatchNorm', sub: '64', col: 4 },
        { id: 'r2', shape: 'layer', prim: 'act', modality: 'act', label: 'ReLU', col: 5 }],
        edges: [{ from: 'c1', to: 'b1' }, { from: 'b1', to: 'r1' }, { from: 'r1', to: 'c2' },
                { from: 'c2', to: 'b2' }, { from: 'b2', to: 'r2' }].map(e => ({ ...e, kind: 'data' })) } },
    { id: 'tn2', shape: 'round', modality: 'vision', label: 'Feature T-Net', sub: '64×64 transform', col: 3, expand: pointnetTnet(64) },
    { id: 'mlp2', shape: 'round', modality: 'vision', label: 'shared MLP (64,128,1024)', sub: 'Conv1d k=1 · BN · ReLU', col: 4,
      expand: { title: 'shared MLP (64,128,1024)', nodes: [
        { id: 'c1', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '64→64', col: 0 },
        { id: 'c2', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '64→128', col: 1 },
        { id: 'c3', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv1d k=1', sub: '128→1024', col: 2 }],
        edges: [{ from: 'c1', to: 'c2', kind: 'data' }, { from: 'c2', to: 'c3', kind: 'data' }] } },
    { id: 'pool', shape: 'layer', prim: 'pool', modality: 'vision', label: 'MaxPool', sub: 'over n points (symmetric fn)', col: 5 },
    { id: 'glob', shape: 'rect', modality: 'vision', label: 'global feature', sub: '1024-D', col: 6 },
    { id: 'proj', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'Linear', sub: `1024→${d}`, col: 7 }],
    edges: [{ from: 'pin', to: 'tn1' }, { from: 'tn1', to: 'mlp1' }, { from: 'mlp1', to: 'tn2' },
            { from: 'tn2', to: 'mlp2' }, { from: 'mlp2', to: 'pool' }, { from: 'pool', to: 'glob' },
            { from: 'glob', to: 'proj' }].map(e => ({ ...e, kind: 'data' })) };
}

// ✦ AUTO-COMPLETE recipes: given a freshly-added modality block, the chain of
// encoder/decoder stages that correctly wires it into the architecture trunk,
// with dimensions resolved against the CURRENT dim_model (d). `tok` = how many
// tokens the completion adds to the fused sequence (drives the auto-resize).
const AUTO_RECIPES = {
  in_cam:      { dir: 'in', mod: 'vision', tok: 8, band: 'patch',
    steps: (d) => [
      { shape: 'trapezoid', modality: 'vision', label: 'ResNet encoder', sub: `resnet18 → ${d}` },
      { shape: 'tokencol', modality: 'vision', label: 'patch tokens', tokens: 8 }] },
  in_goalimg:  { dir: 'in', mod: 'goal', tok: 8, band: 'goal img',
    steps: (d) => [
      { shape: 'trapezoid', modality: 'goal', label: 'ResNet encoder', sub: `shared weights → ${d}` },
      { shape: 'tokencol', modality: 'goal', label: 'goal-img tokens', tokens: 8 }] },
  in_depth:    { dir: 'in', mod: 'vision', tok: 6, band: 'depth',
    steps: (d) => [
      { shape: 'trapezoid', modality: 'vision', label: 'PointNet encoder', sub: `n×3 → ${d}`,
        expand: pointnetExpand(d) },
      { shape: 'tokencol', modality: 'vision', label: 'depth tokens', tokens: 6 }] },
  in_lang:     { dir: 'in', mod: 'text', tok: 6, band: 'text',
    steps: (d) => [
      { shape: 'rect', modality: 'text', label: 'Tokenizer', sub: 'BPE → ids' },
      { shape: 'layer', prim: 'embed', modality: 'text', label: 'Embedding', sub: `vocab→${d}` },
      { shape: 'tokencol', modality: 'text', label: 'text tokens', tokens: 6 }] },
  in_goallang: { dir: 'in', mod: 'goal', tok: 6, band: 'goal text',
    steps: (d) => [
      { shape: 'rect', modality: 'goal', label: 'Tokenizer', sub: 'BPE → ids' },
      { shape: 'layer', prim: 'embed', modality: 'goal', label: 'Embedding', sub: `vocab→${d}` },
      { shape: 'tokencol', modality: 'goal', label: 'goal tokens', tokens: 6 }] },
  in_prop:     { dir: 'in', mod: 'proprio', tok: 1, band: 'state',
    steps: (d, sdim) => [
      { shape: 'round', modality: 'proprio', label: 'MLP', sub: `${sdim || 'n'}→128→${d}` },
      { shape: 'tokencol', modality: 'proprio', label: 'state token', tokens: 1 }] },
  in_goalprop: { dir: 'in', mod: 'goal', tok: 1, band: 'goal state',
    steps: (d, sdim) => [
      { shape: 'round', modality: 'goal', label: 'MLP', sub: `${sdim || 'n'}→128→${d}` },
      { shape: 'tokencol', modality: 'goal', label: 'goal-state token', tokens: 1 }] },
  in_audio:    { dir: 'in', mod: 'audio', tok: 4, band: 'audio',
    steps: (d) => [
      { shape: 'rect', modality: 'audio', label: 'Mel spectrogram', sub: 'STFT' },
      { shape: 'trapezoid', modality: 'audio', label: 'Conv1d encoder', sub: `mel → ${d}` },
      { shape: 'tokencol', modality: 'audio', label: 'audio tokens', tokens: 4 }] },
  out_lang:    { dir: 'out',
    steps: (d) => [
      { shape: 'layer', prim: 'linear', modality: 'text', label: 'LM head', sub: `${d}→vocab` },
      { shape: 'layer', prim: 'softmax', modality: 'text', label: 'Softmax', sub: 'token dist' }] },
  out_value:   { dir: 'out',
    steps: (d) => [{ shape: 'layer', prim: 'linear', modality: 'trunk', label: 'Value head', sub: `${d}→1` }] },
  out_gripper: { dir: 'out',
    steps: (d) => [
      { shape: 'layer', prim: 'linear', modality: 'action', label: 'Discrete head', sub: `${d}→K` },
      { shape: 'layer', prim: 'softmax', modality: 'action', label: 'Softmax', sub: 'argmax → gripper' }] },
  out_action:  { dir: 'out',
    steps: (d, sdim, adim) => [{ shape: 'layer', prim: 'linear', modality: 'action', label: 'Action head', sub: `${d}→${adim || 'act'}` }] },
};
// per-MODULE colours for the proprio-in / action-out joint line graphs (drawn on
// a black panel, data-curation joint-viz style) — arm/hand/base/neck/torso/lift
// each a distinct hue so the module a line belongs to is unmistakable.
const GROUP_COL = {
  base:       '#6FA8DC',   // steel blue
  lift_torso: '#B084E0',   // violet
  neck:       '#57C2A8',   // teal
  left_arm:   '#E8973C',   // orange
  left_hand:  '#E8C84B',   // gold
  right_arm:  '#5FB85A',   // green
  right_hand: '#E070A8',   // pink
};
// Elementary NN-layer palette — the ATOMS. Every composite (ResNet, MHA, MLP…)
// is built from these, and they render with one consistent glyph + colour so the
// same "Conv" or "Linear" reads identically under ACT, DiT or a VLA.
const PRIM = {
  conv:     { f: '#AFD3EC', d: '#5E86A6', ink: '#1e3040', name: 'Conv' },
  pool:     { f: '#CBE0F0', d: '#7FA6C4', ink: '#22323f', name: 'Pool' },
  linear:   { f: '#BEE3A6', d: '#7CB05E', ink: '#213316', name: 'Linear / FC' },
  norm:     { f: '#E9E2C6', d: '#BBAE7C', ink: '#3a3416', name: 'Norm' },
  act:      { f: '#F4C89A', d: '#D89A5C', ink: '#402c14', name: 'Activation' },
  matmul:   { f: '#ECCDF2', d: '#C08FD0', ink: '#381f40', name: 'MatMul' },
  softmax:  { f: '#DAC5EE', d: '#A97FCB', ink: '#2c1f3c', name: 'Softmax' },
  residual: { f: '#E4E1D6', d: '#B0AC9E', ink: '#33302a', name: 'Residual ⊕' },
  embed:    { f: '#CDE8DE', d: '#5FB39C', ink: '#153630', name: 'Embed' },
  dropout:  { f: '#E0DAD2', d: '#A8A196', ink: '#33302a', name: 'Dropout' },
  // extended atoms surfaced by the nn.Module compiler
  convT:    { f: '#BFDCF0', d: '#6E92AE', ink: '#1e3040', name: 'ConvT' },
  concat:   { f: '#DCD6C8', d: '#A8A08C', ink: '#33302a', name: 'Concat' },
  reshape:  { f: '#E4DFD6', d: '#ABA394', ink: '#33302a', name: 'Reshape' },
  scale:    { f: '#F1D9B8', d: '#CFA76A', ink: '#3a2c14', name: 'Scale / FiLM' },
  attention:{ f: '#ECCDF2', d: '#C08FD0', ink: '#381f40', name: 'Attention' },
  posemb:   { f: '#CFE0DA', d: '#7FA69A', ink: '#153630', name: 'Pos. embed' },
  recurrent:{ f: '#D8C8E8', d: '#9A82BC', ink: '#2c1f3c', name: 'RNN / LSTM' },
};
// the elementary layers offered in the always-on "＋ layer" palette (any level,
// any architecture). label + the config sub-text shown on the inserted block.
const PRIM_MENU = [
  { prim: 'conv', label: 'Conv 2D', sub: '3×3' },
  { prim: 'linear', label: 'Linear', sub: 'fully-connected' },
  { prim: 'norm', label: 'LayerNorm', sub: '' },
  { prim: 'act', label: 'GELU', sub: 'activation' },
  { prim: 'matmul', label: 'MatMul', sub: 'Q·Kᵀ' },
  { prim: 'softmax', label: 'Softmax', sub: '' },
  { prim: 'pool', label: 'MaxPool', sub: '2×2' },
  { prim: 'residual', label: '⊕ Add', sub: 'residual' },
  { prim: 'embed', label: 'Embedding', sub: '' },
  { prim: 'dropout', label: 'Dropout', sub: 'p=0.1' },
];
// Which upstream MODALITIES a layer can validly consume (the ML domain rules
// that drive connection suggestions). '*' = accepts anything (norm/act/residual).
const PRIM_ACCEPTS = {
  conv: ['vision'], pool: ['vision'],
  linear: ['proprio', 'latent', 'trunk', 'action', 'vision'], embed: ['*'],
  norm: ['*'], act: ['*'], dropout: ['*'], residual: ['*'],
  matmul: ['trunk', 'vision', 'attention'], softmax: ['attention', 'trunk'],
};
// Composite blocks offered in the ＋ Add block menu. `accepts` = valid input
// modalities (for suggestions); `expand` = the inner elementary layers (so the
// added block is a real nested block, not a generic box).
const ADD_BLOCKS = [
  { id: 'mlp', label: 'MLP block', modality: 'proprio', accepts: ['proprio', 'latent', 'trunk', 'action'],
    sub: 'Linear·act·Linear',
    expand: { title: 'MLP', nodes: [
      { id: 'l1', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'Linear', sub: 'd→h' },
      { id: 'a', shape: 'layer', prim: 'act', modality: 'act', label: 'GELU' },
      { id: 'l2', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'Linear', sub: 'h→d' }],
      edges: [{ from: 'l1', to: 'a', kind: 'data' }, { from: 'a', to: 'l2', kind: 'data' }] } },
  { id: 'cnn', label: 'CNN / Conv block', modality: 'vision', accepts: ['vision'],
    sub: 'Conv·BN·ReLU',
    expand: { title: 'Conv block', nodes: [
      { id: 'c', shape: 'layer', prim: 'conv', modality: 'vision', label: 'Conv 3×3' },
      { id: 'b', shape: 'layer', prim: 'norm', modality: 'norm', label: 'BatchNorm' },
      { id: 'r', shape: 'layer', prim: 'act', modality: 'act', label: 'ReLU' }],
      edges: [{ from: 'c', to: 'b', kind: 'data' }, { from: 'b', to: 'r', kind: 'data' }] } },
  { id: 'attn', label: 'Attention', modality: 'attention', accepts: ['trunk', 'vision', 'proprio', 'latent'],
    sub: 'multi-head',
    expand: { title: 'Attention', nodes: [
      { id: 'wq', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'W_Q' },
      { id: 'wk', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'W_K' },
      { id: 'wv', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'W_V' },
      { id: 'qk', shape: 'layer', prim: 'matmul', modality: 'attention', label: 'Q·Kᵀ/√d' },
      { id: 'sm', shape: 'layer', prim: 'softmax', modality: 'attention', label: 'Softmax' },
      { id: 'av', shape: 'layer', prim: 'matmul', modality: 'attention', label: '·V' },
      { id: 'wo', shape: 'layer', prim: 'linear', modality: 'proprio', label: 'W_O' }],
      edges: [{ from: 'wq', to: 'qk' }, { from: 'wk', to: 'qk' }, { from: 'qk', to: 'sm' },
              { from: 'sm', to: 'av' }, { from: 'wv', to: 'av' }, { from: 'av', to: 'wo' }] } },
  { id: 'transformer', label: 'Transformer layer', modality: 'trunk', accepts: ['trunk', 'vision', 'proprio', 'latent'],
    sub: 'LN·attn·⊕·LN·MLP·⊕',
    expand: { title: 'Transformer layer', nodes: [
      { id: 'ln1', shape: 'layer', prim: 'norm', modality: 'norm', label: 'LayerNorm' },
      { id: 'at', shape: 'round', modality: 'attention', label: 'Self-Attention' },
      { id: 'ad1', shape: 'layer', prim: 'residual', modality: 'residual', label: '⊕ add' },
      { id: 'ln2', shape: 'layer', prim: 'norm', modality: 'norm', label: 'LayerNorm' },
      { id: 'mlp', shape: 'round', modality: 'proprio', label: 'Feed-forward MLP' },
      { id: 'ad2', shape: 'layer', prim: 'residual', modality: 'residual', label: '⊕ add' }],
      edges: [{ from: 'ln1', to: 'at' }, { from: 'at', to: 'ad1' }, { from: 'ad1', to: 'ln2' },
              { from: 'ln2', to: 'mlp' }, { from: 'mlp', to: 'ad2' }] } },
];
const TOKEN = { w: 22, h: 9, r: 3 };   // ONE uniform token glyph size everywhere
// vertical stacking priority for auto-organize: VIDEOS on top, then proprio, then
// goal/latent/text, then the trunk, with the action output last.
const MOD_PRI = { vision: 0, goal: 3, proprio: 1, latent: 4, text: 5, noise: 6,
  trunk: 7, attention: 7, dit: 7, diffusion: 7, flow: 7, vla_backbone: 7, act: 7,
  norm: 7, residual: 7, action: 9 };
const modPri = (n) => (MOD_PRI[n.modality] != null ? MOD_PRI[n.modality] : 7);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function sizeOf(n) {
  switch (n.shape) {
    case 'image': return [118, 78];       // big enough to actually play the camera video
    case 'trapezoid': return [96, 60];
    case 'round': return n.viz ? [150, 96] : (n.expand ? [128, 64] : [104, 48]);   // nested cards need room for the inner-preview
    case 'rect': return [92, 40];
    case 'layer': return [110, 46];       // elementary NN layer — ONE uniform size
    case 'text': return (n.id === 'in_prop' || n.id === 'in_vae') ? [128, 86] : [104, 42];
    case 'tall': return [78, 150];
    case 'tokencol': return [36, clamp((Math.min(n.tokens || 3, 9)) * (TOKEN.h + 3) + 16, 44, 132)];
    case 'actions': return [128, 86];   // black joint-scope panel (module-coloured lines)
    default: return [96, 46];
  }
}

export class ArchDiagram {
  constructor(canvas, legendEl, tipEl, opts = {}) {
    this.cv = canvas; this.legendEl = legendEl; this.tipEl = tipEl;
    this.onEdit = opts.onEdit || (() => {});
    this.data = opts.data || null;          // {epArrays(ds,ep), frameUrl(ds,ep,cam,i)}
    this.ep = null; this._epT = 0;
    this.readOnly = false;   // true while VIEWING a run — the arch is a locked record
    this.toolbar = document.getElementById('archToolbar');
    this.animate = true; this.t = 0; this._last = performance.now();
    this.view = { tx: 40, ty: 20, s: 1 };
    this.nav = [];               // breadcrumb stack of {graph,title}
    this.pos = {};               // manual positions: level -> id -> {x,y}
    this.hover = null; this.drag = null; this.arch = ''; this._needFit = false;
    this._wire();
    new ResizeObserver(() => { this._needFit = true; }).observe(canvas.parentElement);
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  setSpec(spec) {
    this.compiled = false;   // designed spec view (compile toggle resets to this)
    const archChanged = spec.arch !== this.arch;
    // remember the current drill path so an edit (which rebuilds the spec) doesn't
    // kick the user back to the top level — we re-descend the same block labels.
    const prevPath = archChanged ? [] : this.nav.slice(1).map(n => n.title);
    this.arch = spec.arch; this.spec = spec;
    // module colour map for the proprio-in / action-out joint line graphs
    this._groups = spec.groups || [];
    this._idxGrp = {};
    for (const g of this._groups) for (const i of (g.indices || [])) this._idxGrp[i] = g.name;
    this.nav = [{ graph: { nodes: spec.nodes, edges: spec.edges }, title: spec.arch }];
    for (const title of prevPath) {   // re-descend the previous drill-down
      const node = this._cur().nodes.find(n => n.label === title && n.expand);
      if (!node) break;
      this.nav.push({ graph: JSON.parse(JSON.stringify(node.expand)), title: node.label });
    }
    if (archChanged) { this.pos = {}; this.view = { tx: 40, ty: 20, s: 1 }; this._needFit = true; }
    // keep the playing episode's camera buffers in sync with the (possibly new) spec
    if (this.ep) {
      const cams = (spec.inputs && spec.inputs.cameras) || [];
      for (const c of cams) if (!this.ep.imgs[c]) this.ep.imgs[c] = this._makeBuf();
      this.ep.cams = cams;
    }
    this._layout();
    this._renderLegend(); this._renderToolbar();
  }
  setAnimate(on) { this.animate = on; }
  zoomBy(f) { this._zoomAt(this.cv.clientWidth / 2, this.cv.clientHeight / 2, f); }

  // render the COMPILED graph (from the real nn.Module) — the ground-truth view
  showCompiled(g) {
    this.compiled = true;
    this.nav = [{ graph: { nodes: g.nodes, edges: g.edges }, title: `compiled · ${g.params_str || ''} params` }];
    this.pos = {}; this._needFit = true;
    this._layout(); this._renderToolbar(); this._renderLegend();
    window.__toast && window.__toast(`compiled ${g.arch} · ${g.params_str} params · ${g.note || ''}`, 'ok');
  }
  isCompiled() { return !!this.compiled; }

  _cur() { return this.nav[this.nav.length - 1].graph; }
  _levelKey() { return this.nav.map(n => n.title).join('/'); }

  // ── layout: barycenter (Sugiyama-style) — left→right, balanced, few crossings,
  //    no long vertical tails. Dragged nodes are pinned; auto-organize re-flows. ─
  _layout() {
    const g = this._cur(), key = this._levelKey();
    const P = this.pos[key] || (this.pos[key] = {});
    const GAP = 30, COLW = 200;   // roomier → fewer wires crossing over blocks
    const byCol = new Map();
    for (const n of g.nodes) (byCol.get(n.col || 0) || byCol.set(n.col || 0, []).get(n.col || 0)).push(n);
    const cols = [...byCol.keys()].sort((a, b) => a - b);
    const adjP = new Map(), adjN = new Map(), h = {}, y = {};
    for (const n of g.nodes) { adjP.set(n.id, []); adjN.set(n.id, []); h[n.id] = sizeOf(n)[1]; }
    for (const e of g.edges) if (adjN.has(e.from) && adjP.has(e.to)) { adjN.get(e.from).push(e.to); adjP.get(e.to).push(e.from); }
    // primary order = MODALITY (videos → prop → …); secondary = lane, then barycenter
    const order = (a, b) => (modPri(a) - modPri(b)) || ((a.lane || 0) - (b.lane || 0)) || ((a._d || 0) - (b._d || 0));
    for (const c of cols) { let yy = 0; for (const n of byCol.get(c).sort(order)) { y[n.id] = yy + h[n.id] / 2; yy += h[n.id] + GAP; } }
    const stack = (list) => {
      const total = list.reduce((s, n) => s + h[n.id], 0) + GAP * (list.length - 1);
      const meanD = list.reduce((s, n) => s + (n._d != null ? n._d : y[n.id]), 0) / list.length;
      let yy = meanD - total / 2;
      for (const n of list) { y[n.id] = yy + h[n.id] / 2; yy += h[n.id] + GAP; }
    };
    for (let pass = 0; pass < 12; pass++) {
      const rev = pass % 2 === 1;
      for (const c of (rev ? cols.slice().reverse() : cols)) {
        const list = byCol.get(c);
        for (const n of list) { const nb = (rev ? adjN : adjP).get(n.id); n._d = nb.length ? nb.reduce((s, i) => s + y[i], 0) / nb.length : y[n.id]; }
        // keep VIDEOS-top / prop-next grouping, minimise crossings WITHIN each band
        list.sort((a, b) => (modPri(a) - modPri(b)) || (a._d - b._d));
        stack(list);
      }
    }
    const H = this.cv.clientHeight || 600;
    let mn = 1e9, mx = -1e9;
    for (const n of g.nodes) { mn = Math.min(mn, y[n.id] - h[n.id] / 2); mx = Math.max(mx, y[n.id] + h[n.id] / 2); }
    const off = H / 2 - (mn + mx) / 2;
    for (const n of g.nodes) {
      if (P[n.id] && P[n.id].manual) continue;
      const [w] = sizeOf(n);
      P[n.id] = { x: 40 + (n.col || 0) * COLW + 60 - w / 2, y: y[n.id] + off };
    }
    this._P = P;
  }

  autoOrganize() {
    // sketch/auto-added nodes carry col=999 (or none) — derive a sane column from
    // their neighbours first, else tidy shoves them into a far-right ghost column.
    const g = this._cur();
    const byId = Object.fromEntries(g.nodes.map(n => [n.id, n]));
    const maxSpec = Math.max(0, ...g.nodes.map(n => (n.col != null && n.col < 900) ? n.col : 0));
    for (const n of g.nodes) {
      if (n.col != null && n.col < 900) continue;
      const froms = g.edges.filter(e => e.to === n.id).map(e => byId[e.from]).filter(x => x && x.col < 900);
      const tos = g.edges.filter(e => e.from === n.id).map(e => byId[e.to]).filter(x => x && x.col < 900);
      if (froms.length) n.col = Math.max(...froms.map(x => x.col)) + 1;
      else if (tos.length) n.col = Math.max(0, Math.min(...tos.map(x => x.col)) - 1);
      else n.col = Math.min(2, maxSpec);          // unwired stray → park near the inputs
    }
    delete this.pos[this._levelKey()];
    this._layout(); this._needFit = true;
  }

  // ── custom block library (Simulink-style save + reuse) ────────────────────
  _blkKey() { return 'mabel.trainer.blocks'; }
  _loadBlocks() { try { return JSON.parse(localStorage.getItem(this._blkKey()) || '{}'); } catch { return {}; } }
  _saveBlocks(o) { try { localStorage.setItem(this._blkKey(), JSON.stringify(o)); } catch {} }
  listBlocks() { return Object.keys(this._loadBlocks()); }
  deleteBlock(name) { const b = this._loadBlocks(); delete b[name]; this._saveBlocks(b); }
  saveBlock(name) {
    if (!name) return;
    const g = this._cur(), blocks = this._loadBlocks();
    blocks[name] = { nodes: JSON.parse(JSON.stringify(g.nodes)), edges: JSON.parse(JSON.stringify(g.edges)), title: name };
    this._saveBlocks(blocks);
    window.__toast && window.__toast(`saved block "${name}"`, 'ok');
  }
  insertBlock(name) {
    if (this.readOnly) { window.__toast && window.__toast('read-only while viewing a run', 'err'); return; }
    const b = this._loadBlocks()[name]; if (!b) return;
    const g = this._cur();
    const maxCol = Math.max(0, ...g.nodes.map(n => n.col || 0));
    const id = 'blk_' + Math.random().toString(36).slice(2, 8);
    g.nodes.push({ id, shape: 'round', modality: 'trunk', label: name, sub: 'custom block',
      col: maxCol + 1, lane: 0, deletable: true, expand: JSON.parse(JSON.stringify(b)) });
    delete this.pos[this._levelKey()]; this._layout(); this._needFit = true;
    window.__toast && window.__toast(`inserted "${name}" — double-click to look inside`, 'ok');
  }

  // load a real episode so input/output blocks play actual data
  _makeBuf() { const a = new Image(), b = new Image(); a.decoding = b.decoding = 'async'; return { a, b, front: 'a', loading: false, has: false }; }
  setEpisode(ds) {
    if (!ds || !this.data) { this.ep = null; return; }
    if (this.ep && this.ep.ds === ds) return;
    const cams = (this.spec && this.spec.inputs && this.spec.inputs.cameras) || [];
    const ep = { ds, frame: 0, len: 0, arrays: null, imgs: {}, cams };
    for (const c of cams) ep.imgs[c] = this._makeBuf();
    this.ep = ep;
    this.data.epArrays(ds, 0).then(a => {
      if (this.ep && this.ep.ds === ds) { this.ep.arrays = a; this.ep.len = (a.action || []).length; }
    }).catch(() => {});
  }
  _advanceEp(dt) {
    const ep = this.ep; if (!ep || !ep.len) return;
    this._epT += dt;
    if (this._epT < 0.11) return; this._epT = 0;      // ~9fps
    ep.frame = (ep.frame + 1) % ep.len;
    for (const c of ep.cams) {
      const buf = ep.imgs[c]; if (!buf || buf.loading) continue;
      // load the next frame into the BACK buffer; swap only once fully loaded,
      // so we always draw a complete image (no flicker).
      const back = buf.front === 'a' ? buf.b : buf.a;
      buf.loading = true;
      back.onload = () => { buf.front = buf.front === 'a' ? 'b' : 'a'; buf.loading = false; buf.has = true; };
      back.onerror = () => { buf.loading = false; };
      back.src = this.data.frameUrl(ep.ds, 0, c, ep.frame);
    }
  }
  _camImg(camera) {
    const buf = this.ep && this.ep.imgs[camera];
    if (!buf || !buf.has) return null;
    const im = buf.front === 'a' ? buf.a : buf.b;
    return (im.complete && im.naturalWidth) ? im : null;
  }
  _npos(n) { return (this._P && this._P[n.id]) || { x: 60, y: 60 }; }
  _rect(n) { const d = sizeOf(n); const p = this._npos(n); const w = p.w || d[0], h = p.h || d[1]; return { x: p.x - w / 2, y: p.y - h / 2, w, h, cx: p.x, cy: p.y }; }

  fit() {
    const g = this._cur(); if (!g.nodes.length) return;
    if (this.cv.clientWidth < 20 || this.cv.clientHeight < 20) { this._needFit = true; return; }
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of g.nodes) { const r = this._rect(n); x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y); x1 = Math.max(x1, r.x + r.w); y1 = Math.max(y1, r.y + r.h); }
    const W = this.cv.clientWidth, H = this.cv.clientHeight, pad = 30;
    // reserve the left region covered by any floating panel (e.g. the log window)
    // so the architecture lays out BESIDE it, never underneath.
    let leftPad = 0;
    const cr = this.cv.getBoundingClientRect();
    document.querySelectorAll('.fpanel').forEach(p => {
      const r = p.getBoundingClientRect();
      if (r.width > 4 && r.right > cr.left + 4 && r.left < cr.left + cr.width * 0.55 && r.bottom > cr.top && r.top < cr.bottom)
        leftPad = Math.max(leftPad, r.right - cr.left + 14);
    });
    const availW = Math.max(80, W - leftPad - pad * 2);
    const s = clamp(Math.min(availW / (x1 - x0), (H - pad * 2) / (y1 - y0)), 0.22, 1.6);
    this.view = { s, tx: leftPad + pad - x0 * s + (availW - (x1 - x0) * s) / 2, ty: pad - y0 * s };
  }

  // ── screen <-> world ───────────────────────────────────────────────────────
  _w2s(x, y) { return [x * this.view.s + this.view.tx, y * this.view.s + this.view.ty]; }
  _s2w(x, y) { return [(x - this.view.tx) / this.view.s, (y - this.view.ty) / this.view.s]; }
  _hit(sx, sy) {
    const [wx, wy] = this._s2w(sx, sy);
    const g = this._cur();
    for (let i = g.nodes.length - 1; i >= 0; i--) {
      const r = this._rect(g.nodes[i]);
      if (wx >= r.x && wx <= r.x + r.w && wy >= r.y && wy <= r.y + r.h) return g.nodes[i];
    }
    return null;
  }

  // ── interaction ────────────────────────────────────────────────────────────
  _wire() {
    const cv = this.cv;
    cv.addEventListener('wheel', (e) => {
      e.preventDefault(); const r = cv.getBoundingClientRect();
      this._zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.1 : 1 / 1.1);
    }, { passive: false });
    cv.addEventListener('pointerdown', (e) => {
      const r = cv.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
      try { cv.setPointerCapture(e.pointerId); } catch {}   // synthetic/stale pointers throw
      // 1) port hit (ports live OUTSIDE the block rect, so test before _hit)
      const port = this._portHit(sx, sy);
      if (port && !this.readOnly) {
        this.selected = port.n;
        const [wx, wy] = this._s2w(sx, sy);
        this.drag = { wire: true, n: port.n, side: port.side, sx, sy, cx: wx, cy: wy, moved: false };
        return;
      }
      // 2) ✦ auto-complete chip hit (drawn under floating blocks)
      const chip = this._chipHit(sx, sy);
      if (chip && !this.readOnly) { this.autoComplete(chip); return; }
      const n = this._hit(sx, sy);
      this.selected = n || null;   // click a block to SELECT it (highlighted; Del removes it)
      if (n) {
        const nr = this._rect(n);
        // delete button hit-test (top-right corner)
        const [bx, by] = this._w2s(nr.x + nr.w, nr.y);
        if (Math.hypot(sx - bx, sy - by) < 13) { this._tryDelete(n); return; }
        // resize handle hit-test (bottom-right corner)
        const [rbx, rby] = this._w2s(nr.x + nr.w, nr.y + nr.h);
        if (Math.hypot(sx - rbx, sy - rby) < 12) { this.drag = { resize: true, n, tlx: nr.x, tly: nr.y, sx, sy, moved: false }; return; }
        const p = this._npos(n);
        this.drag = { n, dx: p.x - this._s2w(sx, sy)[0], dy: p.y - this._s2w(sx, sy)[1], moved: false, sx, sy };
      } else {
        this.drag = { pan: true, sx, sy, tx: this.view.tx, ty: this.view.ty, moved: false };
      }
    });
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
      if (this.drag) {
        if (Math.hypot(sx - this.drag.sx, sy - this.drag.sy) > 3) this.drag.moved = true;
        if (this.drag.wire) {   // dragging a connection wire from a port
          const [wx, wy] = this._s2w(sx, sy);
          this.drag.cx = wx; this.drag.cy = wy;
          this.drag.cand = this._wireCandidate(this.drag);
        }
        else if (this.drag.pan) { this.view.tx = this.drag.tx + (sx - this.drag.sx); this.view.ty = this.drag.ty + (sy - this.drag.sy); }
        else if (this.drag.resize) {
          const [wx, wy] = this._s2w(sx, sy); const p = this._npos(this.drag.n);
          const w = Math.max(46, wx - this.drag.tlx), h = Math.max(30, wy - this.drag.tly);
          p.w = w; p.h = h; p.x = this.drag.tlx + w / 2; p.y = this.drag.tly + h / 2; p.manual = true;
        } else {
          const [wx, wy] = this._s2w(sx, sy); const p = this._npos(this.drag.n); p.x = wx + this.drag.dx; p.y = wy + this.drag.dy; p.manual = true;
          // a floating (un-wired) block being dragged → live valid-connection hint
          if (this.drag.n.floating) this._suggest = this._connectionSuggestion(this.drag.n);
        }
      } else { this.hover = this._hit(sx, sy); this._tip(this.hover, sx, sy); }
    });
    cv.addEventListener('pointerup', (e) => {
      const d = this.drag; this.drag = null;
      if (d && d.wire) { if (d.cand) this._applyWire(d); return; }
      if (d && d.n && d.n.floating && d.moved && this._suggest) this._applyConnection(this._suggest);
      this._suggest = null;
      if (d && !d.pan && !d.resize && !d.moved) this._click(d.n);
    });
    cv.addEventListener('dblclick', (e) => {
      const r = cv.getBoundingClientRect(); const n = this._hit(e.clientX - r.left, e.clientY - r.top);
      const pop = document.getElementById('archEditPop'); if (pop) pop.classList.add('hidden');
      if (n && n.expand) this._drill(n);   // double-click drills (even edit+expand blocks like the CNN)
    });
    // keyboard: Delete / Backspace removes the SELECTED block (unless typing)
    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      const t = document.activeElement, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      if (this.selected) { e.preventDefault(); this._tryDelete(this.selected); }
    });
  }
  _zoomAt(sx, sy, f) {
    const [wx, wy] = this._s2w(sx, sy);
    this.view.s = clamp(this.view.s * f, 0.2, 3);
    this.view.tx = sx - wx * this.view.s; this.view.ty = sy - wy * this.view.s;
  }
  _click(n) {
    // input/output DATA blocks → open the enlarged data-inspector window
    // (not for freshly-added floating blocks — they have no dataset feed yet)
    if (!n.floating && (n.shape === 'image' || n.id === 'in_prop' || n.id === 'in_vae' || n.shape === 'actions')) {
      this._openDataWindow(n); return;
    }
    // editing is a real, LeRobot-backed choice — only when NOT viewing a run.
    // In read-only view, a click just DRILLS IN (looking inside is always allowed).
    if (n.edit && !this.readOnly) { this._openEditor(n); return; }
    if (n.expand) this._drill(n);    // single-click drills (incl. read-only view of the ResNet/transformer)
  }
  _curHpVal(key) {
    const st = window.__studioState; const hp = (st && st.hpByArch[st.arch]) || {};
    return hp[key];
  }
  // popup picker for a node's `edit` descriptor → writes a REAL LeRobot hp field
  _openEditor(n) {
    const ed = n.edit;
    let pop = document.getElementById('archEditPop');
    if (!pop) { pop = document.createElement('div'); pop.id = 'archEditPop'; pop.className = 'arch-edit-pop hidden'; document.body.appendChild(pop); }
    const cur = this._curHpVal(ed.hp);
    const opts = ed.kind === 'bool'
      ? [{ v: true, l: 'On' }, { v: false, l: 'Off' }]
      : ed.options.map(o => ({ v: o, l: o }));
    pop.innerHTML = `<div class="aep-title">${ed.label}</div>` +
      opts.map(o => `<button data-v='${JSON.stringify(o.v)}' class="${String(o.v) === String(cur) ? 'on' : ''}">${o.l}</button>`).join('') +
      `<div class="aep-note">→ LeRobot · <code>${ed.hp}</code></div>`;
    const r = this._rect(n); const [sx, sy] = this._w2s(r.cx, r.y + r.h);
    const cr = this.cv.getBoundingClientRect();
    pop.style.left = Math.max(8, cr.left + sx - 70) + 'px';
    pop.style.top = (cr.top + sy + 8) + 'px';
    pop.classList.remove('hidden');
    pop.querySelectorAll('[data-v]').forEach(b => b.onclick = (e) => {
      e.stopPropagation(); const v = JSON.parse(b.dataset.v);
      this.onEdit({ hp: { [ed.hp]: v } });
      window.__toast && window.__toast(`→ LeRobot: ${ed.hp} = ${v}`, 'ok');
      pop.classList.add('hidden');
    });
    const close = (ev) => { if (pop.contains(ev.target)) return; pop.classList.add('hidden'); document.removeEventListener('mousedown', close); };
    setTimeout(() => document.addEventListener('mousedown', close), 0);
  }

  // ── enlarged data-inspector window for camera / proprio / action blocks ──────
  _closeDataWin() {
    if (this._dataWin) { this._dataWin.back.remove(); this._dataWin = null; }
    if (this._dwEsc) { document.removeEventListener('keydown', this._dwEsc); this._dwEsc = null; }
  }
  _openDataWindow(n) {
    this._closeDataWin();
    const kind = n.shape === 'image' ? 'camera' : (n.shape === 'actions' ? 'action' : (n.id === 'in_vae' ? 'action' : 'proprio'));
    const ds = (window.__datasets || []).find(d => d.dataset_id === (this.ep && this.ep.ds)) || {};
    const dur = ds.hours != null ? `${ds.hours} h · ${ds.seconds || '?'} s` : '';
    const stat = (k, v) => v ? `<span class="dw-stat"><i>${k}</i>${v}</span>` : '';
    let title, sub, stats, legend = '';
    if (kind === 'camera') {
      const sh = (ds.camera_shapes || {})[n.camera] || [];
      const res = sh.length >= 2 ? `${sh[1]}×${sh[0]}` : '';
      title = n.camera || n.label; sub = 'RGB camera stream';
      stats = stat('resolution', res) + stat('rate', ds.fps ? ds.fps + ' Hz' : '') + stat('frames', ds.total_frames) + stat('length', dur) + stat('dataset', ds.dataset_id);
    } else {
      const idx = kind === 'action'
        ? (this.spec && this.spec.output && this.spec.output.action_indices) || []
        : (this.spec && this.spec.inputs && this.spec.inputs.state_indices) || [];
      const present = {};
      for (const i of idx) { const gname = this._idxGrp[i]; if (gname) present[gname] = (present[gname] || 0) + 1; }
      legend = (this._groups || []).filter(g => present[g.name]).map(g =>
        `<span class="dw-lg"><i style="background:${GROUP_COL[g.name] || '#9aa'}"></i>${g.label} <b>${present[g.name]}</b></span>`).join('');
      title = kind === 'action' ? 'Action · target joint positions' : 'Proprioception · joint state';
      sub = kind === 'action' ? "dataset `action` (absolute targets)" : "dataset `observation.state`";
      stats = stat('dims', `${idx.length}-D`) + stat('rate', ds.fps ? ds.fps + ' Hz' : '') + stat('frames', ds.total_frames) + stat('length', dur) + stat('size', ds.size_mb ? ds.size_mb + ' MB' : '');
    }
    const back = document.createElement('div'); back.className = 'dataw-back';
    back.innerHTML = `<div class="dataw"><div class="dataw-head">
        <span class="dataw-title">${title}</span><span class="dataw-sub">${sub}</span>
        <button class="dataw-close">✕</button></div>
      <div class="dataw-stats">${stats}</div>
      <div class="dataw-body"><canvas class="dataw-cv"></canvas></div>
      <div class="dataw-legend">${legend}</div></div>`;
    document.body.appendChild(back);
    back.onclick = (e) => { if (e.target === back) this._closeDataWin(); };
    back.querySelector('.dataw-close').onclick = () => this._closeDataWin();
    this._dwEsc = (e) => { if (e.key === 'Escape') this._closeDataWin(); };
    document.addEventListener('keydown', this._dwEsc);
    this._dataWin = { node: n, kind, back, cv: back.querySelector('.dataw-cv') };
  }
  _drawDataWin() {
    const w = this._dataWin; if (!w) return;
    const cv = w.cv, r = Math.min(devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight; if (W < 10 || H < 10) return;
    if (cv.width !== W * r || cv.height !== H * r) { cv.width = W * r; cv.height = H * r; }
    const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
    const n = w.node;
    if (w.kind === 'camera') {
      const im = this._camImg(n.camera);
      ctx.fillStyle = '#0c0b0a'; ctx.fillRect(0, 0, W, H);
      if (im) {
        const ar = im.naturalWidth / im.naturalHeight, rr = W / H;
        let dw = W, dh = H, dx = 0, dy = 0;
        if (ar > rr) { dh = W / ar; dy = (H - dh) / 2; } else { dw = H * ar; dx = (W - dw) / 2; }
        ctx.drawImage(im, dx, dy, dw, dh);
      } else { ctx.fillStyle = 'rgba(151,144,138,.5)'; ctx.font = '13px Geist'; ctx.textAlign = 'center'; ctx.fillText('loading camera…', W / 2, H / 2); }
      return;
    }
    // proprio / action: a big module-coloured multi-line plot with axes + playhead
    const arr = w.kind === 'action' ? (this.ep && this.ep.arrays && this.ep.arrays.action)
              : (this.ep && this.ep.arrays && this.ep.arrays['observation.state']);
    const idx = w.kind === 'action' ? (this.spec.output.action_indices || []) : (this.spec.inputs.state_indices || []);
    ctx.fillStyle = '#0c0b0a'; ctx.fillRect(0, 0, W, H);
    if (!arr || !arr.length) { ctx.fillStyle = 'rgba(151,144,138,.5)'; ctx.font = '13px Geist'; ctx.textAlign = 'center'; ctx.fillText('loading…', W / 2, H / 2); return; }
    const pad = { l: 40, r: 12, t: 12, b: 22 };
    const dim = arr[0].length; const cols = idx.filter(i => i < dim);
    let mn = 1e9, mx = -1e9; for (const row of arr) for (const c of cols) { const v = row[c]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx === mn) { mx += 1; mn -= 1; }
    const px = (i) => pad.l + i / (arr.length - 1) * (W - pad.l - pad.r);
    const py = (v) => H - pad.b - (v - mn) / (mx - mn) * (H - pad.t - pad.b);
    ctx.strokeStyle = 'rgba(242,238,230,.08)'; ctx.fillStyle = 'rgba(151,144,138,.7)'; ctx.font = '9px "Geist Mono"'; ctx.textAlign = 'right';
    for (let g = 0; g <= 4; g++) { const v = mn + (mx - mn) * g / 4, y = py(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText((+v.toFixed(2)).toString(), pad.l - 4, y + 3); }
    ctx.strokeStyle = 'rgba(242,238,230,.16)'; ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
    ctx.fillStyle = 'rgba(151,144,138,.6)'; ctx.textAlign = 'center';
    for (let g = 0; g <= 5; g++) { const fi = Math.round((arr.length - 1) * g / 5); ctx.fillText(String(fi), px(fi), H - pad.b + 12); }
    for (const c of cols) {
      ctx.strokeStyle = (GROUP_COL[this._idxGrp[c]] || '#9aa'); ctx.globalAlpha = .9; ctx.lineWidth = 1.2; ctx.beginPath();
      arr.forEach((row, i) => { const X = px(i), Y = py(row[c]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (this.ep) { const fx = px(this.ep.frame || 0); ctx.strokeStyle = 'rgba(255,255,255,.6)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(fx, pad.t); ctx.lineTo(fx, H - pad.b); ctx.stroke(); }
  }
  _drill(n) {
    const g = JSON.parse(JSON.stringify(n.expand));
    this.nav.push({ graph: g, title: n.label });
    this._layout(); this._renderToolbar(); this.fit();
  }
  _navTo(i) { this.nav = this.nav.slice(0, i + 1); this._layout(); this._renderToolbar(); this.fit(); }
  setReadOnly(v) { this.readOnly = !!v; this._renderToolbar(); }
  _tryDelete(n) {
    if (!n) return;
    if (this.readOnly) {
      window.__toast && window.__toast('viewing a run — this architecture is read-only (＋ New job to edit)', 'err');
      return;
    }
    if (!n.deletable) {
      window.__toast && window.__toast(`can't delete "${n.label}" — ${n.protect || 'it would break the architecture'}`, 'err');
      return;
    }
    if (this.selected === n) this.selected = null;
    if (n.editHp) this.onEdit({ hp: n.editHp });        // config-backed removal (VAE, etc.)
    else if (n.camera) this.onEdit({ camera: n.camera }); // drop a camera input
    else this._removeNode(n);                            // added/sketch block → remove directly
    window.__toast && window.__toast(`removed ${n.label}`, 'ok');
  }
  _removeNode(n) {
    const g = this._cur();
    g.nodes = g.nodes.filter(x => x !== n);
    g.edges = g.edges.filter(e => e.from !== n.id && e.to !== n.id);
    const P = this.pos[this._levelKey()]; if (P) delete P[n.id];
    this._layout();
  }

  // ── toolbar: JUST the breadcrumb (＋ Add block lives top-right in arch-ctl) ──
  _renderToolbar() {
    if (!this.toolbar) return;
    const crumbs = this.nav.map((n, i) =>
      `<button class="crumb ${i === this.nav.length - 1 ? 'on' : ''}" data-i="${i}">${n.title}</button>`)
      .join('<span class="crumb-sep">▸</span>');
    const back = this.nav.length > 1 ? `<button class="mini" id="tbBack">‹ back</button>` : '';
    const tail = this.readOnly
      ? `<span class="ro-chip" title="Viewing a trained run — the architecture is locked. ＋ New job to edit.">🔒 read-only</span>`
      : '';
    this.toolbar.innerHTML = `${back}<div class="crumbs">${crumbs}</div>${tail}`;
    this.toolbar.querySelectorAll('.crumb').forEach(b => b.onclick = () => this._navTo(+b.dataset.i));
    const bk = this.toolbar.querySelector('#tbBack'); if (bk) bk.onclick = () => this._navTo(this.nav.length - 2);
  }

  // The unified add-block menu (opened from the top-right ＋ Add block button).
  // Lists elementary layers + composite blocks; the chosen block FLOATS into free
  // space so the user drags it and connects it (with valid-connection hints).
  openAddBlockMenu(btn) {
    if (this.readOnly) { window.__toast && window.__toast('read-only — ＋ New job to edit', 'err'); return; }
    let menu = document.getElementById('addBlockMenu');
    if (!menu) { menu = document.createElement('div'); menu.id = 'addBlockMenu'; menu.className = 'addblock-menu hidden'; document.body.appendChild(menu); }
    if (!menu.classList.contains('hidden')) { menu.classList.add('hidden'); return; }
    const btn2 = (kind, id, mod, label) => `<button data-kind="${kind}" data-id="${id}"><span class="pm-dot" style="background:${(MOD[mod] || MOD.trunk).f}"></span>${label}</button>`;
    const ins = INPUT_BLOCKS.map(b => btn2('io', b.id, b.modality, b.label)).join('');
    const outs = OUTPUT_BLOCKS.map(b => btn2('io', b.id, b.modality, b.label)).join('');
    const comp = ADD_BLOCKS.map(b => btn2('block', b.id, b.modality, b.label)).join('');
    const prim = PRIM_MENU.map(p => `<button data-kind="prim" data-id="${p.prim}"><span class="pm-dot" style="background:${(PRIM[p.prim] || {}).f}"></span>${p.label}</button>`).join('');
    const saved = this.listBlocks();
    const savedHtml = saved.length ? `<div class="ab-sec">Saved blocks</div>` +
      saved.map(s => `<button data-kind="saved" data-id="${s}"><span class="pm-dot" style="background:${MOD.trunk.f}"></span>${s}</button>`).join('') : '';
    menu.innerHTML = `<div class="ab-sec">Inputs</div>${ins}<div class="ab-sec">Outputs</div>${outs}`
      + `<div class="ab-sec">Blocks</div>${comp}<div class="ab-sec">Elementary layers</div>${prim}${savedHtml}`;
    const rc = btn.getBoundingClientRect();
    menu.style.left = Math.max(8, rc.right - 190) + 'px'; menu.style.top = (rc.bottom + 5) + 'px';
    menu.classList.remove('hidden');
    menu.querySelectorAll('button').forEach(b => b.onclick = (e) => {
      e.stopPropagation(); menu.classList.add('hidden');
      if (b.dataset.kind === 'prim') { const p = PRIM_MENU.find(x => x.prim === b.dataset.id); this.addFloating({ prim: p.prim, label: p.label, sub: p.sub }); }
      else if (b.dataset.kind === 'saved') this.addFloating({ saved: b.dataset.id });
      else if (b.dataset.kind === 'io') this.addFloating({ io: b.dataset.id });
      else this.addFloating({ block: b.dataset.id });
    });
    const close = (ev) => { if (menu.contains(ev.target) || ev.target === btn) return; menu.classList.add('hidden'); document.removeEventListener('mousedown', close); };
    setTimeout(() => document.addEventListener('mousedown', close), 0);
  }

  // Add a block that FLOATS in free space (no edges) — the user drags it and the
  // diagram suggests valid connections. `def` is either {prim,label,sub} for an
  // elementary layer or {block:<id>} for a composite from ADD_BLOCKS.
  addFloating(def) {
    if (this.readOnly) { window.__toast && window.__toast('read-only while viewing a run', 'err'); return; }
    const g = this._cur();
    const id = 'A_' + Math.random().toString(36).slice(2, 7);
    let node;
    if (def.block) {
      const b = ADD_BLOCKS.find(x => x.id === def.block); if (!b) return;
      node = { id, shape: 'round', modality: b.modality, label: b.label, sub: b.sub,
               deletable: true, accepts: b.accepts,
               expand: JSON.parse(JSON.stringify(b.expand)) };
    } else if (def.saved) {
      const b = this._loadBlocks()[def.saved]; if (!b) return;
      node = { id, shape: 'round', modality: 'trunk', label: def.saved, sub: 'custom block',
               deletable: true, accepts: ['*'], expand: JSON.parse(JSON.stringify(b)) };
    } else if (def.io) {
      const b = [...INPUT_BLOCKS, ...OUTPUT_BLOCKS].find(x => x.id === def.io); if (!b) return;
      let sub = def.sub || b.sub;
      if (b.prompt === 'joints' && !def.sub && window.__confirm) {
        // proprio input: pick which joint area it covers — a STYLED modal with the
        // embodiment's real scopes (not a dated window.prompt)
        const emb = (window.__studioState || {}).emb || {};
        const opts = Object.entries(emb.scopes || {}).map(([k, v]) => ({ v: k, l: v.label || k }));
        window.__confirm({
          title: 'Which joints?', sub: b.label, confirmLabel: '＋ Add block', kind: 'go',
          rows: [['Block', b.label], ['Covers', 'the joint group this signal represents']],
          inputs: [{ key: 'area', label: 'Joint area', type: 'select',
                     value: 'whole_body', options: opts.length ? opts : [{ v: 'whole_body', l: 'Whole body' }] }],
        }).then(res => {
          if (!res) return;
          const k = res.values.area;
          const lab = (emb.scopes && emb.scopes[k] && emb.scopes[k].label) || k;
          this.addFloating({ ...def, sub: `${lab} · joints` });   // re-enter with the answer
        });
        return;
      }
      node = { id, shape: b.shape || (b.io === 'out' ? 'round' : 'text'), modality: b.modality,
               label: b.label, sub, deletable: true, io: b.io, ioId: b.id,
               accepts: b.accepts || (b.io === 'out' ? ['trunk', 'proprio', 'action'] : undefined),
               camera: b.shape === 'image' ? b.id : undefined };
    } else {
      node = { id, shape: 'layer', prim: def.prim, modality: (PRIM[def.prim] ? def.prim : 'trunk'),
               label: def.label, sub: def.sub, deletable: true, accepts: PRIM_ACCEPTS[def.prim] || ['*'] };
    }
    // drop it near the centre of the current view, as a MANUAL (free) position
    const cx = (this.cv.clientWidth / 2 - this.view.tx) / this.view.s;
    const cy = (this.cv.clientHeight * 0.32 - this.view.ty) / this.view.s;
    node.floating = true;             // marks it un-wired (draw a dashed outline + hint)
    node.col = 999; node.lane = 0;
    g.nodes.push(node);
    const P = this.pos[this._levelKey()] || (this.pos[this._levelKey()] = {});
    P[id] = { x: cx, y: cy, manual: true };
    this._P = P;
    window.__toast && window.__toast(`added ${node.label} — drag it onto a block to connect`, 'ok');
  }
  _curHp(key) {
    const st = window.__studioState; const hp = st && st.hpByArch[st.arch] || {};
    const n = this._cur().nodes.find(x => x.editHp && x.editHp[key] !== undefined);
    return hp[key] != null ? hp[key] : (n ? (n.layers || 1) : 1);
  }

  _renderLegend() {
    if (!this.legendEl) return;
    const nodes = this._cur().nodes;
    // elementary layers present at this level → show the PRIM legend instead of
    // (or alongside) the modality legend, so the atoms are named.
    const prims = [...new Set(nodes.filter(n => n.shape === 'layer' && n.prim).map(n => n.prim))].filter(p => PRIM[p]);
    const mods = [...new Set(nodes.map(n => n.modality))].filter(m => MOD[m] && !prims.length);
    const chips = [
      ...mods.map(m => `<span class="lg-chip"><i style="background:${MOD[m].f}"></i>${MOD[m].name}</span>`),
      ...prims.map(p => `<span class="lg-chip"><i style="background:${PRIM[p].f}"></i>${PRIM[p].name}</span>`),
    ];
    this.legendEl.innerHTML = chips.join('');
  }

  _tip(n, sx, sy) {
    if (!this.tipEl) return;
    if (!n) { this.tipEl.style.display = 'none'; return; }
    const bits = [`<b>${n.label}</b>`, n.sub || ''];
    if (n.edit && !this.readOnly) bits.push(`<i>click to change · ${n.edit.hp}</i>`);
    if (n.expand) bits.push('<i>double-click to look inside</i>');
    if (this.readOnly) bits.push('<i>read-only · trained run</i>');
    else if (n.deletable) bits.push('<i>× to remove</i>');
    else if (n.protect) bits.push(`<i>locked · ${n.protect}</i>`);
    this.tipEl.innerHTML = bits.filter(Boolean).join('<br>');
    this.tipEl.style.display = 'block';
    this.tipEl.style.left = (sx + 14) + 'px'; this.tipEl.style.top = (sy + 12) + 'px';
  }

  // ── render loop ────────────────────────────────────────────────────────────
  _loop(now) {
    const dt = Math.min(0.05, (now - this._last) / 1000); this._last = now;
    if (this.animate) this.t += dt;
    this._advanceEp(dt);
    this._draw();
    if (this._dataWin) this._drawDataWin();   // live-render the open data-inspector
    requestAnimationFrame(this._loop);
  }

  _draw() {
    const cv = this.cv, r = Math.min(devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * r || cv.height !== H * r) { cv.width = W * r; cv.height = H * r; }
    const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
    if (!this.spec) return;
    if (this._needFit && W > 20 && H > 20) { this.fit(); this._needFit = false; }
    ctx.save(); ctx.translate(this.view.tx, this.view.ty); ctx.scale(this.view.s, this.view.s);
    const g = this._cur();
    for (const e of g.edges) this._edge(ctx, e, g);
    for (const e of g.edges) this._tokens(ctx, e, g);
    if (this._suggest) this._drawSuggestion(ctx);   // ghost edge for the pending connection
    for (const n of g.nodes) this._node(ctx, n);
    this._drawWireDrag(ctx);                        // live wire while port-dragging
    ctx.restore();
  }

  _anchor(n, side) {
    const r = this._rect(n);
    return side === 'r' ? [r.x + r.w, r.cy] : side === 'l' ? [r.x, r.cy]
      : side === 't' ? [r.cx, r.y] : [r.cx, r.y + r.h];
  }
  _findNode(id) { return this._cur().nodes.find(n => n.id === id); }

  // Domain-aware connection suggestion for a floating block: the nearest existing
  // node whose OUTPUT modality this block can validly consume (CNN←vision,
  // MLP/Linear←proprio/tokens, attention←token sequences, …), preferring a source
  // to the LEFT (so the new block sits downstream).
  _connectionSuggestion(n) {
    if (!n || !n.floating) return null;
    const g = this._cur(); const rn = this._rect(n);
    const acc = n.accepts || ['*']; const any = acc.includes('*');
    let best = null, bestScore = 170;
    for (const o of g.nodes) {
      if (o === n || o.floating) continue;
      if (!any && !acc.includes(o.modality)) continue;   // modality must be compatible
      const ro = this._rect(o);
      const d = Math.hypot(rn.cx - ro.cx, rn.cy - ro.cy);
      const rightBias = rn.cx >= ro.cx + ro.w * 0.3 ? 0 : 55;   // favour downstream placement
      const score = d + rightBias;
      if (score < bestScore) { bestScore = score; best = o; }
    }
    return best ? { from: best.id, to: n.id, ok: true } : null;
  }
  _applyConnection(s) {
    const g = this._cur();
    const src = g.nodes.find(n => n.id === s.from), dst = g.nodes.find(n => n.id === s.to);
    if (!src || !dst) return;
    if (!g.edges.some(e => e.from === s.from && e.to === s.to))
      g.edges.push({ from: s.from, to: s.to, kind: 'data', tensor: '' });
    dst.floating = false; dst.col = (src.col || 0) + 1;
    window.__toast && window.__toast(`connected ${src.label} → ${dst.label}`, 'ok');
  }
  _drawSuggestion(ctx) {
    const s = this._suggest; if (!s) return;
    const src = this._findNode(s.from), dst = this._findNode(s.to); if (!src || !dst) return;
    const [x0, y0] = this._anchor(src, 'r'); const [x1, y1] = this._anchor(dst, 'l');
    const mx = (x0 + x1) / 2;
    ctx.save();
    ctx.strokeStyle = 'rgba(95,179,156,.95)'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1); ctx.stroke();
    const rs = this._rect(src);
    ctx.setLineDash([]); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(rs.x - 3, rs.y - 3, rs.w + 6, rs.h + 6, 10); ctx.stroke();
    ctx.fillStyle = 'rgba(95,179,156,.98)'; ctx.font = '9px "Geist Mono"'; ctx.textAlign = 'center';
    ctx.fillText('↳ connect after ' + src.label, mx, (y0 + y1) / 2 - 6);
    ctx.restore();
  }

  // ── manual port-connect: ports, wire drag, candidates, auto-adapter ─────────
  // ports show on floating blocks + the selected block: left = input, right = output
  _hasPorts(n) { return !this.readOnly && (n.floating || this.selected === n); }
  _portPos(n, side) {
    const r = this._rect(n);
    return side === 'l' ? [r.x - 9, r.cy] : [r.x + r.w + 9, r.cy];
  }
  _portHit(sx, sy) {
    for (const n of this._cur().nodes) {
      if (!this._hasPorts(n)) continue;
      for (const side of ['l', 'r']) {
        const [wx, wy] = this._portPos(n, side);
        const [px, py] = this._w2s(wx, wy);
        if (Math.hypot(sx - px, sy - py) < 12) return { n, side };
      }
    }
    return null;
  }
  // is `src → dst` a modality-valid connection? (the same domain rules the
  // proximity suggestions use — CNN←vision, MLP←proprio/tokens, norm/act←anything)
  _compatible(src, dst) {
    if (!src || !dst || src === dst) return false;
    const acc = dst.accepts || (dst.prim ? PRIM_ACCEPTS[dst.prim] : null);
    if (!acc) return true;               // spec composites: allow manual wiring
    return acc.includes('*') || acc.includes(src.modality);
  }
  _wireCandidate(d) {
    const g = this._cur();
    let best = null, bestD = 150 / this.view.s + 40;
    for (const o of g.nodes) {
      if (o === d.n) continue;
      const r = this._rect(o);
      const dist = Math.hypot(d.cx - r.cx, d.cy - r.cy);
      if (dist > bestD) continue;
      const ok = d.side === 'r' ? this._compatible(d.n, o) : this._compatible(o, d.n);
      if (!ok) continue;
      bestD = dist; best = o;
    }
    return best;
  }
  _applyWire(d) {
    const src = d.side === 'r' ? d.n : d.cand;
    const dst = d.side === 'r' ? d.cand : d.n;
    this._connect(src, dst);
  }
  // connect src→dst; if their dims are known and mismatched, auto-insert a
  // Linear adapter so the wiring stays dimensionally consistent.
  _connect(src, dst) {
    const g = this._cur();
    if (g.edges.some(e => e.from === src.id && e.to === dst.id)) return;
    const od = this._outDim(src), id_ = this._inDim(dst);
    if (od && id_ && od !== id_) {
      const aid = 'A_' + Math.random().toString(36).slice(2, 7);
      const ps = this._npos(src), pd = this._npos(dst);
      g.nodes.push({ id: aid, shape: 'layer', prim: 'linear', label: 'Linear',
                     sub: `${od}→${id_} · auto-adapter`, deletable: true });
      const P = this.pos[this._levelKey()] || (this.pos[this._levelKey()] = {});
      P[aid] = { x: (ps.x + pd.x) / 2, y: (ps.y + pd.y) / 2 + 26, manual: true };
      g.edges.push({ from: src.id, to: aid, kind: 'data', tensor: String(od) });
      g.edges.push({ from: aid, to: dst.id, kind: 'data', tensor: String(id_) });
      window.__toast && window.__toast(`dims mismatched (${od} vs ${id_}) — inserted a Linear ${od}→${id_} adapter`, 'ok');
    } else {
      g.edges.push({ from: src.id, to: dst.id, kind: 'data', tensor: od ? String(od) : '' });
      window.__toast && window.__toast(`connected ${src.label} → ${dst.label}`, 'ok');
    }
    src.floating = false; dst.floating = false;
  }
  // best-effort tensor dims parsed from a node's shape descriptor
  _specD() {
    const st = window.__studioState;
    const hp = (st && st.hpByArch[st.arch]) || {};
    if (hp.dim_model) return hp.dim_model;
    return { ACT: 512, DiT: 384, pi0: 1024, 'pi0.5': 1024 }[st ? st.arch : 'ACT'] || 512;
  }
  _parseDims(n) {
    const s = n.sub || '';
    // arrow CHAIN "59→128→512": in = first, out = LAST (an MLP is one block)
    const chain = s.match(/\d+(?:\s*→\s*\d+)+/);
    if (chain) {
      const nums = chain[0].split('→').map(x => parseInt(x));
      return { in: nums[0], out: nums[nums.length - 1] };
    }
    const m = s.match(/(\d+)-D/);
    if (m) return { in: +m[1], out: +m[1] };
    if (n.cond_out) return { in: n.cond_in || null, out: n.cond_out };
    if (n.shape === 'tokencol' || n.shape === 'tall') { const d = this._specD(); return { in: d, out: d }; }
    return { in: null, out: null };
  }
  _outDim(n) { return this._parseDims(n).out; }
  _inDim(n) { return this._parseDims(n).in; }
  _drawWireDrag(ctx) {
    const d = this.drag; if (!d || !d.wire) return;
    const [x0, y0] = this._portPos(d.n, d.side);
    ctx.save();
    ctx.strokeStyle = 'rgba(95,179,156,.95)'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(d.cx, d.cy); ctx.stroke(); ctx.setLineDash([]);
    if (d.cand) {          // highlight the valid target under the cursor
      const r = this._rect(d.cand);
      ctx.beginPath(); ctx.roundRect(r.x - 3, r.y - 3, r.w + 6, r.h + 6, 10); ctx.stroke();
      ctx.fillStyle = 'rgba(95,179,156,.98)'; ctx.font = '9px "Geist Mono"'; ctx.textAlign = 'center';
      const lbl = d.side === 'r' ? `→ into ${d.cand.label}` : `← from ${d.cand.label}`;
      ctx.fillText(lbl, r.cx, r.y - 8);
    }
    ctx.restore();
  }

  // ── ⚙ COMPILE THE GRAPH: type-check the user's edited architecture like a
  //    compiler — every connection's dims must match, every block must be wired.
  //    Errors come with one-click fixes (auto-insert Linear adapters, auto-
  //    complete unwired inputs, auto-connect strays) and an "auto-fix all". ─────
  compileGraph() {
    const g = this._cur();
    const issues = [];
    let dimsChecked = 0;
    // 1) dimension consistency on every DATA edge (stack/adaLN/cross-attn/loop
    //    edges concatenate or modulate — they have no single-dim contract)
    for (const e of g.edges) {
      if (['stack', 'cross_attn', 'adaln', 'loop', 'residual'].includes(e.kind)) continue;
      const a = this._findNode(e.from), b = this._findNode(e.to);
      if (!a || !b) { issues.push({ sev: 'error', msg: `dangling edge ${e.from} → ${e.to}`, fix: { type: 'drop-edge', edge: e } }); continue; }
      const od = this._outDim(a), id_ = this._inDim(b);
      if (od && id_) {
        dimsChecked++;
        if (od !== id_) issues.push({
          sev: 'error',
          msg: `${a.label} outputs ${od}-D but ${b.label} expects ${id_}-D`,
          fix: { type: 'adapter', edge: e, od, id: id_ },
          fixLabel: `insert Linear ${od}→${id_}`,
        });
      }
      if (!this._compatible(a, b)) issues.push({
        sev: 'warn', msg: `${a.label} (${a.modality}) feeds ${b.label} — unusual modality for this block`, fix: null });
    }
    // 2) unwired blocks (no edge touches them)
    for (const n of g.nodes) {
      const wired = g.edges.some(e => e.from === n.id || e.to === n.id);
      if (wired) continue;
      const rec = this._recipeFor(n);
      const sug = this._suggestFor(n);
      issues.push({
        sev: 'warn', msg: `"${n.label}" is not connected to anything`,
        fix: rec ? { type: 'auto', n } : (sug ? { type: 'connect', s: sug } : null),
        fixLabel: rec ? '✦ auto-complete' : (sug ? `connect after ${this._findNode(sug.from).label}` : null),
      });
    }
    this._showCompileReport(issues, { blocks: g.nodes.length, edges: g.edges.length, dimsChecked });
    return issues;
  }
  _suggestFor(n) {   // proximity suggestion regardless of the floating flag
    const was = n.floating; n.floating = true;
    const s = this._connectionSuggestion(n);
    n.floating = was;
    return s;
  }
  _applyFix(fix) {
    const g = this._cur();
    if (fix.type === 'adapter') {
      const e = fix.edge;
      g.edges = g.edges.filter(x => x !== e);
      const aid = 'A_' + Math.random().toString(36).slice(2, 7);
      const pa = this._npos(this._findNode(e.from)), pb = this._npos(this._findNode(e.to));
      g.nodes.push({ id: aid, shape: 'layer', prim: 'linear', label: 'Linear',
                     sub: `${fix.od}→${fix.id} · auto-adapter`, deletable: true });
      const P = this.pos[this._levelKey()] || (this.pos[this._levelKey()] = {});
      P[aid] = { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2 + 24, manual: true };
      g.edges.push({ from: e.from, to: aid, kind: 'data', tensor: String(fix.od) });
      g.edges.push({ from: aid, to: e.to, kind: 'data', tensor: String(fix.id) });
    } else if (fix.type === 'auto') this.autoComplete(fix.n);
    else if (fix.type === 'connect') this._applyConnection(fix.s);
    else if (fix.type === 'drop-edge') { g.edges = g.edges.filter(x => x !== fix.edge); }
  }
  _showCompileReport(issues, stats) {
    let back = document.getElementById('compileReport');
    if (back) back.remove();
    back = document.createElement('div'); back.id = 'compileReport'; back.className = 'confirm-back';
    const nErr = issues.filter(i => i.sev === 'error').length;
    const nWarn = issues.length - nErr;
    const head = issues.length === 0
      ? `<span class="confirm-title">✓ Compiles</span><span class="confirm-sub ok">all dimensions match</span>`
      : `<span class="confirm-title">⚙ Compile report</span><span class="confirm-sub">${nErr} error${nErr === 1 ? '' : 's'} · ${nWarn} warning${nWarn === 1 ? '' : 's'}</span>`;
    const items = issues.map((it, i) => `
      <div class="cr-item ${it.sev}">
        <span class="cr-ico">${it.sev === 'error' ? '✕' : '⚠'}</span>
        <span class="cr-msg">${it.msg}</span>
        ${it.fix && !this.readOnly ? `<button class="mini cr-fix" data-i="${i}">${it.fixLabel || 'fix'}</button>` : ''}
      </div>`).join('');
    const summary = `<div class="cr-sum">${stats.blocks} blocks · ${stats.edges} connections · ${stats.dimsChecked} dimension contracts checked</div>`;
    const fixable = issues.filter(i => i.fix).length;
    back.innerHTML = `<div class="confirm cr">
      <div class="confirm-head">${head}</div>
      <div class="confirm-body">${summary}${items || '<div class="cr-item ok"><span class="cr-ico">✓</span><span class="cr-msg">every data connection is dimension-consistent and every block is wired</span></div>'}</div>
      <div class="confirm-acts">
        ${fixable > 1 && !this.readOnly ? `<button class="gbtn" id="crFixAll">⚙ Auto-fix all (${fixable})</button>` : ''}
        <button class="gbtn primary" id="crClose">Close</button></div></div>`;
    document.body.appendChild(back);
    const close = () => back.remove();
    back.onclick = (e) => { if (e.target === back) close(); };
    back.querySelector('#crClose').onclick = close;
    back.querySelectorAll('.cr-fix').forEach(b => b.onclick = () => {
      this._applyFix(issues[+b.dataset.i].fix); close(); this.compileGraph();   // re-check
    });
    const fa = back.querySelector('#crFixAll');
    if (fa) fa.onclick = () => {
      issues.filter(i => i.fix).forEach(i => { try { this._applyFix(i.fix); } catch {} });
      close(); this.compileGraph();
    };
  }

  // ── ✦ auto-complete: from one new modality block, build the full encoder/head
  //    chain, wire it into the trunk, and resize the token sequence to match ────
  _recipeFor(n) { return n.ioId ? AUTO_RECIPES[n.ioId] : null; }
  _chipRect(n, r) { return { x: r.cx - 48, y: r.y + r.h + 7, w: 96, h: 16 }; }
  _chipHit(sx, sy) {
    for (const n of this._cur().nodes) {
      if (!n.floating || !this._recipeFor(n)) continue;
      const c = this._chipRect(n, this._rect(n));
      const [wx, wy] = this._s2w(sx, sy);
      if (wx >= c.x && wx <= c.x + c.w && wy >= c.y && wy <= c.y + c.h) return n;
    }
    return null;
  }
  // the trunk node new INPUT tokens stack into (per architecture family)
  _trunkTarget() {
    const g = this._cur();
    return g.nodes.find(x => x.id === 'seq') || g.nodes.find(x => x.id === 'cond')
        || g.nodes.find(x => x.id === 'prefix') || g.nodes.find(x => x.shape === 'tall');
  }
  // the trunk node OUTPUT heads read from
  _trunkSource() {
    const g = this._cur();
    return g.nodes.find(x => x.id === 'mem') || g.nodes.find(x => x.id === 'dit')
        || g.nodes.find(x => x.id === 'expert')
        || [...g.nodes].reverse().find(x => x.shape === 'tall');
  }
  autoComplete(n) {
    const R = this._recipeFor(n);
    if (!R) { window.__toast && window.__toast('no auto-complete recipe for this block', 'warn'); return; }
    const g = this._cur();
    const d = this._specD();
    const st = window.__studioState;
    const sdim = st && st.resolved ? st.resolved.obs_state_dim : null;
    const adim = st && st.resolved ? st.resolved.action_dim : null;
    const anchor = R.dir === 'in' ? this._trunkTarget() : this._trunkSource();
    if (!anchor) { window.__toast && window.__toast('no trunk found to wire into', 'err'); return; }
    const P = this.pos[this._levelKey()] || (this.pos[this._levelKey()] = {});
    const p0 = this._npos(n), pt = this._npos(anchor);
    const steps = R.steps(d, sdim, adim);
    // create the chain, positioned between the new block and the trunk anchor
    const ids = [];
    steps.forEach((s, i) => {
      const id = 'AC_' + Math.random().toString(36).slice(2, 7);
      g.nodes.push({ ...s, id, deletable: true });
      const f = (i + 1) / (steps.length + 1);
      P[id] = { x: p0.x + (pt.x - p0.x) * f, y: p0.y + (pt.y - p0.y) * f * 0.35, manual: true };
      ids.push(id);
    });
    // wire: input → chain → trunk (stack), or trunk → chain → output block
    const chain = R.dir === 'in' ? [n.id, ...ids, anchor.id] : [anchor.id, ...ids, n.id];
    for (let i = 0; i + 1 < chain.length; i++) {
      const kind = (R.dir === 'in' && i + 2 === chain.length) ? 'stack' : 'data';
      if (!g.edges.some(e => e.from === chain[i] && e.to === chain[i + 1]))
        g.edges.push({ from: chain[i], to: chain[i + 1], kind, tensor: i === 0 ? '' : `${d}` });
    }
    n.floating = false;
    // AUTO-RESIZE the fused token sequence: more tokens + a new concat band so the
    // stacking view shows exactly where the new modality lands.
    if (R.dir === 'in' && anchor.shape === 'tokencol') {
      anchor.seqlen = (anchor.seqlen || anchor.tokens || 0) + R.tok;
      anchor.tokens = Math.min(anchor.seqlen, 24);
      if (anchor.groups) anchor.groups.push({ mod: R.mod, n: R.tok, label: R.band });
      if (anchor.sub) anchor.sub = anchor.sub.replace(/\d+ tokens/, `${anchor.seqlen} tokens`);
      window.__toast && window.__toast(
        `✦ auto-completed: ${steps.map(s => s.label).join(' → ')} → ${anchor.label} (+${R.tok} ${R.band} tokens · d=${d})`, 'ok');
    } else if (R.dir === 'in' && anchor.id === 'cond') {
      anchor.sub = (anchor.sub || '') + ` ⊕ ${R.band}`;
      window.__toast && window.__toast(`✦ auto-completed into the conditioning vector (⊕ ${R.band} · d=${d})`, 'ok');
    } else {
      window.__toast && window.__toast(`✦ auto-completed: ${anchor.label} → ${steps.map(s => s.label).join(' → ')} → ${n.label}`, 'ok');
    }
    this.selected = null;
  }

  _edge(ctx, e, g) {
    const a = this._findNode(e.from), b = this._findNode(e.to); if (!a || !b) return;
    const [x0, y0] = this._anchor(a, 'r'); let [x1, y1] = this._anchor(b, 'l');
    // token DATA path flows into the TOP of tall trunks; cross-attn + adaLN
    // modulation enter from the SIDE (they aren't part of the token sequence).
    if (b.shape === 'tall' && e.kind !== 'cross_attn' && e.kind !== 'adaln') { const r = this._rect(b); [x1, y1] = [r.cx, r.y]; }
    const mx = (x0 + x1) / 2;
    ctx.beginPath(); ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1);
    if (e.kind === 'cross_attn') { ctx.strokeStyle = 'rgba(176,120,206,.85)'; ctx.lineWidth = 2.2; ctx.setLineDash([]); }
    else if (e.kind === 'adaln') { ctx.strokeStyle = 'rgba(149,120,200,.9)'; ctx.lineWidth = 2; ctx.setLineDash([2, 3]); }
    else if (e.kind === 'loop') { ctx.strokeStyle = 'rgba(120,140,200,.6)'; ctx.lineWidth = 1.6; ctx.setLineDash([3, 4]); }
    else if (e.kind === 'residual') { ctx.strokeStyle = 'rgba(150,150,150,.5)'; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); }
    else if (e.kind === 'stack') { ctx.strokeStyle = 'rgba(180,170,160,.5)'; ctx.lineWidth = 1.3; ctx.setLineDash([]); }
    else { ctx.strokeStyle = 'rgba(200,195,185,.55)'; ctx.lineWidth = 1.5; ctx.setLineDash([]); }
    ctx.stroke(); ctx.setLineDash([]);
    if (e.kind === 'cross_attn') { ctx.fillStyle = 'rgba(176,120,206,.95)'; ctx.font = '9px "Geist Mono"'; ctx.textAlign = 'center'; ctx.fillText('cross-attention', mx, (y0 + y1) / 2 - 5); }
    else if (e.kind === 'adaln') { ctx.fillStyle = 'rgba(180,150,220,.98)'; ctx.font = '8.5px "Geist Mono"'; ctx.textAlign = 'center'; ctx.fillText(e.tensor || 'adaLN modulation', mx, (y0 + y1) / 2 - 5); }
    else if (e.tensor && this.view.s > 0.55) { ctx.fillStyle = 'rgba(150,144,138,.8)'; ctx.font = '8.5px "Geist Mono"'; ctx.textAlign = 'center'; ctx.fillText(e.tensor, mx, (y0 + y1) / 2 - 4); }
  }

  _bez(x0, y0, x1, y1, t) {
    const mx = (x0 + x1) / 2, u = 1 - t;
    return [u * u * u * x0 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * x1,
            u * u * u * y0 + 3 * u * u * t * y0 + 3 * u * t * t * y1 + t * t * t * y1];
  }
  _tokens(ctx, e, g) {
    if (!this.animate) return;
    const a = this._findNode(e.from), b = this._findNode(e.to); if (!a || !b) return;
    const [x0, y0] = this._anchor(a, 'r'); let [x1, y1] = this._anchor(b, 'l');
    if (b.shape === 'tall' && e.kind !== 'cross_attn' && e.kind !== 'adaln') { const r = this._rect(b); [x1, y1] = [r.cx, r.y]; }
    const col = e.kind === 'adaln' ? { d: 'rgba(180,150,220,.95)' } : (MOD[a.modality] || MOD.trunk);
    const nT = (e.kind === 'cross_attn' || e.kind === 'adaln') ? 2 : 3;
    const spd = e.kind === 'loop' ? 0.4 : 0.45;
    for (let k = 0; k < nT; k++) {
      const tt = ((this.t * spd) + k / nT) % 1;
      const [px, py] = this._bez(x0, y0, x1, y1, tt);
      this._pill(ctx, px, py, col.d, e.kind === 'adaln' ? 1.6 : 2.4);   // adaLN = small modulation dots
    }
  }
  _pill(ctx, x, y, color, rr) {
    // ONE uniform token glyph everywhere (rr kept for legacy callers → scale)
    const w = rr ? rr * 3.2 : TOKEN.w, h = rr ? rr * 2 : TOKEN.h, rad = rr || TOKEN.r;
    ctx.beginPath(); ctx.roundRect(x - w / 2, y - h / 2, w, h, rad);
    ctx.fillStyle = color; ctx.fill();
  }
  _hash(i) { const x = Math.sin((i + 1) * 12.9898) * 43758.5453; return x - Math.floor(x); }

  // ── an elementary NN layer (the atom): consistent glyph + colour per prim ────
  _layerNode(ctx, n, r) {
    const p = PRIM[n.prim] || PRIM.linear;
    const grad = ctx.createLinearGradient(r.x, r.y, r.x, r.y + r.h);
    grad.addColorStop(0, this._light(p.f, 0.14)); grad.addColorStop(1, p.f);
    ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 7);
    ctx.fillStyle = grad; ctx.strokeStyle = p.d; ctx.lineWidth = 1.3; ctx.fill(); ctx.stroke();
    const gw = 22, gx = r.x + 7, gy = r.y + 7, gh = r.h - 14;
    this._primGlyph(ctx, n.prim, gx, gy, gw, gh, p.d);
    ctx.fillStyle = p.ink; ctx.textAlign = 'left';
    ctx.font = '600 10px Geist';
    ctx.fillText(this._clip(ctx, n.label, r.w - gw - 16), gx + gw + 5, r.cy + (n.sub ? -2 : 3));
    if (n.sub && this.view.s > 0.45) {
      ctx.font = '7px "Geist Mono"'; ctx.globalAlpha = .72;
      ctx.fillText(this._clip(ctx, n.sub, r.w - gw - 16), gx + gw + 5, r.cy + 8); ctx.globalAlpha = 1;
    }
  }
  // ── nested block (has an `expand` subgraph): a special "stack of cards" look
  //    with a MINI PREVIEW of the inner architecture — never a generic box. ─────
  _stackShadow(ctx, r, m, shape) {
    // a clearly-visible STACK of cards behind the block → "there are nested layers
    // inside; open me". Three offset sheets, each with its own outline.
    const rad = shape === 'trapezoid' ? 8 : shape === 'tall' ? 16 : shape === 'round' ? 14 : 8;
    ctx.save();
    for (let i = 3; i >= 1; i--) {
      const dx = i * 4.5, dy = -i * 4.5;
      ctx.globalAlpha = 0.9; ctx.fillStyle = this._light(m.f, 0.45 - i * 0.08);
      ctx.beginPath(); ctx.roundRect(r.x + dx, r.y + dy, r.w, r.h, rad); ctx.fill();
      ctx.globalAlpha = 0.8; ctx.strokeStyle = m.d; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.restore();
  }
  // a persistent "open me" badge on every block that has nested layers inside
  _nestBadge(ctx, r, m, n) {
    const bx = r.x + r.w - 16, by = r.y - 4, w = 20, h = 13;
    ctx.save();
    ctx.fillStyle = m.d; ctx.strokeStyle = this._light(m.f, 0.4); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, w, h, 4); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '700 8.5px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const nlay = (n.expand && n.expand.nodes && n.expand.nodes.length) || '';
    ctx.fillText('⧉' + (nlay ? ' ' + nlay : ''), bx + w / 2, by + h / 2 + 0.5);
    ctx.textBaseline = 'alphabetic';
    ctx.restore();
  }
  _nestedCard(ctx, n, r, m) {
    const grad = ctx.createLinearGradient(r.x, r.y, r.x, r.y + r.h);
    grad.addColorStop(0, this._light(m.f, 0.14)); grad.addColorStop(1, m.f);
    ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 12);
    ctx.fillStyle = grad; ctx.strokeStyle = m.d; ctx.lineWidth = 1.4; ctx.fill(); ctx.stroke();
    // header label (the ⧉ open-me badge is drawn by _nestBadge over every shape)
    ctx.fillStyle = m.ink; ctx.font = '600 10.5px Geist'; ctx.textAlign = 'left';
    ctx.fillText(this._clip(ctx, n.label, r.w - 26), r.x + 8, r.y + 14);
    if (n.sub && this.view.s > 0.5) { ctx.fillStyle = 'rgba(0,0,0,.32)'; ctx.font = '7px "Geist Mono"'; ctx.textAlign = 'left'; ctx.fillText(this._clip(ctx, n.sub, r.w - 14), r.x + 8, r.y + r.h - 4); }
    // the actual mini architecture preview (a chain of the inner layers)
    const py = r.y + 20, ph = r.h - (n.sub && this.view.s > 0.5 ? 30 : 26);
    if (ph > 7) this._miniChain(ctx, n.expand, r.x + 8, py, r.w - 16, ph, m);
  }
  _miniChain(ctx, graph, x, y, w, h, m) {
    const nodes = (graph.nodes || []).filter(nn => nn.shape !== 'tokencol').slice(0, 6);
    const k = nodes.length; if (!k) return;
    const bw = Math.max(6, Math.min(18, (w - (k - 1) * 5) / k));
    const bh = Math.min(h, 11), by = y + (h - bh) / 2;
    ctx.strokeStyle = 'rgba(120,115,108,.45)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, by + bh / 2); ctx.lineTo(x + w, by + bh / 2); ctx.stroke();
    for (let i = 0; i < k; i++) {
      const bx = x + i * (bw + (k > 1 ? (w - k * bw) / (k - 1) : 0));
      const nn = nodes[i];
      const col = nn.prim ? (PRIM[nn.prim] || PRIM.linear) : (MOD[nn.modality] || m);
      ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 3);
      ctx.fillStyle = col.f; ctx.strokeStyle = col.d; ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
    }
  }
  // little schematic icon for each primitive so the layer TYPE is readable at a glance
  _primGlyph(ctx, prim, x, y, w, h, col) {
    ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
    const cx = x + w / 2, cy = y + h / 2;
    if (prim === 'conv' || prim === 'pool') {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        ctx.globalAlpha = (prim === 'pool' && (i > 0 || j > 0)) ? .3 : .8;
        ctx.strokeRect(x + 1 + j * (w - 2) / 3, y + 2 + i * (h - 4) / 3, (w - 2) / 3 - 1, (h - 4) / 3 - 1);
      }
      if (prim === 'pool') { ctx.globalAlpha = 1; ctx.fillRect(x + 1, y + 2, (w - 2) / 3 - 1, (h - 4) / 3 - 1); }
      ctx.globalAlpha = 1;
    } else if (prim === 'linear' || prim === 'embed') {
      const ys = [y + 3, cy, y + h - 3];
      for (const yy of ys) { ctx.beginPath(); ctx.arc(x + 2, yy, 1.4, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(x + w - 2, yy, 1.4, 0, 7); ctx.fill(); }
      ctx.globalAlpha = .5; for (const a of ys) for (const b of ys) { ctx.beginPath(); ctx.moveTo(x + 2, a); ctx.lineTo(x + w - 2, b); ctx.stroke(); }
      ctx.globalAlpha = 1;
    } else if (prim === 'act') {
      ctx.beginPath();   // a GELU/ReLU-ish activation curve
      for (let i = 0; i <= w; i++) { const t = i / w * 2 - 1; const v = t < 0 ? 0.08 * Math.exp(t * 3) : t; const yy = cy + h * 0.4 - v * h * 0.7; i ? ctx.lineTo(x + i, yy) : ctx.moveTo(x + i, yy); }
      ctx.lineWidth = 1.4; ctx.stroke();
    } else if (prim === 'norm') {
      ctx.beginPath(); ctx.moveTo(x, y + h - 2); // a normalized bell
      for (let i = 0; i <= w; i++) { const t = (i / w - 0.5) * 3; const yy = y + h - 2 - Math.exp(-t * t) * (h - 4); ctx.lineTo(x + i, yy); }
      ctx.lineWidth = 1.4; ctx.stroke();
    } else if (prim === 'softmax') {
      const hs = [.35, .6, 1, .5, .25];
      hs.forEach((hv, i) => { const bw = w / hs.length; ctx.fillRect(x + i * bw + 1, y + h - hv * h, bw - 1.5, hv * h); });
    } else if (prim === 'matmul') {
      ctx.font = '600 13px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('×', cx, cy + 1); ctx.textBaseline = 'alphabetic';
    } else if (prim === 'residual') {
      ctx.beginPath(); ctx.arc(cx, cy, Math.min(w, h) / 2 - 1, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - 3, cy); ctx.lineTo(cx + 3, cy); ctx.moveTo(cx, cy - 3); ctx.lineTo(cx, cy + 3); ctx.stroke();
    } else if (prim === 'dropout') {
      const ys = [y + 3, cy, y + h - 3];
      ys.forEach((yy, i) => { ctx.globalAlpha = i === 1 ? .25 : .9; ctx.beginPath(); ctx.arc(x + 3, yy, 1.5, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(x + w - 3, yy, 1.5, 0, 7); ctx.fill(); });
      ctx.globalAlpha = 1;
    } else { ctx.strokeRect(x + 2, y + 2, w - 4, h - 4); }
    ctx.restore();
  }

  // ── diffusion / flow-matching: noise → meaningful action (animated) ──────────
  _vizNode(ctx, n, r, m) {
    // dark inner stage so the particle animation reads clearly
    ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 14);
    const grad = ctx.createLinearGradient(r.x, r.y, r.x, r.y + r.h);
    grad.addColorStop(0, 'rgba(18,17,15,.96)'); grad.addColorStop(1, 'rgba(12,11,10,.98)');
    ctx.fillStyle = grad; ctx.strokeStyle = m.d; ctx.lineWidth = 1.5; ctx.fill(); ctx.stroke();
    const pad = 6, ix = r.x + pad, iy = r.y + 16, iw = r.w - pad * 2, ih = r.h - 22;
    if (n.viz === 'diffusion') this._diffusionViz(ctx, ix, iy, iw, ih, m, n.steps || 10);
    else this._flowViz(ctx, ix, iy, iw, ih, m, n.steps || 10);
    // title + sub
    ctx.fillStyle = m.f; ctx.font = '600 10px Geist'; ctx.textAlign = 'left'; ctx.fillText(n.label, r.x + 8, r.y + 12);
    ctx.fillStyle = 'rgba(190,184,176,.7)'; ctx.font = '7px "Geist Mono"'; ctx.textAlign = 'right'; ctx.fillText(n.sub || '', r.x + r.w - 7, r.y + 12);
  }
  // Gaussian noise cloud (t=0) iteratively denoising onto a clean action curve (t=1)
  _diffusionViz(ctx, x, y, w, h, m, steps) {
    const N = 46, cy = y + h / 2;
    const tau = this.animate ? (this.t * 0.22) % 1.25 : 1;   // brief hold past 1
    const p = Math.min(1, tau);
    const e = p * p * (3 - 2 * p);                            // smoothstep
    // target: a smooth multi-DOF action trajectory (a few phase-shifted sines)
    const target = (i) => {
      const u = i / (N - 1);
      const lane = i % 3;
      return cy + Math.sin(u * Math.PI * 2 + lane * 1.6) * h * 0.32 * (1 - lane * 0.18);
    };
    // step ticks (the discrete reverse-diffusion iterations)
    ctx.strokeStyle = 'rgba(242,238,230,.05)';
    for (let s = 0; s <= steps && s <= 12; s++) { const sx = x + s / Math.min(steps, 12) * w; ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(sx, y + h); ctx.stroke(); }
    for (let i = 0; i < N; i++) {
      const tx = x + (i / (N - 1)) * w, ty = target(i);
      const nx = x + this._hash(i) * w, ny = y + this._hash(i + 91) * h;   // pure noise
      const px = nx + (tx - nx) * e, py = ny + (ty - ny) * e;
      const col = this._mix('#B7B2A6', m.d, e);
      ctx.globalAlpha = 0.35 + 0.6 * e;
      this._pill(ctx, px, py, col, 1.6); ctx.globalAlpha = 1;
    }
    // when converged, trace the clean action curve
    if (e > 0.85) {
      ctx.strokeStyle = m.f; ctx.globalAlpha = (e - 0.85) / 0.15; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let i = 0; i < N; i++) { const X = x + (i / (N - 1)) * w, Y = target(i); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.fillStyle = 'rgba(160,154,146,.7)'; ctx.font = '6.5px "Geist Mono"'; ctx.textAlign = 'left';
    ctx.fillText('x_T ~ 𝒩', x + 1, y + h - 1); ctx.textAlign = 'right'; ctx.fillText('x_0', x + w - 1, y + h - 1);
  }
  // flow matching: particles integrate a velocity field from noise (t=0) → data (t=1)
  _flowViz(ctx, x, y, w, h, m, steps) {
    const N = 26, cy = y + h / 2;
    const t = this.animate ? (this.t * 0.28) % 1.2 : 1;
    const tc = Math.min(1, t);
    const srcY = (i) => cy + (this._hash(i) - 0.5) * h * 0.9;               // Gaussian blob (left)
    const dstY = (i) => cy + Math.sin(i / (N - 1) * Math.PI * 2) * h * 0.34; // data manifold (right)
    // faint velocity field (straight conditional paths)
    ctx.strokeStyle = 'rgba(95,179,156,.16)'; ctx.lineWidth = 1;
    for (let i = 0; i < N; i += 2) { ctx.beginPath(); ctx.moveTo(x, srcY(i)); ctx.lineTo(x + w, dstY(i)); ctx.stroke(); }
    // moving particles
    for (let i = 0; i < N; i++) {
      const px = x + tc * w, py = srcY(i) + (dstY(i) - srcY(i)) * tc;
      const col = this._mix('#B7B2A6', m.d, tc);
      ctx.globalAlpha = 0.4 + 0.55 * tc; this._pill(ctx, px, py, col, 1.7); ctx.globalAlpha = 1;
    }
    if (tc > 0.85) {   // data curve appears as particles land
      ctx.strokeStyle = m.f; ctx.globalAlpha = (tc - 0.85) / 0.15; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let i = 0; i < N; i++) { const X = x + (i / (N - 1)) * w, Y = dstY(i); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.fillStyle = 'rgba(160,154,146,.7)'; ctx.font = '6.5px "Geist Mono"'; ctx.textAlign = 'left';
    ctx.fillText('t=0 noise', x + 1, y + h - 1); ctx.textAlign = 'right'; ctx.fillText('t=1 data', x + w - 1, y + h - 1);
  }
  _mix(a, b, t) {   // blend two hex colours
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const ar = (pa >> 16) & 255, ag = (pa >> 8) & 255, ab = pa & 255;
    const br = (pb >> 16) & 255, bg = (pb >> 8) & 255, bb = pb & 255;
    return `rgb(${ar + (br - ar) * t | 0},${ag + (bg - ag) * t | 0},${ab + (bb - ab) * t | 0})`;
  }

  _node(ctx, n) {
    const r = this._rect(n), m = MOD[n.modality] || MOD.trunk;
    ctx.save();
    const drawBox = (path) => {
      const grad = ctx.createLinearGradient(r.x, r.y, r.x, r.y + r.h);
      grad.addColorStop(0, this._light(m.f, 0.06)); grad.addColorStop(1, m.f);
      ctx.fillStyle = grad; ctx.strokeStyle = m.d; ctx.lineWidth = 1.4;
      path(); ctx.fill(); ctx.stroke();
    };
    // nested blocks (anything with an `expand` subgraph) get a "stack of cards"
    // shadow behind them → visually distinct from a flat/generic block.
    if (n.expand && n.shape !== 'tokencol' && !n.viz) this._stackShadow(ctx, r, m, n.shape);
    if (n.expand && (n.shape === 'round' || n.shape === 'rect')) {
      this._nestedCard(ctx, n, r, m);            // header + mini inner-architecture preview
    } else if (n.shape === 'trapezoid') {
      drawBox(() => { ctx.beginPath(); ctx.moveTo(r.x, r.y + 6); ctx.lineTo(r.x + r.w, r.y); ctx.lineTo(r.x + r.w, r.y + r.h); ctx.lineTo(r.x, r.y + r.h - 6); ctx.closePath(); });
    } else if (n.shape === 'tall') {
      drawBox(() => { ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 16); });
      this._flowDown(ctx, r, m);
    } else if (n.shape === 'tokencol') {
      this._tokencol(ctx, n, r, m);
    } else if (n.shape === 'actions') {
      // action out — black panel, module-coloured TARGET-joint line graph (the
      // dataset's `action` field: absolute target joint positions)
      const aidx = this.spec && this.spec.output && this.spec.output.action_indices;
      const arr = this.ep && this.ep.arrays && this.ep.arrays.action;
      this._jointPanel(ctx, r, arr, aidx, 'action · target pos');
    } else if (n.shape === 'image') {
      // play the ACTUAL camera video at the input block (double-buffered → no flicker)
      drawBox(() => { ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 8); });
      const im = this._camImg(n.camera);
      if (im) {
        ctx.save(); ctx.beginPath(); ctx.roundRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2, 7); ctx.clip();
        const ar = im.naturalWidth / im.naturalHeight, rr = r.w / r.h;
        let dw = r.w, dh = r.h, dx = r.x, dy = r.y;
        if (ar > rr) { dw = r.h * ar; dx = r.x - (dw - r.w) / 2; } else { dh = r.w / ar; dy = r.y - (dh - r.h) / 2; }
        ctx.drawImage(im, dx, dy, dw, dh); ctx.restore();
        ctx.strokeStyle = m.d; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 8); ctx.stroke();
      } else {
        ctx.strokeStyle = m.d; ctx.globalAlpha = .5; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(r.x + 8, r.y + r.h - 12); ctx.lineTo(r.x + r.w * .4, r.y + 14); ctx.lineTo(r.x + r.w * .62, r.y + r.h - 20); ctx.lineTo(r.x + r.w - 8, r.y + 10); ctx.stroke(); ctx.globalAlpha = 1;
      }
    } else if (n.id === 'in_prop') {
      // proprioception in — black panel, module-coloured CURRENT joint-state line
      // graph (the dataset's `observation.state` field)
      const sidx = this.spec && this.spec.inputs && this.spec.inputs.state_indices;
      const arr = this.ep && this.ep.arrays && this.ep.arrays['observation.state'];
      this._jointPanel(ctx, r, arr, sidx, 'obs · joint state');
    } else if (n.id === 'in_vae') {
      // VAE style-encoder input — the GT action chunk (dataset `action`), same
      // module-coloured panel as the output so the training target is explicit
      const aidx = this.spec && this.spec.output && this.spec.output.action_indices;
      const arr = this.ep && this.ep.arrays && this.ep.arrays.action;
      this._jointPanel(ctx, r, arr, aidx, 'action · GT chunk');
    } else if (n.shape === 'layer') {
      this._layerNode(ctx, n, r);            // an elementary NN layer (atom)
    } else if (n.viz) {
      this._vizNode(ctx, n, r, m);           // diffusion / flow-matching animation
    } else {
      const rad = n.shape === 'round' ? 20 : 8;
      drawBox(() => { ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, rad); });
    }
    // blocks that draw their OWN label/caption (skip the generic label pass)
    const jointPanel = (n.id === 'in_prop' || n.id === 'in_vae' || n.shape === 'actions');
    const nestedCard = n.expand && (n.shape === 'round' || n.shape === 'rect');
    const selfLabel = jointPanel || n.shape === 'layer' || n.viz || nestedCard;
    const dataShown = (n.shape === 'image' && this._camImg(n.camera));
    if (dataShown) this._caption(ctx, r, n.label, m);
    // labels
    if (n.shape !== 'tokencol' && n.shape !== 'actions' && !selfLabel && !dataShown) {
      ctx.fillStyle = m.ink; ctx.textAlign = 'center';
      ctx.font = '600 11px Geist, sans-serif';
      ctx.fillText(this._clip(ctx, n.label, r.w - 10), r.cx, r.cy + (n.sub ? -3 : 3));
      if (n.sub && this.view.s > 0.5) { ctx.font = '8px "Geist Mono"'; ctx.globalAlpha = .8; ctx.fillText(this._clip(ctx, n.sub, r.w - 6), r.cx, r.cy + 9); ctx.globalAlpha = 1; }
    }
    // nested block → a clear "⧉ N" open-me badge (works on every shape)
    if (n.expand && !n.floating) this._nestBadge(ctx, r, m, n);
    // a freshly-added, not-yet-wired block → dashed teal outline + hint
    if (n.floating) {
      ctx.strokeStyle = 'rgba(95,179,156,.9)'; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.roundRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4, 9); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(95,179,156,.95)'; ctx.font = '7px "Geist Mono"'; ctx.textAlign = 'center';
      ctx.fillText('drag block · or drag a port', r.cx, r.y - 4);
      // ✦ auto-complete chip (when a recipe exists for this block)
      if (this._recipeFor(n)) {
        const c = this._chipRect(n, r);
        ctx.fillStyle = 'rgba(95,179,156,.92)';
        ctx.beginPath(); ctx.roundRect(c.x, c.y, c.w, c.h, 8); ctx.fill();
        ctx.fillStyle = '#0d0c0b'; ctx.font = '700 8.5px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('✦ auto-complete', c.x + c.w / 2, c.y + c.h / 2 + 0.5); ctx.textBaseline = 'alphabetic';
      }
    }
    // input/output PORTS on floating + selected blocks (drag a wire to connect)
    if (this._hasPorts(n)) {
      for (const side of ['l', 'r']) {
        const [px, py] = this._portPos(n, side);
        ctx.beginPath(); ctx.arc(px, py, 5, 0, 7);
        ctx.fillStyle = 'rgba(95,179,156,.95)'; ctx.fill();
        ctx.strokeStyle = 'rgba(13,12,11,.9)'; ctx.lineWidth = 1.6; ctx.stroke();
        ctx.fillStyle = '#0d0c0b'; ctx.font = '700 7px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(side === 'l' ? '‹' : '›', px, py + 0.5); ctx.textBaseline = 'alphabetic';
      }
    }
    // editable node → a small rust "▾" chip (click opens a real-LeRobot picker)
    if (n.edit && !this.readOnly) {
      ctx.fillStyle = 'rgba(194,91,42,.95)';
      ctx.beginPath(); ctx.roundRect(r.x + r.w - 15, r.y + 3, 12, 10, 3); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 8px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('▾', r.x + r.w - 9, r.y + 8.4); ctx.textBaseline = 'alphabetic';
    }
    // SELECTED block → accent glow outline (so it's clear what Del will remove)
    if (this.selected === n && !this.readOnly) {
      ctx.save(); ctx.strokeStyle = 'rgba(194,91,42,.95)'; ctx.lineWidth = 2; ctx.shadowColor = 'rgba(194,91,42,.7)'; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.roundRect(r.x - 2.5, r.y - 2.5, r.w + 5, r.h + 5, 11); ctx.stroke(); ctx.restore();
    }
    // hover/selected affordances: delete + resize — SUPPRESSED while read-only
    if ((this.hover === n || this.selected === n) && !this.drag && !this.readOnly) {
      const del = n.deletable;
      ctx.beginPath(); ctx.arc(r.x + r.w, r.y, 8.5, 0, 7); ctx.fillStyle = del ? 'rgba(210,59,42,.95)' : 'rgba(120,115,108,.7)'; ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 12px Geist'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(del ? '×' : '🔒', r.x + r.w, r.y + 0.5); ctx.textBaseline = 'alphabetic';
      // resize grip
      ctx.strokeStyle = 'rgba(242,238,230,.7)'; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(r.x + r.w - 2, r.y + r.h - 8); ctx.lineTo(r.x + r.w - 8, r.y + r.h - 2);
      ctx.moveTo(r.x + r.w - 2, r.y + r.h - 3); ctx.lineTo(r.x + r.w - 3, r.y + r.h - 2); ctx.stroke();
    }
    ctx.restore();
  }

  _flowDown(ctx, r, m) {           // uniform tokens streaming downward inside a transformer
    if (!this.animate) return;
    const cols = 2, rows = 4;
    for (let c = 0; c < cols; c++) for (let k = 0; k < rows; k++) {
      const x = r.x + r.w * (c + 1) / (cols + 1);
      const t = ((this.t * 0.5) + k / rows + c * 0.13) % 1;
      const y = r.y + 12 + t * (r.h - 24);
      ctx.globalAlpha = 0.5 + 0.4 * Math.sin(t * Math.PI);
      this._pill(ctx, x, y, m.d); ctx.globalAlpha = 1;   // TOKEN-sized
    }
    ctx.fillStyle = m.ink; ctx.font = '600 10px Geist'; ctx.textAlign = 'center';
  }
  _tokencol(ctx, n, r, m) {
    // A vertical STACK of tokens, every glyph the SAME uniform size (TOKEN), drawn
    // top-to-bottom in concat order and coloured by source modality — so the
    // spatial token stacking (patch ▸ state ▸ latent) is unambiguous. `bounds`
    // marks where one group ends and the next begins (the concat seams).
    let segs, bounds = [];
    if (n.groups && n.groups.length) {
      const totalN = n.groups.reduce((s, g) => s + g.n, 0);
      const shown = clamp(Math.min(totalN, 15), n.groups.length, 15);
      segs = [];
      for (const g of n.groups) {
        const k = Math.max(1, Math.round(shown * g.n / totalN));
        for (let i = 0; i < k; i++) segs.push({ mod: g.mod, label: g.label });
        bounds.push(segs.length);
      }
    } else {
      const nt = clamp(n.tokens || 3, 1, 12);
      segs = Array.from({ length: nt }, () => ({ mod: n.modality }));
    }
    // fit uniform tokens into the column height; sample down if too many
    const pitch = TOKEN.h + 3;
    const maxFit = Math.max(1, Math.floor((r.h - 6) / pitch));
    if (segs.length > maxFit) segs = Array.from({ length: maxFit }, (_, i) => segs[Math.floor(i * segs.length / maxFit)]);
    const nt = segs.length;
    const y0 = r.cy - (nt * pitch) / 2 + pitch / 2;
    for (let i = 0; i < nt; i++) {
      const cm = MOD[segs[i].mod] || m;
      const y = y0 + i * pitch;
      const pulse = this.animate ? 0.74 + 0.26 * Math.sin(this.t * 3 + i * 0.5) : 1;
      ctx.globalAlpha = pulse;
      ctx.beginPath(); ctx.roundRect(r.cx - TOKEN.w / 2, y - TOKEN.h / 2, TOKEN.w, TOKEN.h, TOKEN.r);
      ctx.fillStyle = cm.f; ctx.strokeStyle = cm.d; ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
      ctx.globalAlpha = 1;
      // concat seam: a faint tick between groups (skip the last boundary)
      if (bounds.length && bounds.includes(i + 1) && i + 1 < nt) {
        ctx.strokeStyle = 'rgba(242,238,230,.5)'; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(r.cx - TOKEN.w / 2 - 3, y + pitch / 2); ctx.lineTo(r.cx + TOKEN.w / 2 + 3, y + pitch / 2); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    ctx.fillStyle = 'rgba(190,184,176,.98)'; ctx.font = '8px "Geist Mono"'; ctx.textAlign = 'center';
    ctx.fillText(n.label + (n.seqlen ? ` ·${n.seqlen}` : ''), r.cx, r.y - 4);
  }

  // proprio-in / action-out: a black joint-scope panel (data-curation style) with
  // lines colour-graded BY MODULE (arm/hand/base/neck/torso/lift), + a compact
  // module legend. `indices` are absolute joint indices → module via _idxGrp.
  _jointPanel(ctx, r, series, indices, title) {
    // dark background panel + hairline border
    ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 9);
    ctx.fillStyle = '#0c0b0a'; ctx.fill();
    ctx.strokeStyle = 'rgba(242,238,230,.18)'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.roundRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2, 8); ctx.clip();

    const dim = series && series.length ? series[0].length : Infinity;
    let cols = (indices && indices.length) ? indices.filter(i => i < dim)
             : (series && series.length ? [...Array(series[0].length).keys()] : []);
    const MAXL = 48;
    if (cols.length > MAXL) { const step = cols.length / MAXL; cols = Array.from({ length: MAXL }, (_, i) => cols[Math.floor(i * step)]); }
    const present = new Set();      // modules actually shown (for the legend)
    for (const c of cols) { const g = this._idxGrp[c]; if (g) present.add(g); }

    if (series && series.length && cols.length) {
      let mn = 1e9, mx = -1e9;
      for (const row of series) for (const c of cols) { const v = row[c]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx === mn) { mx += 1; mn -= 1; }
      const top = r.y + 4, bot = r.y + r.h - 12;      // leave room for the legend strip
      const px = (i) => r.x + 3 + i / (series.length - 1) * (r.w - 6);
      const py = (v) => bot - (v - mn) / (mx - mn) * (bot - top);
      for (const c of cols) {
        const grp = this._idxGrp[c];
        ctx.strokeStyle = (grp && GROUP_COL[grp]) || '#9aa'; ctx.globalAlpha = 0.9; ctx.lineWidth = 1;
        ctx.beginPath();
        series.forEach((row, i) => { const X = px(i), Y = py(row[c]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (this.ep) { const fx = px(this.ep.frame || 0); ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(fx, top); ctx.lineTo(fx, bot); ctx.stroke(); }
    } else {
      ctx.fillStyle = 'rgba(151,144,138,.5)'; ctx.font = '8px Geist'; ctx.textAlign = 'center';
      ctx.fillText('loading…', r.cx, r.cy);
    }
    // legend strip along the bottom: a swatch per present module
    const order = (this._groups || []).map(g => g.name).filter(n => present.has(n));
    ctx.font = '6.5px "Geist Mono"'; ctx.textAlign = 'left';
    let lx = r.x + 5; const ly = r.y + r.h - 6;
    for (const g of order) {
      ctx.fillStyle = GROUP_COL[g] || '#9aa'; ctx.fillRect(lx, ly - 4, 5, 5);
      const lab = (this._groups.find(x => x.name === g) || {}).label || g;
      ctx.fillStyle = 'rgba(214,208,198,.85)'; ctx.fillText(lab, lx + 7, ly);
      lx += 11 + ctx.measureText(lab).width;
      if (lx > r.x + r.w - 24) break;
    }
    ctx.restore();
    // title chip (top-left)
    this._caption(ctx, r, title, null);
    // DOF count (top-right)
    ctx.fillStyle = 'rgba(214,208,198,.9)'; ctx.font = '7.5px "Geist Mono"'; ctx.textAlign = 'right';
    ctx.fillText(cols.length + '-D', r.x + r.w - 5, r.y + 12);
  }

  _caption(ctx, r, text, m) {
    ctx.font = '7.5px "Geist Mono"'; const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = 'rgba(13,12,11,.6)'; ctx.beginPath(); ctx.roundRect(r.x + 3, r.y + 3, w, 12, 4); ctx.fill();
    ctx.fillStyle = '#efeae3'; ctx.textAlign = 'left'; ctx.fillText(text, r.x + 7, r.y + 12);
  }

  _clip(ctx, s, w) { if (ctx.measureText(s).width <= w) return s; while (s.length > 1 && ctx.measureText(s + '…').width > w) s = s.slice(0, -1); return s + '…'; }
  _light(hex, a) { const n = parseInt(hex.slice(1), 16); let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255; r += (255 - r) * a; g += (255 - g) * a; b += (255 - b) * a; return `rgb(${r | 0},${g | 0},${b | 0})`; }
}
