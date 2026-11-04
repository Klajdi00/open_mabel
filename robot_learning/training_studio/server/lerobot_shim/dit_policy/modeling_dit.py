"""
DiTPolicy — a real, trainable Diffusion-Transformer policy for LeRobot.

Reuses the stock DiffusionPolicy pipeline (RGB encoder, observation stacking,
DDPM scheduler, loss, action queue) and replaces ONLY the U-Net denoiser with a
DiT transformer (adaLN-Zero conditioning), so the animated architecture diagram
(transformer blocks + denoising loop) reflects the actual network. The denoiser
matches the U-Net interface exactly: forward(x, timestep, global_cond) -> noise.
"""
import math

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionModel, DiffusionPolicy, DiffusionRgbEncoder, _make_noise_scheduler)
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_dit import DiTConfig


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiTBlock(nn.Module):
    """A DiT block: adaLN-Zero self-attention + MLP, conditioned on (t + obs)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        a = _modulate(self.norm1(x), sh1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(a, a, a, need_weights=False)[0]
        m = _modulate(self.norm2(x), sh2, sc2)
        x = x + g2.unsqueeze(1) * self.mlp(m)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        for m in (self.ada[-1], self.linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class DiTDenoiser(nn.Module):
    """Transformer denoiser with the DiffusionConditionalUnet1d interface."""

    def __init__(self, config: DiTConfig, global_cond_dim: int):
        super().__init__()
        d = config.dim_model
        action_dim = config.action_feature.shape[0]
        self.input_proj = nn.Linear(action_dim, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.horizon, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.time_emb = SinusoidalPosEmb(d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_proj = nn.Linear(global_cond_dim, d)
        self.blocks = nn.ModuleList(
            [DiTBlock(d, config.n_heads, config.transformer_dropout) for _ in range(config.n_layers)])
        self.final = FinalLayer(d, action_dim)

    def forward(self, x: Tensor, timestep, global_cond: Tensor | None = None) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device)
        timestep = timestep.expand(x.shape[0]).to(x.device)
        c = self.time_mlp(self.time_emb(timestep))
        if global_cond is not None:
            c = c + self.cond_proj(global_cond)
        h = self.input_proj(x) + self.pos_emb[:, : x.shape[1]]
        for blk in self.blocks:
            h = blk(h, c)
        return self.final(h, c)


class DiTDiffusionModel(DiffusionModel):
    """Same as DiffusionModel but the denoiser is a DiT (no U-Net is built)."""

    def __init__(self, config: DiTConfig):
        nn.Module.__init__(self)   # skip DiffusionModel.__init__ (it builds a U-Net)
        self.config = config
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = DiTDenoiser(config, global_cond_dim * config.n_obs_steps)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start, beta_end=config.beta_end,
            beta_schedule=config.beta_schedule, clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range, prediction_type=config.prediction_type)
        self.num_inference_steps = (config.num_inference_steps
                                    or self.noise_scheduler.config.num_train_timesteps)


class DiTPolicy(DiffusionPolicy):
    """Diffusion Transformer policy — DiffusionPolicy with a DiT denoiser."""

    config_class = DiTConfig
    name = "dit"

    def __init__(self, config: DiTConfig, **kwargs):
        # mirror DiffusionPolicy.__init__ but with the DiT diffusion model
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = DiTDiffusionModel(config)
        self.reset()
