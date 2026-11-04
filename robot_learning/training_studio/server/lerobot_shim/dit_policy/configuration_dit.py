"""
DiTConfig — a real LeRobot policy config for a Diffusion Transformer (DiT) action
head. Subclasses DiffusionConfig to reuse all the machinery (RGB encoder,
observation stacking, DDPM scheduler, normalization mapping, delta indices) and
only adds the transformer-denoiser knobs. Registered as ``policy.type = "dit"``.
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("dit")
@dataclass
class DiTConfig(DiffusionConfig):
    # ── DiT transformer denoiser ──────────────────────────────────────────────
    dim_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    transformer_dropout: float = 0.1

    # Transformer-friendly diffusion defaults (no U-Net downsampling constraint).
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int | None = 10
    crop_shape: tuple[int, int] | None = None   # use the full frame
    optimizer_lr: float = 1e-4

    def __post_init__(self):
        # Deliberately skip DiffusionConfig.__post_init__: its U-Net horizon /
        # down_dims divisibility check does not apply to a transformer denoiser.
        PreTrainedConfig.__post_init__(self)
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"`prediction_type` must be epsilon|sample. Got {self.prediction_type}.")
        if self.noise_scheduler_type not in ("DDPM", "DDIM"):
            raise ValueError(f"`noise_scheduler_type` must be DDPM|DDIM. Got {self.noise_scheduler_type}.")
        if self.dim_model % self.n_heads != 0:
            raise ValueError(f"dim_model {self.dim_model} must be divisible by n_heads {self.n_heads}.")
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must be positive. Got {self.crop_shape}.")
