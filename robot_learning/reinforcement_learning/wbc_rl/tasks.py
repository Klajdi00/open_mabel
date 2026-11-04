"""tasks.py — registers NOVA-WBC-v0 and its rsl_rl PPO config (asymmetric actor / critic).

Passed to Isaac Lab's train/play as  --external_callback tasks.register  (with this folder on PYTHONPATH).
"""
import gymnasium as gym
import torch
from rsl_rl.algorithms import PPO

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg


class CapsPPO(PPO):
    """PPO + a temporal smoothness loss on the policy's MEAN action (CAPS, Mysore et al. 2021).

    Reward terms only see sampled actions, which the exploration noise dominates; the deterministic policy that runs on
    the robot still flipped its arm commands every 20 ms (v21: 8-25 Hz command power ~4e7, the 3 Hz filter hid it).
    After each PPO update: CAPS_STEPS gradient steps on |mu(o_t) - mu(o_t+1)|^2 over consecutive rollout steps."""
    CAPS_COEF, CAPS_STEPS, CAPS_PAIRS = 0.5, 2, 16384

    def update(self):
        st = self.storage
        obs, done = st.observations, st.dones[:-1, :, 0].bool()
        stats = super().update()
        T, N = done.shape
        ok = (~done).flatten().nonzero()[:, 0]
        tot = 0.0
        for _ in range(self.CAPS_STEPS):
            k = ok[torch.randint(len(ok), (min(self.CAPS_PAIRS, len(ok)),), device=self.device)]
            t, n = k // N, k % N
            a0 = self.actor(obs[t, n]); a1 = self.actor(obs[t + 1, n])
            w = torch.ones(a0.shape[-1], device=self.device); w[:4] = 0.3   # base twist + lift rate are smoothed downstream too
            loss = self.CAPS_COEF * ((a0 - a1).square() * w).sum(-1).mean()
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step(); tot += loss.item()
        stats["caps"] = tot / self.CAPS_STEPS
        return stats


@configclass
class NovaWbcPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 250
    experiment_name = "nova_wbc"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(hidden_dims=[512, 384, 256], activation="elu", obs_normalization=True,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6))
    critic = RslRlMLPModelCfg(hidden_dims=[768, 512, 256], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(class_name="tasks:CapsPPO", value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
                                     learning_rate=5e-4, schedule="adaptive", gamma=0.99, lam=0.95,
                                     desired_kl=0.01, max_grad_norm=1.0,
                                     # mirror loss: the action on a mirrored situation = the mirrored action (env.mirror_batch);
                                     # mirrored PPO samples are too far off-policy for a policy that is already lopsided
                                     symmetry_cfg=RslRlSymmetryCfg(use_mirror_loss=True, mirror_loss_coeff=0.2,
                                                                   data_augmentation_func="env:mirror_batch"))


def register():
    if "NOVA-WBC-v0" not in gym.registry:
        gym.register(id="NOVA-WBC-v0", entry_point="env:NovaWbcEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": "env:NovaWbcEnvCfg",
                             "rsl_rl_cfg_entry_point": "tasks:NovaWbcPPOCfg", "default_agent": "rsl_rl"})
