"""
Diffusion Policy using DiTForDiffusion backbone + TimmObsEncoder.

Obs encoder (e.g. DINOv2 ViT-B) produces a flat feature vector of shape
(B, obs_feature_dim), which includes both RGB features and low-dim obs.
A learned linear projection maps this to (B, n_obs_tokens, n_emb) conditioning
tokens which are fed to DiT via cross-attention.

Training:
  - DDIM scheduler, 100 train timesteps, epsilon-prediction
  - input perturbation 0.1
  - N diffusion noise samples per obs encoder forward pass (default 8)

Inference:
  - 16 DDIM denoising steps
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.dit_for_diffusion import DiTForDiffusion
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionDiTTimmPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDIMScheduler,
        obs_encoder: TimmObsEncoder,
        # DiT backbone hyperparams
        n_layer: int = 10,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: int = 4,
        p_drop_attn: float = 0.1,
        # training / inference
        num_inference_steps: int = 16,
        input_pertub: float = 0.1,
        train_diffusion_n_samples: int = 8,
        # number of conditioning tokens the flat obs feature is projected into.
        # None -> obs_horizon (legacy default). Increase to give vision more
        # capacity in the DiT cross-attention instead of cramming CLS + proprio
        # into obs_horizon (=2) tokens. See memory umi-single-tron-vision-ignored.
        n_obs_tokens: int = None,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["action"]["horizon"]

        # obs encoder outputs flat (B, obs_feature_dim) —
        # this includes both RGB features AND low-dim obs concatenated.
        obs_feature_dim = int(np.prod(obs_encoder.output_shape()))

        # Project arbitrary obs_feature_dim → n_cond_tokens × n_emb.
        # n_cond_tokens = To (one token per observation timestep) is a
        # reasonable default, but we expose it as a configurable parameter.
        obs_horizon = shape_meta["obs"][
            next(k for k in shape_meta["obs"] if shape_meta["obs"][k].get("type","low_dim") == "rgb")
        ]["horizon"]
        # default: one conditioning token per obs step (legacy behavior);
        # override via config to widen the vision bottleneck.
        if n_obs_tokens is None:
            n_obs_tokens = obs_horizon

        self.obs_proj = nn.Linear(obs_feature_dim, n_obs_tokens * n_emb)

        model = DiTForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            mlp_ratio=mlp_ratio,
            max_cond_tokens=n_obs_tokens + 1,   # obs tokens + 1 time token
            p_drop_attn=p_drop_attn,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_emb = n_emb
        self.n_obs_tokens = n_obs_tokens
        self.input_pertub = input_pertub
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.num_inference_steps = num_inference_steps

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #
    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run obs encoder + linear projection → (B, n_obs_tokens, n_emb)."""
        obs_flat = self.obs_encoder(nobs)               # (B, obs_feature_dim)
        B = obs_flat.shape[0]
        obs_proj = self.obs_proj(obs_flat)              # (B, n_obs_tokens * n_emb)
        return obs_proj.view(B, self.n_obs_tokens, self.n_emb)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: torch.Tensor,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, t, cond=cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix=None,
    ) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        cond = self._encode_obs(nobs)   # (B, n_obs_tokens, n_emb)

        cond_data = torch.zeros(
            (B, self.action_horizon, self.action_dim),
            device=self.device, dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=cond,
        )

        action_pred = self.normalizer["action"].unnormalize(nsample)
        return {"action": action_pred, "action_pred": action_pred}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        cond = self._encode_obs(nobs)           # (B, n_obs_tokens, n_emb)

        # repeat obs & actions for multiple diffusion samples per forward pass
        if self.train_diffusion_n_samples != 1:
            cond = torch.repeat_interleave(
                cond, repeats=self.train_diffusion_n_samples, dim=0
            )
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_diffusion_n_samples, dim=0
            )

        trajectory = nactions
        noise = torch.randn_like(trajectory)
        # input perturbation (DDPM-IP) to alleviate exposure bias
        noise_perturbed = noise + self.input_pertub * torch.randn_like(trajectory)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_perturbed, timesteps
        )

        pred = self.model(noisy_trajectory, timesteps, cond=cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
