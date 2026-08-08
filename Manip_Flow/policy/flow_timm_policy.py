"""Flow-matching drop-in replacement for UMI's DiffusionUnetTimmPolicy.

Mirrors ``diffusion_policy.policy.diffusion_unet_timm_policy`` line-for-line
where possible so it is a drop-in for ``TrainDiffusionUnetImageWorkspace``
(which only touches ``policy(batch)``, ``policy.predict_action``,
``policy.set_normalizer`` and the ``.model`` / ``.obs_encoder`` attributes for
optimizer param groups). Only the probabilistic head changes:

    DDPM/DDIM epsilon-prediction  ->  conditional flow matching (rectified flow)

Training:   x0 ~ N(0, I),  x1 = normalized action chunk,  t ~ U(0,1)
            x_t = (1 - t) x0 + t x1,   target v = x1 - x0
            loss = MSE(model(x_t, t), v)
Sampling:   x = x0,  Euler-integrate dx = v(x, t) dt over t: 0 -> 1
            (num_inference_steps steps; 4-8 is usually enough, vs 16 DDIM)

Backbones (``backbone=``):
  * ``'dit'`` (default): FlowDiT1D, adaLN-zero transformer over action-frame
    tokens. No horizon divisibility constraint -- required for configurations
    such as 17 tokens @10Hz, which produce a 50-frame provider window after
    30fps resampling/history merge and break UNet's divisibility constraint.
    Also matches the lower-level Prior_Recon prior (transformer + flow).
  * ``'unet'``: UMI's ConditionalUnet1D unchanged (horizon must be divisible
    by 2^(len(down_dims)-1)). Kept for A/B parity with the UMI reference.

Continuous t in [0, 1] is scaled by ``time_embed_scale`` (Flux/SD3 convention)
before the sinusoidal step embedding of either backbone.

Action-prefix inpainting (same flag semantics as UMI): conditioned entries are
pinned to their interpolant ``(1 - t) x0 + t x1_known`` before every Euler
step, so the constraint is enforced on the same probability path the model was
trained on.

Requires the ``universal_manipulation_interface`` repo root on ``sys.path``
(it provides the ``diffusion_policy`` package).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from Manip_Flow.model.common.normalizer import LinearNormalizer
from Manip_Flow.model.diffusion.conditional_unet1d import ConditionalUnet1D
from Manip_Flow.model.vision.timm_obs_encoder import TimmObsEncoder
from Manip_Flow.policy.base_image_policy import BaseImagePolicy

from Manip_Flow.model.flow_dit_1d import FlowDiT1D
from Manip_Flow.policy import rtc_flow


class FlowTimmPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: TimmObsEncoder,
        num_inference_steps: int = 8,
        obs_as_global_cond: bool = True,
        backbone: str = "dit",
        # dit backbone
        dit_d_model: int = 512,
        dit_depth: int = 6,
        dit_n_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.0,
        # Must match time_embed_scale, exactly as unet_time_log_scale does for
        # the unet. See FlowDiT1D.
        dit_time_log_scale: float = 10000.0,
        # unet backbone (UMI reference parity)
        diffusion_step_embed_dim: int = 128,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        unet_time_log_scale: float = 10000.0,
        # flow head
        time_embed_scale: float = 1000.0,
        time_sample: str = "uniform",  # 'uniform' or 'logit_normal'
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        inpaint_fixed_action_prefix: bool = False,
        train_flow_n_samples: int = 1,
        rtc_execution_horizon: int = 12,
        rtc_max_guidance_weight: float = 5.0,
        rtc_prefix_schedule: str = "exp",
        # Project the flat obs feature ONCE instead of letting every
        # ConditionalResidualBlock1D re-project it. 0 disables.
        cond_bottleneck_dim: int = 0,
        # Classifier-free guidance. Training swaps the whole conditioning for a
        # learnable null embedding with this probability; inference extrapolates
        # away from it when cfg_scale > 1.
        cond_dropout_prob: float = 0.0,
        cfg_scale: float = 1.0,
        # Down-weight the far end of the action chunk, which deploy never
        # executes and cannot predict from one frame anyway.
        loss_horizon_full_steps: int = 0,
        loss_horizon_tail_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["action"]["horizon"]
        obs_feature_dim = int(np.prod(obs_encoder.output_shape()))

        # conditioning bottleneck. ConditionalUnet1D builds one
        # Linear(step_embed + global_cond -> 2 * out_channels) PER residual
        # block, so a 1650-dim raw feature spends more parameters on FiLM
        # projections than on temporal convolution. Projecting once first leaves
        # the temporal capacity untouched. FlowDiT1D already does this
        # internally, hence 'unet' is what this actually rescues.
        cond_bottleneck_dim = int(cond_bottleneck_dim)
        if cond_bottleneck_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(obs_feature_dim, cond_bottleneck_dim),
                nn.Mish(),
                nn.Linear(cond_bottleneck_dim, cond_bottleneck_dim),
            )
            global_cond_dim = cond_bottleneck_dim
        else:
            self.cond_proj = nn.Identity()
            global_cond_dim = obs_feature_dim

        # Observation TOKENS for cross-attention, if the encoder emits them. This
        # is the pathway that a pooled global_cond cannot substitute for: the
        # pooled vector averages 196 patch tokens per frame into one 768-vector,
        # so per-patch spatial detail and per-frame identity are destroyed before
        # the velocity model runs.
        context_shape = None
        if getattr(obs_encoder, "context_tokens", False):
            context_shape = obs_encoder.context_shape()
        context_dim = 0 if context_shape is None else int(context_shape[-1])
        self.context_shape = context_shape

        # velocity-field model
        assert obs_as_global_cond
        if backbone == "dit":
            model = FlowDiT1D(
                input_dim=action_dim,
                global_cond_dim=global_cond_dim,
                horizon=action_horizon,
                d_model=dit_d_model,
                depth=dit_depth,
                n_heads=dit_n_heads,
                mlp_ratio=dit_mlp_ratio,
                time_embed_dim=diffusion_step_embed_dim * 2,
                dropout=dit_dropout,
                context_dim=context_dim,
                time_log_scale=dit_time_log_scale,
            )
        elif backbone == "unet":
            # ConditionalUnet1D.forward swallows **kwargs, so a context tensor
            # would be silently discarded rather than rejected -- you would pay
            # for the token pathway and train without it.
            if context_dim > 0:
                raise rtc_flow.FlowPolicyConfigError(
                    "obs_encoder.context_tokens=True has no effect with "
                    "backbone='unet' (its conditioning is a single flat vector); "
                    "use backbone='dit' or turn context_tokens off"
                )
            levels = len(down_dims) - 1
            if action_horizon % (2**levels) != 0:
                raise rtc_flow.FlowPolicyConfigError(
                    f"unet backbone needs action horizon divisible by {2**levels} "
                    f"(skip-connection concat), got {action_horizon}; use "
                    f"backbone='dit' or change the horizon."
                )
            model = ConditionalUnet1D(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                time_log_scale=unet_time_log_scale,
            )
        else:
            raise rtc_flow.FlowPolicyConfigError(
                f"backbone must be 'dit' or 'unet', got {backbone!r}"
            )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = int(num_inference_steps)
        self.time_embed_scale = float(time_embed_scale)
        assert time_sample in ("uniform", "logit_normal")
        self.time_sample = time_sample
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_flow_n_samples = int(train_flow_n_samples)
        self.rtc_execution_horizon = int(rtc_execution_horizon)
        self.rtc_max_guidance_weight = float(rtc_max_guidance_weight)
        self.rtc_prefix_schedule = str(rtc_prefix_schedule)
        self.cond_bottleneck_dim = cond_bottleneck_dim
        self.global_cond_dim = global_cond_dim
        self.context_dim = context_dim
        self.cond_dropout_prob = float(cond_dropout_prob)
        self.cfg_scale = float(cfg_scale)
        self.null_cond = (
            nn.Parameter(torch.zeros(global_cond_dim))
            if self.cond_dropout_prob > 0.0
            else None
        )
        # The unconditional branch must be blind to the observation through BOTH
        # pathways. A null global_cond next to the real patch tokens would leave
        # the image fully visible, and CFG would then extrapolate along a
        # direction that has nothing to do with conditioning strength.
        self.null_context = (
            nn.Parameter(torch.zeros(1, 1, context_dim))
            if (self.cond_dropout_prob > 0.0 and context_dim > 0)
            else None
        )
        if self.cfg_scale != 1.0 and self.null_cond is None:
            raise rtc_flow.FlowPolicyConfigError(
                f"cfg_scale={self.cfg_scale} needs an unconditional branch; "
                "train with cond_dropout_prob > 0"
            )
        # persistent=False: this is derived from config, so a checkpoint trained
        # before the weighting still loads with strict=True.
        self.register_buffer(
            "horizon_loss_weight",
            self.build_horizon_loss_weight(
                action_horizon, loss_horizon_full_steps, loss_horizon_tail_weight
            ),
            persistent=False,
        )
        self.kwargs = kwargs

    @staticmethod
    def build_horizon_loss_weight(
        horizon: int, full_steps: int, tail_weight: float
    ) -> torch.Tensor:
        """Per-action-token loss weight, renormalized to mean 1.0.

        Weight 1.0 for the first ``full_steps`` tokens, then linear down to
        ``tail_weight`` at the last token. Deploy commits roughly a quarter of the
        chunk and RTC carries ``rtc_execution_horizon`` tokens, so with uniform
        weights most of the loss -- and most of ``val_loss`` -- is spent on a tail
        that is neither executed nor predictable from a single frame.

        Renormalizing to mean 1.0 keeps the loss MAGNITUDE roughly where it was,
        but the number is still a different metric: val_loss is not comparable
        across different (full_steps, tail_weight) settings.
        """
        horizon = int(horizon)
        weight = torch.ones(horizon, dtype=torch.float32)
        full = max(0, min(int(full_steps), horizon))
        n_decay = horizon - full
        if n_decay > 0 and float(tail_weight) != 1.0:
            weight[full:] = torch.linspace(
                1.0, float(tail_weight), n_decay + 1, dtype=torch.float32
            )[1:]
        return weight * (horizon / weight.sum())

    # ========= conditioning  ============
    def encode_obs(
        self, nobs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Normalized obs -> (global_cond after the bottleneck, context tokens).

        One backbone forward feeds both: the pooled vector drives adaLN/FiLM, the
        patch tokens are what the DiT cross-attends to. ``context`` is None unless
        the encoder was built with ``context_tokens=True``.
        """
        feature, context = self.obs_encoder.forward_features(nobs)
        return self.cond_proj(feature), context

    def null_conditioning(self, batch_size: int) -> Optional[torch.Tensor]:
        """The learnable "no observation" branch, expanded to the batch."""
        if self.null_cond is None:
            return None
        return self.null_cond.unsqueeze(0).expand(batch_size, -1)

    def null_context_tokens(
        self, context: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """``context``-shaped tokens that carry no observation.

        Broadcast to the real token count rather than a single token so that a
        dropped row and a kept row have the same shape and can coexist in one
        batch. Every token being identical makes it equivalent to attending to
        one learned token.
        """
        if context is None or self.null_context is None:
            return None
        return self.null_context.to(context.dtype).expand(context.shape)

    def drop_conditioning(
        self,
        global_cond: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Replace whole rows with the null embedding, Bernoulli(cond_dropout_prob).

        Applied AFTER the bottleneck, so dropped rows also stop back-propagating
        into the vision encoder -- the unconditional branch must not be able to
        peek at the observation it is supposed to be blind to. The SAME row mask
        drops the context tokens, for the same reason.
        """
        if self.cond_dropout_prob <= 0.0 or self.null_cond is None:
            return global_cond, context
        batch_size = global_cond.shape[0]
        drop = (
            torch.rand(batch_size, device=global_cond.device)
            < self.cond_dropout_prob
        )
        global_cond = torch.where(
            drop.unsqueeze(-1), self.null_conditioning(batch_size), global_cond
        )
        null_context = self.null_context_tokens(context)
        if null_context is not None:
            context = torch.where(drop.view(-1, 1, 1), null_context, context)
        return global_cond, context

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: Optional[int] = None,
        rtc_action_prefix: Optional[torch.Tensor] = None,
        rtc_inference_delay: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        uncond_global_cond = None
        uncond_context = None
        if self.cfg_scale != 1.0:
            uncond_global_cond = self.null_conditioning(condition_data.shape[0])
            uncond_context = self.null_context_tokens(context)
        return rtc_flow.flow_euler_sample(
            model=self.model,
            condition_data=condition_data,
            condition_mask=condition_mask,
            global_cond=global_cond,
            context=context,
            generator=generator,
            rtc_action_prefix=rtc_action_prefix,
            rtc_inference_delay=rtc_inference_delay,
            uncond_global_cond=uncond_global_cond,
            uncond_context=uncond_context,
            guidance_scale=self.cfg_scale,
            config=rtc_flow.FlowSamplingConfig(
                inference_steps=num_inference_steps or self.num_inference_steps,
                time_embed_scale=self.time_embed_scale,
                action_horizon=self.action_horizon,
                execution_horizon=self.rtc_execution_horizon,
                max_guidance_weight=self.rtc_max_guidance_weight,
                prefix_schedule=self.rtc_prefix_schedule,
            ),
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: Optional[torch.Tensor] = None,
        rtc_action_prefix: Optional[torch.Tensor] = None,
        rtc_inference_delay: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: normalized-shape obs (unnormalized values), batched.
        fixed_action_prefix: UNNORMALIZED action prefix (inpainted if enabled).
        rtc_action_prefix: UNNORMALIZED current-base-relative previous leftovers;
            rtc_inference_delay is predicted latency in policy action tokens.
        Returns {'action': (B, Ta, 20), 'action_pred': same} unnormalized.
        """
        assert "past_action" not in obs_dict  # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        global_cond, context = self.encode_obs(nobs)

        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and rtc_action_prefix is not None:
            raise rtc_flow.FlowPolicyConfigError(
                "fixed_action_prefix and rtc_action_prefix are mutually exclusive"
            )
        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer["action"].normalize(cond_data)

        normalized_rtc_prefix = None
        if rtc_action_prefix is not None and rtc_action_prefix.shape[1] > 0:
            prefix_steps = min(
                rtc_action_prefix.shape[1],
                self.action_horizon,
            )
            normalized_rtc_prefix = self.normalizer["action"].normalize(
                rtc_action_prefix[:, :prefix_steps]
            )
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            global_cond=global_cond,
            context=context,
            rtc_action_prefix=normalized_rtc_prefix,
            rtc_inference_delay=rtc_inference_delay,
            **self.kwargs,
        )

        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        return {"action": action_pred, "action_pred": action_pred}

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        assert self.obs_as_global_cond
        global_cond, context = self.encode_obs(nobs)

        # train on multiple flow samples per obs encoder forward pass
        if self.train_flow_n_samples != 1:
            global_cond = torch.repeat_interleave(
                global_cond, repeats=self.train_flow_n_samples, dim=0
            )
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_flow_n_samples, dim=0
            )
            if context is not None:
                # NOTE this is where train_flow_n_samples stops being nearly free.
                # With a pooled global_cond the repeat costs a (B*n, 256) tensor;
                # with context tokens it costs (B*n, N_tokens, C) plus n times the
                # cross-attention. Cut this first if the step OOMs.
                context = torch.repeat_interleave(
                    context, repeats=self.train_flow_n_samples, dim=0
                )

        # Drop AFTER the repeat so the copies of one observation get independent
        # masks -- the same obs then appears both conditioned and unconditioned
        # inside a batch, which is what the guidance direction is estimated from.
        global_cond, context = self.drop_conditioning(global_cond, context)

        x1 = nactions
        x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype)
        t = rtc_flow.sample_flow_time(
            x1.shape[0], x1.device, x1.dtype,
            self.time_sample,
            self.logit_normal_mean,
            self.logit_normal_std,
        )

        t_pad = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_pad) * x0 + t_pad * x1
        target = x1 - x0

        pred = self.model(
            xt,
            t * self.time_embed_scale,
            local_cond=None,
            global_cond=global_cond,
            context=context,
        )

        loss = F.mse_loss(pred, target, reduction="none")
        # (Ta,) -> (1, Ta, 1): weight per action token, shared across channels
        loss = loss * self.horizon_loss_weight.view(1, -1, 1).to(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()

    def forward(self, batch):
        return self.compute_loss(batch)
