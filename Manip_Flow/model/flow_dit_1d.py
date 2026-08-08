"""DiT-1D velocity-field backbone for the flow-matching manipulation policy.

Call-convention drop-in for ``ConditionalUnet1D`` (``forward(sample, timestep,
local_cond=None, global_cond=None)``), so the policy can switch backbones from
config. Standard DiT with adaLN-zero conditioning (Peebles & Xie), tokens =
action frames:

    x: (B, T, action_dim) -> linear embed + learned pos emb
    c = MLP(sinusoidal(t)) + MLP(global_cond)      # one vector per sample
    N blocks: adaLN-zero(SelfAttn) [+ adaLN-zero(CrossAttn)] + adaLN-zero(MLP)
    final: adaLN + zero-init linear -> (B, T, action_dim)

With ``context_dim > 0`` each block also cross-attends to an observation token
sequence (``context``), which is what a flat ``global_cond`` structurally cannot
carry: the vision encoder's pooled vector averages all patch tokens into one
768-vector, so fine spatial detail and per-frame identity are gone before the
velocity model ever sees them. The cross-attention gate is zero-initialized, so
the context pathway starts as a no-op and is learned in.

Why offer DiT next to the UNet:
  * no horizon divisibility constraint — ConditionalUnet1D needs T divisible
    by 2^(levels-1) (=4 for the default down_dims), while a 17-token action
    horizon is the minimum 10Hz setting that yields a 50-frame provider window.
  * matches the lower level: the Prior_Recon motion prior is already a
    transformer trained with flow matching.
  * zero-init gates make the network an identity map at init — a well-behaved
    starting point for velocity regression.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from Manip_Flow.model.diffusion.positional_embedding import SinusoidalPosEmb


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.use_cross_attn = bool(use_cross_attn)
        if self.use_cross_attn:
            self.norm_ca = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        # adaLN-zero: (shift, scale, gate) per sublayer -- 6 chunks, or 9 with
        # cross-attention; zero-init so every block starts as identity. Only
        # allocating 9 when cross-attention is on keeps context-free DiT
        # checkpoints loadable.
        self.n_chunks = 9 if self.use_cross_attn else 6
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, self.n_chunks * d_model)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        chunks = self.adaLN(c).chunk(self.n_chunks, dim=-1)
        sa_shift, sa_scale, sa_gate = chunks[0:3]
        mlp_shift, mlp_scale, mlp_gate = chunks[3:6]

        h = _modulate(self.norm1(x), sa_shift, sa_scale)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + sa_gate.unsqueeze(1) * h

        if self.use_cross_attn:
            if context is None:
                raise ValueError("cross-attention block called without context")
            ca_shift, ca_scale, ca_gate = chunks[6:9]
            h = _modulate(self.norm_ca(x), ca_shift, ca_scale)
            h, _ = self.cross_attn(h, context, context, need_weights=False)
            x = x + ca_gate.unsqueeze(1) * h

        h = _modulate(self.norm2(x), mlp_shift, mlp_scale)
        x = x + mlp_gate.unsqueeze(1) * self.mlp(h)
        return x


class FlowDiT1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        horizon: int,
        d_model: int = 512,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        dropout: float = 0.0,
        context_dim: int = 0,
        time_log_scale: float = 10000.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.context_dim = int(context_dim)

        self.x_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # time_log_scale must be chosen together with the policy's
        # time_embed_scale, which multiplies t in [0, 1] before it gets here. The
        # 10000 default assumes the Flux/SD3 convention (time_embed_scale=1000);
        # with the HuMI convention (time_embed_scale=1.0) it leaves all but the
        # first few frequency channels at ~0, i.e. a near-dead time embedding.
        # ConditionalUnet1D takes the matching knob as unet_time_log_scale.
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, log_scale=time_log_scale),
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(global_cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-attention context. LayerNorm first because the incoming ViT patch
        # features were never scaled for this network.
        if self.context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, d_model),
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model,
                    n_heads,
                    mlp_ratio,
                    dropout,
                    use_cross_attn=self.context_dim > 0,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final_proj = nn.Linear(d_model, input_dim)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Same signature as ConditionalUnet1D.forward, plus ``context``.

        sample: (B, T, input_dim), T <= horizon
        timestep: (B,) float tensor (already scaled by the policy) or scalar
        global_cond: (B, global_cond_dim)
        context: (B, N, context_dim) tokens to cross-attend to; required when
            the model was built with context_dim > 0, rejected otherwise. This is
            the pathway a pooled global_cond cannot provide: the action tokens
            read individual patches, so "hand 5 cm from the handle" survives
            instead of being averaged into one vector.
        """
        assert local_cond is None, "FlowDiT1D does not support local_cond"
        if self.context_dim > 0:
            if context is None:
                raise ValueError(
                    f"FlowDiT1D built with context_dim={self.context_dim} "
                    "requires context"
                )
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"context last dim {context.shape[-1]} != context_dim "
                    f"{self.context_dim}"
                )
            context = self.context_proj(context)
        elif context is not None:
            raise ValueError("context given but the model has context_dim=0")
        B, T, _ = sample.shape
        assert T <= self.horizon, f"T={T} exceeds trained horizon {self.horizon}"

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=sample.dtype, device=sample.device
            )
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B).to(sample.dtype)

        c = self.t_embed(timesteps)
        if global_cond is not None:
            c = c + self.cond_embed(global_cond)

        x = self.x_embed(sample) + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, c, context)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.final_proj(x)
