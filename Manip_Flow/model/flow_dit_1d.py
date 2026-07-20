"""DiT-1D velocity-field backbone for the flow-matching manipulation policy.

Call-convention drop-in for ``ConditionalUnet1D`` (``forward(sample, timestep,
local_cond=None, global_cond=None)``), so the policy can switch backbones from
config. Standard DiT with adaLN-zero conditioning (Peebles & Xie), tokens =
action frames:

    x: (B, T, action_dim) -> linear embed + learned pos emb
    c = MLP(sinusoidal(t)) + MLP(global_cond)      # one vector per sample
    N blocks: adaLN-zero(SelfAttn) + adaLN-zero(MLP)
    final: adaLN + zero-init linear -> (B, T, action_dim)

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
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
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
        # adaLN-zero: 6 chunks = (shift, scale, gate) x (attn, mlp); zero-init
        # so every block starts as identity.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa_shift, sa_scale, sa_gate, mlp_shift, mlp_scale, mlp_gate = self.adaLN(
            c
        ).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), sa_shift, sa_scale)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + sa_gate.unsqueeze(1) * h
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
    ):
        super().__init__()
        self.horizon = int(horizon)

        self.x_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(global_cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)]
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
        **kwargs,
    ) -> torch.Tensor:
        """Same signature as ConditionalUnet1D.forward.

        sample: (B, T, input_dim), T <= horizon
        timestep: (B,) float tensor (already scaled by the policy) or scalar
        global_cond: (B, global_cond_dim)
        """
        assert local_cond is None, "FlowDiT1D does not support local_cond"
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
            x = block(x, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.final_proj(x)
