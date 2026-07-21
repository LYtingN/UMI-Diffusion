"""
DiT (Diffusion Transformer) for action diffusion with RMSNorm.

Architecture:
  - Decoder-only transformer with cross-attention to observation conditioning
  - RMSNorm instead of LayerNorm (as in the paper)
  - pre-norm design (norm before attention/FFN)
  - MLP ratio 4, GELU activation
"""

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class DiTBlock(nn.Module):
    """
    Single DiT block with:
      pre-norm RMSNorm → self-attention
      pre-norm RMSNorm → cross-attention to conditioning
      pre-norm RMSNorm → FFN (MLP ratio)
    """
    def __init__(self, n_emb: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_emb)
        self.self_attn = nn.MultiheadAttention(
            n_emb, n_head, dropout=dropout, batch_first=True)

        self.norm2 = RMSNorm(n_emb)
        self.cross_attn = nn.MultiheadAttention(
            n_emb, n_head, dropout=dropout, batch_first=True)

        self.norm3 = RMSNorm(n_emb)
        ffn_dim = int(n_emb * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-norm)
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed)[0]
        # Cross-attention to conditioning (pre-norm)
        x = x + self.cross_attn(self.norm2(x), memory, memory)[0]
        # FFN (pre-norm)
        x = x + self.ffn(self.norm3(x))
        return x


class DiTForDiffusion(ModuleAttrMixin):
    """
    Diffusion Transformer for action prediction.

    Takes noisy action sequence + timestep + conditioning embeddings,
    returns denoised action sequence (noise prediction).

    Args:
        input_dim:      dimensionality of the action space
        output_dim:     dimensionality of output (usually == input_dim)
        action_horizon: number of action steps to predict (Tp)
        n_layer:        number of DiT blocks (depth)
        n_head:         number of attention heads
        n_emb:          embedding dimension
        mlp_ratio:      FFN hidden dim = n_emb * mlp_ratio
        max_cond_tokens: max number of conditioning tokens (obs + time)
        p_drop_attn:    dropout probability in attention / FFN
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 10,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: int = 4,
        max_cond_tokens: int = 800,
        p_drop_attn: float = 0.1,
    ) -> None:
        super().__init__()

        # --- action input stem ---
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn(1, action_horizon, n_emb))

        # --- timestep embedding ---
        self.time_emb = SinusoidalPosEmb(n_emb)

        # --- conditioning positional embedding ---
        self.cond_pos_emb = nn.Parameter(torch.randn(1, max_cond_tokens, n_emb))

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(n_emb, n_head, mlp_ratio=mlp_ratio, dropout=p_drop_attn)
            for _ in range(n_layer)
        ])

        # --- output head ---
        self.ln_f = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.action_horizon = action_horizon
        self.n_emb = n_emb

        self.apply(self._init_weights)
        logger.info(
            "DiTForDiffusion: %e parameters",
            sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.ModuleList,
            nn.Sequential,
            nn.GELU,
            DiTBlock,
            DiTForDiffusion,
            RMSNorm,
        )
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                w = getattr(module, name)
                if w is not None:
                    nn.init.normal_(w, mean=0.0, std=0.02)
            for name in ['in_proj_bias', 'bias_k', 'bias_v']:
                b = getattr(module, name)
                if b is not None:
                    nn.init.zeros_(b)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Separate parameters into decay / no-decay groups."""
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, RMSNorm):
                    no_decay.add(fpn)

        no_decay.update({"pos_emb", "cond_pos_emb", "_dummy_variable"})

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        assert len(inter) == 0, f"Params in both sets: {inter}"
        assert len(param_dict.keys() - union) == 0, \
            f"Params not separated: {param_dict.keys() - union}"

        return [
            {"params": [param_dict[pn] for pn in sorted(decay)],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)],
             "weight_decay": 0.0},
        ]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sample:   (B, T, input_dim)  noisy action sequence
            timestep: (B,) or scalar     diffusion timestep
            cond:     (B, N, n_emb)      observation conditioning tokens

        Returns:
            (B, T, output_dim)  predicted noise
        """
        # 1. timestep embedding → (B, 1, n_emb)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_emb)

        # 2. build memory = [cond_tokens, time_token] + positional embeddings
        memory = torch.cat([cond, time_emb], dim=1)  # (B, N+1, n_emb)
        tc = memory.shape[1]
        memory = memory + self.cond_pos_emb[:, :tc, :]

        # 3. embed action input
        x = self.input_emb(sample)  # (B, T, n_emb)
        t = x.shape[1]
        x = x + self.pos_emb[:, :t, :]

        # 4. DiT blocks (self-attn over actions, cross-attn to memory)
        for block in self.blocks:
            x = block(x, memory)

        # 5. output head
        x = self.ln_f(x)
        x = self.head(x)  # (B, T, output_dim)
        return x
