import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, log_scale=10000):
        super().__init__()
        self.dim = dim
        self.log_scale = log_scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.log_scale) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
