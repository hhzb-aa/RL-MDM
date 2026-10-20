"""Small neural-network helpers shared by the backbone and the policy."""
from __future__ import annotations

import torch


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding. ``t``: [B] (int64 or float) -> [B, dim]."""
    if t.ndim != 1:
        t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device))
        * torch.arange(0, half, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
