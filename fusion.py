from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .interfaces import Psi


class SimpleFusion(nn.Module):
    """
    Minimal fusion: map three modulators' actions to scalar gates per UNet level.

    Modulator-S controls shallow (scale1, scaleN), Modulator-F mid, Modulator-T deep.
    """

    def __init__(self, action_dim: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, a_s: torch.Tensor, a_f: torch.Tensor, a_t: torch.Tensor) -> Psi:
        s = torch.tanh(a_s).mean(dim=1, keepdim=True)
        f = torch.tanh(a_f).mean(dim=1, keepdim=True)
        t = torch.tanh(a_t).mean(dim=1, keepdim=True)

        def ex(x: torch.Tensor) -> torch.Tensor:
            return (1.0 + 0.5 * x)[:, :, None, None]

        # Map to 3-level gates regardless of action_dim (legacy behavior)
        return {"scale1": ex(s), "scale2": ex(f), "scale3": ex(t)}


class DomainFusion(nn.Module):
    """
    Time-domain x multi-scale-domain gates per UNet stage.
    scale_i = (1 + 0.5*tanh(scale_i)) * (1 + 0.3*tanh(time_i)).
    Supports variable num_levels via last dim of inputs.
    """

    def forward(self, time_act: torch.Tensor, scale_act: torch.Tensor) -> Psi:
        if time_act.shape != scale_act.shape:
            raise ValueError(f"time_act {time_act.shape} and scale_act {scale_act.shape} must match")
        num_levels = time_act.shape[-1]
        ts = torch.tanh(scale_act)
        tt = torch.tanh(time_act)
        psi = {}
        for i in range(num_levels):
            s = (1.0 + 0.5 * ts[:, i : i + 1]) * (1.0 + 0.3 * tt[:, i : i + 1])
            psi[f"scale{i + 1}"] = s[:, :, None, None]
        return psi

