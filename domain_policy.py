"""
Domain-aware scheduling policy: time (t_emb only path), multi-scale summary, optional freq + spatial.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from .agent_state import DomainAgentState
from .fusion import DomainFusion
from .interfaces import Psi


class DomainPolicy(nn.Module):
    """
    Outputs UNet psi dict. Stochastic part for RL-MDM: Normal on time [B,3] and scale [B,3];
    optional Normal on freq [B,K]. Spatial branch is deterministic from (z, x_ld).
    """

    def __init__(
        self,
        *,
        t_dim: int = 128,
        num_levels: int = 3,
        freq_bands: int = 8,
        enable_freq: bool = False,
        enable_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.freq_bands = freq_bands
        self.enable_freq = enable_freq
        self.enable_spatial = enable_spatial

        self.time_net = nn.Sequential(nn.Linear(t_dim, 128), nn.SiLU(), nn.Linear(128, num_levels))
        self.scale_net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, num_levels))

        if enable_freq:
            self.freq_net = nn.Sequential(nn.Linear(freq_bands, 64), nn.SiLU(), nn.Linear(64, freq_bands))
            nn.init.zeros_(self.freq_net[-1].bias)
            nn.init.normal_(self.freq_net[-1].weight, std=1e-3)

        if enable_spatial:
            self.spatial_net = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 1, 3, stride=2, padding=1),
                nn.Sigmoid(),
            )

        self.domain_fusion = DomainFusion()

    def forward_psi(
        self,
        state: DomainAgentState,
        z: torch.Tensor,
        x_ld: torch.Tensor,
        *,
        time_vec: torch.Tensor,
        scale_vec: torch.Tensor,
        freq_vec: Optional[torch.Tensor] = None,
    ) -> Psi:
        psi: Dict[str, Any] = dict(self.domain_fusion(time_vec, scale_vec))
        if self.enable_freq:
            if freq_vec is None:
                raw = self.freq_net(state.freq_energy)
            else:
                raw = freq_vec
            psi["freq_radial_weights"] = 0.85 + 0.3 * torch.sigmoid(raw)
        if self.enable_spatial:
            xin = torch.cat([z, x_ld], dim=1)
            psi["spatial_gate_lowres"] = self.spatial_net(xin)
        return psi

    def forward_means(self, state: DomainAgentState) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        t_mu = self.time_net(state.t_emb)
        s_mu = self.scale_net(state.multiscale)
        f_mu = self.freq_net(state.freq_energy) if self.enable_freq else None
        return t_mu, s_mu, f_mu

    def sample_actions_and_logp(
        self,
        state: DomainAgentState,
        *,
        std_time: Optional[float] = None,
        std_scale: Optional[float] = None,
        std_freq: float = 0.1,
        global_step: int = 0,
        total_steps: int = 1,
        anneal_action_std: bool = False,
        std_time_start: float = 0.2,
        std_time_end: float = 0.05,
        std_scale_start: float = 0.2,
        std_scale_end: float = 0.05,
        per_sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        t_mu, s_mu, f_mu = self.forward_means(state)
        if anneal_action_std:
            p = min(1.0, float(global_step) / max(1, int(total_steps) - 1))
            std_time = std_time_start + p * (std_time_end - std_time_start)
            std_scale = std_scale_start + p * (std_scale_end - std_scale_start)
        if std_time is None:
            std_time = std_time_start
        if std_scale is None:
            std_scale = std_scale_start
        dist_t = torch.distributions.Normal(t_mu, std_time)
        dist_s = torch.distributions.Normal(s_mu, std_scale)
        time_act = dist_t.sample()
        scale_act = dist_s.sample()
        if per_sample:
            # 逐样本 log-prob [B]：只对动作维求和，保留 batch 维
            logp = dist_t.log_prob(time_act).sum(-1) + dist_s.log_prob(scale_act).sum(-1)
        else:
            logp = dist_t.log_prob(time_act).mean() + dist_s.log_prob(scale_act).mean()
        freq_act: Optional[torch.Tensor] = None
        if self.enable_freq and f_mu is not None:
            dist_f = torch.distributions.Normal(f_mu, std_freq)
            freq_act = dist_f.sample()
            logp = logp + (dist_f.log_prob(freq_act).sum(-1) if per_sample
                           else dist_f.log_prob(freq_act).mean())
        return time_act, scale_act, freq_act, logp

    def logp_for_actions(
        self,
        state: DomainAgentState,
        *,
        time_act: torch.Tensor,
        scale_act: torch.Tensor,
        freq_act: Optional[torch.Tensor] = None,
        std_time: float = 0.2,
        std_scale: float = 0.2,
        std_freq: float = 0.1,
    ) -> torch.Tensor:
        """Recompute log probability for stored actions, used by PPO-clip updates."""
        t_mu, s_mu, f_mu = self.forward_means(state)
        dist_t = torch.distributions.Normal(t_mu, std_time)
        dist_s = torch.distributions.Normal(s_mu, std_scale)
        logp = dist_t.log_prob(time_act).mean() + dist_s.log_prob(scale_act).mean()
        if self.enable_freq and f_mu is not None and freq_act is not None:
            dist_f = torch.distributions.Normal(f_mu, std_freq)
            logp = logp + dist_f.log_prob(freq_act).mean()
        return logp
