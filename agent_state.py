"""
Shared modulator state construction for the RL-MDM policy.

Keeps legacy 128-d vector path and domain_v1 decomposed features (time emb, multiscale, freq).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F

from .nn_utils import timestep_embedding


def build_agent_state_legacy(z: torch.Tensor, t: torch.Tensor, state_dim: int = 128) -> torch.Tensor:
    """Same as former _state_from_zt: [pool(z) | timestep_embed(t)]."""
    pooled = z.mean(dim=(2, 3))
    t_emb = timestep_embedding(t.view(-1), state_dim - pooled.shape[1]).to(z.device)
    return torch.cat([pooled, t_emb], dim=1)


def build_agent_state(
    z: torch.Tensor,
    t: torch.Tensor,
    *,
    mode: str = "legacy",
    state_dim: int = 128,
    t_dim: int = 128,
    n_freq_bands: int = 8,
) -> Union[torch.Tensor, DomainAgentState]:
    """
    Single entry for the modulator state. ``legacy`` -> flat [B,state_dim];
    ``domain_v1`` -> :class:`DomainAgentState` (t_emb, multiscale, freq_energy).
    """
    if mode == "domain_v1":
        return build_domain_agent_state(z, t, t_dim=t_dim, n_freq_bands=n_freq_bands)
    if mode == "legacy":
        return build_agent_state_legacy(z, t, state_dim=state_dim)
    raise ValueError(f"Unknown scheduling mode: {mode!r} (expected 'legacy' or 'domain_v1')")


def multiscale_summary(z: torch.Tensor) -> torch.Tensor:
    """
    Light multi-scale global statistics [B,3]: full mean, 8x8 pool mean, 4x4 pool mean.
    """
    g0 = z.mean(dim=(2, 3))
    g1 = F.adaptive_avg_pool2d(z, 8).mean(dim=(2, 3))
    g2 = F.adaptive_avg_pool2d(z, 4).mean(dim=(2, 3))
    return torch.cat([g0, g1, g2], dim=1)


def radial_freq_band_energy(z: torch.Tensor, n_bands: int) -> torch.Tensor:
    """
    Real FFT magnitude energy per radial bin on mean channel, shape [B, n_bands].
    z: [B,1,H,W]
    """
    b, _, h, w = z.shape
    x = z.mean(dim=1)  # [B,H,W]
    spec = torch.fft.rfft2(x, norm="ortho")
    mag = (spec.real.pow(2) + spec.imag.pow(2)).clamp_min(1e-12).sqrt()
    hp, wp = mag.shape[1], mag.shape[2]
    fy = torch.arange(hp, device=z.device, dtype=torch.float32)[:, None]
    fx = torch.arange(wp, device=z.device, dtype=torch.float32)[None, :]
    # normalized radius in [0, ~1]
    r = (fy.pow(2) + fx.pow(2)).sqrt()
    r_max = r.max().clamp_min(1e-6)
    r = r / r_max
    edges = torch.linspace(0.0, 1.0, n_bands + 1, device=z.device)
    out = []
    for k in range(n_bands):
        mask = (r >= edges[k]) & (r < edges[k + 1])
        if k == n_bands - 1:
            mask = r >= edges[k]
        e = (mag * mask.float()).sum(dim=(1, 2))
        out.append(torch.log1p(e))
    return torch.stack(out, dim=1)


@dataclass
class DomainAgentState:
    """Decomposed state for domain_v1 schedulers."""

    t_emb: torch.Tensor  # [B, t_dim]
    multiscale: torch.Tensor  # [B, 3]
    freq_energy: torch.Tensor  # [B, n_bands]


def build_domain_agent_state(z: torch.Tensor, t: torch.Tensor, *, t_dim: int = 128, n_freq_bands: int) -> DomainAgentState:
    t_emb = timestep_embedding(t.view(-1), t_dim).to(z.device)
    ms = multiscale_summary(z)
    fe = radial_freq_band_energy(z, n_freq_bands)
    return DomainAgentState(t_emb=t_emb, multiscale=ms, freq_energy=fe)


def scheduling_from_yaml(cfg: Dict[str, Any]) -> Tuple[str, bool, bool, int]:
    s = cfg.get("scheduling") or {}
    mode = str(s.get("mode", "legacy"))
    enable_freq = bool(s.get("enable_freq", False))
    enable_spatial = bool(s.get("enable_spatial", False))
    freq_bands = int(s.get("freq_bands", 8))
    return mode, enable_freq, enable_spatial, freq_bands


# ---------------------------------------------------------------------------
# Differentiated states for the RL-MDM modulators (one per modulator role)
# ---------------------------------------------------------------------------


def build_spatial_state(z: torch.Tensor, spatial_ch: int = 5, downsample: int = 4) -> torch.Tensor:
    """
    Local spatial texture features for Modulator-S (SpatialAgent).

    Returns a feature map [B, spatial_ch, H//downsample, W//downsample] containing:
      ch0: local variance (3x3)
      ch1: horizontal gradient (Sobel-x)
      ch2: vertical gradient (Sobel-y)
      ch3: gradient magnitude
      ch4: Laplacian response
      (if spatial_ch >= 5)
      ch5+: padded zeros for future expansion

    The modulator uses a small CNN on this feature map, giving it a high-resolution
    local view of the noisy image — ideal for learning edge/detail preservation.
    """
    B, C, H, W = z.shape
    if C > 1:
        z = z.mean(dim=1, keepdim=True)

    # Downsample first to reduce computation while preserving local patterns
    if downsample > 1:
        z_ds = F.interpolate(z, size=(H // downsample, W // downsample), mode="bilinear", align_corners=False)
    else:
        z_ds = z

    z_pad = F.pad(z_ds, (1, 1, 1, 1), mode="reflect")

    # Local variance (3x3)
    mu = F.avg_pool2d(z_pad, 3, stride=1)
    var = F.avg_pool2d(z_pad * z_pad, 3, stride=1) - mu * mu

    # Sobel gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=z.device, dtype=z.dtype)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=z.device, dtype=z.dtype)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    gx = F.conv2d(z_pad, sobel_x, padding=0)
    gy = F.conv2d(z_pad, sobel_y, padding=0)
    gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)

    # Laplacian
    laplacian_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=z.device, dtype=z.dtype)
    laplacian_k = laplacian_k.view(1, 1, 3, 3)
    lap = F.conv2d(z_pad, laplacian_k, padding=0)

    feats = [var, gx, gy, gmag, lap]
    while len(feats) < spatial_ch:
        feats.append(torch.zeros_like(var))
    return torch.cat(feats[:spatial_ch], dim=1)


def build_frequency_state(z: torch.Tensor, t: torch.Tensor, *, n_bands: int = 8, t_dim: int = 4) -> torch.Tensor:
    """
    Frequency-domain features for Modulator-F (FrequencyAgent).

    Returns [B, n_bands + t_dim]:
      first n_bands: radial frequency band energy (reuses radial_freq_band_energy)
      last t_dim:    compact timestep encoding (first few dims of sinusoidal embedding)

    Modulator-F sees ONLY frequency-domain statistics — it decides frequency-band gains
    without access to spatial layout.
    """
    fe = radial_freq_band_energy(z, n_bands)
    t_short = timestep_embedding(t.view(-1), t_dim).to(z.device)
    return torch.cat([fe, t_short], dim=1)


def build_temporal_state(z: torch.Tensor, t: torch.Tensor, state_dim: int = 128) -> torch.Tensor:
    """
    Global / temporal features for Modulator-T (TemporalAgent).

    Returns [B, state_dim]: concatenation of multi-scale spatial pool and timestep embedding.

    Modulator-T sees coarse global structure and diffusion phase — it learns
    when in the diffusion process to strengthen or ease denoising.
    """
    pooled = z.mean(dim=(2, 3))
    remaining = state_dim - pooled.shape[1]
    t_emb = timestep_embedding(t.view(-1), remaining).to(z.device)
    return torch.cat([pooled, t_emb], dim=1)
