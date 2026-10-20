"""
FoundDiff-inspired deeper U-Net with improved architecture.

Key improvements over SmallUNet:
- Pre-activation ResBlocks with skip scaling
- Optional self-attention at deeper levels
- Better time embedding injection
- Deeper default config (can go 4-5 levels)
- Weight initialization for stable training
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import Psi


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
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


class PreNormResBlock(nn.Module):
    """Pre-activation ResBlock with time embedding modulation (FoundDiff style)."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        gn_groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(gn_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        gn_groups2 = min(8, out_ch)
        self.norm2 = nn.GroupNorm(gn_groups2, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Zero-init the last projection for stable training
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_out = self.t_proj(t_emb)[:, :, None, None]
        t_scale, t_shift = t_out.chunk(2, dim=1)
        h = h * (1 + t_scale) + t_shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        out = h + self.skip(x)
        if scale is not None:
            out = out * scale
        return out


class SelfAttention(nn.Module):
    """Lightweight self-attention for deep feature levels."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeepUNet(nn.Module):
    """
    FoundDiff-inspired deeper U-Net with pre-activation ResBlocks and attention.

    Args:
        base_ch: channels at first level (default 64)
        ch_mult: channel multipliers, e.g. [1,2,4,8] for 4 levels
        t_dim: time embedding dimension
        in_ch: input channels (z_t + x_ld concatenated)
        out_ch: output channels
        attn_resolutions: spatial sizes at which to use attention (e.g. [16,8])
        dropout: dropout rate
    """

    def __init__(
        self,
        base_ch: int = 64,
        ch_mult: tuple = (1, 2, 4, 8),
        t_dim: int = 256,
        in_ch: int = 2,
        out_ch: int = 1,
        attn_resolutions: tuple = (16, 8, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.base_ch = base_ch
        self.num_levels = len(ch_mult)
        self.attn_resolutions = set(attn_resolutions)

        chs = [base_ch * m for m in ch_mult]

        # Time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.num_levels):
            in_c = chs[i - 1] if i > 0 else chs[0]
            out_c = chs[i]
            self.enc_blocks.append(PreNormResBlock(in_c, out_c, t_dim, dropout))
            self.enc_attn.append(SelfAttention(out_c) if out_c in self.attn_resolutions else nn.Identity())
            if i < self.num_levels - 1:
                self.downs.append(DownBlock(out_c))

        # Bottleneck
        self.mid_block1 = PreNormResBlock(chs[-1], chs[-1], t_dim, dropout)
        self.mid_attn = SelfAttention(chs[-1]) if chs[-1] in self.attn_resolutions else nn.Identity()
        self.mid_block2 = PreNormResBlock(chs[-1], chs[-1], t_dim, dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(self.num_levels - 1, 0, -1):
            up_in = chs[i] + chs[i - 1]  # concat
            up_out = chs[i - 1]
            self.ups.append(UpBlock(chs[i]))
            self.dec_blocks.append(PreNormResBlock(up_in, up_out, t_dim, dropout))
            self.dec_attn.append(SelfAttention(up_out) if up_out in self.attn_resolutions else nn.Identity())

        # Final
        self.final_block = PreNormResBlock(chs[0], chs[0], t_dim, dropout)
        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, x_ld: torch.Tensor, psi: Optional[Psi]
    ) -> torch.Tensor:
        if t.ndim == 1:
            t_emb = timestep_embedding(t, self.t_dim)
        else:
            t_emb = t
        t_emb = self.t_mlp(t_emb)

        x_in = torch.cat([z_t, x_ld], dim=1)
        h = self.in_conv(x_in)

        w_freq = _psi_get(psi, "freq_radial_weights")
        if w_freq is not None:
            h = _apply_radial_rfft_gain(h, w_freq)

        gate = _psi_get(psi, "spatial_gate_lowres")
        scales = _get_scales(psi, self.num_levels)

        # Encoder
        skips = []
        for i in range(self.num_levels):
            h = self.enc_blocks[i](h, t_emb, scale=scales[i] if i < len(scales) else None)
            if gate is not None:
                h = _multiply_spatial_gate(h, gate)
            h = self.enc_attn[i](h)
            skips.append(h)
            if i < self.num_levels - 1:
                h = self.downs[i](h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for i in range(self.num_levels - 1):
            h = self.ups[i](h)
            skip = skips[self.num_levels - 2 - i]
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[i](h, t_emb, scale=scales[self.num_levels - 1 - i] if self.num_levels - 1 - i < len(scales) else None)
            if gate is not None:
                h = _multiply_spatial_gate(h, gate)
            h = self.dec_attn[i](h)

        h = self.final_block(h, t_emb, scale=scales[0] if scales else None)
        if gate is not None:
            h = _multiply_spatial_gate(h, gate)

        return self.out_conv(F.silu(self.out_norm(h)))


def _get_scales(psi: Optional[Psi], num_levels: int) -> List[Optional[torch.Tensor]]:
    if psi is None:
        return [None] * num_levels
    if isinstance(psi, torch.Tensor):
        return [psi] * num_levels
    return [psi.get(f"scale{i + 1}") for i in range(num_levels)]


def _psi_get(psi: Optional[Psi], key: str) -> Optional[torch.Tensor]:
    if psi is None:
        return None
    if isinstance(psi, torch.Tensor):
        return psi
    return psi.get(key)


def _apply_radial_rfft_gain(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    B, C, H, W = h.shape
    K = weights.shape[1]
    Wf = W // 2 + 1
    fy = torch.arange(H, device=h.device, dtype=torch.float32)[:, None]
    fx = torch.arange(Wf, device=h.device, dtype=torch.float32)[None, :]
    r = ((fy / max(H - 1, 1)) ** 2 + (fx / max(Wf - 1, 1)) ** 2).sqrt()
    bin_idx = (r * K).long().clamp(0, K - 1)
    flat = bin_idx.reshape(-1)
    gain = weights[:, flat].view(B, 1, H, Wf)
    Hf = torch.fft.rfft2(h, norm="ortho")
    gain_c = gain.expand(-1, C, -1, -1).to(dtype=Hf.dtype)
    return torch.fft.irfft2(Hf * gain_c, s=(H, W), norm="ortho")


def _multiply_spatial_gate(h: torch.Tensor, gate_lowres: torch.Tensor) -> torch.Tensor:
    g = F.interpolate(gate_lowres, size=h.shape[2:], mode="bilinear", align_corners=False)
    return h * g
