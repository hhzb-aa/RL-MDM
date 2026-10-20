from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import torch


# Psi: modulation dict passed to DenoiserBackbone (e.g. SmallUNet).
# Contract (keys are optional unless noted):
#   scale1, scale2, scale3 — [B,1,1,1] multiplicative gates on shallow / mid / deep ResBlocks
#     (aliases in prose: scale_fine ~ rb1/rb5, scale_mid ~ rb2/rb4, scale_coarse ~ rb3).
#   freq_radial_weights — [B, K] positive gains applied per radial bin in rFFT on h0 (after in_conv).
#   spatial_gate_lowres — [B,1,H',W'] bilinear-upsampled and multiplied on ResBlock outputs
#     (after scale); order: h * scale * spatial_gate.
Psi = Union[torch.Tensor, Dict[str, torch.Tensor]]


class DenoiserBackbone(Protocol):
    """
    Backbone interface used by the denoising model.
    Implementations may be a small U-Net (POC) or an adapter wrapping a real model.
    """

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x_ld: torch.Tensor, psi: Optional[Psi]) -> torch.Tensor:
        """
        Args:
          z_t: current noisy estimate, shape [B,1,H,W]
          t: timestep indices or embeddings, shape [B] (int64) or [B,dim]
          x_ld: conditioning image (LD), shape [B,1,H,W]
          psi: fused actions from modulators; can be None for baseline.

        Returns:
          prediction tensor (epsilon/v/x0 depending on config), shape [B,1,H,W]
        """


class MultiAgentTeam(Protocol):
    """
    Three-modulator team interface. For POC, it can output continuous actions.
    """

    def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          a_S, a_F, a_T: actions (continuous tensors), shape [B, A]
        """


class Fusion(Protocol):
    """
    Fuse three actions into psi used to modulate the backbone.
    """

    def forward(self, a_s: torch.Tensor, a_f: torch.Tensor, a_t: torch.Tensor) -> Psi:
        ...


@dataclass(frozen=True)
class BuildContext:
    device: torch.device

