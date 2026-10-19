# 🧠 RL-MDM

**Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising**

Core code release. This repository contains the model definition and the
deterministic inference procedure for reference. The training pipeline, dataset
loader, evaluation metrics, and pretrained weights are not distributed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](requirements.txt)
[![Paper](https://img.shields.io/badge/Paper-under%20review-lightgrey.svg)](#paper)

---

<a id="paper"></a>
## 📄 Paper

- **Title**: Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising
- **Status**: Under review
- **Link**: *Coming soon*

---

## 💡 Overview

**The problem.** Low-dose CT noise is spatially non-stationary and varies with
dose, anatomy, and diffusion timestep, yet existing denoisers rely on fixed
network parameters and a uniform sampling schedule — the same correction is
applied at every step, scale, and position.

**The idea.** RL-MDM keeps the DeepUNet noise-prediction backbone fixed in
architecture and sampling procedure, and learns *what to modulate* instead.
Three lightweight modulators — temporal, scale, and spatial — observe
complementary projections of the diffusion state and emit a packet
`psi = ({scale_i}, spatial_g)` that rescales every residual block and gates the
intermediate feature maps at each step. The policy is optimised with REINFORCE
on a chain-level image-quality reward (`L1 + SSIM`) and updated jointly with the
backbone under an aligned objective, so both pursue the same final-image goal.

**The result.** Under an identical backbone and 16-step DDIM sampler, the learned
modulation changes *what* the denoiser preserves rather than merely how strongly
it denoises: it attains the best EPI and LPIPS on both 10 % chest subsets and the
best PSNR among diffusion-based methods, improves PSNR/SSIM over the
fixed-schedule baseline on every internal and external validation set, and
transfers to unseen patients without fine-tuning.

---

## 📦 Code Scope

| Component | Status |
|---|---|
| DeepUNet backbone, three modulators, `DomainFusion` | ✅ Released |
| 16-step DDIM sampling (modulated / unmodulated) | ✅ Released |
<!-- | Two-stage training pipeline | ❌ Not released |
| Dataset loader and evaluation metrics | ❌ Not released |
| Pretrained weights | ❌ Not distributed | -->

```text
mdmrl/
├── __init__.py            # package exports
├── interfaces.py          # protocol / dataclass contracts
├── nn_utils.py            # sinusoidal time embedding
├── unet_deep.py           # DeepUNet backbone
├── domain_policy.py       # temporal / scale / spatial modulators
├── fusion.py              # DomainFusion -> psi
├── agent_state.py         # per-modulator observations
├── diffusion_schedule.py  # linear schedule + DDIM step
└── infer.py               # 16-step DDIM sampler
```

---

## 📊 Datasets

None of the data is redistributed here; download from the original sources and
follow their data-use terms.

- **Mayo 2020** — [LDCT-and-Projection-data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) (TCIA, DOI [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637)); Moen et al., *Medical Physics*, 2021, [DOI](https://doi.org/10.1002/mp.14594).
- **Mayo 2016** — [2016 Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/); McCollough et al., *Medical Physics*, 2017.
- **Piglet** — [xinario/SAGAN](https://github.com/xinario/SAGAN); Yi & Babyn, *J. Digit. Imaging*, 2018, [DOI](https://doi.org/10.1007/s10278-018-0056-0). Piglet has no normal-dose reference and is used for qualitative assessment only.

Subsets used in the paper: C016-10 %, C002-10 %, L266-25 % for training;
L277-25 %, L067-25 %, L333-25 % for unseen/external evaluation.

---

<!-- ## 🔍 Inference

No checkpoints are provided; load your own trained weights before sampling.

```python
import torch
from mdmrl import DeepUNet, DomainPolicy, sample_with_modulation, sample_no_modulation
from mdmrl.diffusion_schedule import make_linear_schedule

backbone = DeepUNet(base_ch=80, ch_mult=(1, 2, 4, 8), t_dim=256,
                    in_ch=2, out_ch=1, attn_resolutions=(16, 8, 4)).cuda().eval()
policy = DomainPolicy(t_dim=128, num_levels=4, freq_bands=8,
                      enable_freq=False, enable_spatial=True).cuda().eval()

# load your own checkpoints here ...

schedule = make_linear_schedule(200, device="cuda")
x_ld = ...                      # [B, 1, H, W] in [-1, 1]

x_dn = sample_with_modulation(backbone, schedule, policy, x_ld, steps=16)
x_dn_no_rl = sample_no_modulation(backbone, schedule, x_ld, steps=16)  # baseline
``` -->

---

<!-- ## 📝 Notes

- **Diffusion (no RL)** denotes the unmodulated Stage-1 model; **RL-MDM** is the
  full method.
- The optional frequency branch in `domain_policy.py` (`enable_freq`) is ablated
  in the paper and disabled in the final model. -->

---

## 📚 Citation

> *Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising*,
> Neurocomputing (under review).

The BibTeX entry will be added once the paper is accepted.

---

## 📜 License

Released under the [MIT License](LICENSE). The license covers the source code
only; datasets and paper figures remain subject to their original terms.
