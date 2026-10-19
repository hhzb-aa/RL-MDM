# 🧠 RL-MDM

**Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising**

PyTorch implementation.

> 📌 *The full code release — implementation, training scripts, evaluation
> utilities, and usage instructions — will be published upon paper acceptance.
> Pretrained weights are not distributed.*

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

## 📦 Release Plan

The following components will be released upon paper acceptance:

- **Model** — DeepUNet noise-prediction backbone and the temporal / scale /
  spatial modulators with `DomainFusion`.
- **Training** — two-stage pipeline: Stage 1 `eps`-prediction warm-up and
  Stage 2 chain-level REINFORCE alignment.
- **Inference & evaluation** — 16-step deterministic DDIM sampler; PSNR, SSIM,
  EPI, HFER, and LPIPS metrics.
- **Data** — paired LDCT/NDCT PNG dataset loader with the deterministic 80/20
  split.
- **Not included** — pretrained weights.

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

<!-- ## 🗂️ Data Format

[`mdmrl/dataset_real.py`](mdmrl/dataset_real.py) reads **paired 8-bit grayscale
PNG slices** from two sibling folders under one dataset root:

```text
<data_root>/
├── low/     # low-dose slices   (input)
└── full/    # normal-dose slices (ground truth)
```

- Pairing is by **exact filename**: `low/x.png` ↔ `full/x.png`. Only `.png`
  files are read; a missing counterpart raises `FileNotFoundError`.
- Images must be single-channel 8-bit grayscale (PIL mode `L`), converted to
  `[1, H, W]` in `[0, 1]`, resized to `256 × 256`, and normalised to
  `[-1, 1]`. No augmentation is applied.
- The split is deterministic: `torch.randperm(n, generator=manual_seed(42))`,
  80 % train / 20 % test. It is a **within-patient, slice-level** split;
  unseen-patient evaluation uses separate roots that contribute no training
  slices.

---

## 🚀 Training

Built-in defaults (`Config` in `mdmrl/train.py`) reproduce the C002-10 % setup.

```bash
# Stage 1: backbone warm-up (eps-prediction MSE)
python -m mdmrl.train --stage stage1

# Stage 2: aligned joint training (RL-MDM)
python -m mdmrl.train --stage stage2
```

Use `--data_root`, `--run_dir`, `--device`, and optionally `--config <file>.yaml`
to override defaults. Checkpoints are written to `--run_dir` as
`ckpt_stage1.pt`, `ckpt_stage2_best.pt`, and `ckpt_stage2_final.pt`.

| Parameter | Value |
|---|---|
| Backbone | `base_ch = 80`, `ch_mult = (1, 2, 4, 8)`, attention at `(16, 8, 4)` |
| Image / diffusion | `256 × 256`, `T = 200` |
| Policy | width `128`, 4 levels, 8 frequency bands |
| Optimiser | Adam, lr `1.5e-4` (backbone) / `7.5e-5` (policy), batch size `8` |
| RL rollout | `chain_len = 24`, update every `50` steps |
| Reward | `L1 + 0.7 · SSIM_loss`, alignment weight `beta = 1.0` |
| Inference | 16-step DDIM, `eta = 0` |

---

## 🔍 Inference

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
```

---

## 📝 Notes

- **Diffusion (no RL)** denotes the unmodulated Stage-1 model; **RL-MDM** is the
  full method.
- The optional frequency branch in `domain_policy.py` (`enable_freq`) is ablated
  in the paper and disabled in the final model.
- LPIPS (`metrics.lpips_distance`) requires `pip install lpips`.

-->

---

## 📚 Citation

> *Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising*,
> Neurocomputing (under review).

The BibTeX entry will be added once the paper is accepted.

---

## 📜 License

Released under the [MIT License](LICENSE). The license covers the source code
only; datasets and paper figures remain subject to their original terms.
