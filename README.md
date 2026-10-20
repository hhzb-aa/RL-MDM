# 🧠 RL-MDM

**Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising**

This repository provides the PyTorch model definitions of RL-MDM for reference: the DeepUNet noise-prediction backbone, the temporal, scale, and spatial modulators, and `DomainFusion`. Inference and training code, the dataset loader, the evaluation metrics, and pretrained weights are not included.

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

Low-dose CT noise is spatially non-stationary and varies with dose, anatomy, and diffusion timestep, yet existing denoisers rely on fixed network parameters and a uniform sampling schedule, applying the same correction at every step, scale, and position. RL-MDM instead keeps the DeepUNet noise-prediction backbone fixed in architecture and sampling procedure and learns what to modulate. Three lightweight modulators (temporal, scale, and spatial) observe complementary projections of the diffusion state and emit a packet `psi = ({scale_i}, spatial_g)` that rescales every residual block and gates the intermediate feature maps at each step. The policy is optimised with REINFORCE on a chain-level image-quality reward (`L1 + SSIM`) and updated jointly with the backbone under an aligned objective, so that both pursue the same final-image goal.

Under an identical backbone and 16-step DDIM sampler, the learned modulation changes what the denoiser preserves rather than merely how strongly it denoises. It attains the best EPI and LPIPS on both 10 % chest subsets and the best PSNR among the tested diffusion-based methods, improves PSNR and SSIM over the fixed-schedule baseline on every internal and external validation set, and transfers to unseen patients without fine-tuning.

---

## 📦 Code Scope

| Component | Status |
|---|---|
| DeepUNet noise-prediction backbone | ✅ Released |
| Temporal / scale / spatial modulators | ✅ Released |
| `DomainFusion` and `psi` construction | ✅ Released |
| Modulator observation builder (`agent_state.py`) | ✅ Released |

```text
mdmrl/
├── __init__.py            # package exports
├── interfaces.py          # protocol / dataclass contracts
├── nn_utils.py            # sinusoidal time embedding
├── unet_deep.py           # DeepUNet backbone
├── domain_policy.py       # temporal / scale / spatial modulators
├── fusion.py              # DomainFusion -> psi
└── agent_state.py         # per-modulator observations
```



---



## 📊 Datasets

None of the data is redistributed here; download from the original sources and
follow their data-use terms.

- **Mayo 2020** — [LDCT-and-Projection-data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) (TCIA, DOI [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637)); Moen et al., *Medical Physics*, 2021, [DOI](https://doi.org/10.1002/mp.14594).
- **Mayo 2016** — [2016 Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/); McCollough et al., *Medical Physics*, 2017.
- **Piglet** — [xinario/SAGAN](https://github.com/xinario/SAGAN); Yi & Babyn, *J. Digit. Imaging*, 2018, [DOI](https://doi.org/10.1007/s10278-018-0056-0). Piglet has no normal-dose reference and is used for qualitative assessment only.

---

## 📚 Citation

> *Reinforcement-Learned Multi-Domain Modulation for Low-Dose CT Denoising*,
> Neurocomputing (under review).

The BibTeX entry will be added once the paper is accepted.

---

## 📜 License

Released under the [MIT License](LICENSE). The license covers the source code
only; datasets and paper figures remain subject to their original terms.
