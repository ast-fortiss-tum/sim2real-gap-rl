# Simulation‑to‑Reality Gap — SmartGrids & Autonomous Devices Shift visuals

<p align="center">
  <img src="docs/dc_figs/sim2real_smartgrid.png" alt="Sim2Real workflow" width="610"/>
</p>

This repository contains **Graphical extractors** for our research on the **simulation‑to‑reality (Sim2Real) gap** in two domains:

1. **SmartGrid power‑flow control** (CommonPower models)
2. **Self‑driving device perception** (CycleGAN visual transfer)

The work builds on:

* **CommonPower** – modelling framework by the Chair of Autonomous Systems, TUM. \[1].
* **DARC** – *Domain Adaptation from Rewards using Classifiers* \[2].
* **CycleGAN** – for cross‑domain image style transfer \[3].
* Stefano Lambertenghi’s `image_metrics_utils.py` for image‑metric baselines.

---

## Quick Start
Ask for saved Donkey car dataset `dataset_DC/`. Since it is heavy it is not tracked by git.

```bash
# clone inside a fresh virtual‑env or conda env
pip install -r requirements.txt  # installs PyTorch, scikit‑learn, etc.
```

---

## Datasets

| Folder          | Contents                                             | Notes                                          |
| --------------- | ---------------------------------------------------- | ---------------------------------------------- |
| **dataset\_CP** | `clean.npy`, `noisy.npy`                             | 365 days × 24 hours scalar setpoints           |
| **dataset\_DC** | `dataset_sim/`, `dataset_real/`                      | Synchronous PNG frames (`E1_frameXXXX.png`, …) |
| **CycleGAN/**   | `cyclegan_model_G_AB.pth`, `cyclegan_model_G_BA.pth` | Pre‑trained vision transfer weights            |

---

## Visualisation & Shift Diagnostics

Below is a cheat‑sheet for the provided analysis scripts (all accept `‑‑help`).

| Script                                                      | Domain       | Output                                                                                  |
| ----------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| **benchmark\_three.py**                                     | images       | Samples sim/real, applies CycleGAN ↔ transforms, computes PSNR, SSIM, FID, LPIPS (JSON) |
| **cp\_graphic\_generator.py / visualize\_scalar\_shift.py** | scalar 24‑D  | PCA, t‑SNE, mean curves, heat‑maps, histograms                                          |
| **extract\_metrics.py**                                     | images       | Pixel‑diff heat‑maps + aggregate JSON metrics                                           |
| **visualize\_random\_piles.py**                             | images       | Side‑by‑side image grids (`pile_sim.png`, `pile_real.png`)                              |
| **graphic\_dynamic\_shift.py**                              | Δ‑features   | UMAP scatter, Δ‑norm hist., optical‑flow hist.                                          |
| **graphic\_generator.py**                                   | images       | Six‑plot bundle (see script doc‑string)                                                 |
| **plot\_observation\_metrics\_scaled.py**                   | JSON metrics | Bar‑plots & box‑plots (log‑scale)                                                       |

All visualisations are written to the path given by `--out` (default `./dc_figs`).

---

## Citation

> **\[1]** *M. Eichelbeck et al.* “CommonPower: A Framework for Safe Data‑Driven Smart Grid Control.” arXiv:2406.03231, 2023. https://arxiv.org/abs/2406.03231
> **\[2]** *B. Eysenbach et al.*. “Off‑Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers.” International Conference on Learning Representations (ICLR), 2021.
> **\[3]** *J.- Y. Zhu, T. Park, P. Isola, and A. A. Efros*. “Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks”. In: IEEE International Conference on Computer Vision (ICCV). 2017, pp. 2223–2232. doi: 10.1109/ICCV.2017.244.

Citate this works as:
> *Cubides-Herrera, Cristian Sebastian*. “Simulation-to-Reality Gap Mitigation in Reinforcement
Learning: Analyzing Observational and Dynamical Shifts in Smart Grids and Self-Driving Systems”. Master’s Thesis, M.Sc. Robotics Cognition and Intelligence. Technical University of Munich, 2025.

---

## License

MIT — see `LICENSE`.

---

### Contact

Questions / improvements? Open an issue or create a pull request.
