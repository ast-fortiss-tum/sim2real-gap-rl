# Simulation‑to‑Reality Gap in SmartGrids

<p align="center">
  <img src="docs/figs/sim2real_smartgrid.png" alt="Simulation‑to‑Reality workflow" width="600"/>
</p>

> **Research goal:** quantify and mitigate the simulation‑to‑reality (Sim2Real) gap when controlling energy‑flexible **SmartGrid micro‑networks**.

This repository accompanies our experiments on **domain adaptation for power‑flow control**, built on:

* **CommonPower** — network modelling framework by the Chair of Autonomous Systems, TUM \[1].
* **DARC** — *Domain Adaptation from Rewards using Classifiers* \[2].
* **DARAIL** (Backbone of the code) \[3].
---

## Prerequisites

Install Python requirements:

```bash
pip install -r requirements.txt
```

---

## Repository structure

```text
smartgrid/
├── rest of code      
├── run_extract_DAE.sh                    # offline dataset + DAE training
├── run_experiments_total.sh              # complete experiment launcher
├── run_New_experiments.sh                # experiment launcher for many seeds
├── extract_graphics/                     # result‑parsing & plotting scripts
├── runs/                                 # TensorBoard logs will appear here
```

---

## Reproducing the Experiments

### 1. Train a denoising auto‑encoder (DAE)

Offline calibration for dynamic‑different datasets:

```bash
bash run_extract_DAE.sh
```

This extracts the dataset and trains a DAE that later supplies robust latent states to DARC.

### 2. Run Sim2Real discrepancy experiments

Run the following bash for many case scenarios including changing charging efficiency, soc efficiency, discharging efficiency a
```bash
bash run_experiments_total.sh \
```
or for the experiments explained in the main report run (same seed generator):
```bash
bash run_New_experiments.sh \
```
The flags inside the executable are the following:

| Flag / Parameter   | Meaning & Typical Values                                                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lp`             | Apply *limited-power* constraint to control inputs  `0` = off  `1` = on                                                                                               |
| `--noise`          | Additive Gaussian sensor noise (σ). Example script grid uses **0.2** (set 0 = no noise).                                                                              |
| `--broken`         | Switch to **two-house** environment  `0` = single-house (default)  `1` = two-house                                                                                    |
| `--break_src`      | Disable battery of house 2 (sets capacity to 0 kWh)  `0` = healthy  `1` = disabled                                                                                    |
| `--lin_src`        | Source which env behaves linear: `0` = target  `1` = source                                                          |
| `--variety-name`   | Environment variety code:<br>• `s` – SOC-efficiency<br>• `c` – Charging-efficiency<br>• `d` – discharging efficiency<br>• `v` – all effiencies<br>• `lp` – limited power input |
| `--degree`         | Efficience or degree (for `--variety-name` = `s`,`c`,`d` and `v`) (e.g., **0.5**, **0.65**, **0.8**).                                                                                                     |
| `--bias`           | Systematic measurement bias added to observations (e.g., **0.5**).                                                                                                    |
| `--noise_cfrs`     | Extra noise injected to train classifiers (default 0.0).                                                                                                 |
| `--use_denoiser`   | Route observations through the pretrained DAE  `0` = no  `1` = yes                                                                                                    |
| `--train-steps`    | Gradient updates per episode (default 201).                                                                                                                           |
| `--lr`             | Learning rate (Adam), default **8 × 10⁻⁴**.                                                                                                                           |
| `--bs`             | Batch size, default **12**.                                                                                                                                           |
| `--update`         | Target-network sync interval in steps (default 1).                                                                                                                    |
| `--deltar`         | Reward scaling factor (default 1).                                                                                                                                    |
| `--seed`           | Random seed for reproducibility (scripts sweep **10 – 100**).                                                                                                         |
| `--save-model`     | Directory path for storing `.pt` checkpoints.                                                                                                                         |
| `--save_file_name` | Prefix for TensorBoard run names and model files.                                                                                                                     |


---

## Result Extraction & Visualisation

All training/evaluation metrics are logged to **TensorBoard** under `runs/`.

To generate publication‑ready tables and graphics:

```bash
python3 extract_graphics/(selected_script).py   
```

The scripts automatically scan `runs/` and save the outputs into `plots/`.

---
## Citation

If you build on this work, please cite:

[1] M. Eichelbeck et al. “CommonPower: A Framework for Safe Data‑Driven Smart Grid Control.” arXiv:2406.03231, 2023. https://arxiv.org/abs/2406.03231

[2] B. Eysenbach et al. “Off‑Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers.” International Conference on Learning Representations (ICLR), 2021.

[3] Y. Guo, Y. Wang, Y. Shi, P. Xu & A. Liu. “Off‑Dynamics Reinforcement Learning via Domain Adaptation and Reward‑Augmented Imitation.” arXiv:2411.09891, 2024. https://arxiv.org/abs/2411.09891

---
## License
MIT — see LICENSE.

