# Simulation‑to‑Reality Gap in SmartGrids

<p align="center">
  <img src="docs/figs/sim2real_smartgrid.png" alt="Simulation‑to‑Reality workflow" width="600"/>
</p>

> **Research goal:** quantify and mitigate the simulation‑to‑reality (Sim2Real) gap when controlling energy‑flexible **SmartGrid micro‑networks**.

This repository accompanies our experiments on **domain adaptation for power‑flow control**, built on:

* **CommonPower** — network modelling framework by the Chair of Autonomous Systems, TUM.
* **DARC** — *Domain Adaptation from Rewards using Classifiers* \[1].

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
| `--lin_src`        | Source realism level: `1` = perfect linear real data (obs-only shift)  `2` = “damaged” env (obs + dyn shift)                                                          |
| `--variety-name`   | Environment variety code:<br>• `s` – *single*-bus grid<br>• `c` – *coupled* multi-bus grid<br>• `d` – *damaged* grid (efficiency 0.5)<br>• `v` – *variable* load grid |
| `--degree`         | Coupling degree between buses (e.g., **0.5**, **0.65**, **0.8**).                                                                                                     |
| `--bias`           | Systematic measurement bias added to observations (e.g., **0.5**).                                                                                                    |
| `--noise_cfrs`     | Extra noise injected into **counter-factual** rollouts (default 0.0).                                                                                                 |
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
python extract_graphics/make_tables.py   # CSV + LaTeX tables
python extract_graphics/make_plots.py    # PNG/SVG graphics
```

The scripts automatically scan `runs/` and save the outputs into `extract_graphics/out/`.

---

## Citation

If you build on this work, please cite:

> \[1] **Johannes Becker**, *et al.* “DARC: Domain Adaptation from Classifiers for Reward‑Shaping in Energy Systems,” IEEE SmartGridComm 2025.
> **This repo:** *Simulation‑to‑Reality Gap in SmartGrids*, fortiss & TUM, 2025.

---

## License

MIT — see `LICENSE`.
