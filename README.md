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
├── run_experiments_total.sh              # main experiment launcher
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

```bash
bash run_experiments_total.sh \
     lp=1           \ # limit input power
     noise=1        \ # additive Gaussian sensor noise
     broken=0       \ # 1 → two‑house setup with faulty battery
     break_src=0      # 1 → disable battery of second house
```

Flags can be combined freely to recreate the scenarios reported in the paper.

| Flag        | Meaning                                    |
| ----------- | ------------------------------------------ |
| `lp`        | Apply *limited power* constraint to inputs |
| `noise`     | Inject sensor noise with preset σ          |
| `broken`    | Switch to **two‑house** environment        |
| `break_src` | Set battery of house 2 to 0 kWh capacity   |

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
