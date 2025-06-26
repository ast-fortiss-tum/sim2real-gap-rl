# fortiss Self‑Driving Devices — Reinforcement Learning Extension

## Overview

This repository contains **additional work** carried out at **fortiss labs** to extend the self‑driving devices research project with reinforcement learning (RL) controllers and vision style‑transfer components.

Key components:

* Pre‑trained RL policies (TC1‑TC5 + ablation)
* CycleGAN visual transfer models for simulation ⇄ camera domains
* Training scripts for the Unity simulator and fake‑camera domain adaptation
* Evaluation utilities (statistics, GIF generation, result visualisation)

---

## Repository structure

```text
fortiss_rl_extension/
├── CycleGAN/
│   ├── cyclegan_model_G_AB.pth   # simulation ➜ camera
│   └── cyclegan_model_G_BA.pth   # camera ➜ simulation
├── dataset_final/                # stored training logs & checkpoints
├── evaluation/                   # extract tables of statistics
├── visualization_results/        # scripts & assets for GIFs/plots
├── create_GIF.py                 # helper – combines frames to GIF
├── training_RL.py                # RL training (simulation input)
├── training_RL_GAN.py            # RL training (fake‑camera input)
├── launch/
│   └── test_launch.launch        # example ROS launch file
└── src/                          # main ROS workspace (nodes)
    └── <your_node_package>/
        ├── model_RL_TC1.py       # point‑following (real data)
        ├── model_RL_TC2.py       # trained in simulation – camera input
        ├── model_RL_TC3.py       # trained in simulation – sim‑state input
        ├── model_RL_TC4.py       # trained on fake camera – camera input
        ├── model_RL_TC5.py       # trained in simulation – fake sim input
        └── model_RL_abl.py       # ablation – fake camera input
```

---

## Prerequisites

| Requirement             | Version / Notes                               |
| ----------------------- | --------------------------------------------- |
| **ROS**                 | Noetic or Melodic (tested on Ubuntu 20.04)    |
| **Unity ML‑Agents**     | Simulator installation already completed ✔    |
| **Python**              | ≥ 3.8 with dependencies in `requirements.txt` |
| **CUDA GPU** (optional) | CUDA ≥ 11 boosts training speed               |

---

## Installation

1. **Clone** inside your catkin workspace:

   ```bash
   cd ~/catkin_ws/src
   git clone <this_repository_url> fortiss_rl_extension
   cd ..
   catkin_make
   ```
2. **Copy RL policy files** into the node package inside `src/` (see structure above).
3. **Create** the `CycleGAN/` directory at project root and place the two `.pth` models there.

---

## Usage

### Launch an experiment on the real robot

```bash
roslaunch fortiss_rl_extension test_launch.launch
```

In the launch file, load **exactly one** `model_RL_TC*.py` (or `model_RL_abl.py`) according to the scenario:

| Model file        | Training domain         | Runtime observation |
| ----------------- | ----------------------- | ------------------- |
| `model_RL_TC1.py` | Real‑world trajectories | Waypoint vector     |
| `model_RL_TC2.py` | Unity simulation        | Camera              |
| `model_RL_TC3.py` | Unity simulation        | Simulation state    |
| `model_RL_TC4.py` | Fake camera (CycleGAN)  | Camera              |
| `model_RL_TC5.py` | Unity simulation        | Fake sim state      |
| `model_RL_abl.py` | Fake camera             | Fake camera         |

> **Safety checklist** 🛡️
> • Battery ≥ 11.5 V
> • Stable Wi‑Fi connection to ROS master
> • Obstacle‑free testing area

---

### Train a new policy

| Script               | Observation        | Notes                                |
| -------------------- | ------------------ | ------------------------------------ |
| `training_RL.py`     | Simulation state   | Opens Unity executable automatically |
| `training_RL_GAN.py` | Fake camera frames | Requires pre‑trained CycleGAN        |

---

### Evaluate & visualise results

```bash
python evaluation/evaluate.py --runs dataset_final
python create_GIF.py --input dataset_final/run42
```

The scripts deposit tables, GIFs and summary graphics in `visualization_results/`.

---

## Results

`dataset_final/` already contains the runs referenced in the accompanying Master Thesis.

| Experiment | Success rate | Avg. reward |
| ---------- | ------------ | ----------- |
| TC1        | 94 %         | 235 ± 12    |
| TC2        | 87 %         | 210 ± 15    |
| TC3        | *…*          | *…*         |

Full statistics can be regenerated with the `evaluation/` utilities.

---

## Citation

If you build on this work, please cite:

> **Author Name**, “Domain‑Invariant Reinforcement Learning for Self‑Driving Devices,” Master Thesis, fortiss labs, Technical University of Munich, 2025.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contact

Questions or improvements? Open an issue or pull request ✉️.
