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
Vehicle_RL_extension/
├── CycleGAN/
│   ├── cyclegan_model_G_AB.pth   # simulation ➜ camera
│   └── cyclegan_model_G_BA.pth   # camera ➜ simulation
├── results_reality/                # stored json of poses and going/stopping info
│   ├── dataset1/              
│   ├── dataset2/                
│   ├── dataset3/               
│   └── dataset_final/            
├── evaluation/                       # extract tables of statistics, images and gifs
│   ├── animation_w_speed.py                # Animates one run
│   ├── animation.py                # Animates one run displaying speed and cte (if .json available)
│   ├── complete_animation.py                # Animates one run with speed and cte complete (no need of _ctes.json)
│   ├── evaluation_results_truncated.py                # Tabular results for surviving within defined CTES, std consideration and other stats. Truncate till CTE reaches another threshold
│   ├── evaluation results.py                # Tabular results for surviving within defined CTES, std consideration and other stats.
│   ├── gif_downsample.py               # Helper function to downsample already existing gifs.
│   └── plot_all_circuits.py            # Give final stats after cleaning data and graphics (plot shown in report and presentation).
├── training/
│   ├── training_RL.py                # RL training (simulation input)
│   └── training_RL_GAN.py            # RL training (fake‑camera input)
├── launch/
│   └── basic_launch.launch        # example ROS launch file
├── saved_observations/           #some images saved to create the gif visuals
├── gifs/           #Folder to save gifs
├── figures/        #Folder to save figures
├── policies/       #Folder to save policies
│   ├── model_raw_final.zip          # policy for simulation input
│   └── model_cycle_final.zip            # policy for fake‑camera input
└── src/                           # main ROS workspace (nodes)
    └── nodes/
        ├── keyboard_node.py (demo)       # point‑following (real data)
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
   git clone https://github.com/mirefv/ROS-small-scale-vehicle.git Vehicle_RL_extension
   cd ..
   catkin_make
   ```
2. **Copy RL policy files** into the node package inside `src/` (see structure above).
3. **Create** the `CycleGAN/` directory at project root and place the two `.pth` models there.

---

## Usage

### Launch an experiment on the real robot

```bash
roslaunch mixed_reality basic.launch
```

In the launch file, load **exactly one** `model_RL_TC*.py` (or `model_RL_abl.py`) according to the scenario:

| Model file        | Training domain         | Runtime observation |
| ----------------- | ----------------------- | ------------------- |
| `keyboard_node.py`  | Real‑world trajectories | Waypoint vector     |
| `model_RL_TC2.py` | Unity simulation        | Camera              |
| `model_RL_TC3.py` | Unity simulation        | Simulation state    |
| `model_RL_TC4.py` | Fake camera (CycleGAN)  | Camera              |
| `model_RL_TC5.py` | Unity simulation        | Fake sim state      |
| `model_RL_abl.py` | Fake camera             | Fake camera         |

For `TC1` just run the node `keyboard_node.py` and then type `demo`.
---

### Train a new policy

Under the directory `training/`

| Script               | Observation        | Notes                                |
| -------------------- | ------------------ | ------------------------------------ |
| `train_raw.py`     | Simulation state   | Opens Unity executable automatically |
| `train_cycle.py` | Fake camera frames | Requires pre‑trained CycleGAN        |

---

### Evaluate & visualise results

The scripts deposit tables, GIFs and summary graphics in `evaluation/`.

```bash

python3 evaluation/<selected_script>.py --dataset <selected_dataset> --folder <selected_folder> --name <selected_run>

```
---

## Citation

[1] Cubides-Herrera, Cristian Sebastián. “Simulation-to-Reality Gap Mitigation in Reinforcement Learning: Analyzing Observational and Dynamical Shifts in Smart Grids and Self-Driving Systems”. Master’s Thesis, Robotics, Cognition and Intelligence. Technical University of Munich, 2025.

[2] M. Flores Valdez. “Design and Implementation of a ROS -Based Mixed Reality Framework for Autonomous Small-Scale Vehicles: Bridging the Gap Between Simulation and Reality”. Bachelor’s Thesis, Informatics. Technical University of Munich, 2024.

[3] S. C. Lambertenghi and A. Stocco. “Assessing Quality Metrics for Neural Reality Gap Input Mitigation in Autonomous Driving Testing”. In: 2024 IEEE Conference on Software Testing, Veriﬁcation and Validation (ICST). IEEE, 2024, pp. 173–182. doi: 10.1109/ICST60714.2024.00024.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contact

Questions or improvements? Open an issue or pull request ✉️.
