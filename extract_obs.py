#!/usr/bin/env python3
"""
visualize_random_piles.py ────────────────────────────────────────────────
Produces two simple, side‑by‑side image piles that give you a quick visual
feel for the raw observations coming from a **simulation** domain and a
**real‑world** domain.

For each domain the script randomly samples *k* observations (PNG frames) 
from the sub‑folder ``observations`` inside the given dataset root and saves
an image grid to ``<out>/pile_<sim|real>.png``.

Example
-------
    python visualize_random_piles.py \
        --sim  /path/to/dataset_sim \
        --real /path/to/dataset_real \
        --out  ./dc_figs \
        --k    5          # how many frames per pile

Dependencies
------------
    pip install numpy matplotlib imageio tqdm
"""

import argparse, random, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

# ────────────────────────────── helpers ──────────────────────────────────

def list_episode_files(folder: Path, ext: str = "png"):
    """Return image files sorted lexicographically (episode and timestep)."""
    return sorted(folder.glob(f"E*_*.{ext}"))


def sample_observations(root: Path, k: int, seed: int = 42):
    """Randomly pick *k* observation images from <root>/observations.

    Returns a list of numpy arrays with shape (H, W, 3).
    """
    obs_dir = root / "observations"
    if not obs_dir.exists():
        sys.exit(f"[ERR] folder 'observations' not found inside {root}")

    files = list_episode_files(obs_dir)
    if not files:
        sys.exit(f"[ERR] no PNG files found in {obs_dir}")

    if len(files) < k:
        warnings.warn(f"requested k={k} but only {len(files)} images available; using all")
        k = len(files)

    random.seed(seed)
    picks = random.sample(files, k)
    images = []
    for p in tqdm(picks, desc=f"Reading {root.name}"):
        try:
            img = imageio.imread(p)
            images.append(img)
        except Exception as e:
            warnings.warn(f"failed to read {p}: {e}")
    return images


def plot_pile(images, title: str, out_path: Path):
    """Save a horizontal image strip (1×k grid) of the provided images."""
    k = len(images)
    if k == 0:
        warnings.warn("no images to plot – skipping"); return

    # normalise width:height aspect to keep individual images intact
    fig, ax = plt.subplots(1, k, figsize=(3*k, 3))
    if k == 1:
        ax = [ax]
    for i, (a, img) in enumerate(zip(ax, images)):
        a.imshow(img)
        a.set_axis_off()
        a.set_title(f"#{i+1}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ─────────────────────────────── main ────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Create simple piles of random observations for sim and real domains")
    p.add_argument("--sim", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_sim", help="root folder of SIMULATION dataset")
    p.add_argument("--real", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_real", help="root folder of REAL dataset")
    p.add_argument("--out", default="./dc_figs", help="output directory for the image grids")
    p.add_argument("--k", type=int, default=5, help="number of frames per pile (default 5)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_root  = Path(args.sim).expanduser().resolve()
    real_root = Path(args.real).expanduser().resolve()

    #sim_root = args.sim
    #real_root = args.real

    sim_imgs  = sample_observations(sim_root,  args.k, args.seed)
    real_imgs = sample_observations(real_root, args.k, args.seed)

    print("[INFO] saving piles …")
    plot_pile(sim_imgs,  title="Simulation – random observations", out_path=out_dir / "pile_sim.png")
    plot_pile(real_imgs, title="Real – random observations",        out_path=out_dir / "pile_real.png")
    print(f"[DONE] figures written to {out_dir}")


if __name__ == "__main__":
    main()
