import argparse, random, sys, warnings, json
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import image_metrics_utils as imu  # <---- ADD THIS
from extract_obs2 import *

"""Benchmark script for comparing image generation models with real images.
This script samples images from two directories (simulated and real), applies
generators to transform the images, computes various image metrics, and saves
the results in a structured format."""

# ...[keep Generator and other helper classes/functions as before]...

def compare_metrics(img1, img2):
    # All comparisons expect img1, img2 as numpy arrays (H, W, 3) uint8
    # Some metrics expect list, so wrap in [ ] where needed
    # (Order is always: input vs reference)
    return {
        "mse": float(imu.calculate_mse(img1, img2)),
        "psnr": float(imu.calculate_psnr(img1, img2)),
        "ssim": float(imu.calculate_ssim(img1, img2)),
        "corr_coeff": float(imu.calculate_correlation_coefficient(img1, img2)),
        "lbp_hist_sim": float(imu.calculate_lbp_histogram_similarity(img1, img2)),
        "nmi": float(imu.calculate_normalized_mutual_info([img1], [img2])[0]),
        "texture_sim": float(imu.calculate_texture_similarity([img1], [img2])[0]),
        "wasserstein_dist": float(imu.calculate_wd([img1], [img2])[0]),
        "kl_divergence": float(imu.calculate_kl_divergence([img1], [img2])[0]),
        "perceptual_dist": float(imu.calculate_perceptual_distances([img1], [img2])[0]),
        "hist_intersect": float(imu.calculate_histogram_intersection([img1], [img2])[0]),
        "ifd": float(imu.calculate_ifd(img1)),
    }

def plot_metric_bars(metrics_A_vs_B, metrics_AB_vs_B, metrics_BA_vs_A, metric_names, out_dir):
    import matplotlib.pyplot as plt
    k = len(metrics_A_vs_B)
    n_metrics = len(metric_names)
    x = np.arange(k)
    for m_idx, mname in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(8, 4))
        y1 = [d[mname] for d in metrics_A_vs_B]
        y2 = [d[mname] for d in metrics_AB_vs_B]
        y3 = [d[mname] for d in metrics_BA_vs_A]
        ax.plot(x, y1, 'o-', label='A vs B (Sim vs Real)')
        ax.plot(x, y2, 's-', label='A→B vs B (Sim→Real vs Real)')
        ax.plot(x, y3, '^-', label='B→A vs A (Real→Sim vs Sim)')
        ax.set_title(f"{mname} across pairs")
        ax.set_xlabel("Image Index")
        ax.set_ylabel(mname)
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / f"metric_{mname}.png"))
        plt.close()

def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample frames
    sim_imgs = sample_observations(Path(args.sim).expanduser(), args.k, args.seed)
    real_imgs = sample_observations(Path(args.real).expanduser(), args.k, args.seed)

    # 2. Generators
    dev = torch.device(args.device)
    net_ab = load_generator(Path(args.model_ab), dev)
    net_ba = load_generator(Path(args.model_ba), dev)

    # 3. Inference (sizes kept)
    ab_imgs = apply_gen(net_ab, sim_imgs, dev)  # Sim → Real style
    ba_imgs = apply_gen(net_ba, real_imgs, dev)  # Real → Sim style

    # 4. Save strips and quad (as before)
    print("[INFO] saving figures …")
    strip(sim_imgs,  "Simulation",           out_dir / "pile_sim.png")
    strip(ab_imgs,   "Sim→Real (A→B)",      out_dir / "pile_ab.png")
    strip(real_imgs, "Real",                 out_dir / "pile_real.png")
    strip(ba_imgs,   "Real→Sim (B→A)",      out_dir / "pile_ba.png")
    quad(sim_imgs, ab_imgs, ba_imgs, real_imgs, out_dir / "pile_quad.png")

    # 5. Compute metrics
    metrics_A_vs_B  = []
    metrics_AB_vs_B = []
    metrics_BA_vs_A = []
    for i, (A, B, AB, BA) in enumerate(zip(sim_imgs, real_imgs, ab_imgs, ba_imgs)):
        # A vs B
        metrics_A_vs_B.append(compare_metrics(A, B))
        # AB vs B
        metrics_AB_vs_B.append(compare_metrics(AB, B))
        # BA vs A
        metrics_BA_vs_A.append(compare_metrics(BA, A))
        print(f"[{i+1}/{args.k}] metrics calculated.")

    # 6. Save metrics JSON
    results = {
        "args": vars(args),
        "metrics": {
            "A_vs_B":  metrics_A_vs_B,
            "AB_vs_B": metrics_AB_vs_B,
            "BA_vs_A": metrics_BA_vs_A,
        }
    }
    with open(out_dir / "piles_image_compare_metrics.json", "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"[DONE] Metrics JSON saved to {out_dir/'piles_image_compare_metrics.json'}")

    # 7. Plot all metrics
    metric_names = list(metrics_A_vs_B[0].keys())
    plot_metric_bars(metrics_A_vs_B, metrics_AB_vs_B, metrics_BA_vs_A, metric_names, out_dir)
    print(f"[DONE] Metric barplots saved to {out_dir}")

if __name__ == "__main__":
    main()
