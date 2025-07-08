#!/usr/bin/env python3

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
import image_metrics_utils as imu

"""Benchmark script for comparing image generation models with real images.
This script samples images from two directories (simulated and real), applies
generators to transform the images, computes various image metrics, and saves
the results in a structured format."""

# ---- CycleGAN Generator (from your code 3) ----
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_residual_blocks: int = 6):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        in_c, out_c = ngf, ngf * 2
        # Down-sampling
        for _ in range(2):
            layers += [nn.Conv2d(in_c, out_c, 4, 2, 1),
                       nn.InstanceNorm2d(out_c),
                       nn.ReLU(True)]
            in_c, out_c = out_c, out_c * 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(in_c))
        # Up-sampling
        out_c = in_c // 2
        for _ in range(2):
            layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                       nn.InstanceNorm2d(out_c),
                       nn.ReLU(True)]
            in_c, out_c = out_c, out_c // 2
        layers += [nn.Conv2d(in_c, output_nc, 7, 1, 3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---- Utilities ----
def list_episode_files(folder: Path, ext: str = "png") -> List[Path]:
    return sorted(folder.glob(f"E*_*.{ext}"))

def sample_observations(root: Path, k: int, seed: int = 42) -> List[np.ndarray]:
    obs_dir = root / "observations"
    if not obs_dir.exists():
        sys.exit(f"[ERR] 'observations' not found in {root}")
    files = list_episode_files(obs_dir)
    if not files:
        sys.exit(f"[ERR] no PNGs found in {obs_dir}")
    if len(files) < k:
        warnings.warn(f"requested {k} but only {len(files)} available; using all")
        k = len(files)
    random.seed(seed)
    picks = random.sample(files, k)
    return [imageio.imread(p) for p in tqdm(picks, desc=f"Loading {root.name}")]

def strip(imgs: List[np.ndarray], title: str, out_path: Path):
    k = len(imgs)
    fig, ax = plt.subplots(1, k, figsize=(3*k, 3))
    ax = np.atleast_1d(ax)
    for i, (a, img) in enumerate(zip(ax, imgs)):
        a.imshow(img); a.axis('off'); a.set_title(f"#{i+1}")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_path); plt.close()

def quad(sim: List[np.ndarray], ab: List[np.ndarray], ba: List[np.ndarray], real: List[np.ndarray], out_path: Path):
    k = len(sim); headers = ["Sim", "fakeReal", "Real", "fakeSim"]
    fig, axes = plt.subplots(k, 4, figsize=(12, 3*k))
    for r in range(k):
        for c, img in enumerate([sim[r], ab[r], real[r], ba[r]]):
            axes[r, c].imshow(img); axes[r, c].axis('off')
            if r == 0: axes[r, c].set_title(headers[c])
    plt.tight_layout(); plt.savefig(out_path); plt.close()

_tfms_in = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
_tfms_out = transforms.ToPILImage()

def load_generator(weights: Path, device: torch.device) -> nn.Module:
    net = Generator().to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval(); return net

def apply_gen(net: nn.Module, imgs: List[np.ndarray], device: torch.device) -> List[np.ndarray]:
    outs = []
    with torch.no_grad():
        for img in tqdm(imgs, desc="CycleGAN"):
            h, w = img.shape[:2]
            x = _tfms_in(Image.fromarray(img)).unsqueeze(0).to(device)
            y = net(x).mul(0.5).add(0.5)  # [-1,1] → [0,1]
            y_pil = _tfms_out(y.squeeze(0).cpu().clamp(0,1)).resize((w, h), Image.BICUBIC)
            outs.append(np.array(y_pil))
    return outs

def compare_metrics(img1, img2):
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

def plot_metrics_box(metrics_A_vs_B, metrics_AB_vs_B, metrics_BA_vs_A, metric_names, out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    metrics_dict = {
        'A vs B': {name: [m[name] for m in metrics_A_vs_B] for name in metric_names},
        'A→B vs B': {name: [m[name] for m in metrics_AB_vs_B] for name in metric_names},
        'B→A vs A': {name: [m[name] for m in metrics_BA_vs_A] for name in metric_names}
    }
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6), squeeze=False)
    axes = axes[0]
    for idx, mname in enumerate(metric_names):
        data = [
            metrics_dict['A vs B'][mname],
            metrics_dict['A→B vs B'][mname],
            metrics_dict['B→A vs A'][mname]
        ]
        bplot = axes[idx].boxplot(data, labels=['A vs B', 'A→B vs B', 'B→A vs A'], patch_artist=True)
        axes[idx].set_title(mname)
        axes[idx].set_xticklabels(['A vs B', 'A→B vs B', 'B→A vs A'], rotation=15)
        # Show mean and std for each comparison
        for i, group in enumerate(data):
            mean = np.mean(group)
            std = np.std(group)
            axes[idx].annotate(f"μ={mean:.2f}\nσ={std:.2f}", xy=(i+1, mean), 
                               xytext=(0,10), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8, color='blue')
    fig.suptitle("Image Domain-Shift Metrics\n(box: median/IQR, μ=mean, σ=std)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(str(out_dir / "all_metrics_boxplot.png"))
    plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(description="CycleGAN domain-shift metrics/visualization.")
    p.add_argument("--sim", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_sim", help="root folder of SIM dataset")
    p.add_argument("--real", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_real", help="root folder of REAL dataset")
    p.add_argument("--model_ab", default = "/home/cubos98/Desktop/MA/Shift_visuals/CycleGAN/cyclegan_model_G_AB.pth", help="CycleGAN generator (sim→real)")
    p.add_argument("--model_ba", default = "/home/cubos98/Desktop/MA/Shift_visuals/CycleGAN/cyclegan_model_G_BA.pth", help="CycleGAN generator (real→sim)")
    p.add_argument("--out", default="./dc_figs", help="output directory")
    p.add_argument("--k", type=int, default=5, help="number of frames per pile")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="cuda:0 or cpu")
    return p.parse_args()

def print_metrics_table(metrics_json_path, save_csv=False, csv_path=None):
    import numpy as np
    import json

    # Load results
    with open(metrics_json_path, "r") as f:
        results = json.load(f)

    metric_sets = results["metrics"]
    metric_names = list(metric_sets["A_vs_B"][0].keys())
    comps = ["A_vs_B", "AB_vs_B", "BA_vs_A"]
    comp_labels = {
        "A_vs_B": "A vs B",
        "AB_vs_B": "A→B vs B",
        "BA_vs_A": "B→A vs A"
    }

    # Compute means
    mean_vals = {comp: [] for comp in comps}
    for comp in comps:
        for m in metric_names:
            vals = [x[m] for x in metric_sets[comp]]
            mean_vals[comp].append(np.mean(vals))

    # Print markdown table
    header = "| Metric " + " | ".join(f"| {comp_labels[c]}" for c in comps) + " |"
    sep    = "|" + "----|" * (len(comps)+1)
    print(header)
    print(sep)
    for i, m in enumerate(metric_names):
        line = f"| {m} " + " ".join(f"| {mean_vals[c][i]:.4f}" for c in comps) + " |"
        print(line)

    # Optionally, save as CSV
    if save_csv:
        import csv
        csv_path = csv_path or (str(metrics_json_path).replace(".json", "_mean_table.csv"))
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["Metric"] + [comp_labels[c] for c in comps])
            for i, m in enumerate(metric_names):
                writer.writerow([m] + [f"{mean_vals[c][i]:.4f}" for c in comps])
        print(f"\n[INFO] Means table saved to {csv_path}")

def test_significance_two_directions(metrics_json_path):
    """
    Prints two significance tables:
      1. For sim-to-real:     A_vs_B vs AB_vs_B   (does GAN help from sim to real?)
      2. For real-to-sim:     B_vs_A vs BA_vs_A   (does GAN help from real to sim?)
    Each uses the correct 'alternative' direction for each metric.
    """
    import numpy as np
    import json
    from scipy.stats import wilcoxon

    # Typical direction for metrics
    higher_better = {
        "psnr", "ssim", "corr_coeff", "hist_intersect"
    }
    lower_better = {
        "mse", "lbp_hist_sim", "texture_sim", "wasserstein_dist",
        "kl_divergence", "perceptual_dist", "ifd"
    }

    with open(metrics_json_path, "r") as f:
        results = json.load(f)

    metric_sets = results["metrics"]
    metric_names = list(metric_sets["A_vs_B"][0].keys())

    def run_table(first, second, label_first, label_second, what="GAN"):
        print(f"\nSignificance Test: Does {what} reduce the gap for {label_first} vs {label_second}?\n")
        print("| Metric | ↑/↓ better | {} (mean±std) | {} (mean±std) | alternative | p-value | Significant? |".format(label_first, label_second))
        print("|--------|------------|-------------------|--------------------|-------------|---------|--------------|")
        for i, m in enumerate(metric_names):
            vals_a = np.array([x[m] for x in metric_sets[first]])
            vals_ab = np.array([x[m] for x in metric_sets[second]])

            if m in higher_better:
                alternative = 'greater'
                better = "↑"
            elif m in lower_better:
                alternative = 'less'
                better = "↓"
            else:
                alternative = 'less'
                better = "?"
                print(f"Warning: Unknown metric direction for '{m}'. Defaulting to 'less'.")

            try:
                stat, p = wilcoxon(vals_ab, vals_a, alternative=alternative)
            except Exception as e:
                p = float('nan')

            mean_a, std_a = np.mean(vals_a), np.std(vals_a)
            mean_ab, std_ab = np.mean(vals_ab), np.std(vals_ab)
            signif = "**YES**" if (p < 0.05) else "no"
            print(f"| {m} | {better} | {mean_a:.4f}±{std_a:.4f} | {mean_ab:.4f}±{std_ab:.4f} | {alternative:>9} | {p:.4g} | {signif} |")

    # Table 1: sim-to-real (A vs B and AB vs B)
    run_table("A_vs_B", "AB_vs_B", "A vs B", "A→B vs B", what="GAN A→B")
    # Table 2: real-to-sim (B vs A and BA vs A)
    run_table("B_vs_A", "BA_vs_A", "B vs A", "B→A vs A", what="GAN B→A")

def test_significance_gan_vs_gan(metrics_json_path):
    """
    Compares AB_vs_B (sim→real GAN output vs real) to A_vs_BA (sim vs real→sim GAN output).
    Null hypothesis: Distributions are equal.
    Direction is chosen per metric as before.
    """
    import numpy as np
    import json
    from scipy.stats import wilcoxon

    higher_better = {
        "psnr", "ssim", "corr_coeff", "hist_intersect"
    }
    lower_better = {
        "mse", "lbp_hist_sim", "texture_sim", "wasserstein_dist",
        "kl_divergence", "perceptual_dist", "ifd"
    }

    with open(metrics_json_path, "r") as f:
        results = json.load(f)

    metric_sets = results["metrics"]
    metric_names = list(metric_sets["AB_vs_B"][0].keys())

    print("\nFinal Table: GAN vs GAN (AB vs B vs A vs BA)")
    print("Null hypothesis: No difference between sim→real GAN (AB vs B) and real→sim GAN (A vs BA) for each metric.\n")
    print("| Metric | ↑/↓ better | AB_vs_B (mean±std) | A_vs_BA (mean±std) | alternative | p-value | Significant? |")
    print("|--------|------------|--------------------|--------------------|-------------|---------|--------------|")

    for i, m in enumerate(metric_names):
        ab_vs_b = np.array([x[m] for x in metric_sets["AB_vs_B"]])
        a_vs_ba = np.array([x[m] for x in metric_sets["A_vs_BA"]]) if "A_vs_BA" in metric_sets else np.array([x[m] for x in metric_sets["BA_vs_A"]])
        # The BA_vs_A set can be used for the second, if your script saved under that name.

        if m in higher_better:
            alternative = 'greater'
            better = "↑"
        elif m in lower_better:
            alternative = 'less'
            better = "↓"
        else:
            alternative = 'less'
            better = "?"
            print(f"Warning: Unknown metric direction for '{m}'. Defaulting to 'less'.")

        try:
            stat, p = wilcoxon(ab_vs_b, a_vs_ba, alternative=alternative)
        except Exception as e:
            p = float('nan')

        mean_ab, std_ab = np.mean(ab_vs_b), np.std(ab_vs_b)
        mean_ba, std_ba = np.mean(a_vs_ba), np.std(a_vs_ba)
        signif = "**YES**" if (p < 0.05) else "no"
        print(f"| {m} | {better} | {mean_ab:.4f}±{std_ab:.4f} | {mean_ba:.4f}±{std_ba:.4f} | {alternative:>9} | {p:.4g} | {signif} |")

def print_latex_significance_table(metrics_json_path, comp1, comp2, label1, label2, test_label=""):
    import numpy as np
    import json
    from scipy.stats import wilcoxon

    higher_better = {
        "psnr", "ssim", "corr_coeff", "hist_intersect"
    }
    lower_better = {
        "mse", "lbp_hist_sim", "texture_sim", "wasserstein_dist",
        "kl_divergence", "perceptual_dist", "ifd"
    }

    with open(metrics_json_path, "r") as f:
        results = json.load(f)

    metric_sets = results["metrics"]
    metric_names = list(metric_sets[comp1][0].keys())

    print("\\begin{table}[ht!]")
    print("  \\centering")
    print(f"  \\caption{{Significance testing of GAN domain adaptation: {test_label}}}")
    print("  \\label{tab:significance_gan}")
    print("  \\begin{tabular}{lcccccc}")
    print("    \\toprule")
    print("    Metric & $\\uparrow/\\downarrow$ better & %s (mean$\\pm$std) & %s (mean$\\pm$std) & Alternative & $p$-value & Significant? \\\\" % (label1, label2))
    print("    \\midrule")
    for m in metric_names:
        vals_1 = np.array([x[m] for x in metric_sets[comp1]])
        vals_2 = np.array([x[m] for x in metric_sets[comp2]])

        if m in higher_better:
            alternative = 'greater'
            better = "$\\uparrow$"
        elif m in lower_better:
            alternative = 'less'
            better = "$\\downarrow$"
        else:
            alternative = 'less'
            better = "?"
        try:
            stat, p = wilcoxon(vals_2, vals_1, alternative=alternative)
        except Exception:
            p = float('nan')

        mean_1, std_1 = np.mean(vals_1), np.std(vals_1)
        mean_2, std_2 = np.mean(vals_2), np.std(vals_2)
        signif = "\\textbf{YES}" if (p < 0.05) else "no"
        print(f"    {m} & {better} & {mean_1:.4f}$\\pm${std_1:.4f} & {mean_2:.4f}$\\pm${std_2:.4f} & {alternative} & {p:.4g} & {signif} \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

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

    # 4. Save strips and quad (optional, keep for visuals)
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
    metrics_B_vs_A  = []
    metrics_A_vs_BA = []
    for i, (A, B, AB, BA) in enumerate(zip(sim_imgs, real_imgs, ab_imgs, ba_imgs)):
        metrics_A_vs_B.append(compare_metrics(A, B))
        metrics_AB_vs_B.append(compare_metrics(AB, B))
        metrics_BA_vs_A.append(compare_metrics(BA, A))
        metrics_B_vs_A.append(compare_metrics(B, A))   # <--- Add this
        metrics_A_vs_BA.append(compare_metrics(A, BA)) # <--- And this
        print(f"[{i+1}/{args.k}] metrics calculated.")

    # 6. Save metrics JSON
    results = {
        "args": vars(args),
        "metrics": {
            "A_vs_B":  metrics_A_vs_B,
            "AB_vs_B": metrics_AB_vs_B,
            "BA_vs_A": metrics_BA_vs_A,
            "B_vs_A":  metrics_B_vs_A,   # <--- Add here
            "A_vs_BA": metrics_A_vs_BA   # <--- And here
        }
    }
    with open(out_dir / "piles_image_compare_metrics.json", "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"[DONE] Metrics JSON saved to {out_dir/'piles_image_compare_metrics.json'}")

    # 7. Plot all metrics as boxplots in a single figure
    metric_names = list(metrics_A_vs_B[0].keys())
    plot_metrics_box(metrics_A_vs_B, metrics_AB_vs_B, metrics_BA_vs_A, metric_names, out_dir)
    print(f"[DONE] Boxplot saved to {out_dir/'all_metrics_boxplot.png'}")

if __name__ == "__main__":
    main()
    print_metrics_table(
        metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"),
        save_csv=True  # Set to False if you don't want a CSV
    )
    test_significance_two_directions(metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"))
    test_significance_gan_vs_gan(metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"))

    print_latex_significance_table(
        metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"),
        comp1="A_vs_B",
        comp2="AB_vs_B",
        label1="A vs B",
        label2="A$\\rightarrow$B vs B",
        test_label="Sim2Real: Does GAN reduce the observational gap?"
    )

    print_latex_significance_table(
        metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"),
        comp1="B_vs_A",
        comp2="BA_vs_A",
        label1="B vs A",
        label2="B$\\rightarrow$A vs A",
        test_label="Real2Sim: Does GAN reduce the observational gap?"
    )

    print_latex_significance_table(
        metrics_json_path=Path("./dc_figs/piles_image_compare_metrics.json"),
        comp1="AB_vs_B",
        comp2="A_vs_BA",
        label1="A$\\rightarrow$B vs B",
        label2="A vs B$\\rightarrow$A",
        test_label="Which GAN direction performs better?"
    )

