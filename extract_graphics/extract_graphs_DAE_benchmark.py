#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import argparse, random, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy, wilcoxon

""" Main module to train denoising autoencoder on noisy/clean episodes.
This script trains an online recurrent denoising autoencoder on a dataset of noisy and clean episodes.
It uses a custom PyTorch Dataset to load the data, defines the model architecture,
and implements a training loop with early stopping and learning rate scheduling.
The model is designed to denoise sequences of observations from a manual control task.
The training process includes validation and saves the best model based on validation loss.
The model is then used to visualize the denoising performance on a few validation sequences.
"""


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class DenoisingDataset(Dataset):
    """Load clean/noisy episodes saved as .npy (length 24 each)."""
    def __init__(self, clean_file, noisy_file):
        self.clean = np.load(clean_file, allow_pickle=True)
        self.noisy = np.load(noisy_file, allow_pickle=True)
        assert len(self.clean) == len(self.noisy)

    def __len__(self): return len(self.clean)

    def __getitem__(self, idx):
        c, n = self.clean[idx], self.noisy[idx]
        if c.ndim == 1: c = c[:, None]
        if n.ndim == 1: n = n[:, None]
        return torch.tensor(n, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Model (compatible with old & new checkpoint key names)
# ---------------------------------------------------------------------
class OnlineDenoisingAutoencoder(nn.Module):
    def __init__(self, in_dim=1, proj_dim=16, hid_dim=32):
        super().__init__()
        self.input_linear  = nn.Linear(in_dim, proj_dim)
        self.lstm          = nn.LSTM(proj_dim, hid_dim, batch_first=True)
        self.output_linear = nn.Linear(hid_dim, in_dim)

    def forward_online(self, x_seq, hidden=None):
        outs = []
        for x_t in x_seq.transpose(0, 1):              # iterate over time
            x_proj, hidden = self.lstm(
                self.input_linear(x_t).unsqueeze(1), hidden
            )
            outs.append(self.output_linear(x_proj.squeeze(1)).unsqueeze(1))
        return torch.cat(outs, dim=1), hidden
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helper: symmetric KL on 1-D signals
# ---------------------------------------------------------------------
def kl_divergence_1d(p, q, bins=256):
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    p_hist, edges = np.histogram(p, bins, (lo, hi), density=True)
    q_hist, _     = np.histogram(q, bins=edges, density=True)
    eps = 1e-10
    p_hist += eps; q_hist += eps
    return 0.5 * (entropy(p_hist, q_hist) + entropy(q_hist, p_hist))
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # ─── command-line args
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise",  type=float, default=0.2)
    ap.add_argument("--bias",   type=float, default=0.5)
    ap.add_argument("--degree", type=float, default=0.65)
    ap.add_argument("--seed",   type=int,   default=42)
    args = ap.parse_args()

    # ─── reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator().manual_seed(args.seed)

    # ─── data paths
    base = "manual_control_dataset/"
    cfile = f"{base}clean/clean_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"
    nfile = f"{base}noisy/noisy_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"

    ds = DenoisingDataset(cfile, nfile)
    _, ds_val = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))], generator=gen)
    val_loader = DataLoader(ds_val, batch_size=24, shuffle=False)

    # ─── model + checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = OnlineDenoisingAutoencoder().to(device)
    ckpt   = f"Denoising_AE/best_online_noise_{args.noise}_bias_{args.bias}_deg_{args.degree}.pth"
    state  = torch.load(ckpt, map_location=device)

    # handle legacy key names
    if "input_linear.weight" in state:
        model.load_state_dict(state, strict=True)
    else:
        new_state = collections.OrderedDict()
        for k, v in state.items():
            if k.startswith("in_lin."):
                k = k.replace("in_lin.", "input_linear.")
            if k.startswith("out_lin."):
                k = k.replace("out_lin.", "output_linear.")
            new_state[k] = v
        model.load_state_dict(new_state, strict=False)
    model.eval()
    print("Loaded checkpoint:", ckpt)

    # ─── accumulators
    metr = {k: ([], [], []) for k in ["mse", "ssim", "psnr", "kl"]}

    err_n, err_s, err_r = [], [], []     # point-wise squared errors

    first_batch_plotted = False

    # ─── evaluation loop
    with torch.no_grad():
        for n, c in val_loader:
            n, c = n.to(device), c.to(device)
            r, _ = model.forward_online(n)

            n, c, r = n.cpu().numpy(), c.cpu().numpy(), r.cpu().numpy()
            s = n - 0.5                               # shifted noisy

            for n_i, s_i, c_i, r_i in zip(n, s, c, r):
                n_i, s_i, c_i, r_i = map(np.squeeze, (n_i, s_i, c_i, r_i))

                # point-wise squared errors
                err_n.extend(((n_i - c_i) ** 2).ravel())
                err_s.extend(((s_i - c_i) ** 2).ravel())
                err_r.extend(((r_i - c_i) ** 2).ravel())

                # episode-level metrics (mean ± std)
                rng = c_i.max() - c_i.min() + 1e-12

                metr["mse"][0].append(((n_i - c_i) ** 2).mean())
                metr["mse"][1].append(((s_i - c_i) ** 2).mean())
                metr["mse"][2].append(((r_i - c_i) ** 2).mean())

                metr["ssim"][0].append(ssim(c_i, n_i, data_range=rng))
                metr["ssim"][1].append(ssim(c_i, s_i, data_range=rng))
                metr["ssim"][2].append(ssim(c_i, r_i, data_range=rng))

                metr["psnr"][0].append(psnr(c_i, n_i, data_range=rng))
                metr["psnr"][1].append(psnr(c_i, s_i, data_range=rng))
                metr["psnr"][2].append(psnr(c_i, r_i, data_range=rng))

                metr["kl"][0].append(kl_divergence_1d(c_i, n_i))
                metr["kl"][1].append(kl_divergence_1d(c_i, s_i))
                metr["kl"][2].append(kl_divergence_1d(c_i, r_i))

            # ─── first-batch plot
            if not first_batch_plotted:
                T = n.shape[1]; t = np.arange(1, T+1)
                fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
                for i in range(4):
                    ax = axes[i]
                    ax.plot(t, n[i].squeeze(), 'r--', label="Noisy")
                    ax.plot(t, s[i].squeeze(), 'm--', label="Noisy – 0.5")
                    ax.plot(t, r[i].squeeze(), 'b-',  label="Denoised")
                    ax.plot(t, c[i].squeeze(), 'g:',  label="Clean")
                    ax.set_ylabel("Value"); ax.grid(True)
                    if i == 0: ax.legend(loc="upper right")
                axes[-1].set_xlabel("Timestep")
                plt.tight_layout()
                out_png = f"plots/eval_plot_noise_{args.noise}_bias_{args.bias}_deg_{args.degree}.png"
                fig.savefig(out_png, dpi=1000)
                plt.show()
                print("Saved plot →", out_png)
                first_batch_plotted = True

    # -----------------------------------------------------------------
    # Reporting helpers
    # -----------------------------------------------------------------
    def mean_std(a): a = np.asarray(a); return a.mean(), a.std()
    def fmt(a): m,s = mean_std(a); return f"{m:.3g}$\\pm${s:.1g}"

    def epi_wilcox(base, recon, higher):
        """Episode-level one-sided Wilcoxon."""
        alt = "greater" if higher else "less"
        return wilcoxon(recon, base, alternative=alt).pvalue

    # -----------------------------------------------------------------
    # LaTeX table builder  (uses point-wise errors for MSE)
    # -----------------------------------------------------------------
    def latex_table(b_idx, b_name):
        rows = []
        spec = [
            ("PSNR", "↑", "psnr", True),
            ("SSIM", "↑", "ssim", True),
            ("MSE",  "↓", "mse",  False),   # will use point-wise test
            ("KL",   "↓", "kl",   False),
        ]
        for label, arrow, key, higher in spec:
            base_epi, recon_epi = metr[key][b_idx], metr[key][2]

            if label == "MSE":
                # point-wise Wilcoxon on squared errors
                baseline_err = err_n if b_idx == 0 else err_s
                p_val = wilcoxon(err_r, baseline_err,
                                 alternative="less").pvalue   # Recon error < baseline?
            else:
                # episode-wise Wilcoxon
                p_val = epi_wilcox(base_epi, recon_epi, higher)

            sig = "YES" if p_val < 0.05 else "NO"
            rows.append((
                label, arrow,
                fmt(base_epi),
                f"\\textbf{{{fmt(recon_epi)}}}",
                f"{p_val:.3g}", sig
            ))

        caption = (
            f"  \\caption{{Significance test: "
            f"\"\"\"\\textit{{Recon vs {b_name}}}\"\"\"}}"
        )

        tbl = [
            r"\begin{table}[ht!]",
            r"  \centering",
            caption,
            r"  \resizebox{\textwidth}{!}{%",
            r"  \begin{tabular}{lccccc}",
            r"    \toprule",
            r"    Metric & Dir. & \multicolumn{1}{c}{Baseline (B vs A)} & "
            r"\multicolumn{1}{c}{Recon (B$\rightarrow$A vs A)} & $p$-value & Sig.? \\",
            r"    \midrule",
        ]
        for l, a, b, c, d, e in rows:
            tbl.append(f"    {l:<15s} & {a} & {b} & {c} & {d} & {e} \\\\")
        tbl += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"\end{table}",
        ]
        return "\n".join(tbl)

    # -----------------------------------------------------------------
    # Print the two tables
    # -----------------------------------------------------------------
    print("\n" + latex_table(0, "Noisy") + "\n")
    print(latex_table(1, "Noisy$-0.5$") + "\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
