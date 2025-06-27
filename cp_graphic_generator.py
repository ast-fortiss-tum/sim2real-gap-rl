#!/usr/bin/env python3
"""
visualize_scalar_shift.py ──────────────────────────────────────────────────
Diagnostics for a *scalar* state‑space dataset stored as

    dataset_cp/
        ├── clean/clean.npy   # shape (365, 24)
        └── noisy/noisy.npy   # same shape

where axis‑0 = episodes (days) and axis‑1 = time‑steps (hours).

We treat each 24‑h trajectory as a 24‑D feature vector.
The script reproduces the same style of visualisations used for the RGB case:

    1. PCA scatter (episodes as 24‑D vectors)
    2. t‑SNE scatter
    3. Per‑time‑step mean curves + absolute diff heat‑map
    4. Histogram of scalar values (clean vs noisy)
    5. Histogram of Δ‑magnitudes (|x[t+1]-x[t]|) > 0.1 only

Run
----
    python visualize_scalar_shift.py --root ./dataset_cp --out ./scalar_figs
"""
import argparse, sys, warnings, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --------------------------- CLI ------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Scalar‑state visual diagnostics")
    p.add_argument("--root", default="./dataset_CP/shift_dataset", help="dataset_cp root folder")
    p.add_argument("--out",  default="./cp_figs_full_shift", help="folder for figures")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------------------- utilities ------------------------------------

def load_matrix(root: Path, sub: str):
    f = root / sub 
    if not f.exists():
        sys.exit(f"file not found: {f}")
    arr = np.load(f)            # shape (N, 24)
    if arr.ndim != 2 or arr.shape[1] != 24:
        sys.exit(f"expected shape (episodes, 24) but got {arr.shape} in {f}")
    return arr.astype(np.float32)

# --------------------------- plotting helpers -----------------------------

def pca_scatter(mat_clean, mat_noisy, out):
    pca = PCA(n_components=2, random_state=0)
    X   = np.vstack([mat_clean, mat_noisy])
    XY  = pca.fit_transform(X)
    n   = len(mat_clean)
    plt.figure(figsize=(6,3))
    plt.scatter(XY[:n,0], XY[:n,1], alpha=.6, label='Clean', marker='o')
    plt.scatter(XY[n:,0], XY[n:,1], alpha=.6, label='Noisy', marker='x')
    #plt.title('Episode‑level PCA'); plt.xlabel('PC‑1'); plt.ylabel('PC‑2'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def tsne_scatter(mat_clean, mat_noisy, out):
    X   = np.vstack([mat_clean, mat_noisy])
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    XY  = tsne.fit_transform(X)
    n   = len(mat_clean)
    plt.figure(figsize=(6,3))
    plt.scatter(XY[:n,0], XY[:n,1], alpha=.6, label='Clean', marker='o')
    plt.scatter(XY[n:,0], XY[n:,1], alpha=.6, label='Noisy', marker='x')
    #plt.title('Episode‑level t‑SNE'); 
    plt.xlabel('t‑SNE‑1'); plt.ylabel('t‑SNE‑2'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def mean_curve(mat_clean, mat_noisy, out):
    mu_c = mat_clean.mean(axis=0)   # shape (24,)
    mu_n = mat_noisy.mean(axis=0)
    diff = np.abs(mu_c - mu_n)
    hours = np.arange(24)

    fig, ax = plt.subplots(2,1, figsize=(7,6), gridspec_kw={'height_ratios':[2,1]})
    ax[0].plot(hours, mu_c, label='Clean')
    ax[0].plot(hours, mu_n, label='Noisy')
    ax[0].set_xticks(hours); ax[0].set_ylabel('state'); ax[0].set_title('Mean 24‑h profile'); ax[0].legend()

    ax[1].bar(hours, diff)
    ax[1].set_xticks(hours); ax[1].set_ylabel('|Δ|'); ax[1].set_title('|mean_clean - mean_noisy|')
    plt.tight_layout(); plt.savefig(out); plt.close()


def value_hist(mat_clean, mat_noisy, out):
    plt.figure(figsize=(6,3))
    plt.hist(mat_clean.ravel(), bins=50, alpha=.6, label='Clean')
    plt.hist(mat_noisy.ravel(), bins=50, alpha=.6, label='Noisy')
    plt.xlabel('scalar value'); plt.ylabel('#samples'); #plt.title('Value distribution'); 
    plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def delta_hist(mat_clean, mat_noisy, out):
    def deltas(mat):
        return np.abs(mat[:,1:] - mat[:,:-1]).ravel()
    d_c, d_n = deltas(mat_clean), deltas(mat_noisy)
    #d_c = d_c[d_c > 0.1]
    #d_n = d_n[d_n > 0.1]
    if not (d_c.size and d_n.size):
        warnings.warn('all deltas ≤ 0.1; skipping histogram'); return
    bins = np.linspace(0, max(d_c.max(), d_n.max()), 40)
    plt.figure(figsize=(6,3))
    plt.hist(d_c, bins=bins, alpha=.6, label='Clean')
    plt.hist(d_n, bins=bins, alpha=.6, label='Noisy')
    plt.xlabel('|x[t+1]-x[t]|'); plt.ylabel('#transitions'); #plt.title('Δ‑magnitude distribution'); 
    plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

# --------------------------- main -----------------------------------------

def main():
    args   = get_args()
    random.seed(args.seed); np.random.seed(args.seed)
    root   = Path(args.root).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)

    for degree in [0.5,0.65,0.8]:

        clean = load_matrix(root, f'clean_shift/clean_dataset_degree_{degree}_Gaussian_noise_0.2_bias_0.5.npy')
        noisy = load_matrix(root, f'noisy_shift/noisy_dataset_degree_{degree}_Gaussian_noise_0.2_bias_0.5.npy')

        pca_scatter(clean, noisy, outdir/f'degree_{degree}_fig_pca.png')
        tsne_scatter(clean, noisy, outdir/f'degree_{degree}_fig_tsne.png')
        mean_curve(clean, noisy, outdir/f'degree_{degree}_fig_mean.png')
        value_hist(clean, noisy, outdir/f'degree_{degree}_fig_hist.png')
        delta_hist(clean, noisy, outdir/f'degree_{degree}_fig_delta_hist.png')

        print(f'[DONE] figures saved in {outdir}')

if __name__ == '__main__':
    main()
