#!/usr/bin/env python3
"""
visualize_scalar_shift.py ──────────────────────────────────────────────────
Diagnostics for a *scalar* state‑space dataset stored as

    dataset_cp/
        ├── clean/clean.npy   # shape (365, 24)
        └── noisy/noisy.npy   # same shape

where axis‑0 = episodes (days) and axis‑1 = time‑steps (hours).

For each `degree` setting the script produces:
  1. PCA scatter of 24‑D episode vectors
  2. t‑SNE scatter
  3. Mean 24‑h curves + |Δ| bar
  4. Value histogram (clean vs noisy)
  5. Δ‑magnitude (>0.1) histogram
  6. Console print‑out of KL(clean‖noisy), KL(noisy‖clean) and Jensen–Shannon
     divergence for **values** and for **Δ‑magnitudes**

Run
----
    python visualize_scalar_shift.py \
        --root ./dataset_CP/shift_dataset \
        --out  ./cp_figs_full_shift
"""
import argparse, sys, warnings, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import entropy                 # NEW: KL helper

# --------------------------- CLI ------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Scalar‑state visual diagnostics")
    p.add_argument("--root", default="./dataset_CP/shift_dataset", help="dataset_cp root folder")
    p.add_argument("--out",  default="./cp_figs_full_shift", help="folder for figures")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------------------- utilities ------------------------------------

def load_matrix(root: Path, rel_path: str):
    f = root / rel_path
    if not f.exists():
        sys.exit(f"file not found: {f}")
    arr = np.load(f)  # shape (N, 24)
    if arr.ndim != 2 or arr.shape[1] != 24:
        sys.exit(f"expected shape (episodes, 24) but got {arr.shape} in {f}")
    return arr.astype(np.float32)

# --------------------------- KL / JS helper -------------------------------

def kl_js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 50):
    """Return KL(a‖b), KL(b‖a) and Jensen–Shannon (all in nats)."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    hist_a, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    eps = 1e-10
    hist_a += eps; hist_b += eps
    kl_ab = entropy(hist_a, hist_b)
    kl_ba = entropy(hist_b, hist_a)
    js    = 0.5 * (kl_ab + kl_ba)
    return kl_ab, kl_ba, js

# --------------------------- plotting helpers -----------------------------

def pca_scatter(mat_clean, mat_noisy, out):
    pca = PCA(n_components=2, random_state=0)
    XY  = pca.fit_transform(np.vstack([mat_clean, mat_noisy]))
    n   = len(mat_clean)
    plt.figure(figsize=(6,5))
    plt.scatter(XY[:n,0], XY[:n,1], alpha=.6, label='Clean', marker='o')
    plt.scatter(XY[n:,0], XY[n:,1], alpha=.6, label='Noisy', marker='x')
    plt.title('Episode‑level PCA'); plt.xlabel('PC‑1'); plt.ylabel('PC‑2'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def tsne_scatter(mat_clean, mat_noisy, out):
    XY  = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(np.vstack([mat_clean, mat_noisy]))
    n   = len(mat_clean)
    plt.figure(figsize=(6,3))
    plt.scatter(XY[:n,0], XY[:n,1], alpha=.6, label='Clean', marker='o')
    plt.scatter(XY[n:,0], XY[n:,1], alpha=.6, label='Noisy', marker='x')
    plt.title('Episode‑level t‑SNE'); plt.xlabel('t‑SNE‑1'); plt.ylabel('t‑SNE‑2'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def mean_curve(mat_clean, mat_noisy, out):
    mu_c, mu_n = mat_clean.mean(0), mat_noisy.mean(0)
    diff = np.abs(mu_c - mu_n)
    hrs  = np.arange(24)

    fig, ax = plt.subplots(2,1, figsize=(7,6), gridspec_kw={'height_ratios':[2,1]})
    ax[0].plot(hrs, mu_c, label='Clean'); ax[0].plot(hrs, mu_n, label='Noisy')
    ax[0].set_xticks(hrs); ax[0].set_ylabel('state'); ax[0].set_title('Mean 24‑h profile'); ax[0].legend()

    ax[1].bar(hrs, diff); ax[1].set_xticks(hrs); ax[1].set_ylabel('|Δ|'); ax[1].set_title('|mean_clean - mean_noisy|')
    plt.tight_layout(); plt.savefig(out); plt.close()


def value_hist(mat_clean, mat_noisy, out):
    plt.figure(figsize=(6,3))
    plt.hist(mat_clean.ravel(), bins=50, alpha=.6, label='Clean')
    plt.hist(mat_noisy.ravel(), bins=50, alpha=.6, label='Noisy')
    plt.xlabel('scalar value'); plt.ylabel('#samples'); plt.title('Value distribution'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()


def delta_hist(mat_clean, mat_noisy, out):
    deltas = lambda m: np.abs(m[:,1:] - m[:,:-1]).ravel()
    d_c, d_n = deltas(mat_clean), deltas(mat_noisy)
    #d_c, d_n = d_c[d_c>0.1], d_n[d_n>0.1]
    if not (d_c.size and d_n.size):
        warnings.warn('Δ-hist skipped (all ≤0.1)'); return
    bins = np.linspace(0, max(d_c.max(), d_n.max()), 40)
    plt.figure(figsize=(6,4))
    plt.hist(d_c, bins=bins, alpha=.6, label='Clean'); plt.hist(d_n, bins=bins, alpha=.6, label='Noisy')
    plt.xlabel('|x[t+1]-x[t]|'); plt.ylabel('#transitions'); plt.title('Δ‑magnitude'); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

# --------------------------- main -----------------------------------------

def main():
    args   = get_args()
    random.seed(args.seed); np.random.seed(args.seed)
    root   = Path(args.root).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)

    for degree in [0.5, 0.65, 0.8]:
        clean_path = f'clean_shift/clean_dataset_degree_{degree}_Gaussian_noise_0.2_bias_0.5.npy'
        noisy_path = f'noisy_shift/noisy_dataset_degree_{degree}_Gaussian_noise_0.2_bias_0.5.npy'
        clean = load_matrix(root, clean_path)
        noisy = load_matrix(root, noisy_path)

        # visualisations
        pca_scatter(clean, noisy, outdir/f'degree_{degree}_fig_pca.png')
        tsne_scatter(clean, noisy, outdir/f'degree_{degree}_fig_tsne.png')
        mean_curve(clean, noisy, outdir/f'degree_{degree}_fig_mean.png')
        value_hist(clean, noisy, outdir/f'degree_{degree}_fig_hist.png')
        delta_hist(clean, noisy, outdir/f'degree_{degree}_fig_delta_hist.png')

        # ------------------- KL / JS diagnostics ---------------------------
        kl_cn, kl_nc, js_vals = kl_js_divergence(clean.ravel(), noisy.ravel())
        print(f'degree {degree}:  KL(values  clean‖noisy)={kl_cn:.4f}, '
              f'KL(noisy‖clean)={kl_nc:.4f},  JS={js_vals:.4f} nats')

        dc, dn = np.abs(clean[:,1:] - clean[:,:-1]).ravel(), np.abs(noisy[:,1:] - noisy[:,:-1]).ravel()
        #dc, dn = dc[dc>0.1], dn[dn>0.1]
        if dc.size and dn.size:
            kl_dc_dn, kl_dn_dc, js_delta = kl_js_divergence(dc, dn)
            print(f'             Δ-mag   KL={kl_dc_dn:.4f}/{kl_dn_dc:.4f}, JS={js_delta:.4f}')
        else:
            print('             Δ-mag   skipped (all ≤0.1)')

    print(f'[DONE] figures in {outdir}')

if __name__ == '__main__':
    main()
