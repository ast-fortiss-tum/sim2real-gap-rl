#!/usr/bin/env python3
"""
visualize_dynamical_shift.py ────────────────────────────────────────────────
A one‑stop script that loads *source* and *target* roll‑out datasets stored
in the folder structure you described and produces SIX visual diagnostics of
possible **dynamical shifts**:

  1. 2‑D PCA scatter of flattened next‑state RGB frames
  2. 2‑D t‑SNE scatter of flattened next‑state RGB frames
  3. 2‑D UMAP scatter of ResNet‑18 feature vectors (CNN + non‑linear embed)
  4. Mean‑frame visualisation + absolute difference heat‑map
  5. Channel‑wise intensity histograms (R/G/B)
  6. Violin plot of per‑transition Structural–SIMilarity (SSIM) scores

Each plot is written to <output_dir>/fig_<name>.png.

──────────────────────────────────────────────────────────────────────────────
Usage
-----
    python visualize_dynamical_shift.py \
        --src  /path/to/source/root  \
        --tgt  /path/to/target/root  \
        --out  ./dc_figs  \
        --n    2000         # transitions per domain (default 2000)
        --down 4            # spatial down‑sample factor before PCA/t‑SNE

Dependencies (PyPI)
-------------------
    pip install numpy matplotlib scikit‑learn umap‑learn pillow imageio
                torch torchvision seaborn scikit‑image tqdm

The script autodetects CUDA if available for the CNN pass.
"""

import argparse, json, math, os, sys, random, warnings
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

# ───────────────────────────── optional (CNN) ──────────────────────────────
try:
    import torch, torchvision.transforms as T
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_OK = True
except ImportError:  # let the rest of the script run without CNN part
    warnings.warn("torch/torchvision not found – CNN‑based UMAP plot will be skipped")
    TORCH_OK = False

# ------------------------- command‑line arguments ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual diagnostics for dynamical shift between two datasets")
    p.add_argument("--src", required=True, help="root folder of SOURCE dataset")
    p.add_argument("--tgt", required=True, help="root folder of TARGET dataset")
    p.add_argument("--out", default="./dc_figs", help="output directory for images")
    p.add_argument("--n", type=int, default=2000, help="#transitions per domain")
    p.add_argument("--down", type=int, default=4, help="down‑sample factor (>=1)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args()

# ------------------------- data‑loading helpers ----------------------------

def list_episode_files(folder: Path, ext: str):
    """Return all files with given extension sorted by episode and timestep."""
    return sorted(folder.glob(f"E*_*.{ext}"))  # relies on lexicographic ordering


def load_transitions(root: Path, n_max: int = 2000):
    """Load up to n_max transitions from <root>/observations and resulting_observations.

    Returns: list of tuples (obs_t, obs_tp1)
    """
    obs_dir  = root / "observations"
    next_dir = root / "resulting_observations"
    if not obs_dir.exists() or not next_dir.exists():
        sys.exit(f"[ERR] expected folders 'observations' and 'resulting_observations' inside {root}")

    obs_files = list_episode_files(obs_dir, "png")[:n_max]
    transitions = []
    for obs_path in tqdm(obs_files, desc=f"Loading {root.name}"):
        nxt_path = next_dir / obs_path.name  # same stem, png extension
        if not nxt_path.exists():
            warnings.warn(f"missing next‑state image for {obs_path}")
            continue
        try:
            obs  = imageio.imread(obs_path)
            nxt  = imageio.imread(nxt_path)
        except Exception as e:
            warnings.warn(f"failed to read {obs_path}: {e}")
            continue
        transitions.append((obs, nxt))
        if len(transitions) == n_max:
            break
    if not transitions:
        sys.exit(f"[ERR] no transitions found in {root}")
    return transitions

# ----------------------- visualisation utilities ---------------------------

def downsample(img: np.ndarray, factor: int):
    return img if factor <= 1 else img[::factor, ::factor, :]


# 1 ────────────────────────── PCA scatter (2‑D) ────────────────────────────

def plot_pca(trans_S, trans_T, down_factor, out_path):
    flt = lambda arr: np.array([downsample(t[1], down_factor).ravel() for t in arr])
    X_S, X_T = flt(trans_S), flt(trans_T)
    n = min(len(X_S), len(X_T), 1000)  # speed cap
    idx_S = np.random.choice(len(X_S), n, replace=False)
    idx_T = np.random.choice(len(X_T), n, replace=False)
    X     = np.vstack([X_S[idx_S], X_T[idx_T]])
    pca   = PCA(n_components=2, random_state=0)
    XY    = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(XY[:n,0], XY[:n,1], marker='o', alpha=.5, label='Source')
    plt.scatter(XY[n:,0], XY[n:,1], marker='x', alpha=.5, label='Target')
    plt.xlabel('PC‑1'); plt.ylabel('PC‑2'); plt.legend(); plt.title('PCA of next‑state images')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 2 ────────────────────────── t‑SNE scatter (2‑D) ──────────────────────────

def plot_tsne(trans_S, trans_T, down_factor, out_path):
    flt = lambda arr: np.array([downsample(t[1], down_factor).ravel() for t in arr])
    X_S, X_T = flt(trans_S), flt(trans_T)
    n = min(len(X_S), len(X_T), 700)  # t‑SNE is O(N²)
    X = np.vstack([X_S[:n], X_T[:n]])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=0)
    XY   = tsne.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(XY[:n,0], XY[:n,1], marker='o', alpha=.5, label='Source')
    plt.scatter(XY[n:,0], XY[n:,1], marker='x', alpha=.5, label='Target')
    plt.xlabel('t‑SNE‑1'); plt.ylabel('t‑SNE‑2'); plt.title('t‑SNE of next‑state pixels'); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 3 ────────────── ResNet‑18 features + UMAP (2‑D embedding) ────────────────

def plot_umap_cnn(trans_S, trans_T, out_path):
    if not TORCH_OK:
        warnings.warn("torch not available; skipping CNN‑UMAP plot"); return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    resnet.eval(); cnn = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # 512‑D
    tfms = T.Compose([T.ToPILImage(), T.Resize(224), T.ToTensor(),
                      T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    def featurise(img):
        with torch.no_grad():
            x = tfms(img).unsqueeze(0).to(device)
            f = cnn(x).flatten().cpu().numpy()
        return f

    # restrict to first 1000 each (speed)
    F_S = np.vstack([featurise(t[1]) for t in trans_S[:1000]])
    F_T = np.vstack([featurise(t[1]) for t in trans_T[:1000]])
    X   = np.vstack([F_S, F_T])
    mapper = umap.UMAP(n_components=2, random_state=0)
    XY = mapper.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(XY[:len(F_S),0], XY[:len(F_S),1], marker='o', alpha=.5, label='Source')
    plt.scatter(XY[len(F_S):,0], XY[len(F_S):,1], marker='x', alpha=.5, label='Target')
    plt.xlabel('UMAP‑1'); plt.ylabel('UMAP‑2'); plt.title('ResNet‑18 features → UMAP'); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 4 ─────────────────────── Mean‑frame & delta heat‑map ─────────────────────

def plot_mean_delta(trans_S, trans_T, out_path):
    mean_S = np.mean([t[1].astype(np.float32) for t in trans_S], axis=0)
    mean_T = np.mean([t[1].astype(np.float32) for t in trans_T], axis=0)
    delta  = np.abs(mean_S - mean_T).mean(axis=2)  # greyscale diff

    fig, ax = plt.subplots(1,3, figsize=(10,3))
    ax[0].imshow(mean_S.astype(np.uint8)); ax[0].set_title('⟨img⟩ source')
    ax[1].imshow(mean_T.astype(np.uint8)); ax[1].set_title('⟨img⟩ target')
    im = ax[2].imshow(delta, cmap='inferno'); ax[2].set_title('|meanS − meanT|')
    for a in ax: a.axis('off')
    fig.colorbar(im, ax=ax[2], fraction=.046)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 5 ───────────────────── Channel‑wise intensity histograms ──────────────────

def plot_channel_hist(trans_S, trans_T, out_path):
    def channel_pixels(tr, c):
        return np.concatenate([img[:,:,c].ravel() for _, img in tr])

    colors = ['r','g','b']; names = ['R','G','B']
    fig, ax = plt.subplots(1,3, figsize=(12,3))
    bins   = np.linspace(0,255,65)
    for c in range(3):
        hS, _ = np.histogram(channel_pixels(trans_S,c), bins=bins, density=True)
        hT, _ = np.histogram(channel_pixels(trans_T,c), bins=bins, density=True)
        centres = 0.5*(bins[:-1]+bins[1:])
        ax[c].plot(centres, hS, color=colors[c], label='Source')
        ax[c].plot(centres, hT, color='k', linestyle='--', label='Target')
        ax[c].set_title(f'{names[c]}‑channel PDF'); ax[c].set_xlabel('intensity'); ax[c].set_ylabel('density')
        ax[c].legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# 6 ─────────────────── Violin plot of per‑transition SSIM ───────────────────

def plot_ssim_violin(trans_S, trans_T, out_path):
    m = min(len(trans_S), len(trans_T))
    scores = [ssim(trans_S[i][1], trans_T[i][1], channel_axis=2, data_range=255) for i in range(m)]
    sns.violinplot(data=scores, inner='box'); plt.ylabel('SSIM(source → target)')
    plt.title('Image‑wise SSIM distribution')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# ------------------------------  main()  -----------------------------------

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    src_root = Path(args.src).expanduser().resolve()
    tgt_root = Path(args.tgt).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    trans_S = load_transitions(src_root, args.n)
    trans_T = load_transitions(tgt_root, args.n)

    print("[INFO] generating plots …")
    plot_pca(         trans_S, trans_T, args.down, out_dir / 'fig_pca.png')
    plot_tsne(        trans_S, trans_T, args.down, out_dir / 'fig_tsne.png')
    plot_mean_delta(  trans_S, trans_T,           out_dir / 'fig_mean_delta.png')
    plot_channel_hist(trans_S, trans_T,           out_dir / 'fig_hist.png')
    plot_ssim_violin( trans_S, trans_T,           out_dir / 'fig_ssim.png')
    if TORCH_OK:
        plot_umap_cnn(trans_S, trans_T,           out_dir / 'fig_umap_cnn.png')

    print(f"[DONE] all figures saved to {out_dir}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()


"""
python3 graphic_generator.py \
       --src  /home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_sim \
       --tgt  /home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_real\
       --out  ./dc_figs \
       --n    2000        # how many transitions per domain
       --down 4           # spatial down-sample factor before PCA/t-SNE
"""