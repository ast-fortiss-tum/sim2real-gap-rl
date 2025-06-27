#!/usr/bin/env python3
"""
visualize_pure_dynamics.py ────────────────────────────────────────────────
Focus *only* on **dynamical shift**: how the environment’s transition kernel
P(o_{t+1}|o_t,a_t) differs between a source and a target domain after the
same action sequence.  Static observational differences (colour balance,
fixed overlays, camera response) are suppressed by comparing *changes*
(rather than absolute frames) in a feature space.

Diagnostics produced (all written to --out):
  • fig_dyn_umap.png      – UMAP scatter of ResNet-18 Δ-features
  • fig_delta_hist.png    – Histograms of Δ-feature L2 norms (source vs target)
  • fig_flow_hist.png*    – Histograms of Farnebäck optical-flow magnitudes
                              (*only if opencv-python is installed*)

Usage
-----
    python3 graphic_dynamic_shift.py \
        --src  /home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_sim \
        --tgt  /home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_real \
        --out  ./dyn_figs \
        --n    2000           # transitions per domain

Dependencies
------------
    numpy   matplotlib   scikit-learn   umap-learn   tqdm
    torch + torchvision                 (for ResNet-18)
    opencv-python       (optional; for optical flow plot)
"""
from pathlib import Path
import argparse, json, random, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import umap
from scipy.stats import entropy   # KL / JS helper

# ───── Torch (CNN features) ─────
try:
    import torch, torchvision.transforms as T
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_OK = True
except Exception as e:
    warnings.warn("CNN-based feature extractor unavailable – aborting.")
    TORCH_OK = False

# ───── OpenCV (optical-flow) ─────
try:
    import cv2
    CV_OK = True
except Exception:
    CV_OK = False

# --------------------------- argument parsing -----------------------------

def get_args():
    p = argparse.ArgumentParser(description="Visualise pure dynamical shift")
    p.add_argument("--src", required=True, help="root folder of SOURCE dataset")
    p.add_argument("--tgt", required=True, help="root folder of TARGET dataset")
    p.add_argument("--out", default="./dyn_figs", help="output folder for figs")
    p.add_argument("--n", type=int, default=2000, help="# transitions per domain")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------------------- data loading ---------------------------------

def list_files(folder: Path):
    return sorted(folder.glob("E1*_*.png"))


def load_transitions(root: Path, n_max: int):
    obs_dir  = root / "observations"
    nxt_dir  = root / "resulting_observations"
    files    = list_files(obs_dir)[:n_max]
    data = []  # (o_t, o_tp1)
    for f in tqdm(files, desc=f"{root.name} imgs"):
        nxt = nxt_dir / f.name
        if not nxt.exists():
            continue
        try:
            o_t   = plt.imread(f)
            o_tp1 = plt.imread(nxt)
        except Exception as e:
            warnings.warn(f"read error {f}: {e}")
            continue
        data.append((o_t, o_tp1))
        if len(data) == n_max:
            break
    if not data:
        sys.exit(f"No transitions loaded from {root}")
    return data

# --------------------------- feature extractor ----------------------------

def build_featuriser(device="cpu"):
    res = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = torch.nn.Sequential(*(list(res.children())[:-1]))
    tx = T.Compose([T.ToPILImage(), T.Resize(224), T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    def feat(img: np.ndarray):
        with torch.no_grad():
            x = tx(img).unsqueeze(0).to(device)
            f = model(x).flatten().cpu().numpy()
        return f
    return feat

# --------------------------- optical flow util ----------------------------

def flow_mag(img0: np.ndarray, img1: np.ndarray):
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    flow  = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag   = np.linalg.norm(flow, axis=2)
    return mag.mean()

# --------------------------- plotting helpers ----------------------------

def plot_feature_delta_umap(ΔS, ΔT, out_path):
    X = np.vstack([ΔS, ΔT])
    scaler = StandardScaler().fit(X)
    X_std  = scaler.transform(X)
    mapper = umap.UMAP(n_components=2, random_state=0)
    XY = mapper.fit_transform(X_std)

    plt.figure(figsize=(6,5))
    plt.scatter(XY[:len(ΔS),0], XY[:len(ΔS),1], marker='o', alpha=.5, label='Source')
    plt.scatter(XY[len(ΔS):,0], XY[len(ΔS):,1], marker='x', alpha=.5, label='Target')
    plt.title('Δ-ResNet18-feature UMAP'); plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2'); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_delta_norm_hist2(ΔS, ΔT, out_path):
    l2_S = np.linalg.norm(ΔS, axis=1)
    l2_T = np.linalg.norm(ΔT, axis=1)
    bins = np.linspace(0.1, l2_S.max().max() if l2_S.size else 1, 50)
    plt.figure(figsize=(6,4))
    plt.hist(l2_S, bins=bins, alpha=.6, label='Source')
    plt.hist(l2_T, bins=bins, alpha=.6, label='Target')
    plt.xlabel('||Δfeature||₂'); plt.ylabel('#transitions'); plt.title('Δ-feature magnitude distribution')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_delta_norm_hist(ΔS, ΔT, out_path):
    # 1. L2 norms, keep only meaningful steps (>0.1)
    l2_S = np.linalg.norm(ΔS, axis=1);  #l2_S = l2_S[l2_S > 0.1]
    l2_T = np.linalg.norm(ΔT, axis=1);  #l2_T = l2_T[l2_T > 0.1]

    if not (l2_S.size and l2_T.size):
        warnings.warn("Δ-hist skipped (all ≤ 0.1)");  return

    # 2. Shared binning for both PDFs
    lo, hi = min(l2_S.min(), l2_T.min()), max(l2_S.max(), l2_T.max())
    bins   = np.linspace(lo, hi, 50)

    p, _ = np.histogram(l2_S, bins=bins, density=True)
    q, _ = np.histogram(l2_T, bins=bins, density=True)

    # 3. KL & JS  (add tiny ε to avoid log(0))
    eps   = 1e-10
    p, q  = p + eps, q + eps
    kl_pq = entropy(p, q)                  # KL(source‖target)
    kl_qp = entropy(q, p)                  # KL(target‖source)
    js    = 0.5 * (kl_pq + kl_qp)          # Jensen–Shannon (nats)

    print(f"Δ-norms  KL(src‖tgt)={kl_pq:.4f}, KL(tgt‖src)={kl_qp:.4f}, JS={js:.4f} nats")

    # 4. Plot
    plt.figure(figsize=(6,4))
    plt.hist(l2_S, bins=bins, alpha=.6, label='Source')
    plt.hist(l2_T, bins=bins, alpha=.6, label='Target')
    plt.xlabel('||Δfeature||₂'); plt.ylabel('#transitions')
    plt.title('Δ-feature magnitude distribution (>0.1)')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_flow_hist(flow_S, flow_T, out_path):
    bins = np.linspace(0, max(max(flow_S), max(flow_T)), 40)
    plt.figure(figsize=(6,4))
    plt.hist(flow_S, bins=bins, alpha=.6, label='Source')
    plt.hist(flow_T, bins=bins, alpha=.6, label='Target')
    plt.xlabel('mean optical-flow |v|'); plt.ylabel('#transitions')
    plt.title('Motion-field magnitude distribution'); plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# --------------------------- main routine ----------------------------------

def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed)

    src_root = Path(args.src).expanduser().resolve()
    tgt_root = Path(args.tgt).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    trans_S = load_transitions(src_root, args.n)
    trans_T = load_transitions(tgt_root, args.n)

    if not TORCH_OK:
        sys.exit("PyTorch required for dynamics-only visualisation (feature deltas).")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat   = build_featuriser(device)

    def delta_feat(pair):
        f0, f1 = feat(pair[0]), feat(pair[1])
        return f1 - f0

    ΔS = np.vstack([delta_feat(p) for p in trans_S])
    ΔT = np.vstack([delta_feat(p) for p in trans_T])

    print("[INFO] plotting Δ-feature UMAP …")
    plot_feature_delta_umap(ΔS, ΔT, out_dir / 'fig_dyn_umap.png')

    print("[INFO] plotting magnitude histogram …")
    plot_delta_norm_hist(ΔS, ΔT, out_dir / 'fig_delta_hist.png')

    if CV_OK:
        print("[INFO] computing optical flow magnitudes …")
        flow_S = [flow_mag(p[0], p[1]) for p in tqdm(trans_S[:1000], desc='flow Src')]
        flow_T = [flow_mag(p[0], p[1]) for p in tqdm(trans_T[:1000], desc='flow Tgt')]
        plot_flow_hist(flow_S, flow_T, out_dir / 'fig_flow_hist.png')
    else:
        warnings.warn("OpenCV not found – skipping optical-flow plot")

    print(f"[DONE] figures written to {out_dir}")

if __name__ == "__main__":
    main()
