#!/usr/bin/env python3
"""
visualize_random_piles.py ────────────────────────────────────────────────
Visual quick‑look for a domain‑shift experiment with **CycleGAN**.

Columns (left → right):
1. Simulation (domain **A**) raw frames
2. CycleGAN **A→B** (sim → real style) – **size matched to original**
3. CycleGAN **B→A** (real → sim style) – **size matched to original**
4. Real (domain **B**) raw frames

Output files (inside `--out`):
* `pile_sim.png`      – 1×k strip of simulation frames
* `pile_ab.png`       – 1×k strip of A→B frames (128×128 → resized)
* `pile_ba.png`       – 1×k strip of B→A frames (128×128 → resized)
* `pile_real.png`     – 1×k strip of real frames
* `pile_quad.png`     – k×4 grid `[Sim | A→B | B→A | Real]`

Run example
-----------
```bash
python visualize_random_piles.py \
       --sim       /path/to/dataset_sim \
       --real      /path/to/dataset_real \
       --model_ab  /path/to/G_AB.pth \
       --model_ba  /path/to/G_BA.pth \
       --out       ./dc_figs \
       --k         5
```

Dependencies
------------
```bash
pip install numpy matplotlib imageio tqdm torch torchvision pillow
```
"""

import argparse, random, sys, warnings
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

# ──────────────────────────── CycleGAN GENERATOR ──────────────────────────

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
        # Down‑sampling
        for _ in range(2):
            layers += [nn.Conv2d(in_c, out_c, 4, 2, 1),
                       nn.InstanceNorm2d(out_c),
                       nn.ReLU(True)]
            in_c, out_c = out_c, out_c * 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(in_c))
        # Up‑sampling
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

# ───────────────────────────── Helpers ────────────────────────────────────

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

# ───────────────────────── Plotting utilities ─────────────────────────────

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
        for c, img in enumerate([sim[r], ab[r], real[r],ba[r]]):
            axes[r, c].imshow(img); axes[r, c].axis('off')
            if r == 0: axes[r, c].set_title(headers[c])
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# ─────────────────────────── CycleGAN helpers ─────────────────────────────

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

# ─────────────────────────────── main ────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate random image piles for sim/real plus CycleGAN transform")
    p.add_argument("--sim", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_sim", help="root folder of SIMULATION dataset")
    p.add_argument("--real", default="/home/cubos98/Desktop/MA/Shift_visuals/dataset_DC/dataset_real", help="root folder of REAL dataset")
    p.add_argument("--model_ab", default = "/home/cubos98/Desktop/MA/Shift_visuals/CycleGAN/cyclegan_model_G_AB.pth", help="Path to CycleGAN generator weights (sim→real)")
    p.add_argument("--model_ba", default = "/home/cubos98/Desktop/MA/Shift_visuals/CycleGAN/cyclegan_model_G_BA.pth", help="Path to CycleGAN generator weights (real→sim)")
    p.add_argument("--out", default="./dc_figs", help="output directory")
    p.add_argument("--k", type=int, default=5, help="number of frames per pile")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="cuda:0 or cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample frames
    sim_imgs  = sample_observations(Path(args.sim).expanduser(),  args.k, args.seed)
    real_imgs = sample_observations(Path(args.real).expanduser(), args.k, args.seed)

    # 2. Generators
    dev = torch.device(args.device)
    net_ab = load_generator(Path(args.model_ab), dev)
    net_ba = load_generator(Path(args.model_ba), dev)

    # 3. Inference (sizes kept)
    ab_imgs = apply_gen(net_ab, sim_imgs,  dev)  # Sim → Real style
    ba_imgs = apply_gen(net_ba, real_imgs, dev)  # Real → Sim style

    # 4. Save strips and quad
    print("[INFO] saving figures …")
    strip(sim_imgs,  "Simulation",           out_dir / "pile_sim.png")
    strip(ab_imgs,   "Sim→Real (A→B)",      out_dir / "pile_ab.png")
    strip(real_imgs, "Real",                 out_dir / "pile_real.png")
    strip(ba_imgs,   "Real→Sim (B→A)",      out_dir / "pile_ba.png")
    quad(sim_imgs, ab_imgs, ba_imgs, real_imgs, out_dir / "pile_quad.png")
    print(f"[DONE] figures saved to {out_dir}")


if __name__ == "__main__":
    main()


