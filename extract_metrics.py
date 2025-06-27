#!/usr/bin/env python3
"""
compare_dataset_images.py

Compare corresponding images in Dataset_DC/dataset_real/observations and
Dataset_DC/dataset_sim/observations that start with the prefix "E1".

For each common image (same base filename) the script computes the same rich
suite of metrics used in the original rosbag‑based workflow and stores:

  • An absolute‑difference heat‑map (PNG) for every pair
  • A single JSON file aggregating all metrics

Directory layout expected
=========================
Dataset_DC/
  ├─ dataset_real/observations/  (real camera frames)
  └─ dataset_sim/observations/   (simulated frames)

Both *observations* folders contain images such as:
    E1_frame0001.png, E1_frame0002.png, …
Only files whose base‑name begins with "E1" (case‑sensitive) are processed.

Configuration constants near the top let you tweak the dataset root,
output directory, image sampling stride, etc.

Run with:
    python3 compare_dataset_images.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Local metric utilities – make sure this file is discoverable in PYTHONPATH
import image_metrics_utils as imu  # noqa: E402  pylint: disable=wrong-import-position

# ---------- CONFIGURATION ---------------------------------------------------
# Root directory containing dataset_real/ and dataset_sim/ sub‑folders
DATASET_ROOT = Path("dataset_DC")
# Sub‑folder (under each dataset_*/) holding the images we want to compare
OBS_SUBDIR = "observations"
# Prefix that the filenames of interest start with
FILENAME_PREFIX = "E1"
# File extensions to consider (case‑insensitive)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
# Maximum number of image pairs to analyse; set to None to process all pairs
MAX_PAIRS: int | None = 10
# Output directory for JSON metrics and diff images
OUT_DIR = DATASET_ROOT / "compare_observations"
# ----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sample_indices(total: int, n: int | None) -> List[int]:
    """Return *n* indices equally spaced across ``range(total)``.

    If *n* is None or >= *total*, every index is returned.
    """
    if n is None or total <= n:
        return list(range(total))
    return list(np.linspace(0, total - 1, n, dtype=int))


def find_images(folder: Path) -> List[Path]:
    """Return a sorted list of all image files in *folder* whose name starts
    with *FILENAME_PREFIX* and whose extension is in *EXTENSIONS*.
    """
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and p.stem.startswith(FILENAME_PREFIX)
    )


def load_image(path: Path) -> np.ndarray | None:
    """Load *path* with OpenCV (in BGR order). Returns *None* if loading fails."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Failed to load image: {path}", file=sys.stderr)
    return img


def save_diff_image(img_a: np.ndarray, img_b: np.ndarray, path: Path) -> None:
    """Save a normalised absolute‑difference heatmap to *path*."""
    diff = cv2.absdiff(img_a, img_b)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(path), norm)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    real_dir = DATASET_ROOT / "dataset_real" / OBS_SUBDIR
    sim_dir = DATASET_ROOT / "dataset_sim" / OBS_SUBDIR

    if not real_dir.is_dir() or not sim_dir.is_dir():
        print(
            f"Error: expected directories:\n  {real_dir}\n  {sim_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_imgs_paths = find_images(real_dir)
    sim_imgs_paths = {p.name: p for p in find_images(sim_dir)}  # map for lookup

    if not real_imgs_paths:
        print(f"No images starting with '{FILENAME_PREFIX}' found in {real_dir}", file=sys.stderr)
        sys.exit(0)

    # Sub‑sample to the requested number of pairs, equally spaced
    sel_indices = sample_indices(len(real_imgs_paths), MAX_PAIRS)
    metrics: List[dict] = []

    for rel_idx in sel_indices:
        real_path = real_imgs_paths[rel_idx]
        sim_path = sim_imgs_paths.get(real_path.name)
        if sim_path is None:
            print(f"[SKIP] No matching sim image for {real_path.name}")
            continue

        real_img = load_image(real_path)
        sim_img = load_image(sim_path)
        if real_img is None or sim_img is None:
            continue

        # --- compute metrics (cast to Python float for JSON serialisation) --
        m = {
            "mse": float(imu.calculate_mse(sim_img, real_img)),
            "psnr": float(imu.calculate_psnr(sim_img, real_img)),
            "ssim": float(imu.calculate_ssim(sim_img, real_img)),
            "corr_coeff": float(imu.calculate_correlation_coefficient(sim_img, real_img)),
            "lbp_hist_sim": float(imu.calculate_lbp_histogram_similarity(sim_img, real_img)),
            "nmi": float(imu.calculate_normalized_mutual_info([sim_img], [real_img])[0]),
            "texture_sim": float(imu.calculate_texture_similarity([sim_img], [real_img])[0]),
            "wasserstein_dist": float(imu.calculate_wd([sim_img], [real_img])[0]),
            "kl_divergence": float(imu.calculate_kl_divergence([sim_img], [real_img])[0]),
            "perceptual_dist": float(imu.calculate_perceptual_distances([sim_img], [real_img])[0]),
            "hist_intersect": float(imu.calculate_histogram_intersection([sim_img], [real_img])[0]),
            "ifd": float(imu.calculate_ifd(sim_img)),
        }

        # Save per‑pair diff image
        diff_path = OUT_DIR / f"{real_path.stem}_sim_vs_real_diff.png"
        save_diff_image(sim_img, real_img, diff_path)

        metrics.append({"image": real_path.name, "sim_vs_real": m})
        print(f"Processed {real_path.name}")

    # Write summary JSON
    json_path = OUT_DIR / "observations_image_compare_metrics.json"
    with json_path.open("w") as jf:
        json.dump({"dataset": str(DATASET_ROOT), "metrics": metrics}, jf, indent=2)

    print(f"\nSaved metrics JSON → {json_path.relative_to(Path.cwd())}")
    print(f"Diff images stored in   → {OUT_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()