#!/usr/bin/env python3
"""
compare_bag_images_enhanced_sampling.py

Enhanced comparison of camera, sim, and mixed images from rosbag with equally-spaced frame sampling:
  - Uses BAG_DIR with .bag files under camera/lanes
  - Extracts up to MAX_FRAMES_PER_TOPIC equally spaced frames from
    /camera, /sim/image, /mixed_image
  - Computes a rich set of metrics using image_metrics_utils
  - Saves JSON metrics and difference images to OUT_DIR
"""
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge
import image_metrics_utils as imu

# ---------- CONFIG ----------
BAG_DIR = Path("/media/ast/1868aaf7-73b4-4e22-9fd8-ff4f3eb47346/ASE/PERCEPTION_INPUTS/camera/lanes")
OUT_DIR = BAG_DIR.parent / "compare_lanes"
TOPICS = ["/camera", "/sim/image", "/mixed_image"]
MAX_FRAMES_PER_TOPIC = 10
# ----------------------------------

def sample_indices(total, n):
    """Return n indices equally spaced across [0, total)."""
    if total <= n:
        return list(range(total))
    # linspace from 0 to total-1, integer positions
    return list(np.linspace(0, total - 1, n, dtype=int))


def read_frames(bag_path, topic, max_frames):
    """
    Read up to max_frames equally spaced frames from topic in bag, return list of BGR images.
    """
    # first pass: count total messages
    with rosbag.Bag(str(bag_path), 'r') as bag:
        total = sum(1 for _ in bag.read_messages(topics=[topic]))
    if total == 0:
        return []
    # compute sample indices
    idxs = sample_indices(total, max_frames)
    sel = set(idxs)

    # second pass: extract only selected messages
    images = []
    bridge = CvBridge()
    with rosbag.Bag(str(bag_path), 'r') as bag:
        msg_idx = 0
        for _, msg, _ in bag.read_messages(topics=[topic]):
            if msg_idx in sel:
                try:
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except Exception as e:
                    print(f"Failed to decode {topic} in {bag_path.stem}: {e}")
                    break
                images.append(img)
                if len(images) >= max_frames:
                    break
            msg_idx += 1
    return images


def save_diff_image(imgA, imgB, path):
    """
    Save absolute-difference heatmap to path.
    """
    diff = cv2.absdiff(imgA, imgB)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(path), norm)


def main():
    if not BAG_DIR.is_dir():
        print(f"Error: BAG_DIR {BAG_DIR} not found", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bags = sorted(BAG_DIR.glob("*.bag"))
    if not bags:
        print(f"No .bag files in {BAG_DIR}", file=sys.stderr)
        sys.exit(0)

    for bag_file in bags:
        name = bag_file.stem
        print(f"\n--- Comparing images in {name}.bag ---")

        frames = { t: read_frames(bag_file, t, MAX_FRAMES_PER_TOPIC) for t in TOPICS }
        cam_list = frames.get('/camera', [])
        if not cam_list:
            print(f"No camera frames for {name}, skipping.")
            continue

        metrics = []
        for idx, cam_img in enumerate(cam_list):
            row = {'frame': idx}
            for key, topic in [('sim_vs_camera', '/sim/image'), ('mixed_vs_camera', '/mixed_image')]:
                imgs = frames.get(topic, [])
                if idx < len(imgs):
                    cmp_img = imgs[idx]
                    # compute core metrics and cast to native Python floats
                    m = {
                        'mse': float(imu.calculate_mse(cmp_img, cam_img)),
                        'psnr': float(imu.calculate_psnr(cmp_img, cam_img)),
                        'ssim': float(imu.calculate_ssim(cmp_img, cam_img))
                    }
                    # additional metrics
                    m.update({
                        'corr_coeff': float(imu.calculate_correlation_coefficient(cmp_img, cam_img)),
                        'lbp_hist_sim': float(imu.calculate_lbp_histogram_similarity(cmp_img, cam_img)),
                        'nmi': float(imu.calculate_normalized_mutual_info([cmp_img], [cam_img])[0]),
                        'texture_sim': float(imu.calculate_texture_similarity([cmp_img], [cam_img])[0]),
                        'wasserstein_dist': float(imu.calculate_wd([cmp_img], [cam_img])[0]),
                        'kl_divergence': float(imu.calculate_kl_divergence([cmp_img], [cam_img])[0]),
                        'perceptual_dist': float(imu.calculate_perceptual_distances([cmp_img], [cam_img])[0]),
                        'hist_intersect': float(imu.calculate_histogram_intersection([cmp_img], [cam_img])[0]),
                        'ifd': float(imu.calculate_ifd(cmp_img))
                    })
                    row[key] = m
                    # save diff image
                    diff_path = OUT_DIR / f"{name}_{key}_frame{idx}_diff.png"
                    save_diff_image(cmp_img, cam_img, diff_path)
                else:
                    row[key] = None
            metrics.append(row)

        # save JSON metrics
        json_path = OUT_DIR / f"{name}_image_compare_metrics.json"
        with open(json_path, 'w') as jf:
            json.dump({'bag': name, 'metrics': metrics}, jf, indent=2)
        print(f"Saved metrics: {json_path}")

if __name__ == '__main__':
    main()
