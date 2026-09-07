#!/usr/bin/env python3
"""
==============================================================================
BTS Monocular Depth Estimation: Automated Checkpoint Downloader
==============================================================================
Allows 1-click downloading of pretrained / reproduced 50-Epoch weights:
- KITTI DenseNet-161 (Peak Step 242,500 - AbsRel 0.05748 - Beat Paper 9/9)
- NYU Depth V2 DenseNet-161 (Peak Step 271,000 - AbsRel 0.10967)

Usage:
    python download_weights.py --dataset kitti
    python download_weights.py --dataset nyu
    python download_weights.py --all
"""

import os
import sys
import argparse
import urllib.request
import hashlib

CHECKPOINT_REGISTRY = {
    "kitti": {
        "filename": "bts_kitti_densenet161_step242500_best_absrel_0.05748.pth",
        "description": "KITTI Eigen Split 50-Epoch Peak Checkpoint (Step 242,500, AbsRel 0.05748)",
        "url": "https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation/releases/download/v1.0.0-reproduction/bts_kitti_densenet161_step242500_best_absrel_0.05748.pth",
        "filesize_mb": 537.6,
        "sha256": None  # Populated on release
    },
    "nyu": {
        "filename": "bts_nyuv2_densenet161_step271000_best_absrel_0.10967.pth",
        "description": "NYU Depth V2 50-Epoch Peak Checkpoint (Step 271,000, AbsRel 0.10967)",
        "url": "https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation/releases/download/v1.0.0-reproduction/bts_nyuv2_densenet161_step271000_best_absrel_0.10967.pth",
        "filesize_mb": 537.6,
        "sha256": None
    }
}

def download_file(url: str, dest_path: str, expected_size_mb: float):
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path} (~{expected_size_mb:.1f} MB)")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r[{percent:5.1f}%] {downloaded / 1024 / 1024:6.1f} MB / {total_size / 1024 / 1024:6.1f} MB")
        else:
            sys.stdout.write(f"\rDownloaded {downloaded / 1024 / 1024:6.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print("\nDownload complete successfully!")
    except Exception as e:
        print(f"\n[Warning] Direct download failed ({e}).")
        print(f"Please visit the official release page to download manually:")
        print(f"👉 https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation/releases/tag/v1.0.0-reproduction")
        print(f"Place the downloaded .pth file into: {os.path.abspath(dest_path)}")

def main():
    parser = argparse.ArgumentParser(description="Download BTS Pretrained / Reproduced Checkpoints")
    parser.add_argument("--dataset", choices=["kitti", "nyu", "all"], default="kitti",
                        help="Dataset checkpoint to download (kitti, nyu, or all)")
    parser.add_argument("--dest_dir", type=str, default="checkpoints",
                        help="Target directory for saving weights (default: checkpoints/)")
    args = parser.parse_args()

    targets = ["kitti", "nyu"] if args.dataset == "all" else [args.dataset]
    
    for key in targets:
        meta = CHECKPOINT_REGISTRY[key]
        dest = os.path.join(args.dest_dir, meta["filename"])
        print(f"\n=== Target: {meta['description']} ===")
        if os.path.exists(dest):
            sz = os.path.getsize(dest) / 1024 / 1024
            print(f"Checkpoint already exists at {dest} ({sz:.1f} MB). Skipping download.")
        else:
            download_file(meta["url"], dest, meta["filesize_mb"])

if __name__ == "__main__":
    main()
