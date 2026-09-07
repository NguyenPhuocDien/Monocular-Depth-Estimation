"""K1: isolated BTS KITTI dataloader gate on Kaggle."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/k1_dataloader_report.json")
DATA_ROOT = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
CONFIG_SHA256 = "9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7"
TRAIN_SHA256 = "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094"
TEST_SHA256 = "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_runtime_root() -> Path:
    matches = []
    for path in INPUT.rglob("bts_dataloader.py"):
        root = path.parent.parent
        required = [
            root / "config" / "arguments_train_eigen_kaggle.txt",
            root / "train_test_inputs" / "eigen_train_files_with_gt.txt",
            root / "train_test_inputs" / "eigen_test_files_with_gt.txt",
            root / "provenance_manifest.json",
        ]
        if all(item.is_file() for item in required):
            matches.append(root)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one BTS runtime root, found {unique}")
    return unique[0]


def parse_config(path: Path) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parts = line.split(maxsplit=1)
            result[parts[0].removeprefix("--")] = parts[1] if len(parts) == 2 else True
    return result


def run() -> dict:
    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)

    runtime = find_runtime_root()
    config_path = runtime / "config" / "arguments_train_eigen_kaggle.txt"
    train_list = runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt"
    test_list = runtime / "train_test_inputs" / "eigen_test_files_with_gt.txt"
    config = parse_config(config_path)
    canonical_protocol = {
        "dataset": "kitti",
        "encoder": "densenet161_bts",
        "batch_size": "4",
        "num_epochs": "50",
        "learning_rate": "1e-4",
        "input_height": "352",
        "input_width": "704",
        "max_depth": "80",
        "do_kb_crop": True,
        "do_random_rotate": True,
        "degree": "1.0",
        "garg_crop": True,
    }

    sys.path.insert(0, str(runtime / "pytorch"))
    from bts_dataloader import BtsDataLoader

    args = SimpleNamespace(
        dataset="kitti",
        data_path=str(DATA_ROOT) + "/",
        gt_path=str(DATA_ROOT / "data_depth_annotated") + "/",
        filenames_file=str(train_list),
        filenames_file_eval=str(test_list),
        data_path_eval=str(DATA_ROOT) + "/",
        gt_path_eval=str(DATA_ROOT / "data_depth_annotated") + "/",
        input_height=352,
        input_width=704,
        batch_size=1,
        num_threads=1,
        distributed=False,
        use_right=False,
        do_kb_crop=True,
        do_random_rotate=True,
        degree=1.0,
    )
    loader = BtsDataLoader(args, "train")
    batch = next(iter(loader.data))
    image = batch["image"]
    depth = batch["depth"]
    focal = batch["focal"]

    checks = {
        "data_root_exists": DATA_ROOT.is_dir(),
        "runtime_config_sha256": sha256(config_path) == CONFIG_SHA256,
        "train_list_sha256": sha256(train_list) == TRAIN_SHA256,
        "test_list_sha256": sha256(test_list) == TEST_SHA256,
        "canonical_protocol_unchanged": all(config.get(k) == v for k, v in canonical_protocol.items()),
        "dataset_rows": len(loader.training_samples) == 23158,
        "batch_keys": set(batch) == {"image", "depth", "focal"},
        "image_shape": list(image.shape) == [1, 3, 352, 704],
        "depth_shape": list(depth.shape) == [1, 1, 352, 704],
        "focal_shape": list(focal.shape) == [1],
        "image_float32": image.dtype == torch.float32,
        "depth_float32": depth.dtype == torch.float32,
        "finite_image": bool(torch.isfinite(image).all()),
        "finite_depth": bool(torch.isfinite(depth).all()),
        "finite_focal": bool(torch.isfinite(focal).all()),
        "positive_depth_present": bool((depth > 0).any()),
        "positive_focal": bool((focal > 0).all()),
    }
    return {
        "gate": "K1",
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observed": {
            "runtime_root": str(runtime),
            "dataset_rows": len(loader.training_samples),
            "gate_batch_size": 1,
            "canonical_batch_size": int(config["batch_size"]),
            "image_shape": list(image.shape),
            "depth_shape": list(depth.shape),
            "focal_shape": list(focal.shape),
            "image_dtype": str(image.dtype),
            "depth_dtype": str(depth.dtype),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "focal": [float(value) for value in focal],
        },
        "failures": [name for name, passed in checks.items() if not passed],
        "authorization_boundary": {
            "model_constructed": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "checkpoint_written": False,
            "training_executed": False,
            "dataset_written": False,
        },
    }


if __name__ == "__main__":
    try:
        report = run()
    except Exception as exc:
        report = {
            "gate": "K1",
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {
                "model_constructed": False,
                "backward_executed": False,
                "optimizer_step_executed": False,
                "checkpoint_written": False,
                "training_executed": False,
                "dataset_written": False,
            },
        }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)
