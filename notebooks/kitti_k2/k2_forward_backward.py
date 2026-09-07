"""K2: one BTS KITTI forward/loss/backward gate, without optimizer step."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/k2_forward_backward_report.json")
DATA_ROOT = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
CONFIG_SHA256 = "9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7"
WEIGHT_SHA256 = "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359"


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
        if (root / "pytorch" / "bts.py").is_file() and (root / "provenance_manifest.json").is_file():
            matches.append(root)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one BTS runtime root, found {unique}")
    return unique[0]


def find_weight() -> Path:
    matches = list(INPUT.rglob("densenet161-8d451a50.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DenseNet-161 weight, found {matches}")
    return matches[0]


def install_torch_weight(source: Path) -> Path:
    torch_home = Path("/kaggle/working/torch_home")
    target = torch_home / "hub" / "checkpoints" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    os.environ["TORCH_HOME"] = str(torch_home)
    return target


def run() -> dict:
    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    if not torch.cuda.is_available():
        raise RuntimeError("K2 requires a Kaggle GPU")
    torch.cuda.manual_seed_all(20260826)

    runtime = find_runtime_root()
    weight_source = find_weight()
    weight_cache = install_torch_weight(weight_source)
    config_path = runtime / "config" / "arguments_train_eigen_kaggle.txt"
    train_list = runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt"
    sys.path.insert(0, str(runtime / "pytorch"))
    from bts import BtsModel, silog_loss
    from bts_dataloader import BtsDataLoader

    args = SimpleNamespace(
        encoder="densenet161_bts",
        bts_size=512,
        max_depth=80.0,
        dataset="kitti",
        data_path=str(DATA_ROOT) + "/",
        gt_path=str(DATA_ROOT / "data_depth_annotated") + "/",
        filenames_file=str(train_list),
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
    device = torch.device("cuda:0")
    model = BtsModel(args).to(device)
    model.train()
    image = batch["image"].to(device)
    depth_gt = batch["depth"].to(device)
    focal = batch["focal"].to(device)
    outputs = model(image, focal)
    depth_est = outputs[-1]
    mask = depth_gt > 1.0
    loss = silog_loss(variance_focus=0.85)(depth_est, depth_gt, mask.to(torch.bool))
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "runtime_config_sha256": sha256(config_path) == CONFIG_SHA256,
        "weight_source_sha256": sha256(weight_source) == WEIGHT_SHA256,
        "weight_cache_sha256": sha256(weight_cache) == WEIGHT_SHA256,
        "dataset_rows": len(loader.training_samples) == 23158,
        "model_parameter_count": sum(p.numel() for p in model.parameters()) == 47000688,
        "five_outputs": len(outputs) == 5,
        "output_shapes": all(list(value.shape) == [1, 1, 352, 704] for value in outputs),
        "finite_outputs": all(bool(torch.isfinite(value).all()) for value in outputs),
        "valid_loss_mask": int(mask.sum()) > 0,
        "finite_positive_loss": bool(torch.isfinite(loss) and loss > 0),
        "gradients_present": len(gradients) > 0,
        "all_gradients_finite": all(bool(torch.isfinite(grad).all()) for grad in gradients),
        "decoder_gradient_present": any(
            parameter.grad is not None for name, parameter in model.named_parameters() if name.startswith("decoder.")
        ),
        "encoder_gradient_present": any(
            parameter.grad is not None for name, parameter in model.named_parameters() if name.startswith("encoder.")
        ),
    }
    return {
        "gate": "K2",
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observed": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "runtime_root": str(runtime),
            "weight_source": str(weight_source),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_tensor_count": len(trainable),
            "gradient_tensor_count": len(gradients),
            "valid_depth_pixels": int(mask.sum()),
            "loss": float(loss.detach().cpu()),
            "output_shapes": [list(value.shape) for value in outputs],
            "max_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "failures": [name for name, passed in checks.items() if not passed],
        "authorization_boundary": {
            "optimizer_constructed": False,
            "optimizer_step_executed": False,
            "checkpoint_written": False,
            "tiny_train_executed": False,
            "full_training_executed": False,
            "dataset_written": False,
        },
    }


if __name__ == "__main__":
    try:
        report = run()
    except Exception as exc:
        report = {
            "gate": "K2",
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {
                "optimizer_step_executed": False,
                "checkpoint_written": False,
                "tiny_train_executed": False,
                "full_training_executed": False,
                "dataset_written": False,
            },
        }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)
