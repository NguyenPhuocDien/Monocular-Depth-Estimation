"""Validate the real step-500 production checkpoint without continuing the full run."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
REPORT = WORKING / "step500_resume_preflight_report.json"
EXPECTED = {
    "checkpoint": "35a36b5f7b4d972b34b6ef850237174f3c1b8d3d65fdc96f4e5918cf6f3ae579",
    "bts_main": "c0b4703fa85ca1edb2ddb9103a5fa3fb34c906de17004bd9d42d0ece3bf0599d",
    "retention_helper": "9adc2f5f5eb6d49f870aadbd7b2fa9a7290dcdc95bfd36f4d516adc331262c59",
    "config": "d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81",
    "weight": "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(name: str, expected_hash: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if sha256(path) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def run() -> dict:
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Preflight requires exactly one effective Tesla T4")

    bts_main_path = find_one("bts_main.py", EXPECTED["bts_main"])
    runtime = bts_main_path.parent.parent
    helper = runtime / "pytorch" / "checkpoint_retention.py"
    config = runtime / "config" / "arguments_train_eigen_kaggle.txt"
    if sha256(helper) != EXPECTED["retention_helper"] or sha256(config) != EXPECTED["config"]:
        raise RuntimeError("Runtime hash mismatch")

    weight = find_one("densenet161-8d451a50.pth", EXPECTED["weight"])
    torch_home = Path("/tmp/bts_step500_resume_preflight")
    cached = torch_home / "hub" / "checkpoints" / weight.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight, cached)
    os.environ["TORCH_HOME"] = str(torch_home)
    sys.path[:0] = [str(runtime / "vendor"), str(runtime / "pytorch")]
    sys.argv = ["bts_main.py", str(config)]
    bts_main = importlib.import_module("bts_main")
    retention = importlib.import_module("checkpoint_retention")

    checkpoint_path = find_one("latest.pth", EXPECTED["checkpoint"])
    before_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    retention.validate_checkpoint(checkpoint, 500)

    args = copy.copy(bts_main.args)
    data_root = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
    args.data_path = str(data_root) + "/"
    args.gt_path = str(data_root / "data_depth_annotated") + "/"
    args.filenames_file = str(runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt")
    args.batch_size = 1
    args.num_threads = 1
    args.distributed = False
    args.multiprocessing_distributed = False

    loader = bts_main.BtsDataLoader(args, "train")
    base_model = bts_main.BtsModel(args).train()
    base_model.decoder.apply(bts_main.weights_init_xavier)
    bts_main.set_misc(base_model)
    trainable_tensors = sum(int(parameter.requires_grad) for parameter in base_model.parameters())
    frozen_tensors = sum(int(not parameter.requires_grad) for parameter in base_model.parameters())
    model = torch.nn.DataParallel(base_model).cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.encoder.parameters(), "weight_decay": args.weight_decay},
            {"params": model.module.decoder.parameters(), "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        eps=args.adam_eps,
    )
    load = model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    state_before = len(optimizer.state)
    group_counts = [len(group["params"]) for group in optimizer.param_groups]

    tracked_name = next(
        name for name, value in model.state_dict().items()
        if name.startswith("module.decoder.") and value.is_floating_point()
    )
    tracked_before = model.state_dict()[tracked_name].detach().clone()
    batch = next(iter(loader.data))
    image = batch["image"].cuda()
    depth = batch["depth"].cuda()
    focal = batch["focal"].cuda()
    optimizer.zero_grad()
    estimate = model(image, focal)[-1]
    mask = depth > 1.0
    loss = bts_main.silog_loss(variance_focus=args.variance_focus)(
        estimate, depth, mask.to(torch.bool)
    )
    loss.backward()
    optimizer.step()
    delta = float((model.state_dict()[tracked_name] - tracked_before).abs().max().cpu())
    state_after = len(optimizer.state)

    checks = {
        "checkpoint_hash_stable": sha256(checkpoint_path) == before_hash,
        "six_key_schema": sorted(checkpoint) == sorted(retention.REQUIRED_KEYS),
        "global_step_500": int(checkpoint["global_step"]) == 500,
        "resume_next_step_501": int(checkpoint["global_step"]) + 1 == 501,
        "strict_model_load": not load.missing_keys and not load.unexpected_keys,
        "optimizer_group_parameters_550": sum(group_counts) == 550,
        "production_state_entries_227_before": state_before == 227,
        "production_state_entries_227_after": state_after == 227,
        "set_misc_trainable_tensors_227": trainable_tensors == 227,
        "set_misc_frozen_tensors_323": frozen_tensors == 323,
        "best_metric_slots": [
            len(checkpoint["best_eval_measures_higher_better"]),
            len(checkpoint["best_eval_measures_lower_better"]),
            len(checkpoint["best_eval_steps"]),
        ] == [3, 6, 9],
        "finite_loss": bool(np.isfinite(float(loss.detach().cpu()))),
        "parameter_updated": delta > 0,
    }
    result = {
        "schema_version": 1,
        "gate": "STEP500_PRODUCTION_RESUME_PREFLIGHT",
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_boundary": "isolated one-step resume preflight; not full-run continuation",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu_names": [name.strip() for name in physical if name.strip()],
            "effective_gpu_name": torch.cuda.get_device_name(0),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": before_hash,
            "bytes": checkpoint_path.stat().st_size,
            "global_step": 500,
            "optimizer_state_entries_before": state_before,
            "optimizer_state_entries_after": state_after,
            "optimizer_group_param_counts": group_counts,
        },
        "set_misc": {
            "trainable_parameter_tensors": trainable_tensors,
            "frozen_parameter_tensors": frozen_tensors,
        },
        "loss": float(loss.detach().cpu()),
        "tracked_parameter": tracked_name,
        "tracked_parameter_delta": delta,
        "checks": checks,
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    try:
        result = run()
    except Exception as exc:
        result = {
            "gate": "STEP500_PRODUCTION_RESUME_PREFLIGHT",
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)
