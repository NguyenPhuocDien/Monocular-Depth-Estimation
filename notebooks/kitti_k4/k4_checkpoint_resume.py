"""K4: BTS KITTI checkpoint, fresh-process restart, and resume gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "k4_checkpoint_resume_report.json"
CHECKPOINT = WORKING / "k4_checkpoint_step_2.pth"
PHASE1_REPORT = WORKING / "k4_phase1_report.json"
PHASE2_REPORT = WORKING / "k4_phase2_report.json"
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


def install_torch_weight(source: Path, phase: str) -> Path:
    torch_home = Path(f"/tmp/bts_k4_torch_home_{phase}")
    target = torch_home / "hub" / "checkpoints" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    os.environ["TORCH_HOME"] = str(torch_home)
    return target


def initialize(phase: str):
    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    if not torch.cuda.is_available():
        raise RuntimeError("K4 requires a Kaggle GPU")
    torch.cuda.manual_seed_all(20260826)
    runtime = find_runtime_root()
    weight = find_weight()
    cached_weight = install_torch_weight(weight, phase)
    sys.path.insert(0, str(runtime / "pytorch"))
    from bts import BtsModel, silog_loss
    from bts_dataloader import BtsDataLoader

    args = SimpleNamespace(
        encoder="densenet161_bts", bts_size=512, max_depth=80.0,
        dataset="kitti", data_path=str(DATA_ROOT) + "/",
        gt_path=str(DATA_ROOT / "data_depth_annotated") + "/",
        filenames_file=str(runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt"),
        input_height=352, input_width=704, batch_size=1, num_threads=1,
        distributed=False, use_right=False, do_kb_crop=True,
        do_random_rotate=True, degree=1.0,
    )
    loader = BtsDataLoader(args, "train")
    device = torch.device("cuda:0")
    model = BtsModel(args).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "weight_decay": 1e-2},
            {"params": model.decoder.parameters(), "weight_decay": 0.0},
        ],
        lr=1e-4,
        eps=1e-3,
    )
    return runtime, weight, cached_weight, loader, device, model, optimizer, silog_loss(variance_focus=0.85)


def train_step(iterator, device, model, optimizer, criterion) -> tuple[float, int]:
    batch = next(iterator)
    image = batch["image"].to(device)
    depth_gt = batch["depth"].to(device)
    focal = batch["focal"].to(device)
    optimizer.zero_grad()
    depth_est = model(image, focal)[-1]
    mask = depth_gt > 1.0
    loss = criterion(depth_est, depth_gt, mask.to(torch.bool))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("Missing or non-finite gradients")
    optimizer.step()
    return float(loss.detach().cpu()), int(mask.sum())


def phase1() -> dict:
    runtime, weight, cached_weight, loader, device, model, optimizer, criterion = initialize("produce")
    iterator = iter(loader.data)
    losses = []
    valid_pixels = []
    global_step = 0
    for _ in range(2):
        loss, valid = train_step(iterator, device, model, optimizer, criterion)
        losses.append(loss)
        valid_pixels.append(valid)
        global_step += 1
    checkpoint = {
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT)
    report = {
        "phase": "produce", "status": "passed",
        "process_id": os.getpid(), "global_step": global_step,
        "losses": losses, "valid_depth_pixels": valid_pixels,
        "checkpoint_path": str(CHECKPOINT), "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_keys": sorted(checkpoint),
        "optimizer_state_entries": len(optimizer.state),
        "runtime_config_sha256": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
        "weight_source_sha256": sha256(weight), "weight_cache_sha256": sha256(cached_weight),
    }
    PHASE1_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def phase2() -> dict:
    runtime, weight, cached_weight, loader, device, model, optimizer, criterion = initialize("resume")
    checkpoint_hash_before = sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    loaded_model = model.state_dict()
    exact_model_load = all(
        torch.equal(loaded_model[name].detach().cpu(), value.detach().cpu())
        for name, value in checkpoint["model"].items()
    )
    tracked_name = next(name for name in loaded_model if name.startswith("decoder.") and loaded_model[name].is_floating_point())
    tracked_before = loaded_model[tracked_name].detach().clone()
    loaded_step = int(checkpoint["global_step"])
    loss, valid_pixels = train_step(iter(loader.data), device, model, optimizer, criterion)
    resumed_step = loaded_step + 1
    tracked_delta = float((model.state_dict()[tracked_name] - tracked_before).abs().max().cpu())
    report = {
        "phase": "resume", "status": "passed",
        "process_id": os.getpid(), "loaded_global_step": loaded_step,
        "resumed_global_step": resumed_step, "resume_loss": loss,
        "valid_depth_pixels": valid_pixels, "exact_model_load": exact_model_load,
        "optimizer_state_entries_after_load": len(optimizer.state),
        "tracked_parameter": tracked_name,
        "tracked_parameter_max_abs_delta_after_resume": tracked_delta,
        "checkpoint_sha256_before_resume": checkpoint_hash_before,
        "checkpoint_sha256_after_resume": sha256(CHECKPOINT),
        "runtime_config_sha256": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
        "weight_source_sha256": sha256(weight), "weight_cache_sha256": sha256(cached_weight),
    }
    PHASE2_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_parent() -> dict:
    commands = []
    for phase in ["produce", "resume"]:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--phase", phase],
            text=True, capture_output=True, check=False,
        )
        commands.append({
            "phase": phase, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:],
        })
        if completed.returncode != 0:
            break
    phase1_data = json.loads(PHASE1_REPORT.read_text(encoding="utf-8")) if PHASE1_REPORT.is_file() else {}
    phase2_data = json.loads(PHASE2_REPORT.read_text(encoding="utf-8")) if PHASE2_REPORT.is_file() else {}
    checks = {
        "two_fresh_processes_completed": len(commands) == 2 and all(item["returncode"] == 0 for item in commands),
        "distinct_process_ids": phase1_data.get("process_id") != phase2_data.get("process_id"),
        "runtime_config_sha256": phase1_data.get("runtime_config_sha256") == CONFIG_SHA256 == phase2_data.get("runtime_config_sha256"),
        "weight_sha256": phase1_data.get("weight_source_sha256") == WEIGHT_SHA256 == phase2_data.get("weight_source_sha256"),
        "cached_weight_sha256": phase1_data.get("weight_cache_sha256") == WEIGHT_SHA256 == phase2_data.get("weight_cache_sha256"),
        "checkpoint_exists": CHECKPOINT.is_file(),
        "checkpoint_schema": phase1_data.get("checkpoint_keys") == ["global_step", "model", "optimizer"],
        "checkpoint_nonempty": phase1_data.get("checkpoint_bytes", 0) > 0,
        "checkpoint_hash_stable": phase1_data.get("checkpoint_sha256") == phase2_data.get("checkpoint_sha256_before_resume") == phase2_data.get("checkpoint_sha256_after_resume"),
        "produce_step_is_2": phase1_data.get("global_step") == 2,
        "resume_loaded_step_is_2": phase2_data.get("loaded_global_step") == 2,
        "resume_step_is_3": phase2_data.get("resumed_global_step") == 3,
        "exact_model_load": phase2_data.get("exact_model_load") is True,
        "optimizer_state_restored": phase1_data.get("optimizer_state_entries") == 550 == phase2_data.get("optimizer_state_entries_after_load"),
        "finite_positive_losses": all(
            np.isfinite(value) and value > 0
            for value in phase1_data.get("losses", []) + [phase2_data.get("resume_loss", float("nan"))]
        ),
        "parameter_updated_after_resume": phase2_data.get("tracked_parameter_max_abs_delta_after_resume", 0) > 0,
    }
    return {
        "gate": "K4", "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "phase_commands": commands,
        "phase1": phase1_data, "phase2": phase2_data,
        "failures": [name for name, passed in checks.items() if not passed],
        "authorization_boundary": {
            "produce_optimizer_steps": 2, "resume_optimizer_steps": 1,
            "full_training_executed": False, "dataset_written": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["produce", "resume"])
    cli = parser.parse_args()
    try:
        if cli.phase == "produce":
            result = phase1()
        elif cli.phase == "resume":
            result = phase2()
        else:
            result = run_parent()
            OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        result = {
            "gate": "K4" if cli.phase is None else cli.phase,
            "status": "failed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "authorization_boundary": {"full_training_executed": False, "dataset_written": False},
        }
        if cli.phase is None:
            OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)
