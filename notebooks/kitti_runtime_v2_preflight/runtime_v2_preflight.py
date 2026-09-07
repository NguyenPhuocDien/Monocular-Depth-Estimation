"""Final no-full-run Kaggle preflight for the BTS KITTI runtime v2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Kaggle currently allocates T4s in pairs. The frozen BTS topology is one T4,
# so isolate GPU 0 before importing torch or querying CUDA.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torchvision


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
REPORT = WORKING / "runtime_v2_preflight_report.json"
EXPERIMENT_LOG = WORKING / "experiment_log.json"
METRICS = WORKING / "metrics.jsonl"
ARTIFACTS = WORKING / "artifacts_manifest.json"
CHECKPOINT = WORKING / "runtime_v2_production_preflight.pth"
PHASE_PRODUCE = WORKING / "runtime_v2_phase_produce.json"
PHASE_RESUME = WORKING / "runtime_v2_phase_resume.json"
DATA_ROOT = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
EXPECTED = {
    "official_commit": "5e3406b35d1497b2e55d2dd600524d1f4efacaed",
    "retention_patch": "21a94527dae9bf7971bd334a083b2e75f2396d9159f293004a8376be824757c8",
    "bts_main": "c0b4703fa85ca1edb2ddb9103a5fa3fb34c906de17004bd9d42d0ece3bf0599d",
    "retention_helper": "9adc2f5f5eb6d49f870aadbd7b2fa9a7290dcdc95bfd36f4d516adc331262c59",
    "config": "d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81",
    "vendor_manifest": "0206c24df45e0bc11d5504c159b3486bf2c090a6a623cd8b0cd3cceff84a467d",
    "weight": "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_runtime() -> Path:
    matches = []
    for path in INPUT.rglob("bts_main.py"):
        root = path.parent.parent
        helper = root / "pytorch" / "checkpoint_retention.py"
        config = root / "config" / "arguments_train_eigen_kaggle.txt"
        if helper.is_file() and config.is_file() and sha256(path) == EXPECTED["bts_main"]:
            matches.append(root)
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact runtime v2 root, found {matches}")
    return matches[0]


def find_weight() -> Path:
    matches = [path for path in INPUT.rglob("densenet161-8d451a50.pth")]
    if len(matches) != 1 or sha256(matches[0]) != EXPECTED["weight"]:
        raise RuntimeError(f"Expected one exact DenseNet-161 weight, found {matches}")
    return matches[0]


def install_weight(source: Path, phase: str) -> Path:
    target = Path(f"/tmp/bts_runtime_v2_{phase}") / "hub" / "checkpoints" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    os.environ["TORCH_HOME"] = str(target.parents[2])
    return target


def environment() -> dict:
    physical_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    physical_gpu_names = [
        line.strip() for line in physical_query.stdout.splitlines() if line.strip()
    ]
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    values = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "physical_gpu_count": len(physical_gpu_names),
        "physical_gpu_names": physical_gpu_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kaggle_image": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }
    checks = {
        "python_3_12": values["python"].startswith("3.12."),
        "torch_2_10_cu128": values["torch"].startswith("2.10.0+cu128"),
        "torchvision_0_25_cu128": values["torchvision"].startswith("0.25.0+cu128"),
        "cuda_12_8": values["cuda"] == "12.8",
        "one_effective_t4": gpu_count == 1 and "T4" in gpu_names[0],
        "physical_allocation_is_t4": bool(physical_gpu_names)
        and all("T4" in name for name in physical_gpu_names),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Environment mismatch before optimizer steps: {values}, {checks}")
    return {"values": values, "checks": checks}


def import_runtime(runtime: Path):
    sys.path.insert(0, str(runtime / "vendor"))
    sys.path.insert(0, str(runtime / "pytorch"))
    import tensorboardX

    if tensorboardX.__version__ != "2.6.5":
        raise RuntimeError(f"Unexpected tensorboardX version: {tensorboardX.__version__}")
    sys.argv = ["bts_main.py", str(runtime / "config" / "arguments_train_eigen_kaggle.txt")]
    bts_main = importlib.import_module("bts_main")
    retention = importlib.import_module("checkpoint_retention")
    return bts_main, retention


def runtime_checks(runtime: Path) -> dict:
    manifest = json.loads((runtime / "provenance_manifest.json").read_text())
    hashes = {
        "bts_main": sha256(runtime / "pytorch" / "bts_main.py"),
        "retention_helper": sha256(runtime / "pytorch" / "checkpoint_retention.py"),
        "config": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
        "vendor_manifest": sha256(runtime / "vendor_manifest.json"),
    }
    checks = {
        "official_commit": manifest["official_commit"] == EXPECTED["official_commit"],
        "retention_patch": any(
            item["sha256"] == EXPECTED["retention_patch"] for item in manifest["patches"]
        ),
        "bts_main": hashes["bts_main"] == EXPECTED["bts_main"],
        "retention_helper": hashes["retention_helper"] == EXPECTED["retention_helper"],
        "config": hashes["config"] == EXPECTED["config"],
        "vendor_manifest": hashes["vendor_manifest"] == EXPECTED["vendor_manifest"],
        "historical_worktree_unused": manifest["historical_working_tree_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Runtime provenance mismatch: {checks}, {hashes}")
    return {"hashes": hashes, "checks": checks}


def geometry(runtime: Path) -> dict:
    bts_main, _ = import_runtime(runtime)
    args = copy.copy(bts_main.args)
    args.data_path = str(DATA_ROOT) + "/"
    args.gt_path = str(DATA_ROOT / "data_depth_annotated") + "/"
    args.filenames_file = str(runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt")
    args.batch_size = 4
    args.num_threads = 1
    args.distributed = False
    args.multiprocessing_distributed = False
    loader = bts_main.BtsDataLoader(args, "train")
    values = {
        "train_rows": len(loader.training_samples),
        "local_batch_size": args.batch_size,
        "gpu_count": torch.cuda.device_count(),
        "global_batch_size": args.batch_size * torch.cuda.device_count(),
        "steps_per_epoch": len(loader.data),
        "epochs": args.num_epochs,
        "total_steps": len(loader.data) * args.num_epochs,
    }
    expected = {
        "train_rows": 23158,
        "local_batch_size": 4,
        "gpu_count": 1,
        "global_batch_size": 4,
        "steps_per_epoch": 5790,
        "epochs": 50,
        "total_steps": 289500,
    }
    if values != expected:
        raise RuntimeError(f"Geometry mismatch before optimizer steps: {values}")
    return {"values": values, "expected": expected, "status": "passed"}


def initialize(phase: str):
    random.seed(20260827)
    np.random.seed(20260827)
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)
    runtime = find_runtime()
    weight = find_weight()
    cached = install_weight(weight, phase)
    bts_main, retention = import_runtime(runtime)
    args = copy.copy(bts_main.args)
    args.data_path = str(DATA_ROOT) + "/"
    args.gt_path = str(DATA_ROOT / "data_depth_annotated") + "/"
    args.filenames_file = str(runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt")
    args.batch_size = 1
    args.num_threads = 1
    args.distributed = False
    args.multiprocessing_distributed = False
    loader = bts_main.BtsDataLoader(args, "train")
    device = torch.device("cuda:0")
    model = bts_main.BtsModel(args).to(device).train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "weight_decay": args.weight_decay},
            {"params": model.decoder.parameters(), "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        eps=args.adam_eps,
    )
    criterion = bts_main.silog_loss(variance_focus=args.variance_focus)
    return runtime, weight, cached, retention, loader, device, model, optimizer, criterion


def train_step(iterator, device, model, optimizer, criterion) -> tuple[float, int]:
    batch = next(iterator)
    image = batch["image"].to(device)
    depth = batch["depth"].to(device)
    focal = batch["focal"].to(device)
    optimizer.zero_grad()
    estimate = model(image, focal)[-1]
    mask = depth > 1.0
    loss = criterion(estimate, depth, mask.to(torch.bool))
    loss.backward()
    optimizer.step()
    if not np.isfinite(float(loss.detach().cpu())):
        raise RuntimeError("Non-finite preflight loss")
    return float(loss.detach().cpu()), int(mask.sum())


def produce() -> dict:
    runtime, weight, cached, retention, loader, device, model, optimizer, criterion = initialize("produce")
    iterator = iter(loader.data)
    losses = []
    for completed_step in (0, 1):
        loss, _ = train_step(iterator, device, model, optimizer, criterion)
        losses.append(loss)
    checkpoint = retention.build_training_checkpoint(
        completed_step,
        model,
        optimizer,
        torch.zeros(3),
        torch.zeros(6) + 1e3,
        np.zeros(9, dtype=np.int32),
    )
    receipt = retention.save_verified_checkpoint(checkpoint, CHECKPOINT)
    result = {
        "phase": "produce",
        "status": "passed",
        "process_id": os.getpid(),
        "losses": losses,
        "checkpoint": receipt,
        "checkpoint_keys": sorted(checkpoint),
        "optimizer_state_entries": len(optimizer.state),
        "weight_sha256": sha256(weight),
        "cached_weight_sha256": sha256(cached),
        "runtime_config_sha256": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
    }
    PHASE_PRODUCE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def resume() -> dict:
    runtime, weight, cached, retention, loader, device, model, optimizer, criterion = initialize("resume")
    before_hash = sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    retention.validate_checkpoint(checkpoint, 1)
    load = model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    tracked_name = next(name for name in model.state_dict() if name.startswith("decoder.") and model.state_dict()[name].is_floating_point())
    tracked_before = model.state_dict()[tracked_name].detach().clone()
    loss, valid_pixels = train_step(iter(loader.data), device, model, optimizer, criterion)
    delta = float((model.state_dict()[tracked_name] - tracked_before).abs().max().cpu())
    result = {
        "phase": "resume",
        "status": "passed",
        "process_id": os.getpid(),
        "loaded_global_step": int(checkpoint["global_step"]),
        "resumed_completed_global_step": 2,
        "loss": loss,
        "valid_depth_pixels": valid_pixels,
        "strict_model_load": not load.missing_keys and not load.unexpected_keys,
        "optimizer_state_entries": len(optimizer.state),
        "best_metric_slots": [
            len(checkpoint["best_eval_measures_higher_better"]),
            len(checkpoint["best_eval_measures_lower_better"]),
            len(checkpoint["best_eval_steps"]),
        ],
        "tracked_parameter": tracked_name,
        "tracked_parameter_delta": delta,
        "checkpoint_sha256_before": before_hash,
        "checkpoint_sha256_after": sha256(CHECKPOINT),
        "weight_sha256": sha256(weight),
        "cached_weight_sha256": sha256(cached),
        "runtime_config_sha256": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
    }
    PHASE_RESUME.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parent() -> dict:
    started = datetime.now(timezone.utc).isoformat()
    env = environment()
    runtime = find_runtime()
    provenance = runtime_checks(runtime)
    geometry_result = geometry(runtime)
    commands = []
    for phase in ("produce", "resume"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--phase", phase],
            text=True,
            capture_output=True,
            check=False,
        )
        commands.append(
            {
                "phase": phase,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            break
    produced = json.loads(PHASE_PRODUCE.read_text()) if PHASE_PRODUCE.is_file() else {}
    resumed = json.loads(PHASE_RESUME.read_text()) if PHASE_RESUME.is_file() else {}
    checks = {
        "environment": all(env["checks"].values()),
        "runtime_provenance": all(provenance["checks"].values()),
        "dataset_mount": DATA_ROOT.is_dir(),
        "geometry": geometry_result["status"] == "passed",
        "fresh_processes": len(commands) == 2 and all(item["returncode"] == 0 for item in commands),
        "six_key_schema": produced.get("checkpoint_keys") == sorted([
            "global_step", "model", "optimizer",
            "best_eval_measures_higher_better",
            "best_eval_measures_lower_better", "best_eval_steps",
        ]),
        "optimizer_state": produced.get("optimizer_state_entries") == 550 == resumed.get("optimizer_state_entries"),
        "strict_model_load": resumed.get("strict_model_load") is True,
        "best_metric_state": resumed.get("best_metric_slots") == [3, 6, 9],
        "resume_semantics": produced.get("checkpoint", {}).get("global_step") == 1 and resumed.get("loaded_global_step") == 1 and resumed.get("resumed_completed_global_step") == 2,
        "checkpoint_hash_stable": produced.get("checkpoint", {}).get("sha256") == resumed.get("checkpoint_sha256_before") == resumed.get("checkpoint_sha256_after"),
        "parameter_updated_after_resume": resumed.get("tracked_parameter_delta", 0) > 0,
    }
    status = "passed" if all(checks.values()) else "failed"
    checkpoint_evidence = produced.get("checkpoint", {})
    if CHECKPOINT.is_file():
        CHECKPOINT.unlink()
    result = {
        "schema_version": 1,
        "gate": "KAGGLE_RUNTIME_V2_PREFLIGHT",
        "status": status,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "runtime": provenance,
        "geometry": geometry_result,
        "checks": checks,
        "phase_commands": commands,
        "produce": produced,
        "resume": resumed,
        "checkpoint_evidence_before_prune": checkpoint_evidence,
        "temporary_checkpoint_pruned": not CHECKPOINT.exists(),
        "authorization_boundary": {
            "full_training_started": False,
            "preflight_optimizer_steps": 3,
            "session_1_started": False,
        },
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    EXPERIMENT_LOG.write_text(json.dumps({
        "run_id": "bts_kitti_runtime_v2_preflight",
        "status": status,
        "started_utc": started,
        "completed_utc": result["completed_utc"],
        "hardware": env["values"],
        "full_training_started": False,
        "error": None if status == "passed" else "One or more preflight checks failed",
    }, indent=2) + "\n")
    METRICS.write_text(json.dumps({
        "stage": "runtime_v2_preflight",
        "metric": "checks_passed",
        "value": sum(checks.values()),
        "total": len(checks),
        "timestamp": result["completed_utc"],
    }) + "\n")
    ARTIFACTS.write_text(json.dumps({
        "run_id": "bts_kitti_runtime_v2_preflight",
        "artifacts": [REPORT.name, EXPERIMENT_LOG.name, METRICS.name, PHASE_PRODUCE.name, PHASE_RESUME.name],
        "temporary_checkpoint_retained": False,
    }, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("produce", "resume"))
    cli = parser.parse_args()
    try:
        if cli.phase == "produce":
            value = produce()
        elif cli.phase == "resume":
            value = resume()
        else:
            value = parent()
    except Exception as exc:
        value = {
            "gate": "KAGGLE_RUNTIME_V2_PREFLIGHT",
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {"full_training_started": False},
        }
        if cli.phase is None:
            REPORT.write_text(json.dumps(value, indent=2) + "\n")
            EXPERIMENT_LOG.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2), flush=True)
    if value["status"] != "passed":
        raise SystemExit(1)
