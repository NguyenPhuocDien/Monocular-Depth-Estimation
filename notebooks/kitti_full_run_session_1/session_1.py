"""Launch BTS KITTI full-run Session 1 under immutable Freeze v2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING_RUNTIME = WORKING / "bts_runtime"
HANDOFF = WORKING / "session_1_handoff"
STARTUP_RECEIPT = WORKING / "session_1_startup_audit.json"
START_RECEIPT = WORKING / "session_1_start_receipt.json"
BOUNDARY_RECEIPT = WORKING / "session_1_boundary_receipt.json"
SESSION_LOG = WORKING / "session_1_training.log"
EXPERIMENT_LOG = WORKING / "experiment_log.json"
METRICS = WORKING / "metrics.jsonl"
ARTIFACTS = WORKING / "artifacts_manifest.json"
RUN_ID = "bts_kitti_densenet161_full_run_01"
MODEL_NAME = "bts_kitti_densenet161_kaggle"
TARGET_RECOVERY_STEP = 500
EXPECTED = {
    "freeze_manifest": "c770cdeebfb0515347e868dbcd810be054bd235a49589aa5bb3184ac4990716a",
    "freeze_retention": "e7fd7174f3a8132c6787b837f7cb686dfad2c6210624bcc0b7a7c1d10630be86",
    "freeze_receipt": "f775dd007721bd5fb8fda5b6203de43d1e08dbfdadacbc592f18b98e8ba5c60a",
    "bts_main": "c0b4703fa85ca1edb2ddb9103a5fa3fb34c906de17004bd9d42d0ece3bf0599d",
    "retention_helper": "9adc2f5f5eb6d49f870aadbd7b2fa9a7290dcdc95bfd36f4d516adc331262c59",
    "config": "d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81",
    "vendor_manifest": "0206c24df45e0bc11d5504c159b3486bf2c090a6a623cd8b0cd3cceff84a467d",
    "weight": "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359",
    "train_list": "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094",
    "test_list": "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd",
}
REQUIRED_KEYS = {
    "global_step", "model", "optimizer",
    "best_eval_measures_higher_better",
    "best_eval_measures_lower_better", "best_eval_steps",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def find_one(name: str, expected_hash: str | None = None) -> Path:
    matches = list(INPUT.rglob(name))
    if expected_hash is not None:
        matches = [path for path in matches if sha256(path) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def verify_checkpoint(path: Path, expected_step: int) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_KEYS.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing production keys: {sorted(missing)}")
    if int(checkpoint["global_step"]) != expected_step:
        raise RuntimeError(
            f"Expected checkpoint step {expected_step}, got {checkpoint['global_step']}"
        )
    if [
        len(checkpoint["best_eval_measures_higher_better"]),
        len(checkpoint["best_eval_measures_lower_better"]),
        len(checkpoint["best_eval_steps"]),
    ] != [3, 6, 9]:
        raise RuntimeError("Checkpoint best-metric state has invalid dimensions")
    return {
        "path": str(path),
        "global_step": expected_step,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "schema": sorted(REQUIRED_KEYS),
        "optimizer_state_entries": len(checkpoint["optimizer"].get("state", {})),
    }


def startup_audit() -> dict:
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    effective = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal 0")
    if len(effective) != 1 or "T4" not in effective[0]:
        raise RuntimeError(f"Expected one effective T4, got {effective}")

    freeze_manifest = find_one("full_run_manifest.v2.json", EXPECTED["freeze_manifest"])
    freeze_retention = find_one("checkpoint_retention_plan.v2.json", EXPECTED["freeze_retention"])
    freeze_receipt = find_one("freeze_receipt.v2.json", EXPECTED["freeze_receipt"])
    receipt = json.loads(freeze_receipt.read_text())
    if receipt["gate_state"]["session_1"] != "authorized_not_started":
        raise RuntimeError("Freeze v2 does not authorize Session 1")
    if receipt["artifacts"]["full_run_manifest_v2"]["sha256"] != EXPECTED["freeze_manifest"]:
        raise RuntimeError("Freeze receipt manifest hash mismatch")
    if receipt["artifacts"]["checkpoint_retention_plan_v2"]["sha256"] != EXPECTED["freeze_retention"]:
        raise RuntimeError("Freeze receipt retention hash mismatch")

    bts_main = find_one("bts_main.py", EXPECTED["bts_main"])
    runtime = bts_main.parent.parent
    helper = runtime / "pytorch" / "checkpoint_retention.py"
    config = runtime / "config" / "arguments_train_eigen_kaggle.txt"
    vendor_manifest = runtime / "vendor_manifest.json"
    checks = {
        "bts_main": sha256(bts_main) == EXPECTED["bts_main"],
        "retention_helper": sha256(helper) == EXPECTED["retention_helper"],
        "config": sha256(config) == EXPECTED["config"],
        "vendor_manifest": sha256(vendor_manifest) == EXPECTED["vendor_manifest"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Runtime hash mismatch: {checks}")

    data_root = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
    if not data_root.is_dir():
        raise RuntimeError(f"KITTI mount missing: {data_root}")
    train_list = runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt"
    test_list = runtime / "train_test_inputs" / "eigen_test_files_with_gt.txt"
    train_rows = sum(1 for line in train_list.read_text().splitlines() if line.strip())
    test_rows = sum(1 for line in test_list.read_text().splitlines() if line.strip())
    geometry = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "local_batch_size": 4,
        "global_batch_size": 4,
        "steps_per_epoch": (train_rows + 3) // 4,
        "epochs": 50,
        "total_steps": ((train_rows + 3) // 4) * 50,
    }
    if geometry != {
        "train_rows": 23158, "test_rows": 697,
        "local_batch_size": 4, "global_batch_size": 4,
        "steps_per_epoch": 5790, "epochs": 50, "total_steps": 289500,
    }:
        raise RuntimeError(f"Frozen geometry mismatch: {geometry}")
    if sha256(train_list) != EXPECTED["train_list"] or sha256(test_list) != EXPECTED["test_list"]:
        raise RuntimeError("Official list hash mismatch")
    config_text = config.read_text()
    if "--checkpoint_path" in config_text or "--retrain" in config_text:
        raise RuntimeError("Session 1 must start without a checkpoint")

    if WORKING_RUNTIME.exists():
        raise RuntimeError("Working runtime destination unexpectedly exists")
    shutil.copytree(runtime, WORKING_RUNTIME)
    for relative, expected in (
        ("pytorch/bts_main.py", EXPECTED["bts_main"]),
        ("pytorch/checkpoint_retention.py", EXPECTED["retention_helper"]),
        ("config/arguments_train_eigen_kaggle.txt", EXPECTED["config"]),
    ):
        if sha256(WORKING_RUNTIME / relative) != expected:
            raise RuntimeError(f"Copied runtime hash mismatch: {relative}")

    weight = find_one("densenet161-8d451a50.pth", EXPECTED["weight"])
    torch_home = Path("/tmp/bts_session_1_torch_home")
    cached = torch_home / "hub" / "checkpoints" / weight.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight, cached)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(WORKING_RUNTIME / "vendor"),
            str(WORKING_RUNTIME / "pytorch"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "session": 1,
        "freeze": "v2",
        "created_utc": now(),
        "status": "passed_before_optimizer_step_0",
        "initial_checkpoint": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu_names": [name.strip() for name in physical if name.strip()],
            "effective_gpu_names": effective,
            "effective_gpu_count": len(effective),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "runtime_hashes": checks,
        "freeze_hashes": {
            "manifest": sha256(freeze_manifest),
            "retention": sha256(freeze_retention),
            "receipt": sha256(freeze_receipt),
        },
        "geometry": geometry,
    }
    write_json(STARTUP_RECEIPT, result)
    return result


def create_start_receipt() -> None:
    if START_RECEIPT.exists():
        return
    write_json(
        START_RECEIPT,
        {
            "run_id": RUN_ID,
            "session": 1,
            "freeze": "v2",
            "created_utc": now(),
            "initial_checkpoint": None,
            "first_completed_global_step": 0,
            "effective_gpu_count": 1,
            "global_batch_size": 4,
            "steps_per_epoch": 5790,
            "total_steps": 289500,
            "status": "running",
        },
    )


def launch() -> dict:
    startup = startup_audit()
    config = WORKING_RUNTIME / "config" / "arguments_train_eigen_kaggle.txt"
    command = [
        sys.executable,
        str(WORKING_RUNTIME / "pytorch" / "bts_main.py"),
        str(config),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
        cwd=str(WORKING_RUNTIME / "pytorch"),
        start_new_session=True,
    )
    recovery = WORKING / "models" / MODEL_NAME / "checkpoints" / "recovery" / "latest.pth"
    boundary_checkpoint = None
    with SESSION_LOG.open("w", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if re.search(r"\[0\]\[0/5790/0\]", line):
                create_start_receipt()
            if "Saved verified recovery checkpoint:" in line and recovery.is_file():
                candidate = verify_checkpoint(recovery, TARGET_RECOVERY_STEP)
                if candidate["optimizer_state_entries"] != 550:
                    raise RuntimeError("Recovery checkpoint optimizer state is incomplete")
                boundary_checkpoint = candidate
                os.killpg(process.pid, signal.SIGTERM)
                break
    try:
        returncode = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        returncode = process.wait(timeout=30)
    if boundary_checkpoint is None:
        raise RuntimeError(f"Training exited before verified step-500 recovery: {returncode}")
    if not START_RECEIPT.is_file():
        raise RuntimeError("Step-0 start receipt was not emitted")

    HANDOFF.mkdir(parents=True, exist_ok=False)
    latest_target = HANDOFF / "latest.pth"
    shutil.copyfile(recovery, latest_target)
    latest = verify_checkpoint(latest_target, TARGET_RECOVERY_STEP)
    model_dir = WORKING / "models" / MODEL_NAME
    best_candidates = sorted(model_dir.glob("model-500-best_abs_rel_*"))
    if len(best_candidates) != 1:
        raise RuntimeError(f"Expected one step-500 best abs_rel checkpoint, found {best_candidates}")
    best_target = HANDOFF / "best_abs_rel.pth"
    shutil.copyfile(best_candidates[0], best_target)
    best = verify_checkpoint(best_target, TARGET_RECOVERY_STEP)

    best_ledger = []
    for path in sorted(model_dir.glob("model-500-best_*")):
        receipt = verify_checkpoint(path, TARGET_RECOVERY_STEP)
        receipt["source_name"] = path.name
        best_ledger.append(receipt)
    write_json(HANDOFF / "best_metric_evidence.json", {"checkpoints": best_ledger})
    for path in model_dir.rglob("*.pth"):
        path.unlink()
    for path in model_dir.glob("model-500-best_*"):
        path.unlink()

    boundary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "session": 1,
        "freeze": "v2",
        "completed_utc": now(),
        "status": "boundary_checkpoint_verified_local_output_pending_remote_handoff",
        "initial_checkpoint": None,
        "first_completed_global_step": 0,
        "last_committed_global_step": TARGET_RECOVERY_STEP,
        "latest": latest,
        "best_abs_rel": best,
        "best_metric_evidence_count": len(best_ledger),
        "training_process_returncode_after_controlled_stop": returncode,
        "remote_handoff_verified": False,
        "session_2_authorized": False,
    }
    write_json(BOUNDARY_RECEIPT, boundary)
    write_json(HANDOFF / "handoff_manifest.json", boundary)
    write_json(
        EXPERIMENT_LOG,
        {
            "run_id": RUN_ID,
            "session": 1,
            "status": "passed_boundary_pending_remote_handoff",
            "startup": startup,
            "boundary": boundary,
            "full_training_complete": False,
        },
    )
    METRICS.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "session": 1,
                "metric": "last_committed_global_step",
                "value": TARGET_RECOVERY_STEP,
                "timestamp": boundary["completed_utc"],
            }
        )
        + "\n"
    )
    write_json(
        ARTIFACTS,
        {
            "run_id": RUN_ID,
            "session": 1,
            "artifacts": [
                STARTUP_RECEIPT.name,
                START_RECEIPT.name,
                BOUNDARY_RECEIPT.name,
                SESSION_LOG.name,
                "session_1_handoff/latest.pth",
                "session_1_handoff/best_abs_rel.pth",
                "session_1_handoff/handoff_manifest.json",
                "session_1_handoff/best_metric_evidence.json",
            ],
        },
    )
    return boundary


if __name__ == "__main__":
    try:
        result = launch()
    except Exception as exc:
        result = {
            "run_id": RUN_ID,
            "session": 1,
            "status": "failed",
            "completed_utc": now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "session_2_authorized": False,
        }
        write_json(EXPERIMENT_LOG, result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] not in {
        "boundary_checkpoint_verified_local_output_pending_remote_handoff"
    }:
        raise SystemExit(1)
