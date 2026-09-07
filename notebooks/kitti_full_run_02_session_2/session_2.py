"""Resume BTS KITTI full-run Session 2 under immutable 2xT4 Freeze v4."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING_RUNTIME = WORKING / "bts_runtime"
RESUME_DIR = WORKING / "resume_source"
RESUME_CHECKPOINT = RESUME_DIR / "latest.pth"
SESSION_CONFIG = WORKING / "session_2_config.txt"
HANDOFF = WORKING / "session_2_handoff"
STARTUP_RECEIPT = WORKING / "session_2_startup_audit.json"
START_RECEIPT = WORKING / "session_2_start_receipt.json"
BOUNDARY_RECEIPT = WORKING / "session_2_boundary_receipt.json"
SESSION_LOG = WORKING / "session_2_training.log"
EXPERIMENT_LOG = WORKING / "experiment_log.json"
METRICS = WORKING / "metrics.jsonl"
ARTIFACTS = WORKING / "artifacts_manifest.json"
RUN_ID = "bts_kitti_densenet161_full_run_02"
MODEL_NAME = "bts_kitti_densenet161_kaggle"
SESSION = 2
INITIAL_STEP = 500
FIRST_RESUMED_STEP = 501
TARGET_RECOVERY_STEP = 25000
EXPECTED = {
    "freeze_manifest": "2bf8f9fd1ca968b89a2f807c649f5e3783f33e3b84a36ac801cf5e58f697c464",
    "freeze_retention": "db31311e6f549aa9ed66e645402644c83a1eacfc6664530fd8bcf92d8de2554b",
    "freeze_receipt": "b743c2ea6404fa82c5fa33809be8bc1cd6304afcd0d59c921e51018ae8e26e83",
    "initial_checkpoint": "f3c778cfca0d73718295b75404b20e951e992c0e4ed440533afa81d3e208e3dd",
    "bts_main": "7651c9a1b5c27153d7c92c38446fadb06fda7d869e4e15c1c544214d686546ab",
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


def find_one(name: str, expected_hash: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if sha256(path) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def verify_checkpoint(path: Path, expected_step: int | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_KEYS.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing production keys: {sorted(missing)}")
    step = int(checkpoint["global_step"])
    if expected_step is not None and step != expected_step:
        raise RuntimeError(f"Expected checkpoint step {expected_step}, got {step}")
    metric_slots = [
        len(checkpoint["best_eval_measures_higher_better"]),
        len(checkpoint["best_eval_measures_lower_better"]),
        len(checkpoint["best_eval_steps"]),
    ]
    state_entries = len(checkpoint["optimizer"].get("state", {}))
    group_counts = [
        len(group["params"]) for group in checkpoint["optimizer"].get("param_groups", [])
    ]
    if metric_slots != [3, 6, 9] or state_entries != 227 or group_counts != [482, 68]:
        raise RuntimeError(
            f"Checkpoint contract mismatch: slots={metric_slots}, "
            f"states={state_entries}, groups={group_counts}"
        )
    return {
        "path": str(path),
        "global_step": step,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "schema": sorted(REQUIRED_KEYS),
        "optimizer_state_entries": state_entries,
        "optimizer_group_param_counts": group_counts,
    }


def startup_audit() -> dict:
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    effective = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(effective) != 2 or not all("T4" in name for name in effective):
        raise RuntimeError(f"Expected two effective T4 GPUs, got {effective}")

    contract_paths = {
        "freeze_manifest": find_one("full_run_manifest.v4.json", EXPECTED["freeze_manifest"]),
        "freeze_retention": find_one("checkpoint_retention_plan.v4.json", EXPECTED["freeze_retention"]),
        "freeze_receipt": find_one("freeze_receipt.v4.json", EXPECTED["freeze_receipt"]),
    }

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
        "local_batch_size": 2,
        "global_batch_size": 4,
        "steps_per_epoch": (train_rows + 3) // 4,
        "epochs": 50,
        "total_steps": ((train_rows + 3) // 4) * 50,
    }
    expected_geometry = {
        "train_rows": 23158, "test_rows": 697,
        "local_batch_size": 2, "global_batch_size": 4,
        "steps_per_epoch": 5790, "epochs": 50, "total_steps": 289500,
    }
    if geometry != expected_geometry:
        raise RuntimeError(f"Frozen geometry mismatch: {geometry}")
    if sha256(train_list) != EXPECTED["train_list"] or sha256(test_list) != EXPECTED["test_list"]:
        raise RuntimeError("Official list hash mismatch")

    source_checkpoint = find_one("latest.pth", EXPECTED["initial_checkpoint"])
    source_receipt = verify_checkpoint(source_checkpoint, INITIAL_STEP)
    shutil.copytree(runtime, WORKING_RUNTIME)
    for relative, expected in (
        ("pytorch/bts_main.py", EXPECTED["bts_main"]),
        ("pytorch/checkpoint_retention.py", EXPECTED["retention_helper"]),
        ("config/arguments_train_eigen_kaggle.txt", EXPECTED["config"]),
    ):
        if sha256(WORKING_RUNTIME / relative) != expected:
            raise RuntimeError(f"Copied runtime hash mismatch: {relative}")

    RESUME_DIR.mkdir()
    shutil.copyfile(source_checkpoint, RESUME_CHECKPOINT)
    shutil.copyfile(WORKING_RUNTIME / "pytorch" / "bts.py", RESUME_DIR / "resume_source.py")
    staged_receipt = verify_checkpoint(RESUME_CHECKPOINT, INITIAL_STEP)
    if staged_receipt["sha256"] != EXPECTED["initial_checkpoint"]:
        raise RuntimeError("Staged checkpoint hash mismatch")

    base_config = WORKING_RUNTIME / "config" / "arguments_train_eigen_kaggle.txt"
    config_text = base_config.read_text()
    if "--checkpoint_path" in config_text or "--retrain" in config_text:
        raise RuntimeError("Frozen base config unexpectedly contains resume fields")
    SESSION_CONFIG.write_text(
        config_text.rstrip() + f"\n--checkpoint_path {RESUME_CHECKPOINT}\n"
    )

    weight = find_one("densenet161-8d451a50.pth", EXPECTED["weight"])
    torch_home = Path("/tmp/bts_session_2_torch_home")
    cached = torch_home / "hub" / "checkpoints" / weight.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight, cached)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(WORKING_RUNTIME / "vendor"),
        str(WORKING_RUNTIME / "pytorch"),
        os.environ.get("PYTHONPATH", ""),
    ])

    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "session": SESSION,
        "freeze": "v4",
        "created_utc": now(),
        "status": "passed_before_resumed_optimizer_step_501",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu_names": [name.strip() for name in physical if name.strip()],
            "effective_gpu_names": effective,
            "effective_gpu_count": len(effective),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract_hashes": {key: sha256(path) for key, path in contract_paths.items()},
        "runtime_hashes": checks,
        "base_config_sha256": sha256(base_config),
        "session_config_sha256": sha256(SESSION_CONFIG),
        "resume_override": f"--checkpoint_path {RESUME_CHECKPOINT}",
        "source_checkpoint": source_receipt,
        "staged_checkpoint": staged_receipt,
        "geometry": geometry,
        "target_recovery_step": TARGET_RECOVERY_STEP,
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
            "session": SESSION,
            "freeze": "v4",
            "created_utc": now(),
            "resumed_from_checkpoint_sha256": EXPECTED["initial_checkpoint"],
            "loaded_global_step": INITIAL_STEP,
            "first_completed_global_step": FIRST_RESUMED_STEP,
            "effective_gpu_count": 2,
            "global_batch_size": 4,
            "steps_per_epoch": 5790,
            "total_steps": 289500,
            "target_boundary_global_step": TARGET_RECOVERY_STEP,
            "status": "running",
        },
    )


def launch() -> dict:
    startup = startup_audit()
    command = [
        sys.executable,
        str(WORKING_RUNTIME / "pytorch" / "bts_main.py"),
        str(SESSION_CONFIG),
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
    latest_emitted_step = INITIAL_STEP
    with SESSION_LOG.open("w", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if re.search(r"\[0\]\[501/5790/501\]", line):
                create_start_receipt()
            match = re.search(r"Saved verified recovery checkpoint:\s*(.+\.pth)\s*$", line)
            if match:
                emitted_recovery = Path(match.group(1))
                if emitted_recovery != recovery:
                    raise RuntimeError(
                        f"Recovery path mismatch: emitted={emitted_recovery}, expected={recovery}"
                    )
                if not recovery.is_file():
                    raise RuntimeError(f"Emitted recovery checkpoint is missing: {recovery}")
                candidate = verify_checkpoint(recovery)
                latest_emitted_step = candidate["global_step"]
                if latest_emitted_step >= TARGET_RECOVERY_STEP:
                    boundary_checkpoint = candidate
                    os.killpg(process.pid, signal.SIGTERM)
                    break
    try:
        returncode = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        returncode = process.wait(timeout=30)

    if not START_RECEIPT.is_file():
        raise RuntimeError("Step-501 start receipt was not emitted")
    if not recovery.is_file():
        raise RuntimeError("Final recovery checkpoint is missing")

    final_checkpoint_data = verify_checkpoint(recovery)
    final_committed_step = final_checkpoint_data["global_step"]

    HANDOFF.mkdir(parents=True, exist_ok=False)
    latest_target = HANDOFF / "latest.pth"
    shutil.copyfile(recovery, latest_target)
    latest = verify_checkpoint(latest_target, final_committed_step)

    model_dir = WORKING / "models" / MODEL_NAME
    best_candidates = sorted(model_dir.glob("model-*-best_abs_rel_*"))
    best = None
    if best_candidates:
        best_target = HANDOFF / "best_abs_rel.pth"
        shutil.copyfile(best_candidates[-1], best_target)
        best = verify_checkpoint(best_target)

    best_ledger = []
    for path in sorted(model_dir.glob("model-*-best_*")):
        receipt = verify_checkpoint(path)
        receipt["source_name"] = path.name
        best_ledger.append(receipt)
    write_json(HANDOFF / "best_metric_evidence.json", {"checkpoints": best_ledger})

    for path in model_dir.rglob("*.pth"):
        path.unlink()
    for path in model_dir.glob("model-*-best_*"):
        path.unlink()

    boundary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "session": SESSION,
        "freeze": "v4",
        "completed_utc": now(),
        "status": "boundary_checkpoint_verified_local_output_pending_remote_handoff",
        "initial_checkpoint": EXPECTED["initial_checkpoint"],
        "first_completed_global_step": FIRST_RESUMED_STEP,
        "last_committed_global_step": final_committed_step,
        "latest": latest,
        "best_abs_rel": best,
        "best_metric_evidence_count": len(best_ledger),
        "training_process_returncode_after_controlled_stop": returncode,
        "remote_handoff_verified": False,
        "next_session_authorized": False,
    }
    write_json(BOUNDARY_RECEIPT, boundary)
    write_json(HANDOFF / "handoff_manifest.json", boundary)
    write_json(
        EXPERIMENT_LOG,
        {
            "run_id": RUN_ID,
            "session": SESSION,
            "status": "passed_boundary_pending_remote_handoff",
            "startup": startup,
            "boundary": boundary,
            "full_training_complete": final_committed_step >= 289500,
        },
    )
    METRICS.write_text(
        json.dumps({
            "run_id": RUN_ID,
            "session": SESSION,
            "metric": "last_committed_global_step",
            "value": final_committed_step,
            "timestamp": boundary["completed_utc"],
        }) + "\n"
    )
    write_json(
        ARTIFACTS,
        {
            "run_id": RUN_ID,
            "session": SESSION,
            "artifacts": [
                STARTUP_RECEIPT.name,
                START_RECEIPT.name,
                BOUNDARY_RECEIPT.name,
                SESSION_LOG.name,
                f"{HANDOFF.name}/{latest_target.name}",
                f"{HANDOFF.name}/handoff_manifest.json",
                f"{HANDOFF.name}/best_metric_evidence.json",
            ],
        },
    )
    return boundary


def main() -> None:
    try:
        launch()
    except Exception as exc:
        write_json(
            EXPERIMENT_LOG,
            {
                "run_id": RUN_ID,
                "session": SESSION,
                "status": "failed",
                "completed_utc": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "full_training_complete": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
