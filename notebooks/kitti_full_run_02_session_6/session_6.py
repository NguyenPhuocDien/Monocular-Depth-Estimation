"""
BTS KITTI Full Run 02 — Session 6
Resume: Step 145,000 → 212,500  (~25.0 → 36.7 Epochs)
Hardware: 2x Tesla T4  |  DDP DistributedDataParallel
Duration: ~11.5 hours (Maximized Kaggle Session)
"""

# ── 1. IMPORTS ──────────────────────────────────────────────────────────────
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

import torch

# ── 2. CONFIG ────────────────────────────────────────────────────────────────
class CFG:
    # Run identity
    run_id      = "bts_kitti_densenet161_full_run_02"
    model_name  = "bts_kitti_densenet161_kaggle"
    session     = 6
    freeze      = "v4"

    # Step boundaries
    resume_step       = 145_000   # last committed step from Session 5
    first_step        = 145_001   # first new optimizer step this session
    target_step       = 190_000   # 45,000 steps (~7.4h running, leaves ~7h buffer for S8)

    # Architecture / data geometry (FROZEN — must match freeze v4)
    encoder           = "densenet161"
    global_batch_size = 4         # 2 GPUs × local_batch 2
    steps_per_epoch   = 5_790     # ceil(23158 / 4)
    total_steps       = 289_500   # 5790 × 50 epochs

    # Dataset / paths (Kaggle read-only mounts)
    data_root   = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
    handoff_src = Path("/kaggle/input/datasets/thudo25/bts-kitti-full-run-02-handoff/latest.pth")

    # Freeze v4 expected SHA256s (immutable contract)
    sha = {
        "freeze_manifest":  "2bf8f9fd1ca968b89a2f807c649f5e3783f33e3b84a36ac801cf5e58f697c464",
        "freeze_retention": "db31311e6f549aa9ed66e645402644c83a1eacfc6664530fd8bcf92d8de2554b",
        "freeze_receipt":   "b743c2ea6404fa82c5fa33809be8bc1cd6304afcd0d59c921e51018ae8e26e83",
        "initial_ckpt":     "e5c7207f5bafa22232045d4a7c9b7cb0767fc4bb63cbb59b53a1425908616710",
        "bts_main":         "7651c9a1b5c27153d7c92c38446fadb06fda7d869e4e15c1c544214d686546ab",
        "retention":        "9adc2f5f5eb6d49f870aadbd7b2fa9a7290dcdc95bfd36f4d516adc331262c59",
        "config":           "d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81",
        "vendor_manifest":  "0206c24df45e0bc11d5504c159b3486bf2c090a6a623cd8b0cd3cceff84a467d",
        "weight":           "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359",
        "train_list":       "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094",
        "test_list":        "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd",
    }

    # Checkpoint schema (production contract)
    ckpt_schema = {
        "keys":     {"global_step","model","optimizer",
                     "best_eval_measures_higher_better",
                     "best_eval_measures_lower_better","best_eval_steps"},
        "higher":   3,
        "lower":    6,
        "steps":    9,
        "opt_states": 227,
        "opt_groups": [482, 68],
    }


# ── 3. WORKING DIRECTORIES ───────────────────────────────────────────────────
WORK          = Path("/kaggle/working")
RUNTIME       = WORK / "bts_runtime"
RESUME_DIR    = WORK / "resume_source"
RESUME_CKPT   = RESUME_DIR / "latest.pth"
SESSION_CFG   = WORK / "session_6_config.txt"
HANDOFF       = WORK / "session_6_handoff"
LOG_FILE      = WORK / "session_6_training.log"
STARTUP_FILE  = WORK / "session_6_startup_audit.json"
START_FILE    = WORK / "session_6_start_receipt.json"
BOUNDARY_FILE = WORK / "session_6_boundary_receipt.json"
EXP_LOG       = WORK / "experiment_log.json"
METRICS_FILE  = WORK / "metrics.jsonl"
ARTIFACTS_FILE= WORK / "artifacts_manifest.json"


# ── 4. UTILITIES ─────────────────────────────────────────────────────────────
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def find_one(name: str, expected: str) -> Path:
    hits = [p for p in Path("/kaggle/input").rglob(name) if sha256(p) == expected]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {name} with hash {expected[:8]}…, got {hits}")
    return hits[0]


def load_checkpoint(path: Path, expect_step: int | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing = CFG.ckpt_schema["keys"] - set(ckpt)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)}")
    step = int(ckpt["global_step"])
    if expect_step is not None and step != expect_step:
        raise RuntimeError(f"Expected step {expect_step}, got {step}")
    opt = ckpt["optimizer"]
    n_states = len(opt.get("state", {}))
    groups   = [len(g["params"]) for g in opt.get("param_groups", [])]
    hi = len(ckpt["best_eval_measures_higher_better"])
    lo = len(ckpt["best_eval_measures_lower_better"])
    st = len(ckpt["best_eval_steps"])
    s  = CFG.ckpt_schema
    if [hi, lo, st] != [s["higher"], s["lower"], s["steps"]] \
            or n_states != s["opt_states"] or groups != s["opt_groups"]:
        raise RuntimeError(
            f"Checkpoint contract mismatch: metrics=[{hi},{lo},{st}] "
            f"opt_states={n_states} groups={groups}"
        )
    return {
        "path": str(path), "global_step": step,
        "bytes": path.stat().st_size, "sha256": sha256(path),
        "schema": sorted(CFG.ckpt_schema["keys"]),
        "optimizer_state_entries": n_states,
        "optimizer_group_param_counts": groups,
    }


# ── 5. STARTUP AUDIT ─────────────────────────────────────────────────────────
def startup_audit() -> dict:
    # GPU check
    phys = subprocess.run(
        ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
        text=True, capture_output=True, check=True
    ).stdout.splitlines()
    eff  = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(eff) != 2 or not all("T4" in n for n in eff):
        raise RuntimeError(f"Expected 2x T4 GPUs, got: {eff}")

    # Freeze v4 contract hashes
    contracts = {
        "freeze_manifest":  find_one("full_run_manifest.v4.json",       CFG.sha["freeze_manifest"]),
        "freeze_retention": find_one("checkpoint_retention_plan.v4.json",CFG.sha["freeze_retention"]),
        "freeze_receipt":   find_one("freeze_receipt.v4.json",           CFG.sha["freeze_receipt"]),
    }

    # Runtime integrity
    bts_main = find_one("bts_main.py", CFG.sha["bts_main"])
    runtime  = bts_main.parent.parent
    for rel, key in [
        ("pytorch/checkpoint_retention.py", "retention"),
        ("config/arguments_train_eigen_kaggle.txt", "config"),
        ("vendor_manifest.json", "vendor_manifest"),
    ]:
        if sha256(runtime / rel) != CFG.sha[key]:
            raise RuntimeError(f"Runtime hash mismatch: {rel}")

    # Data geometry
    train_list = runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt"
    test_list  = runtime / "train_test_inputs" / "eigen_test_files_with_gt.txt"
    if sha256(train_list) != CFG.sha["train_list"] or sha256(test_list) != CFG.sha["test_list"]:
        raise RuntimeError("Official split list hash mismatch")
    if not CFG.data_root.is_dir():
        raise RuntimeError(f"KITTI data mount missing: {CFG.data_root}")

    # Resume checkpoint
    src_ckpt    = find_one("latest.pth", CFG.sha["initial_ckpt"])
    src_receipt = load_checkpoint(src_ckpt, CFG.resume_step)

    # Stage runtime + checkpoint
    shutil.copytree(runtime, RUNTIME)
    RESUME_DIR.mkdir()
    shutil.copyfile(src_ckpt, RESUME_CKPT)
    shutil.copyfile(RUNTIME / "pytorch" / "bts.py", RESUME_DIR / "resume_source.py")
    staged = load_checkpoint(RESUME_CKPT, CFG.resume_step)
    if staged["sha256"] != CFG.sha["initial_ckpt"]:
        raise RuntimeError("Staged checkpoint hash mismatch after copy")

    # Patch config with resume path
    base_cfg = RUNTIME / "config" / "arguments_train_eigen_kaggle.txt"
    SESSION_CFG.write_text(base_cfg.read_text().rstrip() + f"\n--checkpoint_path {RESUME_CKPT}\n")

    # DenseNet-161 ImageNet weight cache
    weight  = find_one("densenet161-8d451a50.pth", CFG.sha["weight"])
    th_home = Path("/tmp/bts_s6_torch_home")
    cached  = th_home / "hub" / "checkpoints" / weight.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight, cached)
    os.environ["TORCH_HOME"] = str(th_home)
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(RUNTIME / "vendor"),
        str(RUNTIME / "pytorch"),
        os.environ.get("PYTHONPATH", ""),
    ])

    result = {
        "schema_version": 1,
        "run_id": CFG.run_id, "session": CFG.session, "freeze": CFG.freeze,
        "created_utc": now(),
        "status": f"passed_before_resumed_optimizer_step_{CFG.first_step}",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "physical_gpu_names": [n.strip() for n in phys if n.strip()],
            "effective_gpu_names": eff, "effective_gpu_count": len(eff),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract_hashes": {k: sha256(p) for k, p in contracts.items()},
        "source_checkpoint": src_receipt,
        "staged_checkpoint": staged,
        "target_recovery_step": CFG.target_step,
    }
    write_json(STARTUP_FILE, result)
    return result


# ── 6. TRAINING LOOP ─────────────────────────────────────────────────────────
def _emit_start_receipt() -> None:
    if START_FILE.exists():
        return
    write_json(START_FILE, {
        "run_id": CFG.run_id, "session": CFG.session, "freeze": CFG.freeze,
        "created_utc": now(),
        "resumed_from_checkpoint_sha256": CFG.sha["initial_ckpt"],
        "loaded_global_step": CFG.resume_step,
        "first_completed_global_step": CFG.first_step,
        "effective_gpu_count": 2, "global_batch_size": CFG.global_batch_size,
        "steps_per_epoch": CFG.steps_per_epoch, "total_steps": CFG.total_steps,
        "target_boundary_global_step": CFG.target_step,
        "status": "running",
    })


def train() -> dict:
    startup = startup_audit()
    cmd = [sys.executable, str(RUNTIME / "pytorch" / "bts_main.py"), str(SESSION_CFG)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=os.environ.copy(),
        cwd=str(RUNTIME / "pytorch"), start_new_session=True,
    )
    recovery = (
        WORK / "models" / CFG.model_name / "checkpoints" / "recovery" / "latest.pth"
    )
    boundary_ckpt = None
    last_step     = CFG.resume_step

    with LOG_FILE.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
            if re.search(str(CFG.first_step), line):
                _emit_start_receipt()
            m = re.search(r"Saved verified recovery checkpoint:\s*(.+\.pth)\s*$", line)
            if m:
                emitted = Path(m.group(1))
                if emitted != recovery:
                    raise RuntimeError(f"Recovery path mismatch: {emitted} != {recovery}")
                info     = load_checkpoint(recovery)
                last_step = info["global_step"]
                if last_step >= CFG.target_step:
                    boundary_ckpt = info
                    os.killpg(proc.pid, signal.SIGTERM)
                    break

    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        rc = proc.wait(timeout=30)

    if not START_FILE.is_file():
        raise RuntimeError("Start receipt was never emitted")
    if not recovery.is_file():
        raise RuntimeError("Recovery checkpoint missing after training")

    # Package handoff
    final_info = load_checkpoint(recovery)
    final_step = final_info["global_step"]
    HANDOFF.mkdir(parents=True, exist_ok=False)
    latest_dst = HANDOFF / "latest.pth"
    shutil.copyfile(recovery, latest_dst)
    latest_rcpt = load_checkpoint(latest_dst, final_step)

    model_dir = WORK / "models" / CFG.model_name
    best = None
    best_cands = sorted(model_dir.glob("model-*-best_abs_rel_*"))
    if best_cands:
        best_dst = HANDOFF / "best_abs_rel.pth"
        shutil.copyfile(best_cands[-1], best_dst)
        best = load_checkpoint(best_dst)

    evidence = []
    for p in sorted(model_dir.glob("model-*-best_*")):
        r = load_checkpoint(p); r["source_name"] = p.name
        evidence.append(r)
    write_json(HANDOFF / "best_metric_evidence.json", {"checkpoints": evidence})

    # Clean up checkpoints from working storage
    for p in model_dir.rglob("*.pth"):
        p.unlink(missing_ok=True)

    # Write receipts
    boundary = {
        "schema_version": 1,
        "run_id": CFG.run_id, "session": CFG.session, "freeze": CFG.freeze,
        "completed_utc": now(),
        "status": "boundary_checkpoint_verified_local_output_pending_remote_handoff",
        "initial_checkpoint": CFG.sha["initial_ckpt"],
        "first_completed_global_step": CFG.first_step,
        "last_committed_global_step": final_step,
        "latest": latest_rcpt, "best_abs_rel": best,
        "best_metric_evidence_count": len(evidence),
        "training_process_returncode_after_controlled_stop": rc,
        "remote_handoff_verified": False, "next_session_authorized": False,
    }
    write_json(BOUNDARY_FILE, boundary)
    write_json(HANDOFF / "handoff_manifest.json", boundary)
    write_json(EXP_LOG, {
        "run_id": CFG.run_id, "session": CFG.session,
        "status": "passed_boundary_pending_remote_handoff",
        "startup": startup, "boundary": boundary,
        "full_training_complete": final_step >= CFG.total_steps,
    })
    METRICS_FILE.write_text(json.dumps({
        "run_id": CFG.run_id, "session": CFG.session,
        "metric": "last_committed_global_step",
        "value": final_step, "timestamp": boundary["completed_utc"],
    }) + "\n")
    write_json(ARTIFACTS_FILE, {
        "run_id": CFG.run_id, "session": CFG.session,
        "artifacts": [
            STARTUP_FILE.name, START_FILE.name, BOUNDARY_FILE.name,
            LOG_FILE.name,
            f"{HANDOFF.name}/{latest_dst.name}",
            f"{HANDOFF.name}/handoff_manifest.json",
            f"{HANDOFF.name}/best_metric_evidence.json",
        ],
    })
    return boundary


# ── 7. MAIN ──────────────────────────────────────────────────────────────────
def main() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        train()
    except Exception as exc:
        write_json(EXP_LOG, {
            "run_id": CFG.run_id, "session": CFG.session, "status": "failed",
            "completed_utc": now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "full_training_complete": False,
        })
        raise


if __name__ == "__main__":
    main()
