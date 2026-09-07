"""Fail-closed two-T4 DDP preflight; never starts the full reproduction run."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA_ROOT = Path("/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1")
REPORT = WORKING / "runtime_v4_2xt4_preflight_report.json"
CHECKPOINT = WORKING / "runtime_v4_2xt4_production_preflight.pth"
EXPECTED = {
    "official_commit": "5e3406b35d1497b2e55d2dd600524d1f4efacaed",
    "base_patch": "03fbe34208408c93b5b8937e8e7431ed5ce2ca07e95caa880f6a63e7ef78dea7",
    "retention_patch": "614347a373f25fa9857aefe41bc152edb689b9a9c4d32ba989d783ed729704ea",
    "bts_main": "7651c9a1b5c27153d7c92c38446fadb06fda7d869e4e15c1c544214d686546ab",
    "retention_helper": "9adc2f5f5eb6d49f870aadbd7b2fa9a7290dcdc95bfd36f4d516adc331262c59",
    "config": "d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81",
    "vendor_manifest": "0206c24df45e0bc11d5504c159b3486bf2c090a6a623cd8b0cd3cceff84a467d",
    "weight": "8d451a50bad6b9a83f477126c443716882a17af26e94d338154b058fb2dfd359",
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
    path.write_text(json.dumps(value, indent=2) + "\n")


def find_exact(name: str, expected: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if sha256(path) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact {name}, found {matches}")
    return matches[0]


def find_runtime() -> Path:
    return find_exact("bts_main.py", EXPECTED["bts_main"]).parent.parent


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def environment() -> dict:
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    effective = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    checks = {
        "python_3_12": platform.python_version().startswith("3.12."),
        "torch_2_10_cu128": torch.__version__.startswith("2.10.0+cu128"),
        "torchvision_0_25_cu128": torchvision.__version__.startswith("0.25.0+cu128"),
        "cuda_12_8": torch.version.cuda == "12.8",
        "two_effective_t4": len(effective) == 2 and all("T4" in name for name in effective),
        "two_physical_t4": len(physical) == 2 and all("T4" in name for name in physical),
        "both_devices_visible": visible not in {"0", "1"},
    }
    values = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "physical_gpu_names": physical, "effective_gpu_names": effective,
        "effective_gpu_count": len(effective), "cuda_visible_devices": visible,
    }
    if not all(checks.values()):
        raise RuntimeError(f"2xT4 environment mismatch: {values}, {checks}")
    return {"values": values, "checks": checks}


def runtime_audit(runtime: Path) -> dict:
    manifest = json.loads((runtime / "provenance_manifest.json").read_text())
    hashes = {
        "bts_main": sha256(runtime / "pytorch" / "bts_main.py"),
        "retention_helper": sha256(runtime / "pytorch" / "checkpoint_retention.py"),
        "config": sha256(runtime / "config" / "arguments_train_eigen_kaggle.txt"),
        "vendor_manifest": sha256(runtime / "vendor_manifest.json"),
    }
    patch_hashes = {item["sha256"] for item in manifest["patches"]}
    checks = {
        "official_commit": manifest["official_commit"] == EXPECTED["official_commit"],
        "base_patch": EXPECTED["base_patch"] in patch_hashes,
        "retention_patch": EXPECTED["retention_patch"] in patch_hashes,
        "bts_main": hashes["bts_main"] == EXPECTED["bts_main"],
        "retention_helper": hashes["retention_helper"] == EXPECTED["retention_helper"],
        "config": hashes["config"] == EXPECTED["config"],
        "vendor_manifest": hashes["vendor_manifest"] == EXPECTED["vendor_manifest"],
        "historical_worktree_unused": manifest["historical_working_tree_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Runtime mismatch: {checks}, {hashes}")
    return {"hashes": hashes, "checks": checks}


def configure_imports(runtime: Path) -> None:
    for path in (runtime / "vendor", runtime / "pytorch"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def worker(rank: int, world_size: int, port: int, phase: str, runtime_text: str) -> None:
    runtime = Path(runtime_text)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size, rank=rank,
    )
    torch.cuda.set_device(rank)
    random.seed(20260827 + rank)
    np.random.seed(20260827 + rank)
    torch.manual_seed(20260827 + rank)
    torch.cuda.manual_seed_all(20260827 + rank)
    configure_imports(runtime)
    sys.argv = ["bts_main.py", str(runtime / "config" / "arguments_train_eigen_kaggle.txt")]
    bts_main = importlib.import_module("bts_main")
    retention = importlib.import_module("checkpoint_retention")
    args = copy.copy(bts_main.args)
    args.data_path = str(DATA_ROOT) + "/"
    args.gt_path = str(DATA_ROOT / "data_depth_annotated") + "/"
    args.filenames_file = str(runtime / "train_test_inputs" / "eigen_train_files_with_gt.txt")
    args.batch_size = 2
    args.num_threads = 1
    args.distributed = True
    args.multiprocessing_distributed = True
    args.rank = rank
    args.gpu = rank

    loader = bts_main.BtsDataLoader(args, "train")
    loader.train_sampler.set_epoch(0 if phase == "produce" else 1)
    model = bts_main.BtsModel(args).train()
    model.decoder.apply(bts_main.weights_init_xavier)
    bts_main.set_misc(model)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.module.encoder.parameters(), "weight_decay": args.weight_decay},
        {"params": model.module.decoder.parameters(), "weight_decay": 0.0},
    ], lr=args.learning_rate, eps=args.adam_eps)
    loaded_step = None
    checkpoint_hash = None
    if phase == "resume":
        checkpoint_hash = sha256(CHECKPOINT)
        checkpoint = torch.load(CHECKPOINT, map_location=f"cuda:{rank}", weights_only=False)
        retention.validate_checkpoint(checkpoint, 0)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_step = int(checkpoint["global_step"])

    batch = next(iter(loader.data))
    image = batch["image"].cuda(rank, non_blocking=True)
    depth = batch["depth"].cuda(rank, non_blocking=True)
    focal = batch["focal"].cuda(rank, non_blocking=True)
    optimizer.zero_grad()
    estimate = model(image, focal)[-1]
    mask = depth > 1.0
    loss = bts_main.silog_loss(args.variance_focus)(estimate, depth, mask.to(torch.bool))
    loss.backward()
    optimizer.step()
    reduced = loss.detach().clone()
    dist.all_reduce(reduced)
    reduced /= world_size

    checkpoint_receipt = None
    if phase == "produce" and rank == 0:
        checkpoint = retention.build_training_checkpoint(
            0, model, optimizer, torch.zeros(3), torch.zeros(6) + 1e3,
            np.zeros(9, dtype=np.int32),
        )
        checkpoint_receipt = retention.save_verified_checkpoint(checkpoint, CHECKPOINT)
    dist.barrier()
    if phase == "resume" and sha256(CHECKPOINT) != checkpoint_hash:
        raise RuntimeError("Resume mutated source checkpoint")
    result = {
        "phase": phase, "rank": rank, "status": "passed",
        "gpu": torch.cuda.get_device_name(rank),
        "local_batch_size": int(image.shape[0]),
        "sampler_samples": len(loader.train_sampler),
        "steps_per_epoch": len(loader.data),
        "loss_mean": float(reduced.cpu()),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_group_param_counts": [len(g["params"]) for g in optimizer.param_groups],
        "loaded_global_step": loaded_step,
        "completed_global_step": 0 if phase == "produce" else 1,
        "checkpoint": checkpoint_receipt,
    }
    write_json(WORKING / f"{phase}_rank_{rank}.json", result)
    dist.destroy_process_group()


def run_phase(phase: str, runtime: Path) -> list[dict]:
    mp.spawn(worker, nprocs=2, args=(2, free_port(), phase, str(runtime)), join=True)
    return [json.loads((WORKING / f"{phase}_rank_{rank}.json").read_text()) for rank in range(2)]


def main() -> dict:
    started = now()
    env = environment()
    runtime = find_runtime()
    provenance = runtime_audit(runtime)
    if not DATA_ROOT.is_dir():
        raise RuntimeError(f"Dataset mount missing: {DATA_ROOT}")
    weight = find_exact("densenet161-8d451a50.pth", EXPECTED["weight"])
    cache = Path("/tmp/bts_v4_2xt4/hub/checkpoints") / weight.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weight, cache)
    os.environ["TORCH_HOME"] = str(cache.parents[2])

    produced = run_phase("produce", runtime)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    schema = sorted(checkpoint)
    checkpoint_sha = sha256(CHECKPOINT)
    resumed = run_phase("resume", runtime)
    checks = {
        "environment": all(env["checks"].values()),
        "runtime": all(provenance["checks"].values()),
        "two_ddp_ranks_produce": len(produced) == 2 and all(x["status"] == "passed" for x in produced),
        "two_ddp_ranks_resume": len(resumed) == 2 and all(x["status"] == "passed" for x in resumed),
        "local_batch_2": all(x["local_batch_size"] == 2 for x in produced + resumed),
        "global_batch_4": sum(x["local_batch_size"] for x in produced) == 4,
        "steps_per_epoch_5790": all(x["steps_per_epoch"] == 5790 for x in produced + resumed),
        "total_steps_289500": 5790 * 50 == 289500,
        "six_key_schema": schema == sorted(REQUIRED_KEYS),
        "optimizer_contract": all(
            x["optimizer_state_entries"] == 227
            and x["optimizer_group_param_counts"] == [482, 68]
            for x in produced + resumed
        ),
        "resume_step_0_to_1": all(x["loaded_global_step"] == 0 and x["completed_global_step"] == 1 for x in resumed),
        "checkpoint_hash_stable": sha256(CHECKPOINT) == checkpoint_sha,
    }
    status = "passed" if all(checks.values()) else "failed"
    result = {
        "schema_version": 1, "gate": "KAGGLE_RUNTIME_V4_2XT4_PREFLIGHT",
        "status": status, "started_utc": started, "completed_utc": now(),
        "environment": env, "runtime": provenance,
        "geometry": {
            "train_rows": 23158, "local_batch_size": 2, "gpu_count": 2,
            "global_batch_size": 4, "steps_per_epoch": 5790,
            "epochs": 50, "total_steps": 289500,
        },
        "checks": checks, "produce": produced, "resume": resumed,
        "checkpoint_sha256_before_prune": checkpoint_sha,
        "authorization_boundary": {
            "full_training_started": False,
            "preflight_optimizer_steps_per_rank": 2,
            "run_02_session_1_started": False,
        },
    }
    CHECKPOINT.unlink()
    result["temporary_checkpoint_pruned"] = not CHECKPOINT.exists()
    write_json(REPORT, result)
    write_json(WORKING / "experiment_log.json", {
        "run_id": "bts_kitti_runtime_v4_2xt4_preflight",
        "status": status, "full_training_started": False,
        "completed_utc": result["completed_utc"],
    })
    return result


if __name__ == "__main__":
    try:
        output = main()
    except Exception as exc:
        output = {
            "gate": "KAGGLE_RUNTIME_V4_2XT4_PREFLIGHT", "status": "failed",
            "completed_utc": now(), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {"full_training_started": False},
        }
        write_json(REPORT, output)
    print(json.dumps(output, indent=2), flush=True)
    if output["status"] != "passed":
        raise SystemExit(1)
