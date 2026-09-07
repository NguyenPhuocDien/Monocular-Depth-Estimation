"""K0: read-only Kaggle mount and official-path gate for BTS KITTI."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


INPUT = Path("/kaggle/input")
OWNER_ROOT = INPUT / "datasets" / "thudo25"
TARGET = OWNER_ROOT / "bts-kitti-eigen-materialized-v1"
OUTPUT = Path("/kaggle/working/k0_mount_path_report.json")
CONFIG_SHA256 = "9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7"
TRAIN_SHA256 = "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094"
TEST_SHA256 = "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd"
DATA_ROOT = "/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1/"
GT_ROOT = DATA_ROOT + "data_depth_annotated/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_root(required: set[str]) -> Path:
    anchor = sorted(required)[0]
    matches = []
    for path in INPUT.rglob(anchor):
        parent = path.parent
        if all((parent / name).is_file() for name in required):
            matches.append(parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one evidence root for {sorted(required)}, found {matches}")
    return matches[0]


def parse_config(path: Path) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parts = line.split(maxsplit=1)
            result[parts[0].removeprefix("--")] = parts[1] if len(parts) == 2 else True
    return result


def rows(path: Path) -> list[list[str]]:
    result = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) < 3 for row in result):
        raise ValueError(f"Malformed list: {path}")
    return result


def relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"Unsafe path: {value!r}")
    return Path(*posix.parts)


def run() -> dict:
    list_root = find_root({"eigen_train_files_with_gt.txt", "eigen_test_files_with_gt.txt"})
    config_root = find_root({"arguments_train_eigen_kaggle.txt", "provenance_manifest.json"})
    config_path = config_root / "arguments_train_eigen_kaggle.txt"
    train_path = list_root / "eigen_train_files_with_gt.txt"
    test_path = list_root / "eigen_test_files_with_gt.txt"
    config = parse_config(config_path)
    train = rows(train_path)
    test = rows(test_path)

    data_root = Path(str(config["data_path"]))
    gt_root = Path(str(config["gt_path"]))
    eval_root = Path(str(config["data_path_eval"]))
    eval_gt_root = Path(str(config["gt_path_eval"]))
    missing = {"train_rgb": 0, "train_gt": 0, "test_rgb": 0, "test_gt": 0}
    test_none = 0

    for row in train:
        missing["train_rgb"] += int(not (data_root / relative(row[0])).is_file())
        missing["train_gt"] += int(not (gt_root / relative(row[1])).is_file())
    for row in test:
        missing["test_rgb"] += int(not (eval_root / relative(row[0])).is_file())
        if row[1] == "None":
            test_none += 1
        else:
            missing["test_gt"] += int(not (eval_gt_root / relative(row[1])).is_file())

    expected_top = sorted([
        "2011_09_26", "2011_09_28", "2011_09_29",
        "2011_09_30", "2011_10_03", "data_depth_annotated",
    ])
    observed_top = sorted(path.name for path in TARGET.iterdir()) if TARGET.is_dir() else []
    checks = {
        "target_mount_exists": TARGET.is_dir(),
        "target_top_level_exact": observed_top == expected_top,
        "config_sha256": sha256(config_path) == CONFIG_SHA256,
        "data_path_exact": config.get("data_path") == DATA_ROOT,
        "gt_path_exact": config.get("gt_path") == GT_ROOT,
        "data_path_eval_exact": config.get("data_path_eval") == DATA_ROOT,
        "gt_path_eval_exact": config.get("gt_path_eval") == GT_ROOT,
        "configured_roots_exist": all(path.is_dir() for path in [data_root, gt_root, eval_root, eval_gt_root]),
        "train_list_sha256": sha256(train_path) == TRAIN_SHA256,
        "test_list_sha256": sha256(test_path) == TEST_SHA256,
        "train_rows": len(train) == 23158,
        "test_rows": len(test) == 697,
        "train_rgb_resolve": missing["train_rgb"] == 0,
        "train_gt_resolve": missing["train_gt"] == 0,
        "test_rgb_resolve": missing["test_rgb"] == 0,
        "test_gt_resolve": missing["test_gt"] == 0,
        "expected_test_gt_none": test_none == 45,
    }
    return {
        "gate": "K0",
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observed": {
            "target_mount": str(TARGET),
            "config_root": str(config_root),
            "list_root": str(list_root),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_gt_none": test_none,
            "missing": missing,
            "top_level": observed_top,
        },
        "failures": [name for name, passed in checks.items() if not passed],
        "authorization_boundary": {
            "k1_k4_executed": False,
            "training_executed": False,
            "dataset_written": False,
        },
    }


if __name__ == "__main__":
    try:
        report = run()
    except Exception as exc:
        report = {
            "gate": "K0",
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {
                "k1_k4_executed": False,
                "training_executed": False,
                "dataset_written": False,
            },
        }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)
