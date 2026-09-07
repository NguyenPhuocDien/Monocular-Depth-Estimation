"""Focused Gate-2 path recheck after the allowlisted Kaggle mount remediation."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input/datasets/thudo25")
OUTPUT_PATH = Path("/kaggle/working/gate2_path_recheck.json")
TARGET_ROOT = INPUT_ROOT / "bts-kitti-eigen-materialized-v1"
LIST_EVIDENCE_HINT = INPUT_ROOT / "bts-kitti-gate2-evidence-v1"
PATH_EVIDENCE_HINT = INPUT_ROOT / "bts-kitti-gate2-path-evidence-v1"
EXPECTED_DATA_ROOT = "/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1/"
EXPECTED_GT_ROOT = EXPECTED_DATA_ROOT + "data_depth_annotated/"
EXPECTED_CONFIG_SHA256 = "9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7"
EXPECTED_PATCH_SHA256 = "03fbe34208408c93b5b8937e8e7431ed5ce2ca07e95caa880f6a63e7ef78dea7"
EXPECTED_COMMIT = "5e3406b35d1497b2e55d2dd600524d1f4efacaed"
TRAIN_LIST_SHA256 = "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094"
TEST_LIST_SHA256 = "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd"
EXPECTED_TRAIN_ROWS = 23_158
EXPECTED_TEST_ROWS = 697
EXPECTED_TEST_GT_NONE = 45


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_config(path: Path) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        result[parts[0].removeprefix("--")] = parts[1] if len(parts) == 2 else True
    return result


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe list path: {value!r}")
    return path


def read_rows(path: Path) -> list[list[str]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) < 3 for row in rows):
        raise ValueError(f"Malformed official list: {path.name}")
    return rows


def find_evidence_root(hint: Path, required_names: set[str]) -> Path:
    if hint.is_dir() and all((hint / name).is_file() for name in required_names):
        return hint
    anchor = sorted(required_names)[0]
    matches = []
    for anchor_path in Path("/kaggle/input").rglob(anchor):
        parent = anchor_path.parent
        if all((parent / name).is_file() for name in required_names):
            matches.append(parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one evidence root for {sorted(required_names)}, found "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def main() -> dict:
    path_evidence_root = find_evidence_root(
        PATH_EVIDENCE_HINT,
        {"arguments_train_eigen_kaggle.txt", "provenance_manifest.json", "smoke_report.json"},
    )
    list_evidence_root = find_evidence_root(
        LIST_EVIDENCE_HINT,
        {"eigen_train_files_with_gt.txt", "eigen_test_files_with_gt.txt", "files.sha256"},
    )
    config_path = path_evidence_root / "arguments_train_eigen_kaggle.txt"
    provenance_path = path_evidence_root / "provenance_manifest.json"
    smoke_path = path_evidence_root / "smoke_report.json"
    train_list = list_evidence_root / "eigen_train_files_with_gt.txt"
    test_list = list_evidence_root / "eigen_test_files_with_gt.txt"
    required = [config_path, provenance_path, smoke_path, train_list, test_list]
    missing_evidence = [str(path) for path in required if not path.is_file()]
    if missing_evidence:
        raise FileNotFoundError(f"Missing Gate-2 evidence: {missing_evidence}")

    config = parse_config(config_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    train_rows = read_rows(train_list)
    test_rows = read_rows(test_list)

    configured_data_root = Path(str(config["data_path"]))
    configured_gt_root = Path(str(config["gt_path"]))
    configured_eval_root = Path(str(config["data_path_eval"]))
    configured_eval_gt_root = Path(str(config["gt_path_eval"]))

    train_rgb_missing = 0
    train_gt_missing = 0
    for row in train_rows:
        rgb = safe_relative(row[0])
        gt = safe_relative(row[1])
        train_rgb_missing += int(not (configured_data_root / rgb).is_file())
        train_gt_missing += int(not (configured_gt_root / gt).is_file())

    test_rgb_missing = 0
    test_gt_none = 0
    test_gt_missing = 0
    for row in test_rows:
        rgb = safe_relative(row[0])
        test_rgb_missing += int(not (configured_eval_root / rgb).is_file())
        if row[1] == "None":
            test_gt_none += 1
        else:
            gt = safe_relative(row[1])
            test_gt_missing += int(not (configured_eval_gt_root / gt).is_file())

    checks = {
        "target_mount_exists": TARGET_ROOT.is_dir(),
        "config_sha256": sha256(config_path) == EXPECTED_CONFIG_SHA256,
        "data_path_exact": config.get("data_path") == EXPECTED_DATA_ROOT,
        "gt_path_exact": config.get("gt_path") == EXPECTED_GT_ROOT,
        "data_path_eval_exact": config.get("data_path_eval") == EXPECTED_DATA_ROOT,
        "gt_path_eval_exact": config.get("gt_path_eval") == EXPECTED_GT_ROOT,
        "configured_data_path_exists": configured_data_root.is_dir(),
        "configured_gt_path_exists": configured_gt_root.is_dir(),
        "configured_data_path_eval_exists": configured_eval_root.is_dir(),
        "configured_gt_path_eval_exists": configured_eval_gt_root.is_dir(),
        "train_list_hash": sha256(train_list) == TRAIN_LIST_SHA256,
        "test_list_hash": sha256(test_list) == TEST_LIST_SHA256,
        "train_row_count": len(train_rows) == EXPECTED_TRAIN_ROWS,
        "test_row_count": len(test_rows) == EXPECTED_TEST_ROWS,
        "train_rgb_resolve": train_rgb_missing == 0,
        "train_gt_resolve": train_gt_missing == 0,
        "test_rgb_resolve": test_rgb_missing == 0,
        "test_gt_resolve": test_gt_missing == 0,
        "expected_test_gt_none": test_gt_none == EXPECTED_TEST_GT_NONE,
        "provenance_commit": provenance.get("official_commit") == EXPECTED_COMMIT,
        "provenance_patch_hash": provenance.get("patch_sha256") == EXPECTED_PATCH_SHA256,
        "provenance_config_hash": (
            provenance.get("adaptation_config_sha256", {}).get("arguments_train_eigen_kaggle.txt")
            == EXPECTED_CONFIG_SHA256
        ),
        "provenance_patch_checks": (
            provenance.get("patch_apply_check") == "passed"
            and provenance.get("reverse_patch_check") == "passed"
        ),
        "provenance_smoke": provenance.get("smoke_status") == "passed" and smoke.get("status") == "passed",
    }

    return {
        "gate": "D_GATE_2_PATH_RECHECK",
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": utc_now(),
        "checks": checks,
        "observed": {
            "path_evidence_root": str(path_evidence_root),
            "list_evidence_root": str(list_evidence_root),
            "config_sha256": sha256(config_path),
            "data_path": config.get("data_path"),
            "gt_path": config.get("gt_path"),
            "data_path_eval": config.get("data_path_eval"),
            "gt_path_eval": config.get("gt_path_eval"),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_rgb_missing": train_rgb_missing,
            "train_gt_missing": train_gt_missing,
            "test_rgb_missing": test_rgb_missing,
            "test_gt_missing": test_gt_missing,
            "test_gt_none": test_gt_none,
        },
        "failures": [name for name, passed in checks.items() if not passed],
        "reused_evidence": {
            "mounted_content_audit": "artifacts/kaggle/bts_kitti_gate2/run02/gate2_mounted_audit.json",
            "files": 47_665,
            "bytes": 23_695_187_138,
            "sha256_mismatches": 0,
        },
        "authorization_boundary": {
            "full_content_rehash_performed": False,
            "k0_k4_executed": False,
            "training_executed": False,
            "dataset_written": False,
        },
    }


if __name__ == "__main__":
    try:
        report = main()
    except Exception as exc:
        report = {
            "gate": "D_GATE_2_PATH_RECHECK",
            "status": "failed",
            "completed_utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "authorization_boundary": {
                "full_content_rehash_performed": False,
                "k0_k4_executed": False,
                "training_executed": False,
                "dataset_written": False,
            },
        }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)
