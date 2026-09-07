"""Read-only BTS KITTI Gate-2 audit for the Kaggle-mounted dataset."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/gate2_mounted_audit.json")
TARGET_SLUG = "bts-kitti-eigen-materialized-v1"
EVIDENCE_SLUG = "bts-kitti-gate2-evidence-v1"
DATASET_ID = "thudo25/bts-kitti-eigen-materialized-v1"
OFFICIAL_COMMIT = "5e3406b35d1497b2e55d2dd600524d1f4efacaed"
TRAIN_LIST_SHA256 = "e9eca9ea3589f6a667db108f5e307d40dd6b7ab7634b32b0bd28ecf1d864c094"
TEST_LIST_SHA256 = "8966957128cce3846a2af5e82e3444cbde4376f7d1dc07f63fcd550b6cba94fd"
FILES_MANIFEST_SHA256 = "d8a545f7beb7a45a35fb594d6d9b5a7c4f48b03114c3754168fb8e92507317b0"
EXPECTED_FILES = 47_665
EXPECTED_BYTES = 23_695_187_138
EXPECTED_TRAIN_ROWS = 23_158
EXPECTED_TEST_ROWS = 697
EXPECTED_TEST_GT_NONE = 45
CHUNK_SIZE = 8 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def read_rows(path: Path) -> list[list[str]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) < 3 for row in rows):
        raise ValueError(f"Malformed official list: {path.name}")
    return rows


def read_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}") from exc
        relative = safe_relative(relative).as_posix()
        if relative in entries:
            raise ValueError(f"Duplicate checksum path: {relative}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA256 at line {line_number}")
        entries[relative] = digest
    return entries


def input_candidates() -> list[Path]:
    if not INPUT_ROOT.is_dir():
        return []
    candidates: list[Path] = []
    frontier = [INPUT_ROOT]
    for _ in range(3):
        next_frontier: list[Path] = []
        for parent in frontier:
            try:
                children = [path for path in parent.iterdir() if path.is_dir()]
            except OSError:
                continue
            candidates.extend(children)
            next_frontier.extend(children)
        frontier = next_frontier
    return candidates


def input_inventory() -> list[str]:
    return sorted(str(path.relative_to(INPUT_ROOT)) for path in input_candidates())


def find_mount(required_names: set[str], preferred_slug: str) -> Path:
    preferred = INPUT_ROOT / preferred_slug
    if preferred.is_dir() and required_names.issubset({path.name for path in preferred.iterdir()}):
        return preferred
    matches = []
    for candidate in input_candidates():
        try:
            names = {path.name for path in candidate.iterdir()}
        except OSError:
            continue
        if required_names.issubset(names):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mount matching {sorted(required_names)}, found "
            f"{[str(path) for path in matches]}; inventory={input_inventory()}"
        )
    return matches[0]


def main() -> dict:
    started = utc_now()
    required_evidence = {
        "files.sha256",
        "eigen_train_files_with_gt.txt",
        "eigen_test_files_with_gt.txt",
        "dataset_manifest.json",
        "materialization_report.json",
        "upload_success_20260826.json",
    }
    expected_top_level_names = {
        "2011_09_26",
        "2011_09_28",
        "2011_09_29",
        "2011_09_30",
        "2011_10_03",
        "data_depth_annotated",
    }
    target_root = find_mount(expected_top_level_names, TARGET_SLUG)
    evidence_root = find_mount(required_evidence, EVIDENCE_SLUG)
    evidence_files = {path.name for path in evidence_root.iterdir() if path.is_file()}
    if evidence_files != required_evidence:
        raise RuntimeError(
            f"Evidence package mismatch: missing={sorted(required_evidence - evidence_files)}, "
            f"unexpected={sorted(evidence_files - required_evidence)}"
        )

    checksum_path = evidence_root / "files.sha256"
    train_list = evidence_root / "eigen_train_files_with_gt.txt"
    test_list = evidence_root / "eigen_test_files_with_gt.txt"
    dataset_manifest = json.loads((evidence_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    materialization_report = json.loads(
        (evidence_root / "materialization_report.json").read_text(encoding="utf-8")
    )
    upload_receipt = json.loads(
        (evidence_root / "upload_success_20260826.json").read_text(encoding="utf-8")
    )

    train_hash = sha256(train_list)
    test_hash = sha256(test_list)
    checksum_manifest_hash = sha256(checksum_path)
    checksum_entries = read_checksum_manifest(checksum_path)
    train_rows = read_rows(train_list)
    test_rows = read_rows(test_list)

    membership: set[str] = set()
    train_rgb_missing = 0
    train_gt_missing = 0
    test_rgb_missing = 0
    test_gt_none = 0

    for row in train_rows:
        rgb = safe_relative(row[0]).as_posix()
        gt = safe_relative(row[1]).as_posix()
        membership.add(rgb)
        membership.add(f"data_depth_annotated/{gt}")
        train_rgb_missing += int(not (target_root / rgb).is_file())
        train_gt_missing += int(not (target_root / "data_depth_annotated" / gt).is_file())

    for row in test_rows:
        rgb = safe_relative(row[0]).as_posix()
        membership.add(rgb)
        test_rgb_missing += int(not (target_root / rgb).is_file())
        if row[1] == "None":
            test_gt_none += 1
        else:
            gt = safe_relative(row[1]).as_posix()
            membership.add(f"data_depth_annotated/{gt}")

    actual_files = sorted(path for path in target_root.rglob("*") if path.is_file())
    actual_relative = {path.relative_to(target_root).as_posix() for path in actual_files}
    actual_bytes = sum(path.stat().st_size for path in actual_files)
    expected_relative = set(checksum_entries)

    missing_manifest_files = sorted(expected_relative - actual_relative)
    unexpected_files = sorted(actual_relative - expected_relative)
    checksum_mismatches: list[str] = []
    for index, relative in enumerate(sorted(expected_relative), 1):
        path = target_root / relative
        if path.is_file() and sha256(path) != checksum_entries[relative]:
            checksum_mismatches.append(relative)
        if index % 1000 == 0 or index == len(expected_relative):
            print(f"[D-GATE-2] SHA256 {index}/{len(expected_relative)}", flush=True)

    top_level = sorted(path.name for path in target_root.iterdir())
    expected_top_level = sorted(expected_top_level_names)

    checks = {
        "target_mount_exists": target_root.is_dir(),
        "evidence_mount_exists": evidence_root.is_dir(),
        "target_top_level_exact": top_level == expected_top_level,
        "no_unexpected_metadata_file": not unexpected_files,
        "file_count_exact": len(actual_files) == EXPECTED_FILES,
        "byte_count_exact": actual_bytes == EXPECTED_BYTES,
        "checksum_manifest_hash": checksum_manifest_hash == FILES_MANIFEST_SHA256,
        "checksum_manifest_count": len(checksum_entries) == EXPECTED_FILES,
        "mounted_membership_exact": actual_relative == expected_relative,
        "all_sha256_match": not checksum_mismatches and not missing_manifest_files,
        "train_list_hash": train_hash == TRAIN_LIST_SHA256,
        "test_list_hash": test_hash == TEST_LIST_SHA256,
        "train_row_count": len(train_rows) == EXPECTED_TRAIN_ROWS,
        "test_row_count": len(test_rows) == EXPECTED_TEST_ROWS,
        "train_rgb_resolve": train_rgb_missing == 0,
        "train_gt_resolve": train_gt_missing == 0,
        "test_rgb_resolve": test_rgb_missing == 0,
        "expected_test_gt_none": test_gt_none == EXPECTED_TEST_GT_NONE,
        "official_membership_equals_manifest": membership == expected_relative,
        "dataset_manifest_lineage": (
            dataset_manifest.get("bts_commit") == OFFICIAL_COMMIT
            and dataset_manifest.get("materialized_files") == EXPECTED_FILES
            and dataset_manifest.get("materialized_bytes") == EXPECTED_BYTES
            and dataset_manifest.get("files_sha256_manifest_sha256") == FILES_MANIFEST_SHA256
        ),
        "materialization_report_lineage": (
            materialization_report.get("status") == "passed"
            and materialization_report.get("files_completed") == EXPECTED_FILES
            and materialization_report.get("bytes_completed") == EXPECTED_BYTES
            and materialization_report.get("source_destination_sha256_equal") is True
        ),
        "upload_receipt_lineage": (
            upload_receipt.get("dataset_id") == DATASET_ID
            and upload_receipt.get("dataset_status") == "ready"
            and upload_receipt.get("metadata_is_private") is True
            and upload_receipt.get("upload_gate") == "passed"
        ),
    }

    result = {
        "gate": "D_GATE_2",
        "status": "passed" if all(checks.values()) else "failed",
        "started_utc": started,
        "completed_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "target_mount": str(target_root),
        "evidence_mount": str(evidence_root),
        "input_inventory": input_inventory(),
        "checks": checks,
        "observed": {
            "files": len(actual_files),
            "bytes": actual_bytes,
            "manifest_entries": len(checksum_entries),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "test_gt_none": test_gt_none,
            "train_rgb_missing": train_rgb_missing,
            "train_gt_missing": train_gt_missing,
            "test_rgb_missing": test_rgb_missing,
            "missing_manifest_files_count": len(missing_manifest_files),
            "unexpected_files_count": len(unexpected_files),
            "checksum_mismatches_count": len(checksum_mismatches),
            "top_level": top_level,
        },
        "failures": [name for name, passed in checks.items() if not passed],
        "samples": {
            "missing_manifest_files": missing_manifest_files[:20],
            "unexpected_files": unexpected_files[:20],
            "checksum_mismatches": checksum_mismatches[:20],
        },
        "authorization_boundary": {
            "k0_k4_executed": False,
            "training_executed": False,
            "dataset_written": False,
        },
    }
    return result


if __name__ == "__main__":
    try:
        report = main()
    except Exception as exc:
        report = {
            "gate": "D_GATE_2",
            "status": "failed",
            "completed_utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "input_inventory": input_inventory(),
            "authorization_boundary": {
                "k0_k4_executed": False,
                "training_executed": False,
                "dataset_written": False,
            },
        }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)
