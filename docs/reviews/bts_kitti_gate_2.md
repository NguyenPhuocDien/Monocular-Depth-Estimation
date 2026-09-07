# BTS KITTI Gate-2 Review (Agent D)

Date: 2026-08-26
Decision: **PASS**

The Kaggle-mounted dataset content is byte-identical and its lineage passes.
The allowlisted runtime mount-path mismatch found in the initial review was
remediated and independently rechecked by kernel version 4.

## Dataset identity

| Check | Result |
| --- | --- |
| Dataset ID | `thudo25/bts-kitti-eigen-materialized-v1` |
| Dataset status | `ready` |
| Present in owner inventory | PASS |
| Server metadata `isPrivate` | `true` |

## Mounted content audit

Kernel `thudo25/bts-kitti-gate-2-mounted-audit`, version 2, completed the
read-only audit. The authoritative output is
`artifacts/kaggle/bts_kitti_gate2/run02/gate2_mounted_audit.json`.

| Check | Observed | Result |
| --- | ---: | --- |
| Mounted files | 47,665 | PASS |
| Mounted bytes | 23,695,187,138 | PASS |
| Checksum manifest entries | 47,665 | PASS |
| SHA256 mismatches | 0 | PASS |
| Missing manifest files | 0 | PASS |
| Unexpected files or temporary metadata | 0 | PASS |
| Train RGB resolved | 23,158 | PASS |
| Train GT resolved | 23,158 | PASS |
| Test RGB resolved | 697 | PASS |
| Expected test GT `None` | 45 | PASS |
| Official membership equals manifest | exact | PASS |
| Upload/materialization/manifest lineage | consistent | PASS |

The mounted hierarchy is exactly:

```text
/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1
```

The first kernel version failed before reading data because it assumed the old
short mount path. Version 2 discovered the mount by content signature and then
completed the full checksum review. The first failure is retained in `run01` as
harness evidence and is not treated as a dataset mismatch.

## Protocol invariants

| Check | Result |
| --- | --- |
| Official mirror clean at `5e3406b35d1497b2e55d2dd600524d1f4efacaed` | PASS |
| Train/test list SHA256 unchanged | PASS |
| Runtime patch/config hashes match provenance | PASS |
| Config differences limited to allowlisted runtime fields | PASS |
| Configured dataset path resolves on audited mount | PASS |
| Runtime smoke after review | PASS |

Current config:

```text
/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1/
```

Observed mount:

```text
/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1/
```

Only `data_path`, `gt_path`, `data_path_eval` and `gt_path_eval` changed. The
new config SHA256 is
`9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7`.
Kernel version 4 confirmed all four configured directories exist, all official
train/test references resolve, list hashes are unchanged, and config/provenance
hashes match. The authoritative focused output is
`artifacts/kaggle/bts_kitti_gate2/run04/gate2_path_recheck.json`.

## Gate boundary

```text
D_MOUNTED_CONTENT = PASS
D_PROVENANCE       = PASS
D_PATH_COMPAT      = PASS
D_GATE_2           = PASS
K0-K4              = UNLOCKED_IN_ORDER
FULL_TRAINING      = LOCKED
```

No K0-K4 step or training was executed during Gate-2. K0 may now begin, followed
strictly by K1, K2, K3 and K4. Full training remains locked until K4 passes.
