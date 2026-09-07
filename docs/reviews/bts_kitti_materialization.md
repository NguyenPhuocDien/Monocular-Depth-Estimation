# BTS KITTI Materialization Review (Agent E)

Date: 2026-08-26
Local materialization decision: PASS
Private Kaggle upload decision: PASS

## Local materialization

| Check | Result |
| --- | --- |
| Source used read-only | PASS |
| Official Eigen lists only | PASS |
| Relative hierarchy preserved | PASS |
| Files copied | 47,665 |
| Exact bytes copied | 23,695,187,138 |
| Source SHA256 equals destination SHA256 | PASS |
| Unexpected missing train RGB/GT | 0 / 0 |
| Unexpected missing test RGB | 0 |
| Expected test GT `None` entries | 45 |
| Independent file count and byte-count audit | PASS |

Materialized root:

```text
C:\Users\GIGA\bts-kitti-eigen-materialized-v1
```

Evidence and Kaggle metadata are stored under `data_manifests/kitti/`. The
temporary `dataset-metadata.json` control file was removed from the dataset root
after dataset creation, restoring the root to exactly 47,665 files.

## Private Kaggle upload

The initial creation attempt was rejected because of private quota and remains
recorded in `data_manifests/kitti/kaggle/upload_attempt_20260826.json`. After
the authorized account cleanup, Agent E reused the existing verified local
materialization and uploaded the same six directory archives.

Final verification:

| Check | Result |
| --- | --- |
| Create response has no semantic error | PASS |
| Dataset status | `ready` |
| Present in owner inventory | PASS |
| Server metadata `isPrivate` | `true` |
| Dataset ID | `thudo25/bts-kitti-eigen-materialized-v1` |

The successful receipt is stored at
`data_manifests/kitti/kaggle/upload_success_20260826.json`. The upload process
timed out after five archives, but Kaggle's resumable endpoint completed the
sixth archive; no local rematerialization was performed.

## Gate boundary

`E_LOCAL_MATERIALIZATION = PASS`

`E_PRIVATE_UPLOAD = PASS`

`D_GATE_2 = PASS`

The independent mounted-content audit passed file count, byte count, membership
and SHA256. The runtime mount-path mismatch was remediated through four
allowlisted config fields and the focused path recheck passed. K0-K4 are
unlocked in order; full training remains unauthorized until K4 passes.
