# Kaggle Private Quota Audit (Agent F)

Date: 2026-08-26
Mode: read-only
Decision: PASS

The initial audit was read-only. A later explicitly authorized remediation
deleted exactly two approved datasets. No dataset was made public or versioned,
no third dataset was deleted, and Agent E did not retry the KITTI upload.

## Quota summary

| Check | Result |
| --- | ---: |
| Datasets owned by `thudo25` | 41 |
| Private datasets | 35 |
| Public datasets | 6 |
| Current private versions listed by CLI | 182,022,797,270 bytes (182.02 GB) |
| Remaining private quota reported by failed create | 123.041 MB |
| Required free quota before retry | at least 30 GB |
| Account UI private quota | 182.53 GB used / 214.75 GB total |
| Measured private quota free | 32.22 GB |
| `F_QUOTA_GATE` | PASS |

## Remediation execution

| Gate | Dataset | Delete command | Independent absence checks | Result |
| --- | --- | --- | --- | --- |
| F1 | `thudo25/batch02-16-17` | reported success | status 403; owner list empty; metadata 403 | PASS |
| F2 | `thudo25/aic-batch2-k1-2-11-12` | reported success | status 403; owner list empty; metadata 403 | PASS |
| F3 | private storage quota | Profile > Account screenshot | 32.22 GB free | PASS |

Post-delete owner inventory:

| Check | Result |
| --- | ---: |
| Owned datasets | 39 |
| Private datasets | 33 |
| Current private versions | 150,112,748,980 bytes (150.113 GB) |
| Approved deleted refs still listed | 0 |

`kaggle quota` reports weekly GPU/TPU accelerator quota only. The authoritative
storage evidence is the user-provided Account UI screenshot showing 182.53 GB
used out of 214.75 GB, which yields 32.22 GB free. This exceeds the 30 GB gate.

The discrepancy between CLI-listed current-version sizes and quota usage is not
attributable to old versions from the available evidence. Kaggle staff explains
that dataset listings may show the compressed bundle archive size while private
quota is evaluated using the fully uncompressed dataset size. Therefore
compressed-versus-uncompressed accounting is a plausible primary explanation.
Old versions remain an item to inspect only if the Account UI still shows less
than 30 GB free; they are not the current leading explanation.

Sources:

- [Kaggle dataset technical specifications](https://www.kaggle.com/docs/datasets)
- [Kaggle staff: bundle size versus uncompressed quota](https://www.kaggle.com/product-feedback/462847)
- [Kaggle staff: quota usage is visible in Profile Account](https://www.kaggle.com/discussions/product-feedback/346049)

## Recommended cleanup set

The smallest currently verified cleanup set that safely exceeds the target is:

| Dataset | Current size | Local reference | Kaggle notebook dependency | Collaborators in metadata | Action |
| --- | ---: | --- | --- | --- | --- |
| `thudo25/batch02-16-17` | 18.70 GB | none found | none returned | none listed | DELETED AND VERIFIED ABSENT |
| `thudo25/aic-batch2-k1-2-11-12` | 13.21 GB | none found | none returned | none listed | DELETED AND VERIFIED ABSENT |
| **Combined** | **31.91 GB** | | | | remediation complete |

Using the 123.041 MB free-quota observation, deleting both current datasets is
projected to leave about 32.03 GB free. This remains a projection because the
post-delete uncompressed quota usage has not been measured in the Account UI.

`thudo25/aic2025-keyframe-batch2` is 61.01 GB and has no notebook or local
reference, but its metadata lists four writers: `chaudo15`, `dochau15`,
`thuvia01`, and `tuyetlan05`. It is therefore only `POSSIBLE CLEANUP` and should
not be the first deletion choice without collaborator confirmation.

## Private dataset inventory

Sizes below are current-version sizes returned by Kaggle CLI, expressed as
decimal GB. `Remote referenced` means `kaggle kernels list -m --dataset ...`
returned at least one notebook owned by `thudo25`.

| Dataset | Size | Dependency evidence | Classification |
| --- | ---: | --- | --- |
| `aic2025-keyframe-batch2` | 61.01 GB | four collaborators; no notebook/local reference | POSSIBLE CLEANUP |
| `aic-video-8910` | 35.85 GB | remote notebook reference | KEEP |
| `batch02-16-17` | 18.70 GB | deleted; independently verified absent | DELETED |
| `aic-batch2-k1-2-11-12` | 13.21 GB | deleted; independently verified absent | DELETED |
| `transnet-keyframe-test` | 12.70 GB | five remote notebook references | KEEP |
| `batch02-data-wav-02` | 10.04 GB | remote notebook reference | KEEP |
| `batch02-data-wav-01` | 9.18 GB | two remote notebook references | KEEP |
| `microsoft-florence-2-large` | 5.76 GB | remote notebook reference | KEEP |
| `mot-20` | 5.03 GB | four remote notebook references | KEEP |
| `nasdaq-external-data-2015-2023` | 3.81 GB | no notebook/local reference | POSSIBLE CLEANUP |
| `data2-nlp` | 3.81 GB | no notebook/local reference | POSSIBLE CLEANUP |
| `mot17-yolo` | 0.88 GB | not required by BTS | POSSIBLE CLEANUP |
| `bts-nyuv2-checkpoint` | 0.50 GB | BTS config, training and inference references | KEEP |
| `dpt-hybrid-midas-checkpoint` | 0.46 GB | Project 1 DPT artifact | KEEP |
| `official-midas-dpt-hybrid-general-checkpoint` | 0.46 GB | Project 1 DPT artifact | KEEP |
| `best-pt` | 0.24 GB | not required by BTS | POSSIBLE CLEANUP |
| `tach-dl-natural` | 0.18 GB | not required by BTS | POSSIBLE CLEANUP |
| `out-150-3-val-labels` | 0.06 GB | not required by BTS | POSSIBLE CLEANUP |
| `video-test` | 0.05 GB | not required by BTS | POSSIBLE CLEANUP |
| `model-onnx` | 0.03 GB | not required by BTS | POSSIBLE CLEANUP |
| `out-llm` | 0.03 GB | not required by BTS | POSSIBLE CLEANUP |
| `best-labels-250-pt` | 0.02 GB | not required by BTS | POSSIBLE CLEANUP |
| `best-250` | 0.02 GB | not required by BTS | POSSIBLE CLEANUP |
| `yolo-bot-sort` | <0.01 GB | not required by BTS | POSSIBLE CLEANUP |
| `financial-sentiment-analysis-nasdaq-exteral-15-23` | <0.01 GB | not required by BTS | POSSIBLE CLEANUP |
| `alllll` | <0.01 GB | not required by BTS | POSSIBLE CLEANUP |
| `kitti-dpt-bts-manifests` | <0.01 GB | Project 1 KITTI manifest artifact | KEEP |
| `results-stage01` | <0.01 GB | unknown historical result | POSSIBLE CLEANUP |
| `results-stage04` | <0.01 GB | unknown historical result | POSSIBLE CLEANUP |
| `testtttt` | <0.01 GB | unknown test artifact | POSSIBLE CLEANUP |
| `dpt-hybrid-simple-train` | <0.01 GB | Project 1 DPT artifact | KEEP |
| `batch02-02-aic-detection-audio` | <0.01 GB | AIC artifact | POSSIBLE CLEANUP |
| `batch02-01-aic-detection-audio` | <0.01 GB | AIC artifact | POSSIBLE CLEANUP |
| `ua-detrac` | <0.01 GB | not required by BTS | POSSIBLE CLEANUP |
| `results-stage1-json` | <0.01 GB | unknown historical result | POSSIBLE CLEANUP |

The six public datasets do not consume private quota and are excluded from
cleanup recommendations. KITTI must remain private; making it public is not an
approved remediation.

## UI quota gate

Read the current value at:

```text
Kaggle -> Profile -> Account -> Quotas / Private Datasets
```

Record either `used / 200 GB` or the directly displayed remaining/free value in
`data_manifests/kitti/kaggle/quota_ui_evidence.json`.

```text
PRIVATE_FREE_UI >= 30 GB
    YES -> F_QUOTA_GATE = PASS; Agent E unlocked
    NO  -> F_QUOTA_GATE = FAIL; stop and audit further
```

Do not infer free quota as `200 GB - CLI inventory`, because the CLI inventory
may represent compressed bundle sizes.

## Version-history fallback

Only if the Account UI still shows less than 30 GB free, inspect large private
datasets in Kaggle UI:

```text
Dataset -> Settings -> Only keep latest version
```

Apply this only when historical versions are not required. Do not create a new
version merely to trigger cleanup because that may require temporary headroom.

## Approval boundary

Agent F ran exactly the two explicitly authorized destructive commands:

```text
kaggle datasets delete thudo25/batch02-16-17
kaggle datasets delete thudo25/aic-batch2-k1-2-11-12
```

Both deletions and the Account UI measurement are complete.
`PRIVATE_FREE_UI = 32.22 GB`, so `F_QUOTA_GATE = PASS` and upload authority is
returned to Agent E. This does not unlock Gate-2, K0-K4 or training.

Agent E must reuse:

```text
C:\Users\GIGA\bts-kitti-eigen-materialized-v1
```

Before retry, rerun membership, exact-byte and SHA256 verification. Upload is
successful only when all of these are true:

1. The CLI command completes without a semantic error in stdout/stderr.
2. Dataset status is `READY` or `COMPLETE`.
3. The dataset appears in the owner's dataset list.
4. Downloaded metadata confirms `isPrivate: true`.

CLI exit code 0 alone is not proof of success. Gate-2 remains locked until the
Kaggle-mounted dataset independently passes file, byte, path and protocol checks.

## Subsequent owner-authorized cleanup and upload

After the original quota remediation passed, the account owner explicitly
authorized deletion of every Kaggle dataset unrelated to Project 1. The cleanup
deleted 32 datasets and independently verified that the post-delete owner
inventory contained exactly the seven approved Project 1 references. No Project
1 dataset was deleted and no dataset was converted to public. Evidence is stored
in `data_manifests/kitti/kaggle/dataset_cleanup_20260826.json`.

Agent E then reused the existing local KITTI materialization. The private dataset
`thudo25/bts-kitti-eigen-materialized-v1` is now `ready`, appears in the owner
inventory, and its downloaded server metadata reports `isPrivate: true`.
Therefore `E_PRIVATE_UPLOAD = PASS` and Gate-2 was unlocked for independent
review. That subsequent review passed mounted content. Its initial runtime path
failure was remediated and the focused recheck passed, so K0-K4 are unlocked in
order; full training remains locked until K4 passes.
