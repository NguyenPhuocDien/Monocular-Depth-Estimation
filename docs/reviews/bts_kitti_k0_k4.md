# BTS KITTI Kaggle K0-K4 Review

**Date:** August 26, 2026  
**Dataset:** `thudo25/bts-kitti-eigen-materialized-v1`  
**Official source:** `cleinc/bts@5e3406b`  
**Overall:** `K0_K4 = PASS`

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| K0 mount/path | PASS | Exact mount, config/list hashes, 23,158 train rows, 697 test rows and 45 expected test GT `None` entries |
| K1 dataloader | PASS | One real train batch: image `[1,3,352,704]`, depth `[1,1,352,704]`, finite tensors and focal |
| K2 forward/backward | PASS | 47,000,688 parameters, five correct outputs, finite silog loss and 550 finite gradient tensors |
| K3 tiny train | PASS | Three AdamW steps with official optimizer fields; finite losses and verified decoder parameter update |
| K4 checkpoint/resume | PASS | Two distinct processes; exact model load, 550 optimizer states restored and global step resumed from 2 to 3 |

## K4 Checkpoint Evidence

```text
path    artifacts/kaggle/bts_kitti_k4/run01/k4_checkpoint_step_2.pth
bytes   565,856,097
sha256  68872d9bcae3512ddc8ba8794fb746ca8fd4872064cff1e1f0bf69d990e1e4db
schema  global_step, model, optimizer
```

The checkpoint is K4 gate evidence only. It is not a full-training checkpoint
and must not be used to claim KITTI reproduction metrics.

## Authorization Boundary

K0-K4 did not execute a full epoch or evaluation run. Full 50-epoch training is
now unlocked by the gate sequence but has not started. A separate full-run
manifest, checkpoint lineage and evaluation audit are still required.
