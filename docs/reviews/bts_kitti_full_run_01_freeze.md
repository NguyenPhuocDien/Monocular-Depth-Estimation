# BTS KITTI Full Run 01 Freeze

**Frozen:** August 26, 2026  
**Run ID:** `bts_kitti_densenet161_full_run_01`  
**Manifest:** `FROZEN`  
**Retention plan:** `FROZEN`  
**Full training:** `AUTHORIZED_NOT_STARTED`

## Frozen Execution

The run uses official BTS commit `5e3406b`, the audited runtime patch/config,
the private byte-identical KITTI dataset and the pinned DenseNet-161 ImageNet
weight. The execution surface is one Tesla T4, local/global batch size 4,
23,158 train rows, 5,790 steps per epoch and 289,500 total steps.

Protocol remains 50 epochs at `352x704`, max depth 80, AdamW, initial learning
rate `1e-4`, polynomial power `0.9`, effective end learning rate `1e-5`, KITTI
benchmark crop, random rotation of 1 degree, Garg evaluation crop and online
evaluation every 500 global steps.

`abs_rel` minimizing is frozen as the primary scientific checkpoint criterion.
All nine metrics tracked by the runtime remain mandatory reporting outputs.

## Retention

Always retain latest, previous, best `abs_rel` and final. Also retain epochs 5,
10, 20, 30, 40 and 50 plus the final checkpoint from every Kaggle session until
the audit closes. Artifacts sharing the same global step and SHA256 are stored
once.

Every handoff must pass save, load/schema validation, SHA256, private upload and
remote verification before any older checkpoint is pruned. The K4 checkpoint is
gate evidence and is prohibited as the initial checkpoint for this run.

## Start Gate

Session 1 must abort before its first optimizer step if runtime geometry or
environment differs from the manifest. Exact Python, torchvision, CUDA and
Kaggle image identifiers must be captured at startup because the stored K0-K4
reports did not expose all exact remote image fields.

The training harness must also support explicit milestone/final emission and a
versioned latest/previous handoff before Session 1 starts. Freezing these
artifacts authorizes training but does not claim that training has begun.
