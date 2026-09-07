# BTS KITTI Full Run 02 Freeze v4

**Frozen:** 2026-08-27T07:43:11.5201407Z

Run 02 is a fresh BTS KITTI reproduction run on two effective Tesla T4 GPUs. It starts at global step 0 with no checkpoint. Run 01 remains preserved as historical 1xT4 evidence and no Run 01 checkpoint may enter this lineage.

The official config batch remains 4. BTS DDP divides this across two processes, producing local batch 2 per GPU and global batch 4. The audited geometry remains 5,790 steps per epoch and 289,500 total steps over 50 epochs.

Local protocol audit and Kaggle two-rank DDP produce/resume preflight both pass. The preflight verified two physical/effective T4 GPUs, the six-key production checkpoint schema, 227 optimizer states, optimizer groups 482/68 and step 0 to step 1 resume. Its temporary checkpoint was pruned and is prohibited as a full-run initial checkpoint.

`full_run_manifest.v4.json` and `checkpoint_retention_plan.v4.json` are immutable after receipt creation. Any hash-affecting runtime, protocol, dataset or topology change requires a new freeze before training.
