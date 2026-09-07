# Scientific Changelog

No architecture, loss, dataset membership, split, crop, augmentation, metric or
full-training hyperparameter was changed.

Runtime-only changes are explicit: modern PyTorch checkpoint loading,
cross-platform filesystem operations, optional smoke termination, checkpoint/
resume support and logging. The NYUv2-only batch assertion was removed because
it contradicted official KITTI `batch_size=4`.

Runtime v2 adds fail-closed one-GPU/batch-4 geometry guards and checkpoint
retention side effects only. Kaggle's physical `2 x T4` allocation is isolated
to effective GPU 0 with `CUDA_VISIBLE_DEVICES=0`, preserving global batch 4.
The runtime dataset vendors hashed `tensorboardX 2.6.5` solely to satisfy the
official logging import; this does not alter model, optimization or evaluation.

The GT package hierarchy was normalized to the direct `data_depth_annotated`
root expected by BTS. Files were moved without rename or re-encoding.

Freeze v3 corrects evidence interpretation only. The earlier preflight omitted
production `set_misc()` and therefore observed AdamW state for all 550 optimizer
parameters. Production freezes 323 parameter tensors, leaving 227 trainable
tensors and 227 lazily materialized AdamW state entries. Runtime code, optimizer
groups, learning rate, scheduler, model, data and evaluation protocol are
unchanged.
