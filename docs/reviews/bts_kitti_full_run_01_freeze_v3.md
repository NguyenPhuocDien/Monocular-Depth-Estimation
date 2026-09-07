# BTS KITTI Full Run 01 Freeze v3

**Status:** FROZEN CORRECTION OVERLAY

Freeze v2 remains immutable historical evidence. Freeze v3 applies only to the
exact v2 manifest and retention hashes and corrects one invalid resume
assertion discovered from the first production checkpoint.

## Finding

- Session 1 passed startup audit and completed global steps 0 through 500.
- Step-500 online evaluation produced `abs_rel = 0.11019`.
- Recovery SHA256 is `35a36b5f7b4d972b34b6ef850237174f3c1b8d3d65fdc96f4e5918cf6f3ae579`.
- The checkpoint has all six production keys and metric slots `3/6/9`.
- AdamW contains 550 parameters in groups `482 + 68`.
- Production `set_misc()` freezes 323 parameter tensors, leaving 227 trainable
  tensors and therefore 227 materialized AdamW state entries.
- The old 550-state preflight omitted `set_misc()` and conflated optimizer
  parameter count with optimizer state count.

## Verification

The private handoff roundtrip matches the local recovery checkpoint byte for
byte. An independent Kaggle one-step resume preflight loaded global step 500,
strict-loaded the model, restored 227 optimizer states over 550 grouped
parameters, resumed at step 501, produced finite loss and updated a decoder
parameter without changing the checkpoint hash.

No BTS algorithm, dataset, runtime source, config, GPU topology, batch size,
optimizer, scheduler, evaluation setting or retention behavior changed.

## Gate

`SESSION_1 = COMPLETED_AT_VERIFIED_STEP_500_BOUNDARY`

`SESSION_2 = AUTHORIZED_NOT_STARTED`

`FULL_TRAINING = IN_PROGRESS`

`REPRODUCTION_COMPLETE = false`
