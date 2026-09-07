# BTS KITTI Full Run 01 Freeze v2

**Frozen UTC:** 2026-08-27T02:56:16.606662+00:00  
**Decision:** `SESSION_1 = AUTHORIZED_NOT_STARTED`

## Acceptance

- Official source clean at `5e3406b35d1497b2e55d2dd600524d1f4efacaed`.
- Canonical retention patch SHA256: `21a94527dae9bf7971bd334a083b2e75f2396d9159f293004a8376be824757c8`.
- Adaptation/generated `bts_main.py` SHA256: `c0b4703fa85ca1edb2ddb9103a5fa3fb34c906de17004bd9d42d0ece3bf0599d`.
- Runtime config SHA256: `d7dc6b5cccd75757e3c5478b7375078267b65bba085591789be78c3d10be4e81`.
- Local smoke, protocol audit, retention tests, handoff roundtrip and production schema/resume PASS.
- Kaggle runtime-v2 preflight PASS with effective `1 x Tesla T4`, batch 4, 5,790 steps/epoch and 289,500 total steps.
- Kaggle checkpoint preflight used the six-key production schema, restored 550 optimizer states and resumed in a fresh process.

## Environment

Kaggle allocated two physical T4 GPUs. The run topology is intentionally isolated to GPU 0 by setting `CUDA_VISIBLE_DEVICES=0` before importing torch; BTS therefore sees exactly one T4 and retains global batch 4. The runtime dataset vendors hashed `tensorboardX 2.6.5` because the Kaggle image does not provide it.

## Boundary

Freeze v1 remains immutable historical evidence. K4 and preflight checkpoints are prohibited as full-run initial or scientific checkpoints. No full-training optimizer step has started under this freeze.
