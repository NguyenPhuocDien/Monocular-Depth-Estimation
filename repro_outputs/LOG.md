# Reproduction Log

- Verified clean official mirror at full SHA
  `5e3406b35d1497b2e55d2dd600524d1f4efacaed`.
- Rebuilt runtime only from official mirror, explicit patch and hashed config
  overlay.
- Extracted 67/67 official KITTI archives; retained archives.
- Normalized official GT train/val drive layout without modifying file bytes.
- B audit: train RGB/GT missing 0/0; test RGB missing 0; test GT has 45
  official `None` entries.
- Private Kaggle environment probe version 1 completed and outputs were pulled.
- Local isolated smoke passed imports, parser, model, dataloader and forward.
- Independent protocol and Gate-1 checks passed.
- Rebuilt runtime v2 with canonical retention patch SHA256 `21a94527...` and
  exact adaptation/generated `bts_main.py` SHA equality.
- Verified K4 private handoff upload/download roundtrip and production six-key
  checkpoint schema with 550 restored optimizer states.
- Kaggle runtime-v2 preflight version 3 passed environment, mount, batch
  geometry, checkpoint emitter, strict load and fresh-process resume checks.
- Frozen full-run manifest and retention plan v2, then launched Session 1 from
  no initial checkpoint. Attempt 1 failed before step 0 due launcher cwd;
  attempt 2 completed steps 0-500 and produced `abs_rel=0.11019`.
- Verified the step-500 six-key recovery checkpoint locally and by private
  Kaggle handoff roundtrip at SHA256 `35a36b5f...`.
- Corrected the v2 resume assertion after production evidence showed 550
  optimizer parameters but 227 materialized AdamW states under `set_misc()`.
- Kaggle step-500 resume preflight strict-loaded the model, restored 227 states,
  resumed at step 501 and updated a decoder parameter with finite loss.
- Preserved Freeze v2 and froze a v3 correction overlay. Its private Kaggle
  dataset is READY and roundtrip verified.
- Launched Session 2 kernel version 1 from the exact step-500 checkpoint toward
  boundary step 25,000. Kernel is running; step-501 output evidence is pending.
