# Patches

The workspace root is not a Git repository, so no reproduction branch or patch
commit exists. The canonical official mirror remains clean and unmodified.

- Highest risk: training resume/checkpoint runtime behavior.
- Changed artifact: `patches/kaggle_runtime_adaptations.patch`.
- Retention artifact: `patches/kaggle_checkpoint_retention.patch` at SHA256
  `21a94527dae9bf7971bd334a083b2e75f2396d9159f293004a8376be824757c8`.
- Config overlay: `adaptations/bts_runtime/config/arguments_train_eigen_kaggle.txt`.
- Rationale: modern PyTorch/Kaggle compatibility and auditable smoke/resume.
- Verification: forward/reverse patch checks, source/build SHA equality, syntax
  compile, local smoke, protocol audit, retention tests and Kaggle runtime-v2
  production checkpoint/resume preflight passed.
- README fidelity: no scientific protocol field changed; full-result fidelity
  remains unverified until training/evaluation completes.
