# BTS KITTI Run 02 Session 1 Attempt 1 Live Audit

Training itself is healthy: both T4 ranks run, losses and learning rates remain finite, evaluation executes every 500 steps, and `abs_rel` improved from 0.123 at step 500 to 0.08416 at step 4500. No traceback, CUDA OOM, NaN-loss or NCCL failure appears in the captured UTF-8 live log.

The attempt fails orchestration acceptance. The launcher watched a `...run02` model directory while the frozen config writes `bts_kitti_densenet161_kaggle`. It therefore missed the verified step-500 recovery emission and did not perform the controlled stop, copy, schema verification or private handoff. Live Kaggle output files are not downloadable, so observed recovery checkpoints cannot be authorized for resume.

Attempt 1 must be stopped and preserved as non-resumable live evidence. The corrected launcher asserts the frozen model name and parses the emitted recovery path fail-closed. Version 2 must start fresh from step 0 under the unchanged Freeze v4 contract.
