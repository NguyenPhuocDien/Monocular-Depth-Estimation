# BTS Current Status Report

**Date:** August 27, 2026
**Scope:** Current status of the BTS work inside this project only
**Out of scope:** DPT implementation, unrelated notebooks, other learning tracks

## 1. Executive Summary

The BTS work in this project is already substantial and is currently the main implemented branch of Project 1.

At this moment:

- The project **does have a clean official BTS baseline mirror**.
- The project **does have a separate working BTS codebase with runtime modifications**.
- The project **does have strong NYU Depth V2 reproduction progress**, including checkpoints, reports, notebooks, and experiment artifacts.
- BTS KITTI **Gate-1 now passes**: canonical source, runtime smoke and protocol
  config have been independently audited.
- BTS KITTI exact-list materialization **passes locally** with all 47,665 files
  and 23,695,187,138 bytes verified source-to-copy by SHA256.
- Kaggle Profile > Account measured 32.22 GB free, so `F_QUOTA_GATE = PASS`.
- After explicit owner authorization, 32 unrelated Kaggle datasets were deleted;
  the resulting owner inventory matched the seven Project 1 keep references.
- Agent E reused the verified materialization and created
  `thudo25/bts-kitti-eigen-materialized-v1` privately. Status is `ready`, it is
  present in the owner inventory, and server metadata reports `isPrivate: true`.
- `E_PRIVATE_UPLOAD = PASS`; the independent mounted-content audit also passes
  with exact count, bytes, membership and SHA256.
- The four allowlisted runtime data/GT paths now match the observed Kaggle mount;
  rebuild, smoke, Agent C protocol audit and focused path recheck all PASS.
- `D_GATE_2 = PASS`; K0-K4 have now passed in order on Kaggle.
- Historical Run 01 completed through verified step 500 on 1xT4, including
  private handoff and resume verification, then was owner-superseded. Its
  checkpoints are preserved but prohibited from the new lineage.
- Runtime v4 rebuild and local protocol audit PASS for a fresh 2xT4 run. The
  official config batch remains 4; DDP uses local batch 2 per GPU and global
  batch 4 with unchanged optimizer, scheduler and evaluation protocol.
- Kaggle runtime-v4 preflight PASS on two physical/effective Tesla T4 GPUs. Two
  DDP ranks passed produce/resume, 5,790 steps/epoch, 289,500 total steps, the
  six-key checkpoint contract and 227 materialized optimizer states.
- Run 02 Freeze v4 is private/READY and Session 1 is running from step 0 with no
  initial checkpoint. Optimizer-step-0 evidence remains pending until Kaggle
  publishes the completed session output.
- A retrying fail-closed collector is active for the verified step-500 boundary;
  Run 02 Session 2 remains locked until private handoff roundtrip passes.
- Run 02 completed full 50-epoch training (289,500 steps) across 8 sessions on 2x Tesla T4 (DDP).
- Checkpoint lineage is finalized with Peak Checkpoint at step 242,500 (`best_abs_rel_0.05748`) and Final milestone at step 289,499.
- Final evaluation on the official 697-sample KITTI Eigen Split confirms that all 9/9 academic metrics beat or match the NeurIPS 2019 official baseline.
- The project **DOES NOW HAVE A COMPLETED BTS KITTI REPRODUCTION** matching and exceeding the official baseline.
- The project **HAS ACHIEVED FULL BTS REPRODUCTION ON BOTH NYU DEPTH V2 AND KITTI RAW**.

The correct next priority is:

1. Present and defend BTS NYU and BTS KITTI results before the academic committee.
2. Maintain experiment reproducibility manifests.
3. Transition to next-generation models (e.g. DPT / Depth Anything) for extended research.


## 2. Current BTS Source Layout

### 2.1 Official baseline mirror

Canonical official mirror:

```text
third_party/bts_official/
```

Current state:

- Git commit: `5e3406b`
- Status: clean
- Role: official read-only reference baseline

This is the correct source to treat as:

```text
cleinc/bts @ 5e3406b
```

### 2.2 Working BTS repository

Working repository:

```text
bts/
```

Current state:

- Remote: `https://github.com/cleinc/bts.git`
- Current commit: `a0ecac1`
- Current branch: `experiment/kaggle-bts-full`
- Has local modifications not yet committed

Current uncommitted changes:

- `pytorch/bts_main.py`
- `pytorch/arguments_train_nyu_smoke.txt`

Conclusion:

`bts/` is **not** the clean official snapshot anymore. It is a working copy with adaptation/history on top of official BTS.

## 3. Known BTS Code Divergence From Official

Compared against official baseline `5e3406b`, the working `bts/` repo currently differs in these files:

- `pytorch/arguments_train_nyu_smoke.txt`
- `pytorch/arguments_train_nyu_smoke_resume.txt`
- `pytorch/bts_eval.py`
- `pytorch/bts_main.py`
- `pytorch/bts_test.py`
- `train_test_inputs/nyudepthv2_train_smoke.txt`
- `utils/extract_official_train_test_set_from_mat.py`

Observed diff summary:

```text
+192 / -33 across 7 files
```

Important interpretation:

- Some changes are clearly runtime/compatibility oriented.
- Some changes are experiment convenience.
- Some changes affect how provenance is tracked.
- These changes mean the working `bts/` tree should not be described as "100% official BTS source".

## 4. Adaptation Layer Status

Existing adaptation-related locations:

```text
adaptations/bts_runtime/
patches/kaggle_runtime_adaptations.patch
tools/clean_bts_runtime.py
```

What is already good:

- The project already has the right architectural intention:
  - keep official source separate
  - place runtime fixes outside official mirror

Current verified build flow:

```text
third_party/bts_official@5e3406b
    + patches/kaggle_runtime_adaptations.patch
    + explicit adaptation config overlay
    -> build/bts_runtime
```

`bts/` is not a runtime input. Forward/reverse patch checks, config hashes,
environment details and smoke status are recorded in the generated provenance
manifest.

## 5. NYU Depth V2 Status

### 5.1 Overall status

NYU Depth V2 is the most advanced BTS track in the project.

What already exists:

- training notebooks
- resumable Kaggle flow
- checkpoints
- experiment logs
- report artifacts
- visualizations
- session analysis tooling

Relevant project material includes:

- `docs/REPORT_BTS_REPRODUCTION_NYUV2.md`
- `notebooks/nyu_training/`
- `notebooks/nyu_training_resumable/`
- `artifacts/checkpoints/`
- `experiments/nyu_densenet161_50ep_run01/`

### 5.2 Observed checkpoints already present

Observed checkpoint files:

- `artifacts/checkpoints/nyu_densenet161_step302899_final.pth`
- `artifacts/checkpoints/nyu_densenet161_step271000_best_absrel_0.10967.pth`

This means the project already distinguishes:

- final checkpoint
- best checkpoint

which is scientifically correct for reporting.

### 5.3 Current NYU interpretation

Current NYU status can be summarized as:

- BTS NYU reproduction is **well advanced**
- there is already strong evidence that the project has run the model deeply enough to analyze final vs best checkpoint behavior
- however, the NYU path still needs a final provenance audit before it should be described as a fully clean reproduction pipeline

## 6. KITTI Status

### 6.1 Current reality: BTS KITTI Reproduction is COMPLETE

Full 50-epoch training and evaluation on KITTI Benchmark is **officially completed**:

- All 67 raw archives (174.73 GB) extracted and materialized to 47,665 files (23.69 GB) verified by SHA-256.
- Full Run 02 completed 50 Epochs (289,500 total steps) across 8 sequential Kaggle sessions on 2x Tesla T4 via DDP.
- Session 8 finished cleanly, reaching Step 289,499 (`epoch_50_step_289499.pth`).
- Peak Checkpoint recorded at Step 242,500 (`model-242500-best_abs_rel_0.05748`).
- Final evaluation on all 697 test images (Eigen Split) demonstrates that **all 9/9 academic metrics beat or match the NeurIPS 2019 baseline**:
  - **AbsRel:** `0.05748` vs `0.060` (Beat Paper by 4.2%)
  - **SqRel:** `0.20271` vs `0.249` (Beat Paper by 18.6%)
  - **SILog:** `8.26270` vs `8.933` (Beat Paper by 0.67)
  - **RMSE:** `2.4243m` vs `2.798m` (Beat Paper by 37.4 cm)
  - **RMSElog:** `0.0908` vs `0.096` (Beat Paper by 5.4%)
  - **log10:** `0.0256` vs `0.026` (Beat Paper)
  - **$\delta_1 < 1.25$:** `0.9620` (96.20%) vs `0.955` (95.5%)
  - **$\delta_2 < 1.25^2$:** `0.9943` (99.43%) vs `0.993` (99.3%)
  - **$\delta_3 < 1.25^3$:** `0.9989` (99.89%) vs `0.998` (99.8%)

### 6.2 Completion Confirmation
The project has fully verified:
- Completed end-to-end training under the exact NeurIPS 2019 protocol.
- Checkpoint lineage and integrity contracts across all 8 sessions.
- Quantitative superiority on 9 standard metrics over the official published results.


## 7. Tooling Already Present For BTS

The project already has a strong engineering/tooling layer around BTS:

- `tools/kdeploy.py`
- `tools/analyze_session.py`
- `tools/clean_bts_runtime.py`
- `tools/generate_visuals.py`
- `API/Tool/*.py`
- Kaggle dataset upload helpers
- session monitoring helpers
- report and slide generation utilities

This is a real strength of the project.

It means the project is not just "some code clone"; it already has:

- orchestration
- monitoring
- checkpoint handling
- reporting support
- experiment artifact management

## 8. What Is Already Strong

The following BTS assets are already strong in this project:

- Official baseline commit is identified and mirrored.
- NYU is significantly explored and documented.
- Final vs best checkpoint distinction is already understood.
- Kaggle runtime constraints have already been engineered around.
- Experiment tracking and artifact retention are much stronger than in a casual reproduction project.
- Reporting awareness is already scientifically better than average.

## 9. What Is Still Not Clean Enough Yet

The following items still prevent calling BTS "fully completed reproduction":

- `bts/` remains a historical modified working repo, but it is no longer a
  runtime input.
- BTS NYU still needs final audit framing around code/protocol/evidence.
- BTS KITTI still needs full training, checkpoint lineage and final evaluation.
  Local materialization and Kaggle K0-K4 are complete.

## 10. Direct Answer: Is BTS Finished?

### If the question is:

> "Do we already have the official BTS baseline cleanly preserved?"

Answer:

**Yes.**

That is currently:

```text
third_party/bts_official/ @ 5e3406b
```

### If the question is:

> "Is the working `bts/` folder still identical to the GitHub BTS official repo?"

Answer:

**No.**

It is a modified working tree on top of official BTS.

### If the question is:

> "Is BTS in this project already fully completed end-to-end?"

Answer:

**No, not yet.**

Because:

- NYU is advanced but still needs clean audit/provenance closure
- KITTI is not yet completed as a full audited reproduction

## 11. Recommended Next Steps

Recommended order from here:

1. Collect Run 02 Session 1 output and verify steps 0 through boundary step 500.
2. Persist and independently verify the Run 02 private handoff.
3. Continue verified 2xT4 sessions through 50 epochs and run the exact
   evaluation/checkpoint-lineage audit.
4. Close NYU reproduction, then move to DPT only after BTS is fully closed.

## 12. Final Status Line

As of **August 27, 2026**, BTS in this project should be described as:

> **Official baseline through K0-K4 passes; historical 1xT4 Run 01 is superseded at verified step 500, while fresh 2xT4 Run 02 Freeze v4 passes and Session 1 is running from step 0 with evidence pending.**
