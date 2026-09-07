# BTS KITTI Reproduction Summary

Gate-1 passed on 2026-08-26. Official BTS source provenance, full local KITTI
source resolution, Kaggle protocol config, isolated runtime imports/model/
dataloader, and one GPU forward are verified. Full reproduction is not yet
claimed: materialization, Gate-2, Kaggle K0-K4 and 50-epoch training remain.

Current blocker: G: has about 28.46 GB free while the measured materialized
subset requires 23.70 GB, leaving unsafe staging/upload headroom.
