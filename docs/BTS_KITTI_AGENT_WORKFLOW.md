# BTS KITTI Agent Workflow

```text
R: local source prep ─> B: KITTI data audit ─┐
A: runtime/env/config ─> C: protocol audit  ─┴─> D Gate-1
                                                    │
                                                    v
                                         E materialize/upload ─> D Gate-2
                                                                  │
                                                                  v
                                                        K0/K1/K2/K3/K4
                                                                  │
                                                                  v
                                                        full KITTI training
```

Ownership is exclusive:

- R: `dataset/kitti_dataset/**` source extraction/layout only; no selection,
  materialization or upload.
- A: `tools/clean_bts_runtime.py`, `patches/kaggle_runtime_adaptations.patch`,
  `adaptations/bts_runtime/**`, and runtime smoke evidence.
- B: `docs/audits/kitti_data_preflight.*`.
- C: `docs/audits/kitti_protocol.*`.
- D: `docs/reviews/*`.
- E: `tools/kitti_materialize.py`, `tools/kitti_verify.py`,
  `data_manifests/kitti/**`, `build/kitti_bts_eigen/**`, and Kaggle metadata.

Invariant sources:

- `third_party/bts_official@5e3406b` is the canonical code source.
- `bts/` is historical only and is never a runtime build input.
- Local full KITTI is the canonical data source.
- Kaggle receives only a byte-identical, official-list-driven materialization.
