# KITTI Data Preflight (Agent B)

Date: 2026-08-26
Status: PASS

## Canonical inputs

- Local source: `dataset/kitti_dataset/`
- Official train list: 23,158 references
- Official test list: 697 references
- Official code baseline: `third_party/bts_official@5e3406b`

Remediation R extracted all 67 official archives with zero extractor errors.
The raw date hierarchy and `_archives/` are retained. The official GT package
produced separate `train/` and `val/` roots; its 151 non-overlapping drive
directories were moved byte-for-byte under `data_depth_annotated/`, which is
the `gt_path` hierarchy consumed by BTS. No image or GT was renamed,
re-encoded, resized, cropped, or selected during source preparation.

## Resolution audit

| Split | References | RGB missing | GT missing |
| --- | ---: | ---: | ---: |
| Train | 23,158 | 0 | 0 |
| Test | 697 | 0 | 45 |

All 45 missing test GT fields are the literal value `None`. This matches the
official BTS evaluation reference (`45 GT files missing`) and is not treated as
dataset corruption.

## Materialization projection

- Unique required RGB paths: 23,855
- Unique GT path tokens: 23,811, including the single `None` token
- Resolved unique GT files: 23,810
- Required RGB bytes: 19,790,383,746
- Required GT bytes: 3,904,803,392
- Projected byte-identical subset: 23,695,187,138 bytes (about 23.70 GB decimal)

The measured subset fits one private Kaggle dataset. Membership remains
strictly controlled by the two official Eigen lists.
