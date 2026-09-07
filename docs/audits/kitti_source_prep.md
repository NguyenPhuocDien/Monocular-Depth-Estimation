# KITTI Canonical Source Preparation (Remediation R)

Date: 2026-08-26
Status: PASS

- Expected archives from the official BTS URL list plus GT: 67
- Local archives found: 67
- Missing/unexpected archives: 0/0
- ZIP central-directory errors: 0
- Unsafe archive paths: 0
- Compressed bytes: 187,614,633,695
- Declared uncompressed bytes: 187,591,363,248
- Free bytes before extraction: 216,891,031,552
- All extractor exit codes: 0
- Free bytes after extraction: 28,460,552,192

Raw date roots `2011_09_26` through `2011_10_03` were extracted without
selection. The official GT ZIP produced `train/` and `val/`; their 151
non-overlapping drive directories were moved intact under
`data_depth_annotated/` to match BTS `gt_path` resolution. Empty `train/` and
`val/` roots and all files in `_archives/` were retained. No image contents or
filenames were modified.
