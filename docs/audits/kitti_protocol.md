# KITTI Protocol Audit (Agent C)

Date: 2026-08-26
Status: PASS

## Compared files

- Official: `third_party/bts_official/pytorch/arguments_train_eigen.txt`
- Kaggle: `adaptations/bts_runtime/config/arguments_train_eigen_kaggle.txt`

The Kaggle config is copied into generated runtime through an explicit config
overlay and its SHA256 is recorded in `provenance_manifest.json`.

## Allowed differences observed

- `model_name`
- `data_path`, `gt_path`, `data_path_eval`, `gt_path_eval`
- `filenames_file`, `filenames_file_eval` (path relocation only)
- `log_directory`, `eval_summary_directory`

The relocated filename files are byte-identical to official:

- Train SHA256: `E9ECA9EA3589F6A667DB108F5E307D40DD6B7AB7634B32B0BD28ECF1D864C094`
- Test SHA256: `8966957128CCE3846A2AF5E82E3444CBDE4376F7D1DC07F63FCD550B6CBA94FD`

No unexpected config field differs. Encoder, dataset, batch size `4`, epochs
`50`, optimizer values, input `352 x 704`, maximum depth `80`, crop,
augmentation, distributed mode, and evaluation settings match official.

`max_steps` remains a runtime smoke-only parser extension and is absent from
the full Kaggle 50-epoch config.

## Mount-path remediation re-audit

The four allowlisted data/GT path fields were updated to the mount observed by
the independent Kaggle audit:

```text
/kaggle/input/datasets/thudo25/bts-kitti-eigen-materialized-v1/
```

New config SHA256:

```text
9bef0b1329a0ff8cb394e1985a240ebc299dab0d15959dab2a04c8aaaebef6e7
```

Agent C independently confirmed that the generated runtime config is
byte-identical to the adaptation config, all differences from official remain
allowlisted, and every protocol field is unchanged. Patch forward/reverse
checks, official source cleanliness and runtime smoke all pass.
