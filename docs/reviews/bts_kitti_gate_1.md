# BTS KITTI Gate-1 Review (Agent D)

Date: 2026-08-26
Decision: PASS

Agent E is unlocked for exact-list materialization, SHA256 generation and a
private Kaggle upload. This decision does not authorize full training; Gate-2
and K0-K4 remain mandatory.

## A: Runtime provenance and smoke

| Check | Result |
| --- | --- |
| Official source clean at `5e3406b35d1497b2e55d2dd600524d1f4efacaed` | PASS |
| Runtime excludes historical `bts/` | PASS |
| Patch forward/reverse applicability | PASS |
| Config overlay SHA256 recorded | PASS |
| Python syntax compile | PASS |
| Imports and argument parser | PASS |
| DenseNet161 model construction (47,000,688 parameters) | PASS |
| KITTI dataloader (23,158 samples) | PASS |
| One GPU forward, five finite `1 x 1 x 352 x 704` outputs | PASS |

Local smoke used isolated Python `3.12.10`, PyTorch `2.10.0+cu128`,
torchvision `0.25.0+cu128`, CUDA `12.8`, and an RTX 3060 Laptop GPU. The
private Kaggle probe independently recorded Python `3.12.13`, the same
torch/torchvision/CUDA versions, and 2 x Tesla T4.

## B: KITTI source

PASS. All train RGB/GT and all test RGB paths resolve. The 45 test GT `None`
entries match official reference behavior. Projected subset size is
23,695,187,138 bytes.

## C: KITTI protocol

PASS. Config differences are limited to allowlisted runtime paths, output
locations and model naming. Both official Eigen lists remain byte-identical.

## Gate boundary

`D_GATE_1 = PASS` unlocks Agent E only. Agent E must still create a
byte-identical subset, source/copy hashes and manifest before private upload.
Independent Gate-2 must pass before K0-K4 and full training.
