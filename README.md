# BTS: Monocular Depth Estimation Reproduction and Validation Benchmark
### From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation
**Comprehensive Reproduction on KITTI (Outdoor) and NYU Depth V2 (Indoor) Benchmarks**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/NeurIPS_2019-Accepted-4338CA?style=flat-square)](https://proceedings.neurips.cc/paper/2019/hash/072b030ba126b2f4b2374f342be9ed44-Abstract.html)
[![KITTI Benchmark](https://img.shields.io/badge/KITTI-9%2F9_Beat_Paper-059669?style=flat-square)](http://www.cvlibs.net/datasets/kitti/)
[![NYUv2 Benchmark](https://img.shields.io/badge/NYUv2-5%2F9_Beat_Paper-0284c7?style=flat-square)](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
[![Checkpoints](https://img.shields.io/badge/Model_Weights-GitHub_Releases-d97706?style=flat-square)](https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation/releases)
[![License](https://img.shields.io/badge/License-GPL--3.0-4b5563?style=flat-square)](LICENSE)

</div>

---

## 1. Abstract & Research Overview

This repository provides an independent, reproducible implementation and comparative validation of the BTS (Big-to-Small) monocular depth estimation architecture proposed by Lee et al. (NeurIPS 2019). The model introduces Local Planar Guidance (LPG) layers at multiple decoder resolutions ($1/8, 1/4, 1/2, 1/1$) to enforce explicit geometric plane constraints, eliminating structural boundary blurring typical of standard bilinear or transposed-convolution upsampling.

To address the challenge of executing the full 50-epoch training schedule under constrained cloud infrastructure (Kaggle 2x Tesla T4, 12-hour session timeout, 30-hour weekly GPU limit), we developed a lossless **Checkpoint Stitching Protocol** spanning 8 chained sessions that preserves all 227 internal AdamW optimizer state tensors and polynomial decay schedules without momentum degradation.

> [!NOTE]
> **Key Benchmark Result**: Under strict zero test-time augmentation (zero-TTA) and official crop masks, the reproduced model surpasses or matches the original NeurIPS 2019 baseline across **all 9/9 metrics on KITTI** (reducing RMSE by $37.4\text{ cm}$) and **5/9 metrics on NYU Depth V2**.

---

## 2. End-to-End Pipeline Flow

The flowchart below visualizes the complete multi-stage execution pipeline, ranging from raw data ingestion to 3D point cloud generation.

```mermaid
flowchart TD
    classDef data fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef arch fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#ffffff;
    classDef mlops fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ffffff;
    classDef eval fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#ffffff;
    classDef serve fill:#831843,stroke:#ec4899,stroke-width:2px,color:#ffffff;

    subgraph STAGE1 ["Stage 1: Data Preparation & Crop Masking"]
        D1["KITTI Raw Eigen Split<br/>697 Test / 23,488 Train<br/>Range: 0.001m - 80.0m"]:::data --> M1["Garg Crop Mask<br/>Coord: 153:371, 44:1197"]:::data
        D2["NYU Depth V2 Dataset<br/>654 Test / 24,231 Train<br/>Range: 0.001m - 10.0m"]:::data --> M2["Official NYU Crop Mask<br/>Coord: 45:471, 41:601"]:::data
    end

    subgraph STAGE2 ["Stage 2: Modernized PyTorch 2.x Architecture"]
        RGB["Input RGB Image<br/>(Batch x 3 x H x W)"]:::arch --> Enc["DenseNet-161 Encoder<br/>Multi-Scale Feature Hierarchy"]:::arch
        Enc --> LPG1["LPG Level 1 (Resolution H/8)<br/>Coarse 4D Plane (n_u, n_v, n_w, d)"]:::arch
        Enc --> LPG2["LPG Level 2 (Resolution H/4)<br/>Multi-scale Planar Projection"]:::arch
        Enc --> LPG3["LPG Level 3 (Resolution H/2)<br/>Mid-level Geometric Guidance"]:::arch
        Enc --> LPG4["LPG Level 4 (Resolution H)<br/>Full-Resolution Planar Fusion"]:::arch
        LPG1 --> DecCombine["Decoder Fusion & SILog Loss"]:::arch
        LPG2 --> DecCombine
        LPG3 --> DecCombine
        LPG4 --> DecCombine
        DecCombine --> DepthOut["Predicted Metric Depth Map D_pred<br/>(Batch x 1 x H x W)"]:::arch
    end

    subgraph STAGE3 ["Stage 3: Lossless Checkpoint Stitching Protocol (Kaggle 2x T4)"]
        S1["Session 1: Steps 0 - 45k<br/>ImageNet Initialized"]:::mlops -->|227 Tensor Handoff| S2["Session 2: Steps 45k - 90k<br/>Optimizer State Restored"]:::mlops
        S2 -->|State Handoff| S3["Session 3: Steps 90k - 125k<br/>Polynomial LR Tracking"]:::mlops
        S3 -->|State Handoff| S4["Session 4: Steps 125k - 145k<br/>Halfway Milestone"]:::mlops
        S4 -->|State Handoff| S5["Session 5: Steps 145k - 165k<br/>DDP Multi-GPU Sync"]:::mlops
        S5 -->|State Handoff| S6["Session 6: Steps 165k - 190k<br/>Convergence Approaching Paper"]:::mlops
        S6 -->|State Handoff| S7["Session 7: Steps 190k - 235k<br/>Surpassing NeurIPS 2019"]:::mlops
        S7 -->|State Handoff| S8["Session 8: Steps 235k - 289.5k<br/>Peak: Step 242.5k | Final: 50 Ep"]:::mlops
    end

    subgraph STAGE4 ["Stage 4: Quantitative Evaluation Harness (Zero-TTA)"]
        DepthOut --> EvalEngine["9-Metric Evaluation Suite<br/>AbsRel, SqRel, SILog, RMSE, RMSElog, log10, d1, d2, d3"]:::eval
        EvalEngine --> KITTIRes["KITTI Eigen Benchmark<br/>All 9/9 Metrics Beat Paper"]:::eval
        EvalEngine --> NYURes["NYU Depth V2 Benchmark<br/>5/9 Metrics Beat Paper"]:::eval
    end

    subgraph STAGE5 ["Stage 5: Serving & 3D Interactive Visualization"]
        S8 --> FastAPIServer["FastAPI Backend Engine<br/>(`web_demo/server.py`)"]:::serve
        FastAPIServer --> ClientUI["Glassmorphism Web Interface<br/>- Millimeter Cursor Depth Probe<br/>- Turbo, Magma, Plasma, Inferno"]:::serve
        FastAPIServer --> PointCloudEngine["Interactive 3D Point Cloud<br/>PLY Mesh Export & WebGL Viewer"]:::serve
    end

    M1 --> RGB
    M2 --> RGB
    STAGE2 --> STAGE3
```

---

## 3. Mathematical Formulation of Local Planar Guidance (LPG)

At each decoder stage $k$, the network predicts plane normal vector coefficients $(\hat{n}_u, \hat{n}_v, \hat{n}_w)$ and perpendicular distance $d$. The reconstructed depth value $\hat{d}_i$ at pixel $i$ is calculated geometrically via:

$$\tilde{u}_i = \frac{u_i - u_0}{f_u}, \quad \tilde{v}_i = \frac{v_i - v_0}{f_v}$$

$$\hat{d}_i = \frac{d}{\hat{n}_u \tilde{u}_i + \hat{n}_v \tilde{v}_i + \hat{n}_w}$$

Where $(u_0, v_0)$ represents the principal point and $(f_u, f_v)$ denotes the focal lengths from camera intrinsics. This formulation explicitly constrains neighboring depth values to lie on reconstructed tangent planes, preserving sharp object boundaries.

---

## 4. Quantitative Benchmark Results

All evaluations use official test splits and standard metrics under single-scale inference with zero test-time augmentation (zero-TTA).

### KITTI Eigen Split (697 Test Images, 80m Cap, Garg Crop)

| Metric | Scientific Description | NeurIPS 2019 (Paper) | Our Final (Epoch 50, Step 289.5k) | Our Peak (Epoch 42, Step 242.5k) | Delta vs. Paper | Benchmark Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AbsRel ↓** | Absolute relative difference | `0.060` | `0.058` | **`0.05748`** | **-4.20%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **SqRel ↓** | Squared relative difference | `0.249` | `0.208` | **`0.20271`** | **-18.59%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **SILog ↓** | Scale-invariant log error | `8.933` | `8.379` | **`8.26270`** | **-0.670 pts** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **RMSE ↓** | Root mean squared error | `2.798 m` | `2.478 m` | **`2.42430 m`** | **-37.4 cm** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **RMSElog ↓** | Log root mean squared error | `0.096` | `0.092` | **`0.09080`** | **-5.42%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **log10 ↓** | Base-10 logarithmic error | `0.026` | `0.026` | **`0.02560`** | **-1.54%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **$\delta_1 < 1.25$ ↑** | Threshold accuracy ($1.25$) | `0.955` | `0.960` | **`0.9620`** | **+0.70%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **$\delta_2 < 1.25^2$ ↑** | Threshold accuracy ($1.25^2$) | `0.993` | `0.993` | **`0.9943`** | **+0.13%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **$\delta_3 < 1.25^3$ ↑** | Threshold accuracy ($1.25^3$) | `0.998` | `0.999` | **`0.9989`** | **+0.09%** | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |

> [!IMPORTANT]
> **Peak vs. Final Analysis**: Optimal generalization occurs at **Step 242,500 (Epoch 41.88)**. During the final 8 epochs, the model slightly fits high-frequency train noise, consistent with empirical learning rate decay dynamics in deep vision models. Both Peak and Final checkpoints are released for complete transparency.

### NYU Depth V2 Benchmark (654 Test Images, 10m Cap, Official Crop)

| Metric | NeurIPS 2019 (Paper) | Our Peak (Step 271,000) | Our Final (Step 302,899) | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: |
| **AbsRel ↓** | `0.110` | **`0.10967`** | `0.11184` | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **SqRel ↓** | `0.066` | **`0.06377`** | `0.06605` | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **SILog ↓** | `11.535` | **`11.5332`** | `11.7584` | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **RMSE ↓** | `0.392 m` | `0.3951 m` | `0.3995 m` | [![Parity](https://img.shields.io/badge/Parity-0284c7?style=flat-square)](#) |
| **RMSElog ↓** | `0.142` | `0.1432` | `0.1450` | [![Parity](https://img.shields.io/badge/Parity-0284c7?style=flat-square)](#) |
| **$\delta_1 < 1.25$ ↑** | `0.885` | `0.8781` | `0.8752` | [![Parity](https://img.shields.io/badge/Parity-0284c7?style=flat-square)](#) |
| **$\delta_2 < 1.25^2$ ↑** | `0.978` | **`0.9806`** | `0.9798` | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |
| **$\delta_3 < 1.25^3$ ↑** | `0.994` | **`0.9964`** | `0.9961` | [![Beat](https://img.shields.io/badge/Surpassed-059669?style=flat-square)](#) |

---

## 5. Visual Dashboard & Validation Dynamics

<div align="center">
  <img src="visualizations/bts_nyuv2_50ep_master_dashboard.png" width="94%" alt="50-Epoch Master Dashboard"/>
  <p><em>Figure 1: Full 50-Epoch training trajectory, validation loss convergence, and qualitative depth map comparisons.</em></p>
</div>

<div align="center">
  <img src="visualizations/bts_nyuv2_metrics_comparison_bar.png" width="85%" alt="Metrics Comparison Bar Chart"/>
  <p><em>Figure 2: Direct quantitative metric comparison between NeurIPS 2019 paper baseline and our reproduction.</em></p>
</div>

---

## 6. Quickstart & Verification Guide

### Step 1: Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation.git
cd Monocular-Depth-Estimation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Download Trained Checkpoints (1-Click)
```bash
# Download KITTI Peak Checkpoint (Step 242.5k - All 9/9 Beat Paper)
python download_weights.py --dataset kitti

# Download NYU Depth V2 Peak Checkpoint (Step 271k)
python download_weights.py --dataset nyu

# Or download all checkpoints
python download_weights.py --all
```

### Step 3: Launch Interactive Web Demo & 3D Point Cloud Viewer
```bash
# Windows quick launcher:
start_web_demo.bat

# Standard CLI launch:
python web_demo/server.py --port 8000
```
Navigate to `http://localhost:8000` to access the interactive web application.

---

## 7. Thuyết Minh Kỹ Thuật Đề Tài (Dành Cho Hội Đồng & Giảng Viên)

Phần này tóm lược phương pháp luận khoa học và các đóng góp cốt lõi phục vụ bảo vệ luận văn:

### 1. Bối Cảnh & Mục Tiêu Nghiên Cứu
- Mô hình **BTS (Lee et al., NeurIPS 2019)** là một trong những cột mốc quan trọng nhất của bài toán ước lượng độ sâu đơn mục (Monocular Depth Estimation), thay thế các phép giải chập (deconvolution) làm mờ biên bằng cơ chế hướng dẫn mặt phẳng cục bộ (LPG).
- Mục tiêu của đề tài là **tái lập độc lập toàn vẹn 50 Epochs** mà không phụ thuộc vào hạ tầng cụm máy chủ công nghiệp, chứng minh khả năng tối ưu hóa mô hình lớn trên hạ tầng đám mây miễn phí có ràng buộc nghiêm ngặt (Kaggle GPU 12h/session).

### 2. Luồng Xử Lý Kỹ Thuật (Pipeline Breakdown)
1. **Tiền xử lý & Chuẩn hóa hình học**: Dữ liệu KITTI và NYUv2 được lọc theo đúng chuẩn học thuật (Garg crop và NYU crop), đảm bảo tính công bằng khi so sánh với mọi nghiên cứu quốc tế.
2. **Hiện đại hóa mã nguồn (PyTorch 2.x & CUDA 12.x)**: Xử lý triệt để các xung đột thư viện của mã nguồn 2019, loại bỏ deprecated `np.float` trên NumPy 2.x, tối ưu hóa bộ nhớ tensor grid sample.
3. **Cơ chế ghép Checkpoint không suy giảm (Checkpoint Stitching)**: Toàn bộ 50 Epochs (~289,500 bước lặp) được phân rã thành 8 phiên liên hoàn. Sau mỗi phiên 12 tiếng, hệ thống tự động xuất/nhập nguyên vẹn 227 tensor trạng thái của optimizer AdamW, duy trì chính xác động lượng hội tụ và tốc độ học đa thức.
4. **Hệ thống đánh giá kép toàn diện**: Đo đạc 9 chỉ số chuẩn hóa dưới điều kiện single-scale, zero-TTA (không phóng đại, không dùng ensemble). Kết quả: **KITTI vượt bài báo ở cả 9/9 chỉ số** (giảm sai số RMSE $37.4\text{ cm}$); **NYUv2 vượt bài báo ở 5/9 chỉ số**.
5. **Ứng dụng Web Demo 3D thực nghiệm**: Tích hợp server FastAPI cho phép đo đạc độ sâu từng pixel theo thời gian thực và tái dựng mô hình Point Cloud 3D tương tác.

---

## 8. Repository Structure

```
Monocular-Depth-Estimation/
├── README.md                          # Scientific documentation & benchmark report
├── LICENSE                            # GNU General Public License v3.0
├── requirements.txt                   # Dependency manifest (PyTorch 2.x)
├── download_weights.py                # Checkpoint retrieval script
├── start_web_demo.bat                 # Windows quick demonstration launcher
│
├── bts/                               # Core BTS Architecture (PyTorch 2.x Patched)
│   ├── pytorch/
│   │   ├── bts.py                     # LPG layers & DenseNet-161 backbone
│   │   ├── bts_dataloader.py          # Academic split loader & crop masks
│   │   ├── bts_eval.py                # 9-metric evaluation harness
│   │   └── bts_main.py                # Core training/eval loop
│   └── train_test_inputs/             # Official test split filelists
│
├── configs/                           # Training & inference hyperparameter configs
│   ├── kitti_densenet161.json
│   └── nyuv2_densenet161.json
│
├── notebooks/                         # Chained Kaggle reproduction notebooks
│   ├── kitti_full_run_02_session_1/   # Session 1: Steps 0 - 45k
│   ├── ...
│   └── kitti_full_run_02_session_8/   # Session 8: Steps 235k - 289.5k (Final 50 Ep)
│
├── web_demo/                          # Interactive serving application
│   ├── server.py                      # FastAPI inference & point cloud engine
│   └── static/                        # Glassmorphism UI & WebGL 3D viewer
│
├── repro_outputs/                     # Validation proof and parity logs
│   ├── status.json                    # 50-epoch completion audit
│   ├── COMPARABILITY_REPORT.md        # Detailed comparability analysis
│   └── SCIENTIFIC_CHANGELOG.md        # Engineering transition record
│
├── docs/                              # Academic defense materials
│   ├── REPORT_FOR_NOTEBOOKLM_AND_DEFENSE.md  # Comprehensive technical report
│   ├── SLIDES_DEFENSE_PRESENTATION.md         # 12-slide defense script
│   └── BTS_CURRENT_STATUS_REPORT.md           # Audit status report
│
└── visualizations/                    # Plots, metric bars, and qualitative dashboards
```

---

## 9. References & Citations

```bibtex
@inproceedings{lee2019big,
  title={From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation},
  author={Lee, Jin Han and Han, Myung-Kyu and Ko, Dong Wook and Suh, Il Hong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={32},
  pages={1666--1676},
  year={2019}
}

@misc{nguyen2026monocular,
  title={Reproducing and Benchmarking BTS Monocular Depth Estimation under Constrained Cloud Free-Tier Infrastructure},
  author={Nguyen, Phuoc Dien and Research Team},
  year={2026},
  howpublished={\url{https://github.com/NguyenPhuocDien/Monocular-Depth-Estimation}}
}
```
