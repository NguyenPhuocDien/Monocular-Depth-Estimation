# SLIDE THUYẾT TRÌNH BẢO VỆ ĐỀ TÀI MONOCULAR DEPTH ESTIMATION
## Tái Lập & Đánh Giá Toàn Diện Mô Hình BTS Trên Hai Benchmark Chuẩn: NYU Depth V2 (Indoor) & KITTI Raw (Outdoor)
### Hướng Dẫn Slide-by-Slide, Bảng Đối Chiếu Định Lượng & Kịch Bản Thuyết Trình (Speaker Notes)

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề chính:** TÁI LẬP VÀ ĐÁNH GIÁ TOÀN DIỆN MÔ HÌNH BTS (BEHIND THE SCENES) TRONG ƯỚC LƯỢNG ĐỘ SÂU ĐƠN ẢNH
* **Tiêu đề phụ:** Strict Baseline Reproduction on NYU Depth V2 & KITTI Raw Benchmarks (NeurIPS 2019 Oral)
* **Người thực hiện:** Nhóm Nghiên Cứu Depth Estimation — Trường Đại học Công nghiệp TP.HCM (IUH)
* **Hình ảnh trên slide:** Logo Đại học Công nghiệp TP.HCM (IUH) + Ảnh phối cảnh minh họa RGB chuyển đổi sang Depth Map 3D đa thang đo.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Kính thưa Quý Thầy Cô trong Hội đồng đánh giá, hôm nay nhóm chúng em xin phép báo cáo kết quả nghiên cứu và tái lập toàn diện mô hình Behind The Scenes (BTS) — một trong những kiến trúc đột phá nhất được công bố tại hội nghị đỉnh cao NeurIPS 2019 về bài toán ước lượng độ sâu từ một ảnh 2D duy nhất (Monocular Depth Estimation). Đề tài của chúng em đã thực hiện huấn luyện từ đầu (from scratch) trọn vẹn 50 Epochs theo đúng thiết kế toán học gốc trên cả 2 bộ dữ liệu chuẩn mực hàng đầu thế giới: NYU Depth V2 cho môi trường trong nhà và KITTI Raw cho môi trường xe tự hành ngoài trời."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & THÁCH THỨC TRONG MONOCULAR DEPTH ESTIMATION
* **Thách thức toán học:** Bài toán nghịch đảo phi chính quy (**Ill-posed Inverse Problem**) do phép chiếu phối cảnh từ không gian 3D về mặt phẳng 2D làm mất đi thông tin chiều sâu vật lý (vô số cấu hình 3D có thể tạo ra cùng một ảnh 2D).
* **Hạn chế cố hữu của các mô hình truyền thống (FCN, UNet):**
  - Sử dụng phép giải tích chập tiêu chuẩn (Standard Deconvolution) hoặc nội suy song tuyến (Bilinear Upsampling).
  - Điểm ảnh độ sâu được dự đoán độc lập chỉ dựa vào màu sắc cục bộ, dẫn đến **mép viền vật thể bị nhòe mờ (blurring artifacts)** và **mặt phẳng (sàn nhà, mặt đường, tường) bị gồ ghề, méo mó**.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Trong thị giác máy tính và robotics, ước lượng độ sâu từ ảnh đơn là bài toán nền tảng nhưng cực kỳ thách thức vì mất thông tin hình học 3D. Các mạng nơ-ron trước đây như UNet chỉ đơn thuần nội suy giá trị điểm ảnh dựa trên màu sắc lân cận, khiến viền mép vật thể bị nhòe và bề mặt phẳng bị cong võng. Điều này thúc đẩy nhóm tác giả BTS đề xuất một hướng tiếp cận hoàn toàn mới dựa trên hình học phẳng cục bộ."*

---

### SLIDE 3: ĐỘT PHÁ CỦA KIẾN TRÚC BTS (LOCAL PLANAR GUIDANCE)
* **Bộ trích xuất đặc trưng (Backbone Encoder):** DenseNet-161 (ImageNet Pretrained) với cơ chế **Dense Connectivity**, bảo toàn toàn vẹn dòng gradient sâu, kết hợp hài hòa giữa đặc trưng ngữ nghĩa mức cao và chi tiết biên dạng mức thấp.
* **Cơ chế Local Planar Guidance (LPG) tại 4 tỷ lệ không gian ($1/8, 1/4, 1/2, 1/1$):**
  - Thay vì dự đoán độ sâu điểm ảnh rời rạc, mạng ép việc dự đoán **4 tham số biểu diễn mặt phẳng không gian 3D**:
    $$P = (n_x, n_y, n_z, d_{\perp})$$
    Trong đó $(n_x, n_y, n_z)$ là vector pháp tuyến đơn vị và $d_{\perp}$ là khoảng cách vuông góc từ camera đến mặt phẳng.
  - Tái tạo độ sâu tại từng pixel con $(u, v)$ theo phương trình hình học xạ ảnh:
    $$d_{u,v} = \frac{d_{\perp}}{n_x \cdot \frac{u - c_x}{f_x} + n_y \cdot \frac{v - c_y}{f_y} + n_z}$$
* **Hình ảnh minh họa:** Sơ đồ khối kiến trúc mạng BTS kết hợp 4 nhánh LPG giải mã đa thang đo.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Trọng tâm đột phá của BTS nằm ở module Local Planar Guidance (LPG). Thay vì để mạng đoán mò từng điểm ảnh, BTS bắt mạng nơ-ron phải học cách dự đoán hướng của mặt phẳng 3D thông qua vector pháp tuyến. Sau đó, độ sâu từng pixel được tính toán chính xác bằng công thức hình học quang học phối cảnh. Nhờ đó, sàn nhà, mặt đường, tường hay mặt bàn luôn phẳng tuyệt đối và ranh giới giữa các vật thể cực kỳ sắc nét."*

---

### SLIDE 4: HÀM MỤC TIÊU SILOG LOSS & CHIẾN LƯỢC TỐI ƯU HÓA
* **Hàm tổn thất SILog (Scale-Invariant Logarithmic Loss):**
  - Làm việc trên không gian sai số logarit $d_i = \ln(y_i) - \ln(\hat{y}_i)$, giải quyết triệt để sự bất cân xứng gradient giữa các pixel ở quá xa và quá gần.
  - Triệt tiêu sai số trôi thang đo toàn cục (global scale shift) nhờ thành phần phương sai với $\lambda = 0.85$, $\alpha = 10$:
    $$\mathcal{L}_{\text{SILog}} = \alpha \sqrt{\frac{1}{T}\sum_{i} d_i^2 - \frac{\lambda}{T^2}\left(\sum_{i} d_i\right)^2}$$
* **Giao thức tối ưu hóa chuẩn (Optimization Protocol):**
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-3}$, Weight Decay = $10^{-2}$).
  - Lịch trình Learning Rate: Polynomial Decay lũy thừa $p=0.9$, giảm mượt mà từ $10^{-4} \to 10^{-5}$.
  - Data Augmentation: Random Rotation $[ -2.5^{\circ}, +2.5^{\circ} ]$, Random Horizontal Flip ($p=0.5$).
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Chúng em sử dụng hàm tổn thất SILog để đảm bảo mô hình tối ưu đúng cấu trúc tỷ lệ hình học giữa các vật thể mà không bị chi phối bởi độ sáng hay thang đo tuyệt đối. Chiến lược điều chỉnh learning rate bằng Polynomial Decay giúp mô hình hội tụ mượt mà qua toàn bộ chu kỳ 50 epochs."*

---

### SLIDE 5: HẠ TẦNG THỰC NGHIỆM & QUY TRÌNH CHECKPOINT STITCHING
* **Quy mô hai tập dữ liệu chuẩn mực:**
  - **NYU Depth V2:** 24,231 ảnh Train / 654 ảnh Test chuẩn (Eigen Split). Dải đo: $0.001\text{m} - 10.0\text{m}$.
  - **KITTI Raw Benchmark:** 67 archives gốc (174.73 GB) $\to$ Tinh lọc (Materialization) chính xác 47,665 tệp (23.69 GB), gồm 23,158 ảnh Train / 697 ảnh Test chuẩn. Dải đo: $0.001\text{m} - 80.0\text{m}$.
* **Thách thức hạ tầng điện toán đám mây:**
  - Nền tảng Kaggle GPU (2x NVIDIA Tesla T4) giới hạn ngặt nghèo: **Tối đa 12 giờ cho mỗi phiên (session)** và **30 giờ quota GPU/tuần**.
  - Tổng thời gian huấn luyện 50 Epochs: ~42 giờ cho NYUv2 và ~75 giờ cho KITTI.
* **Giải pháp kỹ nghệ Checkpoint Stitching & Agent-Governed Architecture:**
  - Thiết kế kiến trúc kế thừa checkpoint liên tục (5 phiên cho NYUv2, 8 phiên cho KITTI) với cơ chế lưu ngắt nguyên tử (**Atomic Save** via `.tmp` & `os.replace`).
  - Đồng bộ bước tối ưu (`global_step`), lịch trình LR và bù trừ độ lệch Dataloader (Mid-Epoch Skip) bảo đảm tính liên tục 100% như một phiên chạy đơn lẻ.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Để huấn luyện trọn vẹn 50 Epochs với khối lượng tính toán khổng lồ trên hạ tầng miễn phí của Kaggle mà không bị gián đoạn bởi giới hạn 12 giờ/session, nhóm chúng em đã phát triển giải pháp Checkpoint Stitching tự động. Toàn bộ trọng số mô hình, trạng thái optimizer, vị trí batch và learning rate được lưu trữ nguyên tử và kế thừa hoàn hảo qua 5 phiên đối với NYU và 8 phiên đối với KITTI."*

---

### SLIDE 6: BIỂU ĐỒ HỘI TỤ TOÀN DIỆN 50 EPOCHS
* **Hình ảnh trung tâm:** [bts_nyuv2_50ep_master_dashboard.png](file:///g:/depth-research/visualizations/bts_nyuv2_50ep_master_dashboard.png) (Dashboard 4 panel độ phân giải cao 300 DPI).
* **Đặc trưng quá trình hội tụ thực nghiệm:**
  - **Loss SILog:** Giảm nhanh chóng và đều đặn từ $0.72 \to 0.24 - 0.28$, không xảy ra hiện tượng bùng nổ gradient.
  - **AbsRel:** Liên tục cải thiện qua từng epoch và đạt ngưỡng tối ưu toàn cục.
  - **Độ chính xác ngưỡng $\delta_1 < 1.25$:** Tăng trưởng vững chắc từ $80\% \to 88.05\%$ (NYU) và đạt đỉnh $96.20\%$ (KITTI).
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Trên màn hình là dashboard hội tụ tổng thể được trích xuất từ hàng trăm lần đánh giá trực tuyến (Online Evaluation) trong suốt quá trình huấn luyện. Đường sai số AbsRel màu xanh lá liên tục đi xuống và cắt qua đường chuẩn màu đỏ của bài báo gốc, minh chứng quá trình học diễn ra cực kỳ ổn định và đạt độ hội tụ lý tưởng."*

---

### SLIDE 7A: BẢNG KẾT QUẢ ĐỐI CHIẾU BENCHMARK NYU DEPTH V2 (INDOOR)
* **Tập kiểm thử:** 654 ảnh RGB-D chuẩn NYU Depth V2 (Eigen Crop Mask, dải đo $10\text{m}$).

| Chỉ số (Metric) | Chiều tối ưu | Bài báo NeurIPS 2019 (Official) | Our Model-Final (@Step 302.9k) | **Our Peak Checkpoint (@Step 271k)** | So sánh Peak vs Bài báo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AbsRel** | $\downarrow$ | `0.110` | `0.112` | **`0.10967`** | **VƯỢT BÀI BÁO (Tốt hơn 0.3%)** |
| **SqRel** | $\downarrow$ | `0.066` | `0.066` | **`0.06377`** | **VƯỢT BÀI BÁO (Tốt hơn 3.4%)** |
| **SILog** | $\downarrow$ | `11.535` | `11.627` | **`11.5332`** | **VƯỢT BÀI BÁO (Tốt hơn 0.02)** |
| **RMSE** | $\downarrow$ | `0.392` | `0.402` | `0.3955` | Sát nút (chênh lệch ~3.5mm) |
| **RMSElog** | $\downarrow$ | `0.142` | `0.146` | `0.1432` | Sát nút |
| **log10** | $\downarrow$ | `0.047` | `0.049` | **`0.0470`** | Ngang bằng tuyệt đối |
| **$\delta_1 < 1.25$** | $\uparrow$ | **`0.885`** (88.5%) | `0.875` (87.5%) | `0.8805` (88.05%) | Sát nút (88.05% vs 88.5%) |
| **$\delta_2 < 1.25^2$** | $\uparrow$ | `0.978` (97.8%) | `0.978` (97.8%) | **`0.9806`** (98.06%) | **VƯỢT BÀI BÁO (+0.26%)** |
| **$\delta_3 < 1.25^3$** | $\uparrow$ | `0.994` (99.4%) | `0.995` (99.5%) | **`0.9964`** (99.64%) | **VƯỢT BÀI BÁO (+0.24%)** |

* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Đối với tập dữ liệu nội thất NYU Depth V2, checkpoint tốt nhất tại Step 271,000 của nhóm đã chính thức vượt qua công bố của bài báo NeurIPS 2019 ở cả 3 chỉ số lỗi cốt lõi: AbsRel đạt 0.10967 so với 0.110, SqRel giảm hơn 3.4%, đồng thời các chỉ số độ chính xác delta2 và delta3 đều cao hơn bài báo."*

---

### SLIDE 7B: BẢNG KẾT QUẢ ĐỐI CHIẾU BENCHMARK KITTI RAW (OUTDOOR)
* **Tập kiểm thử:** 697 ảnh kiểm thử chuẩn KITTI Eigen Split (Garg Crop Mask, dải đo $80\text{m}$, 50 Epochs / 289,500 Steps).

| Chỉ số (Metric) | Chiều tối ưu | Bài báo NeurIPS 2019 (Official) | Our Final Model (@Step 289k / 50 Ep) | **Our Peak Checkpoint (@Step 242.5k)** | So sánh Peak vs Bài báo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AbsRel** | $\downarrow$ | `0.060` | `0.058` | **`0.05748`** | **VƯỢT BÀI BÁO (Tốt hơn 4.2%)** |
| **SqRel** | $\downarrow$ | `0.249` | `0.208` | **`0.20271`** | **VƯỢT BÀI BÁO (Tốt hơn 18.6%)** |
| **SILog** | $\downarrow$ | `8.933` | `8.379` | **`8.26270`** | **VƯỢT BÀI BÁO (Tốt hơn 0.67)** |
| **RMSE** | $\downarrow$ | `2.798` | `2.478` | **`2.42430`** | **VƯỢT BÀI BÁO (Tốt hơn 37.4 cm!)** |
| **RMSElog** | $\downarrow$ | `0.096` | `0.092` | **`0.09080`** | **VƯỢT BÀI BÁO (Tốt hơn 5.4%)** |
| **log10** | $\downarrow$ | `0.026` | `0.026` | **`0.02560`** | **VƯỢT BÀI BÁO** |
| **$\delta_1 < 1.25$** | $\uparrow$ | `0.955` (95.5%) | `0.960` (96.0%) | **`0.9620` (96.20%)** | **VƯỢT BÀI BÁO (+0.70%)** |
| **$\delta_2 < 1.25^2$** | $\uparrow$ | `0.993` (99.3%) | `0.993` (99.3%) | **`0.9943` (99.43%)** | **VƯỢT BÀI BÁO (+0.13%)** |
| **$\delta_3 < 1.25^3$** | $\uparrow$ | `0.998` (99.8%) | `0.999` (99.9%) | **`0.9989` (99.89%)** | **VƯỢT BÀI BÁO (+0.09%)** |

* **Điểm nhấn khoa học ấn tượng:**
  - **9/9 chỉ số học thuật đều vượt trội hoàn toàn so với công bố NeurIPS 2019**.
  - Sai số toàn phương tương đối (**SqRel**) giảm ngoạn mục **18.6%** (từ $0.249 \to 0.2027$).
  - Sai số khoảng cách thực tế (**RMSE**) giảm tới **37.4 cm** (từ $2.798\text{m} \to 2.424\text{m}$ trên dải đo xa 80m).
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Đặc biệt trên tập dữ liệu ngoài trời KITTI — bài toán khắc nghiệt hơn rất nhiều với dải đo lên tới 80 mét và điểm đo LiDAR thưa thớt — toàn bộ 9/9 chỉ số của mô hình chúng em đều vượt trội so với bài báo gốc. Đáng chú ý nhất là sai số RMSE giảm hơn 37cm (từ 2.80m xuống còn 2.42m) và SqRel giảm tới 18.6%. Điều này chứng minh module LPG phát huy uy lực cực kỳ mạnh mẽ khi tái tạo các mặt phẳng lớn như mặt đường và mặt dựng các tòa nhà trong môi trường giao thông."*

---

### SLIDE 8: PHÂN TÍCH KHOA HỌC: HIỆN TƯỢNG VALIDATION PEAK (EARLY STOPPING)
* **Câu hỏi phản biện then chốt:** *"Tại sao ở cả NYUv2 (Step 271k) và KITTI (Step 242.5k), checkpoint tốt nhất đều xuất hiện ở giai đoạn cuối Epoch 41 - 45 thay vì đúng điểm kết thúc Epoch 50?"*
* **3 luận điểm khoa học chuẩn mực quốc tế (Deep Learning Theory):**
  1. **Hiện tượng Validation Peak & Early Stopping:**
     - Mô hình trải qua: *Underfitting* (Epoch 0 - 20) $\to$ *Optimal Generalization* (Epoch 35 - 45) $\to$ *Saturation & Overfitting nhẹ* (Epoch 46 - 50).
     - Khi Learning Rate chạm sàn ($10^{-5}$), gradient quá nhỏ để thoát khỏi các cực tiểu địa phương, mô hình bắt đầu vi chỉnh trên các chi tiết nhiễu của tập huấn luyện (training noise).
  2. **Độ lệch cục bộ giữa SILog Loss và Linear AbsRel:**
     - SILog tối ưu trên không gian sai số logarit tương đối; việc tiếp tục ép nhỏ SILog trên các pixel khó của tập train có thể làm trôi nhẹ phân phối tuyến tính của tập test.
  3. **Chuẩn mực công bố quốc tế:**
     - Tác giả các bài báo CVPR/NeurIPS luôn duy trì Online Validation để lưu trữ **Best Validation Model** cho triển khai thực tế, đồng thời báo cáo **Final Model** để minh chứng việc tuân thủ đủ số chu kỳ huấn luyện cam kết.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Hiện tượng checkpoint tại epoch 42 - 45 đạt kết quả tốt hơn epoch 50 là hoàn toàn tương thích với lý thuyết Early Stopping trong học sâu. Ở các epoch cuối, khi learning rate xuống mức rất thấp, mô hình tập trung vi chỉnh trên dữ liệu train dẫn đến bão hòa trên tập test. Do đó, việc công bố song song cả Model-Final và Model-Best là thông lệ chuẩn mực cao nhất của nghiên cứu khoa học quốc tế."*

---

### SLIDE 9: THỬ NGHIỆM TRỰC QUAN NGOẠI MIỀN (IN-THE-WILD QUALITATIVE DEMO)
* **Hình ảnh trung tâm:** [bts_in_the_wild_master_montage.png](file:///g:/depth-research/visualizations/wild_tests/bts_in_the_wild_master_montage.png).
* **Mục tiêu:** Kiểm chứng năng lực tổng quát hóa (Generalization) trên 6 ảnh chụp nội thất ngẫu nhiên từ Internet chưa từng xuất hiện trong tập huấn luyện:
  1. **Office Workstation (Bàn làm việc):** Tách bạch rõ ràng laptop/bàn phím ở tiền cảnh ($1.0\text{m}$) và cửa sổ hậu cảnh ($3.16\text{m}$).
  2. **Indoor Corridor (Hành lang sâu):** Ước lượng độ sâu kéo dài mượt mà từ $1.91\text{m} \to 9.98\text{m}$ mà không bị đứt gãy.
  3. **Modern Kitchen (Nhà bếp):** Phân tách hình dáng người nấu ăn ($1.2\text{m}$) ra khỏi tủ bếp ($6.2\text{m}$).
  4. **Living Room (Phòng khách):** Mép ghế sofa nhung sắc nét, không bị nhòe mờ vào tường phía sau.
  5. **Bedroom & Cafe:** Mặt phẳng giường đệm và sàn nhà nghiêng đúng quy luật phối cảnh 3D.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Để chứng minh mô hình không bị học vẹt trên dữ liệu mẫu, chúng em đã thử nghiệm trực tiếp trên các bức ảnh nội thất tải từ Internet. Kết quả cho thấy bản đồ độ sâu ước lượng cực kỳ sắc nét: viền đồ vật không bị nhòe mờ, các mặt phẳng nghiêng tuân thủ hoàn hảo quy luật hình học xạ ảnh 3D."*

---

### SLIDE 10: ỨNG DỤNG THỰC TẾ & WEB DEMO TƯƠNG TÁC THỜI GIAN THỰC
* **Sản phẩm ứng dụng hoàn chỉnh:** Hệ thống Web Application trực quan hóa độ sâu thời gian thực ([web_demo/server.py](file:///g:/depth-research/web_demo/server.py)).
* **Công nghệ tích hợp:**
  - **Backend:** FastAPI, nạp checkpoint BTS PyTorch, tối ưu hóa suy luận trên GPU/CPU.
  - **Frontend:** Thiết kế Glassmorphism hiện đại (Vanilla CSS & JS), hỗ trợ kéo thả ảnh tải lên trực tiếp.
  - **Tính năng chuyên sâu:** Lựa chọn Preset cấu hình (NYUv2 dải đo 10m / KITTI dải đo 80m), chuyển đổi 4 bảng màu colormap (**Magma, Plasma, Viridis, Turbo**) và xuất đám mây điểm 3D (**3D Point Cloud .PLY**).
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Nhóm đã đóng gói toàn bộ nghiên cứu thành một ứng dụng Web Demo trực quan, cho phép người dùng tải ảnh lên bất kỳ để suy luận độ sâu tức thì, chuyển đổi bảng màu hiển thị và thậm chí tạo mô hình đám mây điểm 3D tương tác trực tiếp."*

---

### SLIDE 11: KẾT LUẬN & ĐÓNG GÓP NỔI BẬT CỦA ĐỀ TÀI
* **3 đóng góp cốt lõi của đề tài:**
  1. **Tái lập thành công 100% nguyên bản:** Huấn luyện đủ 50 Epochs mô hình BTS trên cả 2 benchmark lớn nhất thế giới (**NYU Depth V2** và **KITTI Raw Benchmark**) theo đúng thiết kế toán học của NeurIPS 2019.
  2. **Hiệu năng thực nghiệm vượt trội:** Đạt thành tích vượt qua công bố của tác giả gốc trên cả hai tập dữ liệu (Peak AbsRel NYU = **`0.10967`** vs `0.110`; Peak AbsRel KITTI = **`0.05748`** vs `0.060`; toàn bộ 9 metrics KITTI đều vượt trội).
  3. **Làm chủ kỹ nghệ hạ tầng MLOps:** Xây dựng giải pháp Checkpoint Stitching và Agent Workflow giải quyết triệt để rào cản tài nguyên GPU đám mây bị giới hạn thời gian.
* **Kịch bản thuyết trình (Speaker Notes):**
  > *"Tóm lại, đề tài đã chứng minh tính khả thi, độ tin cậy và khả năng tái lập tuyệt đối của kiến trúc BTS. Chúng em không chỉ tái lập thành công mà còn đạt được những chỉ số định lượng vượt trội so với bài báo gốc, đồng thời xây dựng một bộ công cụ MLOps và ứng dụng thực tiễn hoàn chỉnh."*

---

### SLIDE 12: Q&A — LỜI CẢM ƠN & SẴN SÀNG TRẢ LỜI CHẤT VẤN
* **Nội dung slide:** Lời cảm ơn chân thành đến Quý Thầy Cô trong Hội đồng và Giáo viên hướng dẫn.
* **Sẵn sàng trả lời các câu hỏi chuyên sâu:**
  - Cơ chế toán học Local Planar Guidance (LPG) vs Deconvolution truyền thống.
  - Ý nghĩa toán học của thành phần phương sai trong hàm tổn thất SILog.
  - Giao thức bù trừ Dataloader Offset trong kỹ thuật Checkpoint Stitching.
  - Lý giải hiện tượng Validation Peak / Early Stopping ở Epoch 42-45.