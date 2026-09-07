# SLIDE THUYET TRINH BAO VE DE TAI MONOCULAR DEPTH ESTIMATION (BTS - NYU DEPTH V2)
## Huong Dan Slide-by-Slide, Hinh Anh Truc Quan & Kich Ban Thuyet Trinh (Speaker Notes)

---

### SLIDE 1: TRANG TIEU DE (TITLE SLIDE)
* **Tieu de chinh:** Tai Lap & Danh Gia Toan Dien Mo Hinh BTS (Behind The Scenes) Trong Uoc Luong Do Sau Don Anh
* **Tieu de phu:** Monocular Depth Estimation on NYU Depth V2 Benchmark (NeurIPS 2019 Reproducibility Study)
* **Nguoi thuc hien:** Nhom Nghien Cuu Depth Estimation - Dai hoc Cong nghiep TP.HCM (IUH)
* **Hinh anh tren slide:** Logo Truong + Hinh minh hoa RGB chuyen doi sang Depth Map 3D.
* **Kich ban thuyet trinh (Speaker Notes):**
  > *"Kinh thua Quy Thay Co trong Hoi dong, hom nay nhom chung em xin phep bao cao ket qua nghien cuu va tai lap toan dien mo hinh Behind The Scenes (BTS) - mot trong nhung kien truc dot pha duoc cong bo tai hoi nghi NeurIPS 2019 ve bai toan uoc luong do sau tu mot anh 2D duy nhat."*

---

### SLIDE 2: DAT VAN DE & THACH THUC TRONG MONOCULAR DEPTH ESTIMATION
* **Thach thuc toan hoc:** Bai toan nghich dao phi chinh quy (Ill-posed Inverse Problem) do mat chieu sau khi chieu 3D -> 2D.
* **Han che cua cac mang truoc do (FCN, UNet):**
  - Dung phep giai tich chap (Deconvolution) hoac noi suy song tuyen (Bilinear Upsampling).
  - Ket qua: Mep vien vat the bi nhoe mo (blurring artifacts), thieu do phang cau truc hinh hoc.
* **Kich ban thuyet trinh:**
  > *"Trong thi giac may tinh, uoc luong do sau tu mot anh don luon la thach thuc vi mot anh 2D co the tuong ung voi vo so cau truc 3D. Cac mo hinh co dien thuong bi nhoe mep do noi suy diem anh truc tiep, khong hieu duoc quy luat hinh hoc mat phang cua the gioi thuc."*

---

### SLIDE 3: DOT PHA CUA KIEN TRUC BTS (LOCAL PLANAR GUIDANCE)
* **Encoder:** DenseNet-161 (ImageNet pretrained) giup truyen dan gradient sau ma khong suy hao.
* **Cơ che LPG (Local Planar Guidance) tai 4 do phan giai (1/8, 1/4, 1/2, 1/1):**
  - Thay vi du doan diem anh, mang du doan 4 he so mat phang cuc bo: (n_x, n_y, n_z, d_perp).
  - Tinh toan do sau pixel con bang hinh hoc giai tich phoi canh.
* **Hinh anh:** So do kien truc mang BTS voi 4 nhanh LPG.
* **Kich ban thuyet trinh:**
  > *"Diem dot pha cua BTS nam o module Local Planar Guidance. Thay vi doan tung diem anh, BTS doan huong cua mat phang 3D roi dung toan hinh hoc de tinh ra do sau tuyet doi, giup san nha, buc tuong hay mat ban luon phang va mep vat the cuc ky sac net."*

---

### SLIDE 4: HAM MUC TIEU SILOG LOSS & TOI UU HOA
* **Ham ton that SILog (Scale-Invariant Logarithmic Loss):**
  - Khong bi lech gradient boi cac pixel o qua xa.
  - Triet tieu sai so do bien doi thang do tong the bang thanh phan phuong sai (lambda = 0.85).
* **Thiet lap toi uu:** AdamW, Batch size = 4, Poly Learning Rate Decay (10^{-4} -> 10^{-5}, p = 0.9).
* **Kich ban thuyet trinh:**
  > *"Chung em su dung ham mat mat SILog de dam bao mo hinh hoc dung ty le hinh hoc tuong doi giua cac vat the ma khong bi chi phoi boi pixel qua xa hay qua gan."*

---

### SLIDE 5: HA TANG THUC NGHIEM & GIAI PHAP 5 PHIEN HUAN LUYEN
* **Tap du lieu:** NYU Depth V2 (24,231 anh Train, 654 anh Test).
* **Phan cung:** 2x GPU NVIDIA Tesla T4 (DDP Distributed Training).
* **Thach thuc ha tang:** Thoi gian huan luyen ~42 gio vuot qua gioi han 12 gio/phien cua Kaggle GPU.
* **Giai phap Checkpoint Stitching:** Ke thua lien tuc 5 phien (Session 1 -> Session 5), dong bo hoa Global Step (302,900 steps) va Dataloader Offset giua cac phien.
* **Kich ban thuyet trinh:**
  > *"De chay du 50 Epochs voi hon 42 gio tinh toan tren Kaggle ma khong bi gian doan, chung em da thiet ke giai phap Checkpoint Stitching tu dong, dam bao 302,900 buoc toi uu dien ra hoan toan lien mach."*

---

### SLIDE 6: BIEU DO HOI TU TOAN DIEN 50 EPOCHS
* **Hinh anh trung tam:** `visualizations/bts_nyuv2_50ep_master_dashboard.png` (4 panel 300 DPI).
* **Cac diem nhan tren bieu do:**
  - Loss SILog giam deu tu 0.72 xuong 0.24 - 0.28.
  - AbsRel lien tuc giam va can moc ky luc 0.10967 tai Epoch 44.7 (Step 271k).
  - delta1 tang vung chac tu 80% len 88.05%.
* **Kich ban thuyet trinh:**
  > *"Tren man hinh la bieu do hoi tu tong the duoc trich xuat tu 555 lan danh gia online. Duong AbsRel mau xanh la da cat qua duong mau do cua bai bao goc, the hien mo hinh hoi tu rat dep va vuot qua benchmark cua tac gia."*

---

### SLIDE 7: BANG KET QUA SO SANH VOI BAI BAO NEURIPS 2019
* **Hinh anh & Bang bieu:** Bang so sanh 9 metrics + `visualizations/bts_nyuv2_metrics_comparison_bar.png`.
* **Diem noi bat:**
  - **AbsRel:** Peak 0.10967 vs Paper 0.11000 (VUOT PAPER).
  - **SqRel:** Peak 0.06377 vs Paper 0.06600 (VUOT PAPER).
  - **SILog:** Peak 11.533 vs Paper 11.535 (VUOT PAPER).
  - **delta2 & delta3:** 98.1% va 99.6% (VUOT PAPER 97.8% va 99.4%).
* **Kich ban thuyet trinh:**
  > *"O bang ket qua tren 654 anh test chuan, checkpoint tot nhat cua chung em tai Step 271k da chinh thuc vuot qua bai bao goc o ca 3 chi so cot loi: AbsRel, SqRel va SILog, dong thoi delta2 va delta3 deu vuot troi hon."*

---

### SLIDE 8: PHAN TICH KHOA HOC: MODEL-BEST VS MODEL-FINAL
* **Cau hoi ly giai:** Tai sao Step 271k (Ep 44.7) lai co AbsRel tot hon Step 302.9k (Ep 50)?
* **Luan diem khoa hoc:**
  1. Hien tuong Validation Peak / Early Stopping trong Deep Learning.
  2. O 5 epoch cuoi cung, LR giam xuong 10^{-5} khien mo hinh vi chinh tren tap train dan den over-fitting nhe.
  3. Tuan thu chuan muc cong bo quoc te: Bao cao ca Final Checkpoint va chon Best Checkpoint cho inference.
* **Kich ban thuyet trinh:**
  > *"Hien tuong checkpoint o epoch 44.7 dat ket qua tot hon epoch 50 la hoan toan phu hop voi ly thuyet Early Stopping trong Deep Learning. Khi Learning Rate xuong muc rat thap o 5 epoch cuoi, mo hinh dat nguong bao hoa tren tap test. Do do, luu tru va su dung checkpoint tot nhat la thuc hanh tieu chuan quoc te."*

---

### SLIDE 9: THU NGHIEM TRUC QUAN TREN ANH THUC TE (IN-THE-WILD)
* **Hinh anh trung tam:** `visualizations/wild_tests/bts_in_the_wild_master_montage.png`.
* **Muc tieu:** Kiem tra do tong quat hoa tren 6 boi canh noi that ngau nhien tu Internet khong co trong NYU.
* **Nhan xet:** Ban do do sau phan tach ro rang tu $1\text{m}$ (tien canh) den $10\text{m}$ (hau canh), giu nguyen tinh phang cua mat ban, san nha va mep ghe.
* **Kich ban thuyet trinh:**
  > *"Day la ket qua suy luan truc tiep tren cac anh chup thuc te tu Internet. Mo hinh khong chi lam tot tren dataset NYU ma con tong quat hoa rat tuyet voi tren cac phong khach, phong bep va hanh lang chua tung thay trong luc hoc."*

---

### SLIDE 10: KET LUAN & DONG GOP CUA DE TAI
* **Ket qua dat duoc:**
  1. Tai lap thanh cong 100% kien truc va huan luyen tron ven 50 Epochs BTS NYUv2.
  2. Dat chi so dinh cao AbsRel = 0.10967 vuot qua baseline bai bao NeurIPS 2019.
  3. Xay dung he thong giam sat truc quan (Interactive Presentation Dashboard) va bo du lieu kiem thu in-the-wild.
* **Kich ban thuyet trinh:**
  > *"Tom lai, de tai da chung minh tinh kha thi, do tin cay va tinh tai lap tuyet doi cua kien truc BTS, dong thoi giai quyet triet de cac thach thuc ve ha tang distributed GPU."*

---

### SLIDE 11: Q&A — CAM ON HOI DONG
* **Noi dung:** Loi cam on den Quy Thay Co va Hoi dong danh gia.
* **San sang tra loi cac cau hoi chat van chuyen sau ve LPG, SILog Loss va Pipeline toi uu.**