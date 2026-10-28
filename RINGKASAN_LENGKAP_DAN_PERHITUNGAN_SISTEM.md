# 🛡️ DOKUMEN KOMPREHENSIF SKRIPSI: RINGKASAN, ARSITEKTUR, DAN PERHITUNGAN LENGKAP SISTEM

> **Judul Skripsi:** Klasifikasi Gambar Postingan Internet Menggunakan Arsitektur MobileNetV2 (Multi-Feature Fusion & Grad-CAM)  
> **Penyusun:** Nanda Riansyah (NPM: 235520110108)  
> **Program Studi / Institusi:** Teknik Informatika — Universitas Almuslim  
> **Repositori Proyek:** `D:\skripsi`  
> **Format Dokumen:** Acuan Teknis Lengkap, Formula Matematis, dan Panduan Sidang  

---

## DAFTAR ISI

1. [Analisis Fundamental 5W + 1H](#1-analisis-fundamental-5w--1h)
2. [Arsitektur Sistem & Alur Multi-Feature Fusion](#2-arsitektur-sistem--alur-multi-feature-fusion)
3. [Rincian Dimensi Vektor & Parameter Komputasi](#3-rincian-dimensi-vektor--parameter-komputasi)
4. [Mekanisme Fallback Routing](#4-mekanisme-fallback-routing)
5. [Strategi Pelatihan Dua Fase (Two-Phase Training)](#5-strategi-pelatihan-dua-fase-two-phase-training)
6. [Formulasi Matematis & Perhitungan Lengkap Metrik Evaluasi](#6-formulasi-matematis--perhitungan-lengkap-metrik-evaluasi)
7. [Perhitungan Loss & Label Smoothing Cross-Entropy](#7-perhitungan-loss--label-smoothing-cross-entropy)
8. [Formulasi Grad-CAM (Explainable AI)](#8-formulasi-grad-cam-explainable-ai)
9. [Struktur Direktori Codebase & Komponen Modul](#9-struktur-direktori-codebase--komponen-modul)
10. [Panduan Operasional & Demo Sistem](#10-panduan-operasional--demo-sistem)
11. [Bank Tanya-Jawab Kritis untuk Sidang Skripsi](#11-bank-tanya-jawab-kritis-untuk-sidang-skripsi)

---

## 1. ANALISIS FUNDAMENTAL 5W + 1H

### What (Apa yang Dibangun?)
Sistem moderasi dan deteksi konten visual otomatis berbasis *Deep Learning* (*Computer Vision*) yang mengklasifikasikan postingan gambar digital ke dalam 3 kelas *mutually-exclusive*:
- **Safe (Label 0):** Gambar aman untuk segala usia (pemandangan, keluarga, hewan, makanan, aktivitas olahraga wajar).
- **Violence (Label 1):** Aksi kekerasan fisik, senjata tajam/api, penyerangan fisik, luka berdarah, ledakan.
- **NSFW (Label 2):** *Not Safe for Work* (konten dewasa, pornografi, ketelanjangan, bagian tubuh sensitif).

Model dibangun menggunakan arsitektur **MobileNetV2** yang diperkuat dengan inovasi **Multi-Feature Fusion** (fitur global, tampilan manusia hasil *crop*, estimasi postur kerangka tubuh, dan geometri spasial antar-orang) serta **Grad-CAM** (*Explainable AI*).

### Why (Mengapa Sistem Ini Dibuat?)
1. **Penyaringan Konten Masif di Internet:** Meningkatnya peredaran konten kekerasan dan materi pornografi di media sosial yang membahayakan anak di bawah umur serta memerlukan filter otomatis berkecepatan tinggi.
2. **Menyelesaikan Masalah *Shortcut Learning* (Bias Warna):** Model CNN konvensional yang hanya melihat gambar utuh sering mengambil jalan pintas dengan mengasosiasikan warna merah dengan darah (*Violence*). Akibatnya, gambar normal seperti gaun merah, mobil merah, atau bunga mawar sering salah terdeteksi sebagai kekerasan. Pendekatan *multi-feature* memverifikasi konteks subjek manusia dan postur tubuh untuk memutus bias ini.

### Who (Siapa Subjek & Target Penggunanya?)
- **Pengembang:** Nanda Riansyah (Universitas Almuslim).
- **Target Pengguna:** Moderator konten platform media sosial, pengelola forum web, penyedia layanan penyimpanan awan (*cloud storage*), orang tua (*parental control*), serta instansi pemeriksa kepatuhan konten digital.

### Where (Di Mana Sistem Dapat Digunakan?)
Sistem dirancang modular (*multi-interface*):
- **Server Cloud/On-Premise:** Melalui layanan REST API [api/app.py](file:///D:/skripsi/api/app.py) berbasis Flask.
- **Aplikasi Web Interaktif:** Melalui dashboard [streamlit_app.py](file:///D:/skripsi/streamlit_app.py) untuk pengujian publik/browser.
- **Aplikasi Komputer Desktop:** Melalui GUI modern [desktop_app.py](file:///D:/skripsi/desktop_app.py) berbasis CustomTkinter untuk penggunaan lokal *offline*.

### When (Kapan Sistem Bekerja?)
- **Real-Time Inference:** Memproses gambar tunggal dengan latensi sangat rendah (~35–50 milidetik per gambar).
- **Batch Processing:** Memproses audit folder ribuan gambar secara terjadwal atau saat proses unggahan massal ke server.

### How (Bagaimana Cara Kerjanya?)
Sistem mengekstraksi representasi dari 4 cabang secara simultan:
1. Konteks visual global via MobileNetV2 (1.280 dimensi).
2. Deteksi subjek manusia dengan *padding* 15% via MobileNetV2 (1.280 dimensi).
3. 12 titik sendi postur tubuh MediaPipe Pose (36 dimensi).
4. Relasi geometri spasial antar manusia untuk hingga 3 orang (18 dimensi).
5. Data dipadukan melalui *Gated Feature Fusion* (2.626 dimensi) menuju *Classification Head* (Dense + Softmax). Keputusan klasifikasi kemudian divisualisasikan menggunakan peta panas *Grad-CAM*.

---

## 2. ARSITEKTUR SISTEM & ALUR MULTI-FEATURE FUSION

Diagram alir lengkap pemrosesan citra dari input hingga output:

```
                          CITRA INPUT (Ukuran Variabel)
                                       │
                                       ▼
                     PREPROCESSING: Resize ke (224, 224, 3)
                            Normalisasi Nilai [0, 1]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     CABANG 1: GLOBAL                                     CABANG DETEKSI MANUSIA
   MobileNetV2 (Pre-trained)                             MediaPipe Pose Landmarker
   Global Average Pooling (GAP)                                   │
            │                                             Ada manusia terdeteksi?
            │                                             ├── YA:
            │                                             │   ├─► Bounding Box + 15% Padding
            │                                             │   │   Crop & Resize (224, 224)
            │                                             │   │   MobileNetV2 Appearance (GAP)
            │                                             │   │   ──► CABANG 2: HUMAN CROP
            │                                             │   │
            │                                             │   ├─► 12 Titik Kunci Kerangka (x, y, vis)
            │                                             │   │   ──► CABANG 3: POSE LANDMARKS
            │                                             │   │
            │                                             │   └─► Kalkulasi Geometri Spasial Antar-Orang
            │                                             │       ──► CABANG 4: SPATIAL RELATIONS
            │                                             │
            │                                             └── TIDAK:
            │                                                 Nol-kan Vektor Cabang 2, 3, dan 4
            │                                                 (Case C: Global Fallback)
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                           GATED FEATURE FUSION LAYER
                    Penggabungan 4 Vektor + 2 Gating Flags
                               Total = 2.626 Dimensi
                                       │
                                       ▼
                              CLASSIFICATION HEAD
               Dense(512, ReLU) ──► BatchNormalization ──► Dropout(0.40)
                                       │
                                       ▼
                        Dense(256, ReLU) ──► Dropout(0.30)
                                       │
                                       ▼
                       Dense(3, Softmax Activation)
                                       │
                                       ▼
                 PROBABILITAS KELAS: [ P(Safe), P(Violence), P(NSFW) ]
                                       │
                                       ▼
                    EXPLAINABLE AI: Peta Panas Grad-CAM
```

---

## 3. RINCIAN DIMENSI VEKTOR & PARAMETER KOMPUTASI

### A. Rincian Dimensi Vektor Peleburan (Feature Fusion)

Total dimensi vektor yang dihasilkan sebelum masuk ke *head classifier* adalah **2.626 dimensi**:

1. **Vektor Global Image ($V_{\text{global}}$): 1.280 dimensi**
   - Output layer konvolusi terakhir MobileNetV2: $(7 \times 7 \times 1280)$.
   - Diterapkan *Global Average Pooling 2D* (GAP):
     $$\text{GAP}(A) = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 A_{i, j, k} \quad \forall k \in [1, 1280] \implies \mathbb{R}^{1280}$$
2. **Vektor Human Appearance ($V_{\text{human}}$): 1.280 dimensi**
   - Potongan area manusia dengan bantalan 15% (*15% padding*) di-resize ke $(224, 224, 3)$.
   - Diekstraksi melalui MobileNetV2 + GAP $\implies \mathbb{R}^{1280}$.
3. **Vektor Pose & Spatial Relations ($V_{\text{pose\_rel}}$): 64 dimensi**
   - **Landmark Pose (36 dimensi):** 12 titik kunci utama (Hidung, Bahu Kiri/Kanan, Siku Kiri/Kanan, Pergelangan Kiri/Kanan, Pinggul Kiri/Kanan, Lutut Kiri/Kanan, Pergelangan Kaki Kiri).
     $$\text{Dimensi} = 12 \text{ titik} \times 3 \text{ komponen } (x, y, \text{visibility}) = 36$$
   - **Relasi Spasial Antar-Manusia (18 dimensi):** Untuk hingga $N=3$ subjek manusia, terdapat $\binom{3}{2} = 3$ pasangan relasi. Tiap pasangan menghitung 6 fitur:
     $$\text{Fitur} = [\text{Jarak Euclidean}, \text{IoU Overlap}, \Delta x_{\text{rel}}, \Delta y_{\text{rel}}, \text{Rasio Lebar}, \text{Rasio Tinggi}]$$
     $$\text{Dimensi} = 3 \text{ pasangan} \times 6 \text{ atribut} = 18$$
   - **Proyeksi Dense Terpadu:**
     $$V_{\text{raw\_pose}} = [36 + 18] = 54 \text{ dimensi} \xrightarrow{\text{Dense}(64, \text{ReLU})} \mathbb{R}^{64}$$
4. **Gating Flags ($V_{\text{gate}}$): 2 dimensi**
   - Indikator boolean bernilai numerik $[0, 1]$:
     - `has_human`: $1.0$ jika manusia terdeteksi, $0.0$ jika tidak ada.
     - `has_pose`: $1.0$ jika kerangka sendi terestimasi, $0.0$ jika tidak ada.

$$\text{Total Dimensi} = 1280 + 1280 + 64 + 2 = 2.626 \text{ dimensi}$$

### B. Perhitungan Jumlah Parameter Model

| Layer / Komponen | Bentuk Input | Bentuk Output | Formula Perhitungan Parameter | Jumlah Parameter |
| :--- | :---: | :---: | :--- | :---: |
| **MobileNetV2 Backbone** | $(224, 224, 3)$ | $(7, 7, 1280)$ | Konvolusi Separable & Inverted Residuals | 2.257.984 |
| **Pose Projection Dense**| 54 | 64 | $(54 \times 64) + 64$ | 3.520 |
| **Dense 1 (Fusion)** | 2.626 | 512 | $(2626 \times 512) + 512$ | 1.345.024 |
| **Batch Normalization** | 512 | 512 | $4 \times 512$ $(\gamma, \beta, \mu, \sigma^2)$ | 2.048 |
| **Dense 2** | 512 | 256 | $(512 \times 256) + 256$ | 131.328 |
| **Dense Output (Softmax)**| 256 | 3 | $(256 \times 3) + 3$ | 771 |
| **TOTAL KESELURUHAN** | — | — | — | **~3.672.067 Parameter** |

*Catatan:* Bobot MobileNetV2 digunakan bersama (*weight sharing*) antara cabang global dan cabang potongan manusia, sehingga efisiensi memori tetap terjaga.

---

## 4. MEKANISME FALLBACK ROUTING

Sistem memiliki 3 jalur rute dinamis (*Case Routing*) saat memproses gambar di dunia nyata:

```
[Inference Routing Logic]
                ┌──────────────────────────────────────┐
                │        HumanAnalyzer Mengevaluasi    │
                └──────────────────┬───────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Manusia Terdeteksi? (Ya)                   Manusia Terdeteksi? (Tidak)
              │                                         │
     ┌────────┴────────┐                                │
     ▼                 ▼                                │
Pose Sukses? (Ya)  Pose Gagal? (Tidak)                  │
     │                 │                                │
  [CASE A]          [CASE B]                         [CASE C]
Full Multi-Feature  Global + Human Crop              Global Fallback Only
(has_human=1)       (has_human=1)                    (has_human=0)
(has_pose=1)        (has_pose=0)                     (has_pose=0)
Vektor: 2.626-dim   Vektor Pose dinolkan             Vektor Human & Pose dinolkan
```

- **Case A (`global+human+pose`):** Berjalan pada foto kerumunan atau orang berjarak sedang/dekat di mana kerangka sendi terlihat jelas.
- **Case B (`global+human`):** Berjalan jika orang terlihat sangat dekat (close-up wajah/tubuh saja) atau tertutup objek sehingga MediaPipe tidak dapat membentuk 12 titik sendi lengkap.
- **Case C (`global_fallback`):** Berjalan pada foto benda mati (senjata di atas meja, gedung terbakar, mobil, lanskap pantai). Menghilangkan ketergantungan wajib pada manusia.

---

## 5. STRATEGI PELATIHAN DUA FASE (TWO-PHASE TRAINING)

Untuk memaksimalkan performa *Transfer Learning*, proses pelatihan dibagi menjadi 2 fase:

### Fase 1: Feature Extraction (Backbone Dibekukan)
- **Status Backbone:** Seluruh layer MobileNetV2 dikunci (`trainable = False`).
- **Objektif:** Melatih bobot awal pada *Classification Head* (Dense 512, Dense 256, Dense 3) agar selaras dengan representasi ImageNet tanpa merusak fitur konvolusi dasar.
- **Learning Rate:** $\eta = 1 \times 10^{-4}$ (Adam Optimizer).
- **Epochs:** 12 epoch dengan *EarlyStopping* (patience = 7 terhadap `val_loss`).

### Fase 2: Fine-Tuning (Penyesuaian Layer Atas)
- **Status Backbone:** 30 layer konvolusi teratas dibuka kuncinya (`trainable = True`).
- **Pengecualian Kritis:** Seluruh layer `BatchNormalization` tetap dibekukan (`layer.trainable = False`) agar statistik rata-rata (*mean*) dan varians tidak rusak oleh ukuran batch kecil.
- **Objektif:** Menyesuaikan filter konvolusi tingkat tinggi agar mengenali tekstur spesifik luka, senjata tajam, atau ketelanjangan.
- **Learning Rate:** $\eta = 1 \times 10^{-5}$ (sangat kecil agar perubahan bobot stabil).
- **Epochs:** 10 epoch dengan *EarlyStopping* (patience = 5).

---

## 6. FORMULASI MATEMATIS & PERHITUNGAN LENGKAP METRIK EVALUASI

### A. Data Aktual Matriks Konfusi (Confusion Matrix)
Pengujian dilakukan terhadap **3.040 citra uji independen** (`dataset_large/test/`):

| Kelas Aktual \ Prediksi | Prediksi: Safe | Prediksi: Violence | Prediksi: NSFW | **Total Aktual** |
| :--- | :---: | :---: | :---: | :---: |
| **Aktual: Safe** | **805 (TP)** | 113 | 115 | **1.033** |
| **Aktual: Violence** | 47 | **787 (TP)** | 122 | **956** |
| **Aktual: NSFW** | 7 | 8 | **1.036 (TP)** | **1.051** |
| **Total Prediksi** | **859** | **908** | **1.273** | **3.040** |

---

### B. Perhitungan Akurasi Keseluruhan (Overall Accuracy)

$$\text{Accuracy} = \frac{\sum \text{Prediksi Benar}}{\text{Total Sampel Uji}} = \frac{TP_{\text{Safe}} + TP_{\text{Violence}} + TP_{\text{NSFW}}}{N}$$

$$\text{Perhitungan:} \quad \text{Accuracy} = \frac{805 + 787 + 1.036}{3.040} = \frac{2.628}{3.040} \approx \mathbf{0,86447 \ (86,45\%)}$$

---

### C. Perhitungan Rinci per Kelas (One-vs-Rest)

#### 1. Kelas Safe (Aman)
- **True Positive (TP):** Gambar Safe diprediksi Safe = **805**
- **False Positive (FP):** Gambar non-Safe diprediksi Safe = $47 (\text{Violence}) + 7 (\text{NSFW}) = \mathbf{54}$
- **False Negative (FN):** Gambar Safe salah diprediksi = $113 (\text{Violence}) + 115 (\text{NSFW}) = \mathbf{228}$
- **True Negative (TN):** $3.040 - (805 + 54 + 228) = \mathbf{1.953}$

$$\text{Precision}_{\text{Safe}} = \frac{TP}{TP + FP} = \frac{805}{805 + 54} = \frac{805}{859} \approx \mathbf{0,93714 \ (93,71\%)}$$

$$\text{Recall}_{\text{Safe}} = \frac{TP}{TP + FN} = \frac{805}{805 + 228} = \frac{805}{1.033} \approx \mathbf{0,77928 \ (77,93\%)}$$

$$\text{F1-Score}_{\text{Safe}} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0,93714 \times 0,77928}{0,93714 + 0,77928} = \frac{1,46056}{1,71642} \approx \mathbf{0,85093 \ (85,10\%)}$$

$$\text{Specificity}_{\text{Safe}} = \frac{TN}{TN + FP} = \frac{1.953}{1.953 + 54} = \frac{1.953}{2.007} \approx \mathbf{0,97309 \ (97,31\%)}$$

---

#### 2. Kelas Violence (Kekerasan)
- **True Positive (TP):** Gambar Violence diprediksi Violence = **787**
- **False Positive (FP):** Gambar non-Violence diprediksi Violence = $113 (\text{Safe}) + 8 (\text{NSFW}) = \mathbf{121}$
- **False Negative (FN):** Gambar Violence salah diprediksi = $47 (\text{Safe}) + 122 (\text{NSFW}) = \mathbf{169}$
- **True Negative (TN):** $3.040 - (787 + 121 + 169) = \mathbf{1.963}$

$$\text{Precision}_{\text{Violence}} = \frac{TP}{TP + FP} = \frac{787}{787 + 121} = \frac{787}{908} \approx \mathbf{0,86674 \ (86,67\%)}$$

$$\text{Recall}_{\text{Violence}} = \frac{TP}{TP + FN} = \frac{787}{787 + 169} = \frac{787}{956} \approx \mathbf{0,82322 \ (82,32\%)}$$

$$\text{F1-Score}_{\text{Violence}} = 2 \times \frac{0,86674 \times 0,82322}{0,86674 + 0,82322} = \frac{1,42704}{1,68996} \approx \mathbf{0,84442 \ (84,44\%)}$$

$$\text{Specificity}_{\text{Violence}} = \frac{TN}{TN + FP} = \frac{1.963}{1.963 + 121} = \frac{1.963}{2.084} \approx \mathbf{0,94194 \ (94,19\%)}$$

---

#### 3. Kelas NSFW (Konten Dewasa)
- **True Positive (TP):** Gambar NSFW diprediksi NSFW = **1.036**
- **False Positive (FP):** Gambar non-NSFW diprediksi NSFW = $115 (\text{Safe}) + 122 (\text{Violence}) = \mathbf{237}$
- **False Negative (FN):** Gambar NSFW salah diprediksi = $7 (\text{Safe}) + 8 (\text{Violence}) = \mathbf{15}$
- **True Negative (TN):** $3.040 - (1.036 + 237 + 15) = \mathbf{1.752}$

$$\text{Precision}_{\text{NSFW}} = \frac{TP}{TP + FP} = \frac{1.036}{1.036 + 237} = \frac{1.036}{1.273} \approx \mathbf{0,81383 \ (81,38\%)}$$

$$\text{Recall}_{\text{NSFW}} = \frac{TP}{TP + FN} = \frac{1.036}{1.036 + 15} = \frac{1.036}{1.051} \approx \mathbf{0,98573 \ (98,57\%)}$$

$$\text{F1-Score}_{\text{NSFW}} = 2 \times \frac{0,81383 \times 0,98573}{0,81383 + 0,98573} = \frac{1,60447}{1,79956} \approx \mathbf{0,89159 \ (89,16\%)}$$

$$\text{Specificity}_{\text{NSFW}} = \frac{TN}{TN + FP} = \frac{1.752}{1.752 + 237} = \frac{1.752}{1.989} \approx \mathbf{0,88084 \ (88,08\%)}$$

---

### D. Perhitungan Rata-Rata Makro (Macro Average)

Memberikan bobot yang sama untuk setiap kelas tanpa memandang jumlah sampel:

$$\text{Macro Precision} = \frac{0,93714 + 0,86674 + 0,81383}{3} = \frac{2,61771}{3} \approx \mathbf{0,87257 \ (87,26\%)}$$

$$\text{Macro Recall} = \frac{0,77928 + 0,82322 + 0,98573}{3} = \frac{2,58823}{3} \approx \mathbf{0,86274 \ (86,27\%)}$$

$$\text{Macro F1-Score} = \frac{0,85093 + 0,84442 + 0,89159}{3} = \frac{2,58694}{3} \approx \mathbf{0,86231 \ (86,23\%)}$$

---

### E. Perhitungan Rata-Rata Tertimbang (Weighted Average)

Memperhitungkan proporsi jumlah sampel (*support*) masing-masing kelas ($N_{\text{Safe}}=1.033, N_{\text{Violence}}=956, N_{\text{NSFW}}=1.051$):

$$\text{Weighted Precision} = \frac{(0,93714 \times 1033) + (0,86674 \times 956) + (0,81383 \times 1051)}{3040} = \frac{968,06 + 828,60 + 855,33}{3040} \approx \mathbf{0,87236 \ (87,24\%)}$$

$$\text{Weighted Recall} = \frac{(0,77928 \times 1033) + (0,82322 \times 956) + (0,98573 \times 1051)}{3040} = \frac{805 + 787 + 1036}{3040} = \frac{2628}{3040} \approx \mathbf{0,86447 \ (86,45\%)}$$

$$\text{Weighted F1-Score} = \frac{(0,85093 \times 1033) + (0,84442 \times 956) + (0,89159 \times 1051)}{3040} = \frac{879,01 + 807,27 + 937,06}{3040} \approx \mathbf{0,86294 \ (86,30\%)}$$

---

## 7. PERHITUNGAN LOSS & LABEL SMOOTHING CROSS-ENTROPY

Untuk mencegah model menjadi terlalu percaya diri (*overconfident*) pada probabilitas 1.0 yang dapat memicu *overfitting*, diterapkan **Label Smoothing** sebesar $\alpha = 0,05$ untuk $K = 3$ kelas:

### Formula Transformasi Label Smoothing
$$y_k^{\text{LS}} = y_k (1 - \alpha) + \frac{\alpha}{K}$$

- **Untuk Kelas Target Sebenarnya ($y_k = 1$):**
  $$y_{\text{true}}^{\text{LS}} = 1 \times (1 - 0,05) + \frac{0,05}{3} = 0,95 + 0,01667 = \mathbf{0,96667}$$
- **Untuk Kelas Lainnya ($y_k = 0$):**
  $$y_{\text{false}}^{\text{LS}} = 0 \times (1 - 0,05) + \frac{0,05}{3} = 0 + 0,01667 = \mathbf{0,01667}$$

### Contoh Perhitungan Nilai Loss
Misalkan sebuah citra Safe ($y = [1, 0, 0]$ sehingga $y^{\text{LS}} = [0,9667, 0,0167, 0,0167]$) menghasilkan prediksi probabilitas Softmax $\hat{y} = [0,85, 0,10, 0,05]$:

$$\mathcal{L}_{\text{CE}} = - \sum_{k=1}^3 y_k^{\text{LS}} \ln(\hat{y}_k)$$
$$\mathcal{L}_{\text{CE}} = - [ (0,9667 \times \ln(0,85)) + (0,0167 \times \ln(0,10)) + (0,0167 \times \ln(0,05)) ]$$
$$\mathcal{L}_{\text{CE}} = - [ (0,9667 \times (-0,1625)) + (0,0167 \times (-2,3026)) + (0,0167 \times (-2,9957)) ]$$
$$\mathcal{L}_{\text{CE}} = - [ -0,15709 - 0,03845 - 0,05003 ] = - [ -0,24557 ] = \mathbf{0,24557}$$

---

## 8. FORMULASI GRAD-CAM (EXPLAINABLE AI)

Grad-CAM (*Gradient-weighted Class Activation Mapping*) menghitung kontribusi fitur spasial pada layer konvolusi terakhir MobileNetV2 (`out_relu`) terhadap skor logit kelas $c$:

### 1. Perhitungan Bobot Kepentingan Fitur ($\alpha_k^c$)
Rata-rata gradien dari skor kelas $Y^c$ terhadap *feature map* $A^k$ berukuran $u \times v$ ($7 \times 7$):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial Y^c}{\partial A_{i, j}^k} \quad \text{dengan } Z = u \times v = 49$$

### 2. Kombinasi Linier & Rektifikasi ReLU
Heatmap lokal diperoleh dengan menjumlahkan seluruh *feature map* yang telah dikalikan bobot $\alpha_k^c$, lalu diterapkan fungsi ReLU untuk hanya mempertahankan piksel yang berkontribusi positif:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{1280} \alpha_k^c A^k \right)$$

### 3. Normalisasi & Overlay Visual
Peta panas dinormalisasi ke rentang $[0, 1]$, diubah ukurannya (*bilinear interpolation*) ke resolusi asli gambar $(224 \times 224)$, lalu diwarnai menggunakan colormap Jet/Turbo untuk ditumpuk (*alpha blend*) di atas gambar asli.

---

## 9. STRUKTUR DIREKTORI CODEBASE & KOMPONEN MODUL

Struktur file implementasi pada repositori `D:\skripsi`:

```text
D:\skripsi\
│
├── src/                                  # Modul Sumber Utama (Python Package)
│   ├── config.py                         # Definisi konstanta, hyperparameter, path, dan ambang batas
│   ├── model.py                          # Konstruksi MobileNetV2 Multi-Feature Keras Functional Model
│   ├── human_analyzer.py                 # Deteksi pose MediaPipe, crop padding, ekstraksi relasi spasial
│   ├── dataset.py                        # Multi-input data generator, augmentasi Grayscale & Color-Jitter
│   ├── train.py                          # Skrip eksekusi pelatihan 2-fase (Transfer Learning & Fine-Tuning)
│   ├── predict.py                        # Mesin inferensi ImagePredictor dengan dukungan fallback
│   ├── evaluate.py                       # Skrip evaluasi komprehensif, ekspor ke Excel & Confusion Matrix
│   ├── evaluate_bias.py                  # Pengujian ketahanan model terhadap bias warna merah (Shortcut Test)
│   └── utils.py                          # Implementasi visualisasi Grad-CAM, penjelasan semantik, logging
│
├── api/
│   └── app.py                            # Layanan Flask REST API (Endpoint /predict, /health, /classes)
│
├── desktop_app.py                        # Aplikasi Desktop GUI Interaktif (CustomTkinter)
├── run_desktop_app.bat                   # Skrip Batch Windows 1-klik untuk menjalankan Desktop App
├── streamlit_app.py                      # Dashboard Web Interaktif (Streamlit)
│
├── models/                               # Model Tersimpan & Laporan Evaluasi
│   ├── final_model.keras                 # Bobot model final terbaik siap pakai
│   ├── best_model_phase1.keras           # Checkpoint model terbaik setelah Fase 1
│   ├── best_model_phase2.keras           # Checkpoint model terbaik setelah Fase 2
│   ├── evaluation_metrics.xlsx           # Rekap metrik evaluasi detail format Excel
│   ├── confusion_matrix.png              # Plot heatmap matriks konfusi
│   └── roc_curves.png                    # Plot kurva One-vs-Rest ROC-AUC
│
├── dataset_large/                        # Dataset Gambar Terstruktur
│   ├── train/ [Safe, Violence, NSFW]     # Data pelatihan (~20.000 citra)
│   ├── validation/ [Safe, Violence, NSFW]# Data validasi (~3.000 citra)
│   └── test/ [Safe, Violence, NSFW]      # Data uji independen (3.040 citra)
│
├── create_recap_doc.py                   # Generator dokumen Word rekapitulasi skripsi (.docx)
├── requirements.txt                      # Spesifikasi dependensi pustaka Python
└── README.md                             # Dokumentasi pengembang
```

---

## 10. PANDUAN OPERASIONAL & DEMO SISTEM

### 1. Menjalankan Dashboard Web Interaktif (Streamlit)
Cocok untuk demonstrasi visual di depan dosen pembimbing/penguji:
```powershell
streamlit run streamlit_app.py
```
*Fitur:* Drag-and-drop gambar, tampilan probabilitas real-time dengan grafik Plotly, status deteksi kerangka pose, serta visualisasi tab Grad-CAM (Asli / Heatmap / Overlay).

### 2. Menjalankan Native Desktop GUI (CustomTkinter)
Cocok untuk penggunaan lokal tanpa koneksi internet dan pengujian batch:
```powershell
python desktop_app.py
# Atau klik ganda file run_desktop_app.bat
```
*Fitur:* Tampilan *dark mode* modern, pemrosesan satu gambar atau seluruh gambar dalam satu folder sekaligus, ekspor log hasil prediksi.

### 3. Menjalankan REST API Server (Flask)
Untuk integrasi dengan aplikasi eksternal (Android, Web Frontend, atau Backend):
```powershell
python api/app.py
```
*Contoh Pengujian API via cURL / PowerShell:*
```bash
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:5000/predict
```

---

## 11. BANK TANYA-JAWAB KRITIS UNTUK SIDANG SKRIPSI

### Pertanyaan 1: Mengapa menggunakan MobileNetV2 dan bukan arsitektur yang lebih besar seperti VGG16 atau ResNet50?
> **Jawaban:**  
> *"MobileNetV2 dirancang khusus untuk efisiensi komputasi menggunakan teknik **Depthwise Separable Convolution** dan **Inverted Residual Blocks**. Jumlah parameternya hanya ~3,5 juta, berbanding terbalik dengan VGG16 yang mencapai 138 juta parameter atau ResNet50 dengan 25 juta parameter. Karena sistem moderasi konten internet membutuhkan throughput tinggi dan inferensi berkecepatan milidetik pada server atau perangkat edge, MobileNetV2 memberikan trade-off terbaik antara akurasi tinggi dan efisiensi memori yang sangat hemat."*

### Pertanyaan 2: Apa yang dimaksud dengan Shortcut Learning dan bagaimana cara Anda membuktikan bahwa model Anda berhasil mengatasinya?
> **Jawaban:**  
> *"Shortcut learning adalah kelemahan di mana model deep learning mengambil 'jalan pintas' berupa korelasi statistik dangkal pada dataset alih-alih mempelajari konsep semantik yang sesungguhnya. Contoh klasiknya: gambar baju merah atau mobil merah sering salah diprediksi sebagai 'Violence' karena model menganggap warna merah adalah darah.  
> Kami membuktikan solusinya melalui dua hal:  
> 1. Mengintegrasikan augmentasi 20% random grayscale dan color-jitter saat pelatihan agar model dipaksa mempelajari bentuk dan tekstur.  
> 2. Menerapkan multi-feature fusion yang memvalidasi ada/tidaknya kontak fisik manusia dan postur tubuh. Hasilnya, pada pengujian bias khusus (evaluate_bias.py), gambar pakaian merah non-kekerasan berhasil diklasifikasikan dengan benar sebagai 'Safe'."*

### Pertanyaan 3: Mengapa kelas NSFW memperoleh Recall tertinggi (98,57%) dibandingkan kelas lainnya?
> **Jawaban:**  
> *"Recall 98,57% pada kelas NSFW berarti sistem memiliki sensitivitas luar biasa tinggi sehingga dari 1.051 gambar dewasa, hanya 15 yang terlewat. Keberhasilan ini didorong secara signifikan oleh cabang Human Appearance Crop ber-padding 15%. Cabang ini secara otomatis mengisolasi area tubuh manusia dari latar belakang, memudahkan model mendeteksi fitur biologis seperti proporsi kulit terbuka dan bagian sensitif tanpa terganggu oleh elemen background."*

### Pertanyaan 4: Mengapa pelatihan harus dibagi menjadi 2 fase dan BatchNormalization dibekukan pada Fase 2?
> **Jawaban:**  
> *"Jika kita langsung melakukan fine-tuning pada seluruh layer secara bersamaan dengan bobot head yang masih terinisialisasi acak, gradien error yang besar akan menghancurkan bobot pra-latih ImageNet (kejadian ini disebut Catastrophic Forgetting). Maka di Fase 1, backbone dibekukan agar head stabil terlebih dahulu.  
> Pada Fase 2, BatchNormalization wajib dibekukan karena statistik running mean dan running variance-nya dihitung berdasarkan dataset ImageNet yang sangat masif. Mengubahnya pada batch pelatihan kecil (batch size 32) akan merusak normalisasi internal dan menyebabkan performa model drop secara drastis."*

### Pertanyaan 5: Apa manfaat ilmiah dari penerapan Grad-CAM pada skripsi ini?
> **Jawaban:**  
> *"Jaringan saraf tiruan sering dikritik sebagai 'kotak hitam' (black-box). Grad-CAM berfungsi sebagai Explainable AI (XAI) yang menghitung gradien kelas terhadap layer konvolusi terakhir untuk menghasilkan visualisasi heatmap. Melalui Grad-CAM, kami dapat memverifikasi secara ilmiah bahwa prediksi model didasarkan pada objek yang valid (misalnya fokus pada area pisau/senjata untuk Violence, atau area tubuh untuk NSFW), dan bukan karena noise latar belakang."*

---

*Dokumen ini disusun sebagai acuan resmi terpadu yang memuat seluruh dasar teori, arsitektur perangkat lunak, formula matematika, dan fakta empiris pengujian sistem skripsi.*
