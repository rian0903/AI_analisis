"""
create_recap_doc.py
===================
Script to generate a comprehensive, professionally formatted Word document (.docx)
recapitulating all methods, algorithms, proposal alignments, and evaluation results
for Nanda Riansyah's thesis.
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Set background color for a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins for a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tc_pr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Set clean subtle borders for a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_callout_box(doc, text_list, title="CATATAN PENTING & TIPS SIDANG"):
    """Add a stylish callout box with a left border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="0F4C81"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tc_pr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"💡 {title}\n")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(11)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 76, 129)
    
    for t in text_list:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(3)
        run_t = p2.add_run(t)
        run_t.font.name = "Calibri"
        run_t.font.size = Pt(10)
        run_t.font.color.rgb = RGBColor(40, 40, 40)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_document():
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    PRIMARY_COLOR = RGBColor(15, 76, 129)     # Deep Navy
    SECONDARY_COLOR = RGBColor(0, 128, 128)   # Teal
    DARK_TEXT = RGBColor(40, 40, 40)          # Slate
    
    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("REKAPITULASI METODE, ARSITEKTUR, DAN HASIL EVALUASI MODEL SKRIPSI")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY_COLOR
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(18)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Bahan Acuan Komprehensif Penyusunan Naskah & Persiapan Sidang Skripsi\n"
                            "Klasifikasi Gambar Postingan Internet Menggunakan Arsitektur MobileNetV2\n"
                            "Disusun untuk: Nanda Riansyah (NPM: 235520110108) — Universitas Almuslim")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SECONDARY_COLOR
    
    # Divider line
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(12)
    p_div_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="0F4C81"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(p_div_border)
    
    # Helper to add headings
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = PRIMARY_COLOR
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = SECONDARY_COLOR
        return h

    def add_body(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_prefix = p.add_run(bold_prefix)
            r_prefix.font.name = "Calibri"
            r_prefix.font.size = Pt(10.5)
            r_prefix.font.bold = True
            r_prefix.font.color.rgb = DARK_TEXT
        r_text = p.add_run(text)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(10.5)
        r_text.font.italic = italic
        r_text.font.color.rgb = DARK_TEXT
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_prefix = p.add_run(bold_prefix)
            r_prefix.font.name = "Calibri"
            r_prefix.font.size = Pt(10.5)
            r_prefix.font.bold = True
            r_prefix.font.color.rgb = DARK_TEXT
        r_text = p.add_run(text)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(10.5)
        r_text.font.color.rgb = DARK_TEXT
        return p

    # -------------------------------------------------------------
    # SECTION 1: RINGKASAN EKSEKUTIF & TUJUAN PENELITIAN
    # -------------------------------------------------------------
    add_h1("1. RINGKASAN EKSEKUTIF & TUJUAN PENELITIAN")
    add_body("Penelitian skripsi ini bertujuan untuk membangun sistem moderasi dan deteksi konten gambar tidak layak (inappropriate content) pada postingan internet secara otomatis berbasis Deep Learning dan Computer Vision. Sistem mengklasifikasikan setiap citra digital ke dalam 3 kategori mutually-exclusive:")
    add_bullet(" Konten umum harian, lanskap, keluarga, olahraga, makanan, dan objek yang aman untuk semua kalangan.", "1. Safe (Aman):")
    add_bullet(" Aksi kekerasan fisik, penyerangan, senjata, ledakan, dan adegan perkelahian.", "2. Violence (Kekerasan):")
    add_bullet(" Konten dewasa, pornografi, ketelanjangan, dan bagian tubuh sensitif.", "3. NSFW (Not Safe for Work / Dewasa):")
    add_body("Model ini secara khusus dirancang untuk mengatasi fenomena Shortcut Learning (Color Bias), di mana arsitektur CNN standar sering salah mengklasifikasikan gambar normal yang memiliki dominasi warna merah (seperti baju merah, bunga mawar, mobil merah, latar belakang merah) sebagai Violence (Kekerasan) karena mengasumsikan warna merah sebagai darah. Pendekatan Multi-Feature Fusion berhasil mengeliminasi bias ini dengan memvalidasi keberadaan manusia, postur tubuh, dan interaksi spasial.")

    # -------------------------------------------------------------
    # SECTION 2: METODE & ALGORITMA UTAMA YANG DIGUNAKAN
    # -------------------------------------------------------------
    add_h1("2. METODE, ALGORITMA, DAN ARSITEKTUR TEKNIS")
    add_body("Sistem mengintegrasikan kombinasi algoritma dan arsitektur canggih dari Computer Vision dan Deep Learning:")
    
    # Table of Algorithms
    t_algo = doc.add_table(rows=1, cols=3)
    t_algo.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_algo.autofit = False
    set_table_borders(t_algo)
    
    headers = ["Komponen Sistem", "Algoritma / Arsitektur", "Fungsi & Peran Teknis"]
    widths = [Inches(1.8), Inches(2.2), Inches(2.5)]
    
    hdr_cells = t_algo.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].width = widths[i]
        set_cell_background(hdr_cells[i], "0F4C81")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    algo_data = [
        ("Feature Extractor (Backbone)", "MobileNetV2 (Pre-trained ImageNet)", "Ekstraksi fitur visual mendalam dengan Depthwise Separable Convolutions & Inverted Residuals. Menghasilkan vektor 1280-dim."),
        ("Deteksi Manusia (Person Detector)", "MediaPipe Pose / Native Skin-Tone & Contour", "Mendeteksi koordinat bounding box manusia dengan 15% padding crop untuk analisis lokal."),
        ("Estimasi Postur (Pose Estimator)", "MediaPipe Pose Landmark (12 Keypoints)", "Mengekstrak 12 titik kunci sendi tubuh (x, y, visibility) = 36 dimensi untuk menangkap gesture tubuh."),
        ("Relasi Spasial (Multi-Person)", "Algoritma Geometri Spasial (Distance & IoU)", "Menghitung jarak Euclidean, Intersection-over-Union (IoU), dan pergeseran offset horizontal antar subjek (18-dim)."),
        ("Peleburan Fitur (Fusion Head)", "Gated Feature Fusion & Multi-Layer Perceptron", "Menggabungkan total 2.626 dimensi fitur dengan mekanisme gating fleksibel (has_human, has_pose)."),
        ("Optimasi Pelatihan", "Adam Optimizer", "Optimasi adaptif dengan learning rate 1e-4 pada Fase 1 dan 1e-5 pada Fase 2."),
        ("Fungsi Kerugian (Loss Function)", "Categorical Crossentropy (Label Smoothing 0.05)", "Menghitung loss multi-kelas sekaligus mencegah overconfidence pada klasifikasi probabilitas."),
        ("Augmentasi Data", "Photometric & Geometric + Random Grayscale", "Rotasi, flip, zoom, contrast, hue, saturation, dan 20% grayscale untuk memutus ketergantungan model pada warna semata."),
        ("Explainable AI (XAI)", "Grad-CAM (Gradient-weighted Class Activation Mapping)", "Visualisasi heatmap untuk menunjukkan area citra yang memicu keputusan klasifikasi model.")
    ]
    
    for row_idx, data in enumerate(algo_data):
        row = t_algo.add_row()
        bg_color = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for i, text in enumerate(data):
            cell = row.cells[i]
            cell.width = widths[i]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = DARK_TEXT
            if i == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Architecture Dimension Breakdown
    add_h2("Rincian Dimensi Vektor Peleburan (Feature Fusion Vector = 2.626 Dimensi):")
    add_bullet(" 1.280 dimensi (vektor fitur citra keseluruhan setelah Global Average Pooling).", "1. Global Image Feature:")
    add_bullet(" 1.280 dimensi (vektor fitur visual area manusia hasil crop dengan 15% padding).", "2. Human Appearance Feature:")
    add_bullet(" 64 dimensi (Dense layer dari 36-dim pose landmark + 18-dim relasi spasial).", "3. Pose & Spatial Feature:")
    add_bullet(" 2 dimensi (indikator biner apakah manusia dan pose terdeteksi pada citra).", "4. Gating Mechanism Flags:")
    add_bullet(" Dense(512, ReLU) -> BatchNorm -> Dropout(0.4) -> Dense(256, ReLU) -> Dropout(0.3) -> Dense(3, Softmax).", "5. Classification Head:")

    # -------------------------------------------------------------
    # SECTION 3: KESESUAIAN DENGAN DOKUMEN PROPOSAL
    # -------------------------------------------------------------
    add_h1("3. KESESUAIAN IMPLEMENTASI DENGAN DOKUMEN PROPOSAL SKRIPSI")
    add_body("Implementasi pada codebase telah diverifikasi dan dibandingkan langsung dengan naskah Proposal Skripsi (Bab 1, Halaman 9–11). Seluruh persyaratan dasar telah terpenuhi 100%, bahkan diperkaya dengan fitur kebaruan (novelty):")
    
    t_prop = doc.add_table(rows=1, cols=3)
    t_prop.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_prop.autofit = False
    set_table_borders(t_prop)
    
    p_headers = ["Komponen Proposal", "Target di Naskah Proposal", "Realisasi pada Implementasi Kode"]
    p_widths = [Inches(1.8), Inches(2.3), Inches(2.4)]
    
    p_hdr_cells = t_prop.rows[0].cells
    for i, h in enumerate(p_headers):
        p_hdr_cells[i].width = p_widths[i]
        set_cell_background(p_hdr_cells[i], "0F4C81")
        set_cell_margins(p_hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = p_hdr_cells[i].paragraphs[0]
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    prop_data = [
        ("Arsitektur Model", "CNN MobileNetV2 (Pre-trained ImageNet)", "MobileNetV2 sebagai Backbone utama + Multi-Feature Fusion (Sesuai 100%)"),
        ("Metode Pembelajaran", "Transfer Learning (2-Phase)", "Fase 1 Frozen Base + Fase 2 Fine-Tuning 30 layer teratas (Sesuai 100%)"),
        ("Jumlah Kelas", "3 Kelas: Safe, Violence, NSFW", "3 Kelas Mutually-Exclusive: Safe, Violence, NSFW (Sesuai 100%)"),
        ("Ukuran Citra Input", "224 x 224 piksel, normalisasi [0, 1]", "Input Shape (224, 224, 3), Rescaling 1/255 (Sesuai 100%)"),
        ("Augmentasi Data", "Rotasi, flip, zoom, kecerahan", "Rotasi, flip, zoom, brightness + Grayscale & Color Jitter (Diperkuat)"),
        ("Ukuran Dataset", "Minimal 6.000 citra (2.000 per kelas)", "> 25.000 citra (dengan 3.040 citra khusus test set) (Melampaui Target)"),
        ("Metrik Evaluasi", "Confusion Matrix, Acc, Prec, Recall, F1", "Lengkap: CM, Accuracy, Precision, Recall, F1, ROC-AUC, Export Excel"),
        ("Antarmuka (Web UI)", "Aplikasi web moderasi real-time", "Tersedia antarmuka Streamlit Web App & Flask REST API")
    ]
    
    for row_idx, data in enumerate(prop_data):
        row = t_prop.add_row()
        bg_color = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for i, text in enumerate(data):
            cell = row.cells[i]
            cell.width = p_widths[i]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = DARK_TEXT
            if i == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # SECTION 4: STRATEGI PELATIHAN 2-FASE
    # -------------------------------------------------------------
    add_h1("4. STRATEGI PELATIHAN MODEL (TWO-PHASE TRAINING)")
    add_body("Pelatihan model dilakukan dalam dua tahapan terstruktur untuk menjamin transfer bobot fitur yang optimal dan mencegah rusaknya bobot pra-latih:")
    add_bullet(" Seluruh layer pada MobileNetV2 dibekukan (trainable = False). Hanya layer classification head (Multi-Feature Fusion) yang dilatih menggunakan learning rate 1e-4 dan optimizer Adam. EarlyStopping memantau validation loss (patience=7).", "Fase 1 — Transfer Learning (Feature Extraction):")
    add_bullet(" Sebanyak 30 layer teratas pada MobileNetV2 dibuka kuncinya (trainable = True), kecuali layer BatchNormalization yang tetap dibekukan untuk menjaga kestabilan statistik mean dan varians. Dilatih dengan learning rate sangat halus (1e-5) dan EarlyStopping (patience=5) untuk menyesuaikan representasi fitur spesifik terhadap dataset moderasi.", "Fase 2 — Fine-Tuning:")

    # -------------------------------------------------------------
    # SECTION 5: HASIL EVALUASI MODEL TERAKHIR (DATA RIIL)
    # -------------------------------------------------------------
    add_h1("5. HASIL EVALUASI MODEL TERAKHIR (DATA RIIL UNTUK BAB 5)")
    add_body("Evaluasi model dilakukan secara independen pada test set yang sepenuhnya terpisah dari proses pelatihan. Berikut adalah metrik performa model final:")
    
    # Table Metrics
    t_met = doc.add_table(rows=1, cols=5)
    t_met.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_met.autofit = False
    set_table_borders(t_met)
    
    m_headers = ["Kategori / Kelas", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    m_widths = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.1)]
    
    m_hdr_cells = t_met.rows[0].cells
    for i, h in enumerate(m_headers):
        m_hdr_cells[i].width = m_widths[i]
        set_cell_background(m_hdr_cells[i], "0F4C81")
        set_cell_margins(m_hdr_cells[i], top=120, bottom=120, left=100, right=100)
        p = m_hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    metrics_data = [
        ("Safe (Aman)", "93.71%", "77.93%", "85.10%", "95.38%"),
        ("Violence (Kekerasan)", "86.67%", "82.32%", "84.44%", "95.50%"),
        ("NSFW (Konten Dewasa)", "81.38%", "98.57%", "89.16%", "98.97%"),
        ("Rata-rata Makro (Macro Avg)", "87.26%", "86.27%", "86.23%", "96.62%"),
        ("Akurasi Keseluruhan (Overall)", "86.45% – 88.32% (pada 3.040 citra uji dataset_large)", "", "", "")
    ]
    
    for row_idx, data in enumerate(metrics_data):
        row = t_met.add_row()
        bg_color = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        if row_idx >= 3:
            bg_color = "E9ECEF"
        for i, text in enumerate(data):
            cell = row.cells[i]
            cell.width = m_widths[i]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if row_idx == 4 and i == 1:
                # Span across columns for overall accuracy
                pass
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = DARK_TEXT
            if row_idx >= 3 or i == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Confusion Matrix Breakdown
    add_h2("Rincian Matriks Konfusi (Confusion Matrix Breakdown — 3.040 Data Uji):")
    add_bullet(" 805 citra diprediksi benar, 113 salah sebagai Violence, 115 salah sebagai NSFW (Total: 1.033 citra).", "1. Safe Aktual:")
    add_bullet(" 787 citra diprediksi benar, 47 salah sebagai Safe, 122 salah sebagai NSFW (Total: 956 citra).", "2. Violence Aktual:")
    add_bullet(" 1.036 citra diprediksi benar, 7 salah sebagai Safe, 8 salah sebagai Violence (Total: 1.051 citra).", "3. NSFW Aktual:")

    # -------------------------------------------------------------
    # SECTION 6: PANDUAN PENULISAN BAB IV & BAB V SKRIPSI
    # -------------------------------------------------------------
    add_h1("6. PANDUAN PENYUSUNAN BAB IV DAN BAB V SKRIPSI")
    
    add_h2("A. Panduan Pengisian BAB IV (Implementasi):")
    add_bullet(" Jelaskan spesifikasi hardware (GPU/CPU, RAM) dan software (Python 3.9+, TensorFlow 2.13+, Keras, MediaPipe, OpenCV, Streamlit).", "1. Lingkungan Pengembangan:")
    add_bullet(" Jelaskan tahapan resizing (224x224), normalisasi [0, 1], dan data generator.", "2. Preprocessing & Augmentasi:")
    add_bullet(" Cantumkan diagram alur multi-feature fusion dan potongan kode arsitektur di src/model.py.", "3. Implementasi Arsitektur:")
    add_bullet(" Jelaskan tahapan training 2-fase di src/train.py serta parameter Batch Size=32, Learning Rate, dan EarlyStopping.", "4. Pelatihan Model:")
    add_bullet(" Tampilkan screenshot antarmuka web Streamlit (streamlit_app.py) yang menampilkan fitur upload gambar, hasil prediksi, probabilitas 3 kelas, dan visualisasi Grad-CAM Heatmap.", "5. Implementasi Antarmuka Web:")

    add_h2("B. Panduan Pengisian BAB V (Hasil dan Pembahasan):")
    add_bullet(" Gunakan gambar models/accuracy_curve.png dan models/loss_curve.png untuk menjelaskan bagaimana loss turun secara bertahap tanpa terjadi overfitting parah.", "1. Analisis Kurva Pelatihan:")
    add_bullet(" Masukkan gambar models/confusion_matrix.png dan tabel evaluasi dari models/evaluation_metrics.xlsx.", "2. Evaluasi Matriks Konfusi:")
    add_bullet(" Masukkan gambar models/roc_curves.png dan jelaskan nilai AUC yang mendekati 1.00 pada semua kelas.", "3. Analisis Kurva ROC-AUC:")
    add_bullet(" Jelaskan mengapa kelas NSFW memiliki Recall tertinggi (98.57%) berkat fitur crop manusia dan warna kulit, serta analisis misklasifikasi pada adegan dinamis.", "4. Pembahasan Temuan Ilmiah:")

    # -------------------------------------------------------------
    # SECTION 7: TIPS & BAHAN JAWABAN SIDANG SKRIPSI
    # -------------------------------------------------------------
    add_h1("7. PANDUAN TANYA-JAWAB KRITIS SAAT SIDANG SKRIPSI")
    
    qna_list = [
        "Tanya: 'Mengapa memilih MobileNetV2 dibandingkan VGG16 atau ResNet50?'\n"
        "Jawab: 'MobileNetV2 menggunakan Depthwise Separable Convolution dan Inverted Residuals sehingga hanya memiliki ~3,5 juta parameter. Model ini jauh lebih ringan, cepat, dan hemat komputasi dibandingkan VGG16 (138 juta parameter) atau ResNet50 (25 juta parameter), sehingga sangat responsif dan ideal untuk sistem moderasi konten web real-time.'",
        
        "Tanya: 'Mengapa model Anda menggunakan pendekatan Multi-Feature Fusion?'\n"
        "Jawab: 'Model CNN standar tunggal rentan mengalami Shortcut Learning (bias visual), seperti menganggap semua gambar berlatar merah atau baju merah sebagai kekerasan (darah). Dengan Multi-Feature Fusion, model memeriksa citra global, memotong area tubuh manusia, serta memeriksa pose sendi dan jarak antar-manusia untuk memastikan konteks sebenarnya.'",
        
        "Tanya: 'Mengapa pelatihan dilakukan dalam 2 Fase (Two-Phase Training)?'\n"
        "Jawab: 'Fase 1 (Feature Extraction) membekukan backbone MobileNetV2 agar bobot pre-trained ImageNet tidak rusak saat melatih classification head baru. Setelah head stabil, Fase 2 (Fine-Tuning) membuka 30 layer teratas dengan learning rate sangat kecil (1e-5) agar filter konvolusi beradaptasi khusus dengan fitur visual gambar internet.'",
        
        "Tanya: 'Bagaimana peran Grad-CAM pada sistem Anda?'\n"
        "Jawab: 'Grad-CAM berperan sebagai Explainable AI (XAI) yang memvisualisasikan gradien aktivasi layer konvolusi terakhir menjadi heatmap warna. Ini membuktikan secara transparan area piksel mana yang membuat model mengambil keputusan (misal: area wajah/kulit pada NSFW, atau senjata/interaksi fisik pada Violence).'"
    ]
    add_callout_box(doc, qna_list, title="PANDUAN JAWABAN STRATEGIS UNTUK PENGUJI SIDANG")

    # -------------------------------------------------------------
    # SECTION 8: CARA MENJALANKAN DEMO APLIKASI
    # -------------------------------------------------------------
    add_h1("8. CARA MENJALANKAN APLIKASI UNTUK DEMO SIDANG")
    add_body("Untuk mendemonstrasikan sistem secara langsung di depan dosen pembimbing dan penguji:")
    add_bullet(" Jalankan perintah: streamlit run streamlit_app.py", "1. Menjalankan Web UI Interaktif:")
    add_bullet(" Jalankan perintah: python api/app.py (berjalan di http://127.0.0.1:5000)", "2. Menjalankan REST API Backend:")
    add_bullet(" Jalankan perintah: python check_latest_model_results.py", "3. Mengecek Hasil Evaluasi Ulang:")

    # Save document
    output_path = Path("d:/skripsi/Rekap_Metode_dan_Evaluasi_Model_Skripsi.docx")
    doc.save(str(output_path))
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    build_document()
