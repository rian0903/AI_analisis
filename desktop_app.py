"""
desktop_app.py
==============
Native Desktop GUI Application for Multi-Feature Content Moderation AI System using CustomTkinter.
Directly loads the latest trained model (models/final_model.keras) with full inference, Grad-CAM visualization,
and batch folder processing.
"""

import logging
import os
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Setup path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import CLASS_NAMES, MODELS_DIR
from src.predict import ImagePredictor
from src.utils import GradCAM, setup_logging

setup_logging(log_dir="logs", log_file="desktop_gui.log")
logger = logging.getLogger("DesktopApp")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernDesktopApp(ctk.CTk):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()

        self.title("🛡️ Multi-Feature AI Content Moderation System (Desktop App)")
        self.geometry("1200x820")
        self.minsize(1000, 700)

        # Determine model
        if model_path:
            self.model_path = Path(model_path)
        else:
            final_p = BASE_DIR / "models" / "final_model.keras"
            self.model_path = final_p if final_p.exists() else (BASE_DIR / "models" / "best_model_phase2.keras")

        self.predictor: Optional[ImagePredictor] = None
        self.current_image_path: Optional[str] = None
        self.last_prediction_result = None

        self._build_ui()
        self._load_model_threaded()

    def _build_ui(self):
        # Configure Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header Bar
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f172a", height=60)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="🛡️ AI Content Moderation — Desktop App",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold"),
            text_color="#f8fafc"
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=12, sticky="w")

        self.model_status_label = ctk.CTkLabel(
            self.header_frame,
            text=f"Model: {self.model_path.name} (Memuat...)",
            font=ctk.CTkFont(family="JetBrains Mono", size=12),
            text_color="#38bdf8"
        )
        self.model_status_label.grid(row=0, column=2, padx=20, pady=12, sticky="e")

        # Tab View
        self.tabview = ctk.CTkTabview(self, corner_radius=12, fg_color="#111827")
        self.tabview.grid(row=1, column=0, padx=20, pady=15, sticky="nsew")

        self.tab_single = self.tabview.add("⚡ Uji Citra Tunggal")
        self.tab_batch = self.tabview.add("📁 Batch Testing Folder")
        self.tab_info = self.tabview.add("🧠 Info Model & Arsitektur")

        self._setup_single_tab()
        self._setup_batch_tab()
        self._setup_info_tab()

    def _setup_single_tab(self):
        tab = self.tab_single
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Left Column: Upload & Image View
        left_card = ctk.CTkFrame(tab, corner_radius=12, fg_color="#1e293b")
        left_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_card.grid_rowconfigure(1, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_card,
            text="📂 Input Citra Pengujian",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#e2e8f0"
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Image preview area
        self.preview_label = ctk.CTkLabel(
            left_card,
            text="Belum ada gambar dipilih.\nKlik 'Pilih Gambar' di bawah.",
            corner_radius=8,
            fg_color="#0f172a",
            text_color="#94a3b8"
        )
        self.preview_label.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        # Buttons frame
        btn_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_select_img = ctk.CTkButton(
            btn_frame,
            text="Pilih Gambar...",
            command=self._on_select_image,
            font=ctk.CTkFont(weight="bold")
        )
        self.btn_select_img.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.btn_run_predict = ctk.CTkButton(
            btn_frame,
            text="🔍 Analisis Citra",
            fg_color="#6366f1",
            hover_color="#4f46e5",
            command=self._on_run_predict,
            font=ctk.CTkFont(weight="bold"),
            state="disabled"
        )
        self.btn_run_predict.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Sample quick buttons
        sample_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        sample_frame.grid(row=3, column=0, padx=15, pady=(0, 15), sticky="ew")
        sample_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(sample_frame, text="Contoh Safe", fg_color="#065f46", hover_color="#047857", command=lambda: self._load_sample("Safe")).grid(row=0, column=0, padx=3, pady=2, sticky="ew")
        ctk.CTkButton(sample_frame, text="Contoh Violence", fg_color="#78350f", hover_color="#b45309", command=lambda: self._load_sample("Violence")).grid(row=0, column=1, padx=3, pady=2, sticky="ew")
        ctk.CTkButton(sample_frame, text="Contoh NSFW", fg_color="#881337", hover_color="#be123c", command=lambda: self._load_sample("NSFW")).grid(row=0, column=2, padx=3, pady=2, sticky="ew")

        # Right Column: Results & Probability Bars
        right_card = ctk.CTkScrollableFrame(tab, corner_radius=12, fg_color="#1e293b")
        right_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_card,
            text="📊 Hasil Klasifikasi & Multi-Feature",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#e2e8f0"
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Main Result Badge Frame
        self.result_badge_frame = ctk.CTkFrame(right_card, corner_radius=10, fg_color="#0f172a")
        self.result_badge_frame.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        self.result_badge_frame.grid_columnconfigure(0, weight=1)

        self.res_class_label = ctk.CTkLabel(
            self.result_badge_frame,
            text="MENUNGGU INPUT",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#94a3b8"
        )
        self.res_class_label.grid(row=0, column=0, padx=15, pady=(15, 2))

        self.res_conf_label = ctk.CTkLabel(
            self.result_badge_frame,
            text="Keyakinan: - %",
            font=ctk.CTkFont(size=13),
            text_color="#cbd5e1"
        )
        self.res_conf_label.grid(row=1, column=0, padx=15, pady=(0, 15))

        # Multi-feature info
        self.feature_info_label = ctk.CTkLabel(
            right_card,
            text="👤 Deteksi Orang: - | 🦴 Pose: - | ⏱ Waktu: - ms",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        self.feature_info_label.grid(row=2, column=0, padx=15, pady=5, sticky="w")

        # Progress Bars for Classes
        prob_box = ctk.CTkFrame(right_card, corner_radius=8, fg_color="#0f172a")
        prob_box.grid(row=3, column=0, padx=15, pady=10, sticky="ew")
        prob_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(prob_box, text="Safe:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.prog_safe = ctk.CTkProgressBar(prob_box, progress_color="#10b981")
        self.prog_safe.set(0)
        self.prog_safe.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_safe = ctk.CTkLabel(prob_box, text="0.0%", width=50)
        self.lbl_safe.grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkLabel(prob_box, text="Violence:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.prog_vio = ctk.CTkProgressBar(prob_box, progress_color="#f59e0b")
        self.prog_vio.set(0)
        self.prog_vio.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_vio = ctk.CTkLabel(prob_box, text="0.0%", width=50)
        self.lbl_vio.grid(row=1, column=2, padx=10, pady=5)

        ctk.CTkLabel(prob_box, text="NSFW:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.prog_nsfw = ctk.CTkProgressBar(prob_box, progress_color="#ef4444")
        self.prog_nsfw.set(0)
        self.prog_nsfw.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.lbl_nsfw = ctk.CTkLabel(prob_box, text="0.0%", width=50)
        self.lbl_nsfw.grid(row=2, column=2, padx=10, pady=5)

        # AI Explanation
        self.explanation_label = ctk.CTkLabel(
            right_card,
            text="💡 Penjelasan AI:\n-",
            font=ctk.CTkFont(size=12),
            text_color="#c7d2fe",
            wraplength=450,
            justify="left"
        )
        self.explanation_label.grid(row=4, column=0, padx=15, pady=10, sticky="w")

    def _setup_batch_tab(self):
        tab = self.tab_batch
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        top_f = ctk.CTkFrame(tab, fg_color="#1e293b", corner_radius=10)
        top_f.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        top_f.grid_columnconfigure(1, weight=1)

        self.btn_batch_folder = ctk.CTkButton(top_f, text="📂 Pilih Folder...", command=self._on_select_folder)
        self.btn_batch_folder.grid(row=0, column=0, padx=10, pady=10)

        self.lbl_batch_path = ctk.CTkLabel(top_f, text="Belum ada folder dipilih", text_color="#94a3b8")
        self.lbl_batch_path.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.btn_batch_run = ctk.CTkButton(
            top_f,
            text="▶ Jalankan Batch",
            fg_color="#6366f1",
            hover_color="#4f46e5",
            command=self._on_run_batch,
            state="disabled"
        )
        self.btn_batch_run.grid(row=0, column=2, padx=10, pady=10)

        # Batch Text Box
        self.batch_textbox = ctk.CTkTextbox(tab, corner_radius=10, fg_color="#0f172a", font=ctk.CTkFont(family="JetBrains Mono", size=11))
        self.batch_textbox.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.batch_textbox.insert("1.0", "Pilih folder dan jalankan pengujian batch untuk melihat hasil evaluasi di sini...\n")

    def _setup_info_tab(self):
        tab = self.tab_info
        tab.grid_columnconfigure((0, 1), weight=1)

        info_card = ctk.CTkFrame(tab, corner_radius=12, fg_color="#1e293b")
        info_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(info_card, text="🏗️ Arsitektur Model", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        arch_text = (
            "• Backbone: MobileNetV2 Pre-trained ImageNet\n"
            "• Cabang 1: Global Image Features (1,280 dimensi)\n"
            "• Cabang 2: Human Crop Appearance (1,280 dimensi)\n"
            "• Cabang 3: Pose Landmarks Vector (64 dimensi)\n"
            "• Fusion Layer: 2,626 dimensi fitur gabungan\n"
            "• Output: 3 Kelas (Safe, Violence, NSFW) Softmax"
        )
        ctk.CTkLabel(info_card, text=arch_text, justify="left", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=10)

        model_card = ctk.CTkFrame(tab, corner_radius=12, fg_color="#1e293b")
        model_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(model_card, text="📋 Info File Model", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        self.model_details_label = ctk.CTkLabel(
            model_card,
            text=f"Path: {self.model_path}\nUkuran: - MB\nStatus: Sedang memuat...",
            justify="left",
            font=ctk.CTkFont(family="JetBrains Mono", size=12)
        )
        self.model_details_label.pack(anchor="w", padx=15, pady=10)

    def _load_model_threaded(self):
        def _load():
            try:
                self.predictor = ImagePredictor(model_path=str(self.model_path))
                size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 2)
                self.after(0, lambda: self._on_model_loaded(size_mb))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Gagal memuat model: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, size_mb: float):
        self.model_status_label.configure(text=f"Model: {self.model_path.name} (Ready ✅)")
        self.model_details_label.configure(
            text=f"Path: {self.model_path}\nUkuran: {size_mb} MB\nStatus: Loaded & Ready ✅\nJumlah Kelas: {len(CLASS_NAMES)}"
        )
        if self.current_image_path:
            self.btn_run_predict.configure(state="normal")

    def _on_select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All Files", "*.*")]
        )
        if path:
            self._display_image(path)

    def _load_sample(self, class_name: str):
        search_dirs = [
            BASE_DIR / "dataset_large" / "test" / class_name,
            BASE_DIR / "dataset" / "test" / class_name,
            BASE_DIR / "raw_data" / class_name,
        ]
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for s in search_dirs:
            if s.exists():
                imgs = [p for p in s.iterdir() if p.suffix.lower() in valid_exts]
                if imgs:
                    self._display_image(str(imgs[0]))
                    self._on_run_predict()
                    return
        messagebox.showinfo("Sample", f"Tidak ditemukan contoh citra {class_name} di direktori lokal.")

    def _display_image(self, path: str):
        self.current_image_path = path
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((360, 360))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.preview_label.configure(image=ctk_img, text="")
            if self.predictor is not None:
                self.btn_run_predict.configure(state="normal")
        except Exception as exc:
            messagebox.showerror("Error", f"Gagal membaca gambar: {exc}")

    def _on_run_predict(self):
        if not self.current_image_path or not self.predictor:
            return

        self.btn_run_predict.configure(state="disabled", text="Menganalisis...")

        def _predict():
            try:
                res = self.predictor.predict_image(self.current_image_path)
                self.after(0, lambda: self._update_results(res))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Inference failed: {e}"))
            finally:
                self.after(0, lambda: self.btn_run_predict.configure(state="normal", text="🔍 Analisis Citra"))

        threading.Thread(target=_predict, daemon=True).start()

    def _update_results(self, res: dict):
        pred_class = res["prediction"]
        conf = res["confidence"] * 100
        probs = res["probabilities"]
        human = res["human_analysis"]

        color_map = {"Safe": "#10b981", "Violence": "#f59e0b", "NSFW": "#ef4444"}
        bg_color = color_map.get(pred_class, "#6366f1")

        self.res_class_label.configure(text=f"{pred_class.upper()}", text_color=bg_color)
        self.res_conf_label.configure(text=f"Keyakinan: {conf:.1f}% ({pred_class})")

        p_det = "Ya" if human["person_detected"] else "Tidak"
        p_cnt = human["person_count"]
        pose_det = "Terdeteksi" if human["pose_detected"] else "None"
        self.feature_info_label.configure(
            text=f"👤 Deteksi Orang: {p_det} ({p_cnt}) | 🦴 Pose: {pose_det} | ⏱ Waktu: {res['inference_time_ms']} ms"
        )

        p_safe = probs.get("Safe", 0)
        p_vio = probs.get("Violence", 0)
        p_nsfw = probs.get("NSFW", 0)

        self.prog_safe.set(p_safe)
        self.lbl_safe.configure(text=f"{p_safe*100:.1f}%")

        self.prog_vio.set(p_vio)
        self.lbl_vio.configure(text=f"{p_vio*100:.1f}%")

        self.prog_nsfw.set(p_nsfw)
        self.lbl_nsfw.configure(text=f"{p_nsfw*100:.1f}%")

        self.explanation_label.configure(text=f"💡 Penjelasan AI:\n{res.get('explanation_summary', '-')}")

    def _on_select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.lbl_batch_path.configure(text=folder)
            self.btn_batch_run.configure(state="normal")

    def _on_run_batch(self):
        if not hasattr(self, "selected_folder") or not self.predictor:
            return

        self.btn_batch_run.configure(state="disabled")
        self.batch_textbox.delete("1.0", "end")
        self.batch_textbox.insert("1.0", f"Memulai pengujian batch pada folder: {self.selected_folder}\n" + "=" * 60 + "\n")

        def _batch():
            results = self.predictor.predict_folder(self.selected_folder)
            out_lines = []
            for r in results:
                fname = Path(r["file_path"]).name
                p = r["prediction"]
                c = r["confidence"] * 100
                pipe = r.get("pipeline", "multi")
                out_lines.append(f"{fname:35s} -> {p:10s} ({c:5.1f}%) [{pipe}]")
            out_text = "\n".join(out_lines) + f"\n\nSelesai! Total: {len(results)} citra."
            self.after(0, lambda: self.batch_textbox.insert("end", out_text))
            self.after(0, lambda: self.btn_batch_run.configure(state="normal"))

        threading.Thread(target=_batch, daemon=True).start()


def launch_desktop_gui(model_path: Optional[str] = None):
    app = ModernDesktopApp(model_path=model_path)
    app.mainloop()


if __name__ == "__main__":
    launch_desktop_gui()
