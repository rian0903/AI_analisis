"""
extract_confused_images.py
===========================
Script untuk mengekstrak dan memisahkan semua gambar yang membuat model "bingung"
(Low Confidence < 60% dan/atau Salah Prediksi / Misclassified).

Gambar dipisahkan ke dalam folder ber-keterangan jelas beserta file metadata JSON.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

# Ensure root directory is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import ImagePredictor
from src.config import LOW_CONFIDENCE_THRESHOLD

def main():
    model_path = BASE_DIR / "models" / "best_model_phase2.keras"
    if not model_path.exists():
        model_path = BASE_DIR / "models" / "final_model.keras"

    print(f"Loading model from: {model_path}")
    predictor = ImagePredictor(model_path=str(model_path))

    dataset_test_dir = BASE_DIR / "dataset_large" / "test"
    output_base_dir = BASE_DIR / "models" / "gambar_model_bingung"

    # Reset/Create output directory
    if output_base_dir.exists():
        shutil.rmtree(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    low_conf_dir = output_base_dir / "1_Low_Confidence_Ragu_Ragu_Dibawah_60persen"
    misclass_dir = output_base_dir / "2_Misclassified_Salah_Prediksi"

    low_conf_dir.mkdir(parents=True, exist_ok=True)
    misclass_dir.mkdir(parents=True, exist_ok=True)

    categories = ["Safe", "Violence", "NSFW"]
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    low_conf_count = 0
    misclass_count = 0
    total_processed = 0

    summary_records = []

    print("\nProcessing dataset test images...")
    for true_label in categories:
        class_folder = dataset_test_dir / true_label
        if not class_folder.exists():
            continue

        image_files = [p for p in class_folder.iterdir() if p.suffix.lower() in supported_exts]
        print(f"Analyzing {len(image_files)} images from class '{true_label}'...")

        for img_path in image_files:
            total_processed += 1
            try:
                res = predictor.predict_image(img_path)
            except Exception as e:
                print(f"Error predicting {img_path.name}: {e}")
                continue

            pred_label = res["prediction"]
            confidence = res["confidence"]  # Float 0.0 - 1.0
            probs = res["probabilities"]
            is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
            is_misclass = (pred_label != true_label)

            if is_low_conf or is_misclass:
                img_name = img_path.name
                file_stem = img_path.stem

                # 1. If Low Confidence (< 0.60)
                if is_low_conf:
                    low_conf_count += 1
                    subfolder_name = f"skor_{confidence*100:.1f}pct_asli_{true_label}_diprediksi_{pred_label}"
                    target_dir = low_conf_dir / subfolder_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(img_path, target_dir / img_name)

                    meta = {
                        "filename": img_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence_percent": round(confidence * 100, 2),
                        "probabilities": probs,
                        "human_analysis": res["human_analysis"],
                        "pipeline": res["pipeline"],
                        "keterangan_kebingungan": (
                            f"Model ragu-ragu karena skor keyakinan ({confidence*100:.2f}%) "
                            f"berada di bawah threshold 60.0%. "
                            f"Label asli: {true_label}, diprediksi sebagai: {pred_label}."
                        )
                    }
                    with open(target_dir / f"{file_stem}.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                # 2. If Misclassified (Salah Prediksi)
                if is_misclass:
                    misclass_count += 1
                    subfolder_name = f"asli_{true_label}_salah_diprediksi_{pred_label}"
                    target_dir = misclass_dir / subfolder_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(img_path, target_dir / img_name)

                    meta = {
                        "filename": img_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence_percent": round(confidence * 100, 2),
                        "probabilities": probs,
                        "human_analysis": res["human_analysis"],
                        "pipeline": res["pipeline"],
                        "keterangan_kebingungan": (
                            f"Model salah mengklasifikasikan gambar. "
                            f"Gambar asli berlabel '{true_label}', namun model memprediksi '{pred_label}' "
                            f"dengan tingkat keyakinan {confidence*100:.2f}%."
                        )
                    }
                    with open(target_dir / f"{file_stem}.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                summary_records.append({
                    "filename": img_name,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence_percent": round(confidence * 100, 2),
                    "is_low_confidence": is_low_conf,
                    "is_misclassified": is_misclass
                })

    # Save summary report
    summary_path = output_base_dir / "ringkasan_gambar_bingung.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_processed": total_processed,
            "total_low_confidence": low_conf_count,
            "total_misclassified": misclass_count,
            "details": summary_records
        }, f, indent=2)

    print("\n=======================================================")
    print(" extraction complete!")
    print(f" Total Gambar Diuji           : {total_processed}")
    print(f" Gambar Low Confidence (<60%) : {low_conf_count}")
    print(f" Gambar Salah Prediksi        : {misclass_count}")
    print(f" Folder Hasil                 : {output_base_dir.resolve()}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
