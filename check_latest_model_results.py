"""
check_latest_model_results.py
==============================
Script untuk mengecek hasil evaluasi dari model train terakhir (models/final_model.keras).
Mengecek akurasi, jumlah low confidence (<60%), dan jumlah salah prediksi (wrong).
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
from src.predict import ImagePredictor
from src.config import LOW_CONFIDENCE_THRESHOLD

def evaluate_folder(predictor, dataset_path, description):
    print(f"\n=======================================================")
    print(f" EVALUASI MODEL PADA: {description}")
    print(f" Path: {dataset_path}")
    print("=======================================================")

    categories = ["Safe", "Violence", "NSFW"]
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    total = 0
    correct = 0
    low_conf_count = 0
    wrong_count = 0

    per_class_stats = {cat: {"total": 0, "correct": 0, "wrong": 0, "low_conf": 0} for cat in categories}

    for true_label in categories:
        folder = dataset_path / true_label
        if not folder.exists():
            continue

        images = [p for p in folder.iterdir() if p.suffix.lower() in supported_exts]

        for img_p in images:
            total += 1
            per_class_stats[true_label]["total"] += 1
            try:
                res = predictor.predict_image(img_p)
                pred_label = res["prediction"]
                conf = res["confidence"]
            except Exception as e:
                print(f"Error reading {img_p.name}: {e}")
                continue

            is_correct = (pred_label == true_label)
            is_low_conf = (conf < LOW_CONFIDENCE_THRESHOLD)

            if is_correct:
                correct += 1
                per_class_stats[true_label]["correct"] += 1
            else:
                wrong_count += 1
                per_class_stats[true_label]["wrong"] += 1

            if is_low_conf:
                low_conf_count += 1
                per_class_stats[true_label]["low_conf"] += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    low_conf_pct = (low_conf_count / total * 100) if total > 0 else 0.0
    wrong_pct = (wrong_count / total * 100) if total > 0 else 0.0

    print(f" Total Sampel Diuji            : {total}")
    print(f" Prediksi Benar (Correct)       : {correct} ({accuracy:.2f}%)")
    print(f" Prediksi Salah (Wrong)         : {wrong_count} ({wrong_pct:.2f}%)")
    print(f" Low Confidence (<60%)         : {low_conf_count} ({low_conf_pct:.2f}%)")
    print("\n Rincian per Kelas:")
    for cat in categories:
        c_stat = per_class_stats[cat]
        c_tot = c_stat["total"]
        c_acc = (c_stat["correct"] / c_tot * 100) if c_tot > 0 else 0.0
        print(f"   * {cat:8s}: Acc {c_acc:.2f}% | Wrong: {c_stat['wrong']}/{c_tot} | Low Conf: {c_stat['low_conf']}/{c_tot}")

    return {
        "description": description,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "wrong": wrong_count,
        "low_confidence": low_conf_count,
        "per_class": per_class_stats
    }

def main():
    for model_name in ["best_model_phase2.keras", "final_model.keras"]:
        model_path = BASE_DIR / "models" / model_name
        if not model_path.exists():
            continue
        print(f"\n=======================================================")
        print(f" TESTING MODEL: {model_name}")
        print(f"=======================================================")
        predictor = ImagePredictor(model_path=str(model_path))

        evaluate_folder(predictor, BASE_DIR / "dataset_large" / "test", f"{model_name} on dataset_large/test")

if __name__ == "__main__":
    main()
