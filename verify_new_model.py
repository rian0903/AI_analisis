"""
verify_new_model.py
===================
Evaluates the freshly trained model on test samples of:
1. Real photos (Safe, Violence, NSFW)
2. 2D Anime/Illustration (Safe, Violence, NSFW)
3. Computes exact confidence distributions, class predictions, and low-confidence rate.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from src.predict import ImagePredictor
from src.config import CLASS_NAMES

MODEL_PATH = "models/final_model.keras"
predictor = ImagePredictor(model_path=MODEL_PATH)

print("=" * 60)
print(f" EVALUATING NEW MODEL: {MODEL_PATH}")
print("=" * 60)

test_categories = {
    "Anime NSFW": list(Path("raw_data/anime_nsfw_raw").glob("*.*"))[:15],
    "Anime Safe": list(Path("raw_data/anime_safe_raw").glob("*.*"))[:15],
    "Anime Violence": list(Path("raw_data/anime_violence_raw").glob("*.*"))[:15],
    "Real Photo NSFW": list(Path("raw_data/nsfw_raw").glob("*.*"))[:15],
    "Real Photo Safe": list(Path("raw_data/safe_raw").glob("*.*"))[:15],
    "Real Photo Violence": list(Path("raw_data/violence_raw").glob("*.*"))[:15],
}

summary = {}

for cat_name, file_list in test_categories.items():
    print(f"\n--- Testing Category: [{cat_name}] ({len(file_list)} images) ---")
    preds = []
    confidences = []
    low_conf_count = 0

    expected_label = "NSFW" if "NSFW" in cat_name else ("Violence" if "Violence" in cat_name else "Safe")

    for f in file_list:
        try:
            res = predictor.predict_image(str(f))
            p_cls = res["prediction"]
            p_conf = res["confidence"]
            preds.append(p_cls)
            confidences.append(p_conf)
            if res.get("low_confidence", False):
                low_conf_count += 1
            print(f"  {f.name[:25]:25s} -> {p_cls:10s} ({p_conf*100:5.1f}%) | Safe: {res['probabilities']['Safe']*100:4.1f}% | Vio: {res['probabilities']['Violence']*100:4.1f}% | NSFW: {res['probabilities']['NSFW']*100:4.1f}%")
        except Exception as e:
            print(f"  Error on {f.name}: {e}")

    correct = sum(1 for p in preds if p == expected_label)
    acc = (correct / len(preds)) * 100 if preds else 0.0
    avg_conf = np.mean(confidences) * 100 if confidences else 0.0
    summary[cat_name] = {
        "accuracy": acc,
        "correct": correct,
        "total": len(preds),
        "avg_conf": avg_conf,
        "low_conf": low_conf_count,
    }

print("\n" + "=" * 60)
print(" SUMMARY BENCHMARK RESULTS ACROSS REAL & ANIME CONTENT")
print("=" * 60)
for cat_name, stats in summary.items():
    print(f" {cat_name:25s} | Accuracy: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']}) | Avg Conf: {stats['avg_conf']:5.1f}% | Low Conf: {stats['low_conf']}")
print("=" * 60)
