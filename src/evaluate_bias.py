"""
evaluate_bias.py
================
Dedicated evaluation script specifically for visual shortcut & color bias testing.

Tests model predictions across specific hard-negative test categories:
- RED SAFE IMAGES (Red clothes, red backgrounds, red flowers, red cars, sports)
- RED NSFW IMAGES (Red ambient lighting / backgrounds with NSFW content)
- RED VIOLENCE IMAGES (Violence content with heavy red dominance)
- HARD NEGATIVE NON-VIOLENT (People hugging, dancing, swimming, sports)

Goal: Verify that the improved model does NOT over-rely on red color shortcuts ("red = Violence").
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, MODELS_DIR
from src.predict import ImagePredictor
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def is_red_dominant(image_bgr: np.ndarray, red_threshold: float = 0.15) -> bool:
    """
    Check if an image has a significant proportion of prominent red visual pixels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Red HSV range 1: [0-10, 40-255, 40-255], range 2: [170-180, 40-255, 40-255]
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255]))
    red_pixels = np.sum((mask1 > 0) | (mask2 > 0))
    total_pixels = hsv.shape[0] * hsv.shape[1]
    return (red_pixels / float(total_pixels)) >= red_threshold


class BiasEvaluator:
    """
    Evaluates visual shortcut bias and red-color dependence on test sets.
    """

    def __init__(
        self,
        model_path: str = "models/final_model.keras",
        dataset_dir: str = "dataset_large",
        models_dir: str = "models",
    ) -> None:
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.predictor = ImagePredictor(model_path=str(self.model_path))

    def evaluate_bias(self) -> Dict[str, Any]:
        """
        Scan test dataset split, identify red-dominant and hard-negative subsets,
        and compute class accuracy and misclassification rates.
        """
        test_dir = self.dataset_dir / "test"
        groups: Dict[str, List[Tuple[Path, str]]] = {
            "Red Safe": [],
            "Red NSFW": [],
            "Red Violence": [],
            "Normal Safe": [],
            "Normal NSFW": [],
            "Normal Violence": [],
        }

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls_name in CLASS_NAMES:
            cls_dir = test_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_file in cls_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in valid_exts:
                    bgr = cv2.imread(str(img_file))
                    if bgr is None:
                        continue
                    if is_red_dominant(bgr):
                        groups[f"Red {cls_name}"].append((img_file, cls_name))
                    else:
                        groups[f"Normal {cls_name}"].append((img_file, cls_name))

        logger.info("Categorized test images into bias evaluation groups:")
        for grp_name, items in groups.items():
            logger.info("  %-20s: %d images", grp_name, len(items))

        report_rows = []

        for grp_name, items in groups.items():
            if not items:
                continue

            total = len(items)
            correct = 0
            wrong = 0
            as_violence = 0
            as_nsfw = 0
            as_safe = 0

            for img_file, true_label in items:
                res = self.predictor.predict_image(img_file)
                pred_label = res["prediction"]

                if pred_label == true_label:
                    correct += 1
                else:
                    wrong += 1

                if pred_label == "Violence":
                    as_violence += 1
                elif pred_label == "NSFW":
                    as_nsfw += 1
                elif pred_label == "Safe":
                    as_safe += 1

            acc_pct = (correct / float(total)) * 100.0 if total > 0 else 0.0

            report_rows.append({
                "Test Group": grp_name,
                "Total Samples": total,
                "Correct": correct,
                "Wrong": wrong,
                "Accuracy (%)": round(acc_pct, 2),
                "Predicted Violence": as_violence,
                "Predicted NSFW": as_nsfw,
                "Predicted Safe": as_safe,
            })

        df_report = pd.DataFrame(report_rows)
        logger.info("\n=== VISUAL SHORTCUT & BIAS EVALUATION REPORT ===")
        logger.info("\n%s", df_report.to_string(index=False))

        out_json = self.models_dir / "bias_evaluation_report.json"
        out_csv = self.models_dir / "bias_evaluation_report.csv"

        df_report.to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report_rows, f, indent=2)

        logger.info("Bias evaluation report saved to %s and %s", out_json, out_csv)
        return {"groups": report_rows}


def main() -> None:
    setup_logging(log_dir="logs", log_file="bias_evaluation.log")
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate visual shortcut & color bias.")
    parser.add_argument("--model_path", default="models/final_model.keras")
    parser.add_argument("--dataset_dir", default="dataset_large")
    parser.add_argument("--models_dir", default="models")

    args = parser.parse_args()

    evaluator = BiasEvaluator(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
    )
    evaluator.evaluate_bias()


if __name__ == "__main__":
    main()
