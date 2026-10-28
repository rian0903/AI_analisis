"""
evaluate.py
===========
Comprehensive evaluation and misclassification analysis for the content moderation model.

Outputs
-------
- Accuracy, Precision, Recall, Macro/Weighted F1-score
- Confusion matrix (models/confusion_matrix.png & CSV)
- One-vs-Rest ROC curves (models/roc_curves.png)
- Excel evaluation report (models/evaluation_metrics.xlsx)
- Automatic error categorization & image export (models/evaluation/errors/)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, MODELS_DIR
from src.dataset import DatasetLoader
from src.utils import save_figure, setup_logging

logger = logging.getLogger(__name__)


def compute_extended_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute extended multiclass metrics including specificity and AUC-based scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    class_names = list(class_names) if class_names is not None else [str(i) for i in range(len(np.unique(y_true)))]
    num_classes = len(class_names)

    metrics: Dict[str, float] = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "overall_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "overall_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "overall_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, class_name in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        tn = int(np.sum((y_true != i) & (y_pred != i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))

        precision = per_class_precision[i] if i < len(per_class_precision) else 0.0
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
        f1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics[f"{class_name}_precision"] = float(precision)
        metrics[f"{class_name}_recall"] = float(recall)
        metrics[f"{class_name}_f1"] = float(f1)
        metrics[f"{class_name}_specificity"] = float(specificity)

    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)

    if y_prob.shape[1] != num_classes:
        if y_prob.shape[1] == 1 and num_classes > 1:
            y_prob = np.hstack([1.0 - y_prob, y_prob])
        else:
            raise ValueError(f"Probability matrix shape {y_prob.shape} does not align with {num_classes} classes.")

    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    metrics["roc_auc_macro"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    metrics["average_precision_macro"] = float(average_precision_score(y_bin, y_prob, average="macro"))

    for i, class_name in enumerate(class_names):
        class_prob = y_prob[:, i]
        class_true = y_bin[:, i]
        metrics[f"{class_name}_roc_auc"] = float(roc_auc_score(class_true, class_prob))
        metrics[f"{class_name}_pr_auc"] = float(average_precision_score(class_true, class_prob))

    return metrics


class ModelEvaluator:
    def __init__(
        self,
        model_path: str = "models/final_model.keras",
        models_dir: str = "models",
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)

        self._model: Optional[keras.Model] = None

    def load_model(self) -> keras.Model:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info("Loading model from %s...", self.model_path)
        self._model = keras.models.load_model(str(self.model_path))
        return self._model

    def _collect_predictions(
        self, dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        all_true: List[np.ndarray] = []
        all_prob: List[np.ndarray] = []
        all_global_imgs: List[np.ndarray] = []

        logger.info("Running inference on test dataset...")
        for inputs, labels in dataset:
            probs = self._model.predict(inputs, verbose=0)
            all_true.append(np.argmax(labels.numpy(), axis=1))
            all_prob.append(probs)
            all_global_imgs.append(inputs["input_global"].numpy())

        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        return y_true, y_pred, y_prob, all_global_imgs

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        metrics = compute_extended_classification_metrics(y_true, y_pred, y_prob, self.class_names)

        logger.info("--- Evaluation Metrics ---")
        for k, v in metrics.items():
            logger.info("  %-35s: %.4f", k, v)

        return metrics

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#1a1a2e")

        for ax, title, data, fmt, vmax in zip(
            axes,
            ["Confusion Matrix (Raw Counts)", "Confusion Matrix (Normalized)"],
            [cm, cm_norm],
            ["d", ".2f"],
            [None, 1.0],
        ):
            ax.set_facecolor("#1a1a2e")
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                vmin=0,
                vmax=vmax,
                linewidths=0.5,
                linecolor="#2c2c54",
                annot_kws={"size": 12, "color": "white"},
            )
            ax.set_title(title, color="white", fontsize=13, pad=10)
            ax.set_xlabel("Predicted Label", color="white")
            ax.set_ylabel("True Label", color="white")
            ax.tick_params(colors="white")

        save_figure(fig, self.models_dir / "confusion_matrix.png")
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_df.to_csv(self.models_dir / "confusion_matrix.csv")

    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        colors = ["#2ecc71", "#e67e22", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        auc_scores: Dict[str, float] = {}
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = float(auc(fpr, tpr))
            auc_scores[class_name] = roc_auc
            ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{class_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", linewidth=1.2, label="Random (AUC = 0.500)")
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.01])
        ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
        ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
        ax.set_title("ROC Curves — One-vs-Rest", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2c2c54", labelcolor="white", fontsize=11, loc="lower right")
        ax.grid(color="#2c2c54", linestyle="--", alpha=0.5)

        save_figure(fig, self.models_dir / "roc_curves.png")
        return auc_scores

    def _save_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        all_global_imgs: List[np.ndarray],
    ) -> None:
        """
        Save misclassified test images into models/evaluation/errors/ categories.
        """
        errors_dir = self.models_dir / "evaluation" / "errors"
        error_categories = [
            "Safe_as_Violence", "Safe_as_NSFW",
            "Violence_as_Safe", "Violence_as_NSFW",
            "NSFW_as_Safe", "NSFW_as_Violence",
        ]
        for cat in error_categories:
            (errors_dir / cat).mkdir(parents=True, exist_ok=True)

        batch_imgs = np.concatenate(all_global_imgs, axis=0)
        error_count = 0

        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                error_count += 1
                t_name = self.class_names[t]
                p_name = self.class_names[p]
                folder_name = f"{t_name}_as_{p_name}"
                dest_folder = errors_dir / folder_name

                img_np = batch_imgs[idx]
                img_bgr = cv2.cvtColor((img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                file_stem = f"err_{error_count:04d}"
                img_path = dest_folder / f"{file_stem}.jpg"
                json_path = dest_folder / f"{file_stem}.json"

                cv2.imwrite(str(img_path), img_bgr)

                probs_dict = {
                    name: round(float(y_prob[idx, i]), 6)
                    for i, name in enumerate(self.class_names)
                }

                meta = {
                    "sample_idx": idx,
                    "true_label": t_name,
                    "predicted_label": p_name,
                    "confidence": round(float(y_prob[idx, p]), 4),
                    "probabilities": probs_dict,
                }

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

        logger.info("Error analysis complete: saved %d misclassified images to %s", error_count, errors_dir)

    def _export_to_excel(
        self,
        metrics: Dict[str, float],
        auc_scores: Dict[str, float],
        report_str: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        excel_path = self.models_dir / "evaluation_metrics.xlsx"
        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
            summary_data = {
                "Metric": [
                    "Overall Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1-Score",
                    "Weighted F1-Score",
                    "Balanced Accuracy",
                    "ROC-AUC Macro",
                    "PR-AUC Macro",
                ],
                "Value": [
                    metrics.get("overall_accuracy", 0),
                    metrics.get("overall_precision_macro", 0),
                    metrics.get("overall_recall_macro", 0),
                    metrics.get("overall_f1_macro", 0),
                    metrics.get("overall_f1_weighted", 0),
                    metrics.get("balanced_accuracy", 0),
                    metrics.get("roc_auc_macro", 0),
                    metrics.get("average_precision_macro", 0),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            per_class_rows = [
                {
                    "Class": class_name,
                    "Precision": metrics.get(f"{class_name}_precision", 0),
                    "Recall": metrics.get(f"{class_name}_recall", 0),
                    "F1-Score": metrics.get(f"{class_name}_f1", 0),
                    "Specificity": metrics.get(f"{class_name}_specificity", 0),
                    "ROC-AUC": metrics.get(f"{class_name}_roc_auc", 0),
                    "PR-AUC": metrics.get(f"{class_name}_pr_auc", 0),
                    "AUC": auc_scores.get(class_name, 0),
                }
                for class_name in self.class_names
            ]
            pd.DataFrame(per_class_rows).to_excel(writer, sheet_name="Per-Class Metrics", index=False)

            report_df = pd.DataFrame({"Classification Report": [report_str]})
            report_df.to_excel(writer, sheet_name="Classification Report", index=False)

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
            cm_df.to_excel(writer, sheet_name="Confusion Matrix")

        logger.info("Evaluation metrics exported to %s", excel_path)

    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        if self._model is None:
            self.load_model()

        y_true, y_pred, y_prob, all_global_imgs = self._collect_predictions(test_dataset)
        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        report_str = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        logger.info("\nClassification Report:\n%s", report_str)

        self._plot_confusion_matrix(y_true, y_pred)
        auc_scores = self._plot_roc_curves(y_true, y_prob)
        metrics.update({f"AUC_{k}": v for k, v in auc_scores.items()})

        self._save_error_analysis(y_true, y_pred, y_prob, all_global_imgs)
        self._export_to_excel(metrics, auc_scores, report_str, y_true, y_pred)

        logger.info("Evaluation complete.")
        return metrics


def main() -> None:
    setup_logging(log_dir="logs", log_file="evaluation.log")
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the content moderation model.")
    parser.add_argument("--model_path", default="models/final_model.keras")
    parser.add_argument("--dataset_dir", default="dataset_large")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    loader = DatasetLoader(dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    _, _, test_ds = loader.get_datasets()

    evaluator = ModelEvaluator(model_path=args.model_path, models_dir=args.models_dir)
    evaluator.load_model()
    evaluator.evaluate(test_ds)


if __name__ == "__main__":
    main()
