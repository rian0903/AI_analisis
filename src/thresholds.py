from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def sweep_confidence_thresholds(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate top-1 confidence thresholds for precision/recall trade-off.

    For each threshold t, predictions with top-1 confidence < t are treated as
    REVIEW/uncertain and excluded from acceptance. This is a practical approach
    for moderation systems where low-confidence cases should be deferred rather
    than forced into a hard class decision.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError(f"y_true length ({y_true.shape[0]}) and y_prob rows ({y_prob.shape[0]}) do not match.")

    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    top1_idx = np.argmax(y_prob, axis=1)
    top1_conf = y_prob[np.arange(len(y_true)), top1_idx]
    total_correct = int(np.sum(top1_idx == y_true))
    total_samples = int(len(y_true))

    swept: List[Dict[str, float]] = []
    for threshold in sorted(float(t) for t in thresholds):
        mask = top1_conf >= threshold
        accepted_count = int(np.sum(mask))
        accepted_correct = int(np.sum((mask) & (top1_idx == y_true)))

        precision = accepted_correct / accepted_count if accepted_count > 0 else 0.0
        recall = accepted_correct / total_correct if total_correct > 0 else 0.0
        accuracy = accepted_correct / accepted_count if accepted_count > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        row = {
            "threshold": round(float(threshold), 2),
            "accepted_count": accepted_count,
            "review_count": total_samples - accepted_count,
            "acceptance_rate": accepted_count / total_samples if total_samples > 0 else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
        swept.append(row)

    return swept


def select_best_threshold(
    rows: Sequence[Dict[str, float]],
    metric: str = "f1",
    prefer_high_acceptance: bool = True,
) -> Dict[str, float]:
    """Select the best threshold from a sweep using metric preference and acceptance-rate tie-break."""
    if not rows:
        raise ValueError("Threshold sweep results are empty.")

    sorted_rows = sorted(rows, key=lambda item: item.get(metric, 0.0), reverse=True)
    best = sorted_rows[0]

    if prefer_high_acceptance:
        candidates = [r for r in rows if abs(float(r.get(metric, 0.0)) - float(best.get(metric, 0.0))) < 1e-12]
        if len(candidates) > 1:
            best = max(candidates, key=lambda r: float(r.get("acceptance_rate", 0.0)))

    return best


def build_threshold_table(rows: Sequence[Dict[str, float]]) -> Dict[str, List[float | int]]:
    """Convert threshold sweep rows into a table-friendly structure for reporting and CSV export."""
    table: Dict[str, List[float | int]] = {
        "Threshold": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Acceptance Rate": [],
        "Review Count": [],
    }

    for row in rows:
        table["Threshold"].append(float(row.get("threshold", 0.0)))
        table["Precision"].append(float(row.get("precision", 0.0)))
        table["Recall"].append(float(row.get("recall", 0.0)))
        table["F1"].append(float(row.get("f1", 0.0)))
        table["Acceptance Rate"].append(float(row.get("acceptance_rate", 0.0)))
        table["Review Count"].append(int(row.get("review_count", 0)))

    return table


def run_validation_threshold_sweep(
    model: object,
    val_dataset: object,
    thresholds: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Evaluate a fitted model over a validation dataset for threshold selection."""
    import numpy as np

    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []

    for inputs, labels in val_dataset:
        probs = model.predict(inputs, verbose=0)
        batch_true = np.argmax(labels.numpy(), axis=1)
        batch_pred = np.argmax(probs, axis=1)
        all_true.extend(int(v) for v in batch_true)
        all_pred.extend(int(v) for v in batch_pred)
        all_probs.extend(list(np.asarray(probs, dtype=np.float32)))

    y_true = np.asarray(all_true, dtype=int)
    y_pred = np.asarray(all_pred, dtype=int)
    y_prob = np.asarray(all_probs, dtype=np.float32)

    rows = sweep_confidence_thresholds(y_true, y_prob, thresholds=thresholds)
    best = select_best_threshold(rows, metric="f1")
    summary = {
        "rows": rows,
        "best_threshold": best,
        "table": build_threshold_table(rows),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return summary
