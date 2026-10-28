import numpy as np

from src.evaluate import compute_extended_classification_metrics


def test_extended_metrics_include_per_class_and_balanced_scores():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_prob = np.array(
        [
            [0.9, 0.08, 0.02],
            [0.4, 0.55, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.05, 0.05, 0.9],
            [0.7, 0.1, 0.2],
        ],
        dtype=np.float32,
    )

    metrics = compute_extended_classification_metrics(y_true, y_pred, y_prob, class_names=["Safe", "Violence", "NSFW"])

    assert "balanced_accuracy" in metrics
    assert "overall_accuracy" in metrics
    assert metrics["Safe_precision"] >= 0.0
    assert metrics["Violence_recall"] >= 0.0
    assert metrics["NSFW_f1"] >= 0.0
    assert metrics["balanced_accuracy"] >= 0.0
    assert metrics["average_precision_macro"] >= 0.0
    assert metrics["roc_auc_macro"] >= 0.0
