from src.thresholds import select_best_threshold, sweep_confidence_thresholds


def test_select_best_threshold_prefers_high_f1_and_acceptance():
    rows = [
        {"threshold": 0.3, "precision": 0.7, "recall": 0.65, "f1": 0.68, "acceptance_rate": 0.75},
        {"threshold": 0.5, "precision": 0.82, "recall": 0.60, "f1": 0.69, "acceptance_rate": 0.60},
        {"threshold": 0.7, "precision": 0.88, "recall": 0.50, "f1": 0.64, "acceptance_rate": 0.45},
    ]

    best = select_best_threshold(rows, metric="f1")

    assert best["threshold"] == 0.5
    assert best["acceptance_rate"] == 0.60
