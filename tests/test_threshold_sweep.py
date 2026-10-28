import numpy as np

from src.thresholds import sweep_confidence_thresholds


def test_threshold_sweep_returns_precision_recall_tradeoff():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    y_prob = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.75, 0.20, 0.05],
            [0.20, 0.75, 0.05],
            [0.30, 0.65, 0.05],
            [0.05, 0.10, 0.85],
            [0.40, 0.05, 0.55],
        ],
        dtype=np.float32,
    )

    rows = sweep_confidence_thresholds(y_true, y_prob, [0.5, 0.7, 0.8])

    assert rows[0]["threshold"] == 0.5
    assert rows[0]["acceptance_rate"] >= 0.0
    assert rows[0]["precision"] >= 0.0
    assert rows[0]["recall"] >= 0.0
    assert rows[0]["f1"] >= 0.0
    assert len(rows) == 3
