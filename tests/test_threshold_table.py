from src.thresholds import build_threshold_table


def test_build_threshold_table_has_research_columns():
    rows = [
        {"threshold": 0.3, "precision": 0.75, "recall": 0.68, "f1": 0.71, "acceptance_rate": 0.8},
        {"threshold": 0.5, "precision": 0.82, "recall": 0.65, "f1": 0.72, "acceptance_rate": 0.65},
    ]

    table = build_threshold_table(rows)

    assert list(table.keys()) == [
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "Acceptance Rate",
        "Review Count",
    ]
    assert table["Threshold"] == [0.3, 0.5]
    assert table["F1"][1] == 0.72
