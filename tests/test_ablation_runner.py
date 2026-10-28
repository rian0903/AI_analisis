from src.ablation import generate_ablation_table


def test_generate_ablation_table_has_research_columns():
    results = [
        {
            "name": "E1",
            "accuracy": 0.73,
            "precision": 0.71,
            "recall": 0.69,
            "macro_f1": 0.70,
            "weighted_f1": 0.72,
        },
        {
            "name": "E2",
            "accuracy": 0.78,
            "precision": 0.77,
            "recall": 0.75,
            "macro_f1": 0.76,
            "weighted_f1": 0.77,
        },
    ]

    table = generate_ablation_table(results)

    assert list(table.keys()) == [
        "Model",
        "Global",
        "Human",
        "Pose",
        "Spatial",
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1",
        "Weighted F1",
    ]
    assert table["Model"] == ["E1", "E2"]
    assert table["Global"] == ["✓", "✓"]
    assert table["Accuracy"][1] == 0.78
