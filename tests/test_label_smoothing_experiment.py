from src.training_experiments import build_label_smoothing_experiments


def test_build_label_smoothing_experiments_includes_baseline_and_three_smoothing_levels():
    experiments = build_label_smoothing_experiments()

    assert [exp["name"] for exp in experiments] == [
        "Baseline",
        "Label Smoothing 0.00",
        "Label Smoothing 0.05",
        "Label Smoothing 0.10",
    ]
    assert experiments[0]["label_smoothing"] == 0.0
    assert experiments[1]["label_smoothing"] == 0.0
    assert experiments[2]["label_smoothing"] == 0.05
    assert experiments[3]["label_smoothing"] == 0.10
