from src.training_experiments import build_hard_negative_experiments


def test_build_hard_negative_experiments_includes_baseline_and_hard_negative_expansion():
    experiments = build_hard_negative_experiments()

    assert [exp["name"] for exp in experiments] == [
        "Baseline",
        "Hard Negative Expansion",
        "Hard Negative Expansion + Reweight",
    ]
    assert experiments[0]["hard_negative_ratio"] == 0.0
    assert experiments[1]["hard_negative_ratio"] > 0.0
    assert experiments[2]["use_hard_negative_reweighting"] is True
