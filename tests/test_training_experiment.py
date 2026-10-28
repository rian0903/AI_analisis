from src.training_experiments import build_training_experiments


def test_build_training_experiments_includes_baseline_class_weights_and_focal_loss():
    experiments = build_training_experiments()

    assert [exp["name"] for exp in experiments] == ["Baseline", "Class Weight", "Focal Loss"]
    assert experiments[0]["loss"] == "categorical_crossentropy"
    assert experiments[1]["use_class_weights"] is True
    assert experiments[2]["use_focal_loss"] is True
