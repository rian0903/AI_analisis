from src.training_experiments import build_calibration_experiments


def test_build_calibration_experiments_includes_temperature_scaling_variants():
    experiments = build_calibration_experiments()

    assert [exp["name"] for exp in experiments] == [
        "Baseline",
        "Temperature Scaling 1.5",
        "Temperature Scaling 2.0",
    ]
    assert experiments[0]["temperature"] == 1.0
    assert experiments[1]["temperature"] == 1.5
    assert experiments[2]["temperature"] == 2.0
    assert experiments[1]["use_temperature_scaling"] is True
