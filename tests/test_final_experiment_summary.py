from src.training_experiments import build_final_experiment_summary


def test_build_final_experiment_summary_produces_combined_comparison_rows():
    rows = build_final_experiment_summary()

    assert len(rows) >= 10
    assert "Experiment" in rows[0]
    assert "Accuracy" in rows[0]
    assert "Macro F1" in rows[0]
    assert any(row["Experiment"] == "Baseline" for row in rows)
    assert any(row["Experiment"] == "Projection Head + Attention" for row in rows)
