from src.training_experiments import build_experiment_narrative


def test_build_experiment_narrative_returns_thesis_ready_summary():
    narrative = build_experiment_narrative()

    assert len(narrative) >= 5
    assert all("Stage" in row and "Finding" in row and "Interpretation" in row for row in narrative)
    assert any(row["Stage"] == "Baseline" for row in narrative)
    assert any(row["Stage"] == "Projection / Fusion" for row in narrative)
