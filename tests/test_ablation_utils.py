from src.ablation import build_ablation_experiments, summarize_ablation_results


def test_ablation_experiments_cover_global_human_pose_spatial_stages():
    experiments = build_ablation_experiments()

    assert [exp["name"] for exp in experiments] == ["E1", "E2", "E3", "E4"]
    assert experiments[0]["global"] is True
    assert experiments[0]["human"] is False
    assert experiments[0]["pose"] is False
    assert experiments[0]["spatial"] is False

    assert experiments[1]["global"] is True
    assert experiments[1]["human"] is True
    assert experiments[2]["pose"] is True
    assert experiments[3]["spatial"] is True


def test_summarize_ablation_results_returns_research_table_columns():
    summary = summarize_ablation_results([
        {"name": "E1", "accuracy": 0.70, "macro_f1": 0.60},
        {"name": "E2", "accuracy": 0.75, "macro_f1": 0.65},
    ])

    assert "Model" in summary
    assert "Accuracy" in summary
    assert "Macro F1" in summary
    assert summary["Model"][0] == "E1"
    assert summary["Accuracy"][1] == 0.75
