from src.training_experiments import build_projection_experiments


def test_build_projection_experiments_includes_projection_and_fusion_variants():
    experiments = build_projection_experiments()

    assert [exp["name"] for exp in experiments] == [
        "Baseline Fusion",
        "Projection Head",
        "Projection Head + Attention",
    ]
    assert experiments[0]["use_projection_head"] is False
    assert experiments[1]["use_projection_head"] is True
    assert experiments[2]["use_attention_fusion"] is True
    assert experiments[2]["projection_dim"] == 256
