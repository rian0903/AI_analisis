import numpy as np

from src.decision_utils import assess_decision


def test_assess_decision_marks_small_margin_as_review():
    probs = np.array([0.02, 0.51, 0.47], dtype=np.float32)
    result = assess_decision(probs, class_names=["Safe", "Violence", "NSFW"])

    assert result["top_1_index"] == 1
    assert result["top_1_confidence"] == 0.51
    assert result["top_2_confidence"] == 0.47
    assert result["margin"] == 0.04
    assert result["entropy"] > 0.0
    assert result["review"] is True
    assert result["decision"] == "REVIEW"


def test_assess_decision_accepts_clear_classification():
    probs = np.array([0.02, 0.9, 0.08], dtype=np.float32)
    result = assess_decision(probs, class_names=["Safe", "Violence", "NSFW"])

    assert result["top_1_index"] == 1
    assert result["review"] is False
    assert result["decision"] == "Violence"
