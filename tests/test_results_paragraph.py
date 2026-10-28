from src.training_experiments import build_results_paragraph


def test_build_results_paragraph_returns_formal_thesis_text():
    paragraph = build_results_paragraph()

    assert isinstance(paragraph, str)
    assert len(paragraph) > 250
    assert "eksperimen" in paragraph.lower()
    assert "baseline" in paragraph.lower()
    assert "model" in paragraph.lower()
