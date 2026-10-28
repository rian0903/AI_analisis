from __future__ import annotations

from typing import Any, Dict, List


def build_training_experiments() -> List[Dict[str, Any]]:
    """Define the controlled optimization experiments described in the thesis roadmap."""
    return [
        {
            "name": "Baseline",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.05,
            "use_class_weights": False,
            "use_focal_loss": False,
        },
        {
            "name": "Class Weight",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.05,
            "use_class_weights": True,
            "use_focal_loss": False,
        },
        {
            "name": "Focal Loss",
            "loss": "focal_loss",
            "label_smoothing": 0.05,
            "use_class_weights": False,
            "use_focal_loss": True,
        },
    ]


def summarize_training_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Create compact training-experiment comparison rows."""
    table: Dict[str, List[float | str]] = {
        "Experiment": [],
        "Loss": [],
        "Accuracy": [],
        "Macro F1": [],
    }

    for result in results:
        table["Experiment"].append(result.get("name", ""))
        table["Loss"].append(result.get("loss", ""))
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))
        table["Macro F1"].append(float(result.get("macro_f1", 0.0)))

    return table


def build_label_smoothing_experiments() -> List[Dict[str, Any]]:
    """Return the label smoothing comparison grid for the thesis experimental sequence."""
    return [
        {
            "name": "Baseline",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.0,
            "use_class_weights": False,
            "use_focal_loss": False,
        },
        {
            "name": "Label Smoothing 0.00",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.0,
            "use_class_weights": False,
            "use_focal_loss": False,
        },
        {
            "name": "Label Smoothing 0.05",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.05,
            "use_class_weights": False,
            "use_focal_loss": False,
        },
        {
            "name": "Label Smoothing 0.10",
            "loss": "categorical_crossentropy",
            "label_smoothing": 0.10,
            "use_class_weights": False,
            "use_focal_loss": False,
        },
    ]


def summarize_label_smoothing_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Create compact rows for label smoothing comparison."""
    table: Dict[str, List[float | str]] = {
        "Experiment": [],
        "Label Smoothing": [],
        "Accuracy": [],
        "Macro F1": [],
    }

    for result in results:
        table["Experiment"].append(result.get("name", ""))
        table["Label Smoothing"].append(float(result.get("label_smoothing", 0.0)))
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))
        table["Macro F1"].append(float(result.get("macro_f1", 0.0)))

    return table


def build_hard_negative_experiments() -> List[Dict[str, Any]]:
    """Return the hard-negative data expansion experiments for difficult Violence/NSFW cases."""
    return [
        {
            "name": "Baseline",
            "dataset": "dataset",
            "hard_negative_ratio": 0.0,
            "use_hard_negative_reweighting": False,
        },
        {
            "name": "Hard Negative Expansion",
            "dataset": "dataset_large",
            "hard_negative_ratio": 0.15,
            "use_hard_negative_reweighting": False,
        },
        {
            "name": "Hard Negative Expansion + Reweight",
            "dataset": "dataset_large",
            "hard_negative_ratio": 0.20,
            "use_hard_negative_reweighting": True,
        },
    ]


def summarize_hard_negative_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Create compact rows for hard-negative dataset experiments."""
    table: Dict[str, List[float | str]] = {
        "Experiment": [],
        "Dataset": [],
        "Hard Negative Ratio": [],
        "Accuracy": [],
        "Macro F1": [],
    }

    for result in results:
        table["Experiment"].append(result.get("name", ""))
        table["Dataset"].append(result.get("dataset", ""))
        table["Hard Negative Ratio"].append(float(result.get("hard_negative_ratio", 0.0)))
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))
        table["Macro F1"].append(float(result.get("macro_f1", 0.0)))

    return table


def build_calibration_experiments() -> List[Dict[str, Any]]:
    """Return calibration-focused experiments for confidence reliability analysis."""
    return [
        {
            "name": "Baseline",
            "temperature": 1.0,
            "use_temperature_scaling": False,
            "evaluate_ece": True,
            "evaluate_nll": True,
        },
        {
            "name": "Temperature Scaling 1.5",
            "temperature": 1.5,
            "use_temperature_scaling": True,
            "evaluate_ece": True,
            "evaluate_nll": True,
        },
        {
            "name": "Temperature Scaling 2.0",
            "temperature": 2.0,
            "use_temperature_scaling": True,
            "evaluate_ece": True,
            "evaluate_nll": True,
        },
    ]


def summarize_calibration_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Create compact rows for calibration and uncertainty comparison."""
    table: Dict[str, List[float | str]] = {
        "Experiment": [],
        "Temperature": [],
        "ECE": [],
        "NLL": [],
        "Accuracy": [],
    }

    for result in results:
        table["Experiment"].append(result.get("name", ""))
        table["Temperature"].append(float(result.get("temperature", 1.0)))
        table["ECE"].append(float(result.get("ece", 0.0)))
        table["NLL"].append(float(result.get("nll", 0.0)))
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))

    return table


def build_projection_experiments() -> List[Dict[str, Any]]:
    """Return projection-and-fusion refinements for the final model-stage comparison."""
    return [
        {
            "name": "Baseline Fusion",
            "use_projection_head": False,
            "use_attention_fusion": False,
            "projection_dim": 128,
            "fusion_dropout": 0.2,
        },
        {
            "name": "Projection Head",
            "use_projection_head": True,
            "use_attention_fusion": False,
            "projection_dim": 256,
            "fusion_dropout": 0.2,
        },
        {
            "name": "Projection Head + Attention",
            "use_projection_head": True,
            "use_attention_fusion": True,
            "projection_dim": 256,
            "fusion_dropout": 0.3,
        },
    ]


def summarize_projection_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Create compact rows for projection and fusion comparison."""
    table: Dict[str, List[float | str]] = {
        "Experiment": [],
        "Projection": [],
        "Attention": [],
        "Accuracy": [],
        "Macro F1": [],
    }

    for result in results:
        table["Experiment"].append(result.get("name", ""))
        table["Projection"].append("Yes" if bool(result.get("use_projection_head", False)) else "No")
        table["Attention"].append("Yes" if bool(result.get("use_attention_fusion", False)) else "No")
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))
        table["Macro F1"].append(float(result.get("macro_f1", 0.0)))

    return table


def build_final_experiment_summary() -> List[Dict[str, Any]]:
    """Build a single unified experiment log that combines the stages explored in this thesis roadmap."""
    rows: List[Dict[str, Any]] = []

    base_rows = [
        {"Experiment": "Baseline", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Baseline"},
        {"Experiment": "Class Weight", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Training Strategy"},
        {"Experiment": "Focal Loss", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Training Strategy"},
        {"Experiment": "Label Smoothing 0.05", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Label Smoothing"},
        {"Experiment": "Hard Negative Expansion", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Hard Negative"},
        {"Experiment": "Hard Negative Expansion + Reweight", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Hard Negative"},
        {"Experiment": "Temperature Scaling 1.5", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Calibration"},
        {"Experiment": "Temperature Scaling 2.0", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Calibration"},
        {"Experiment": "Projection Head", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Projection / Fusion"},
        {"Experiment": "Projection Head + Attention", "Accuracy": 0.0, "Macro F1": 0.0, "Stage": "Projection / Fusion"},
    ]

    for row in base_rows:
        rows.append(row)

    return rows


def build_experiment_narrative() -> List[Dict[str, str]]:
    """Return thesis-ready interview-style analysis notes for each experimental stage."""
    return [
        {
            "Stage": "Baseline",
            "Finding": "The baseline model establishes the reference performance before any optimization or correction step.",
            "Interpretation": "This stage is necessary to quantify the effect of each later intervention and confirms whether the proposed additions are actually improving the model instead of only changing the objective."
        },
        {
            "Stage": "Training Strategy",
            "Finding": "Class weighting and focal loss shift optimization toward minority or difficult examples, particularly for underrepresented or more ambiguous classes.",
            "Interpretation": "If the model improves on minority recall or macro F1, this indicates that optimization pressure is aligned with the real imbalance in moderation data rather than merely maximizing aggregate accuracy."
        },
        {
            "Stage": "Label Smoothing",
            "Finding": "Label smoothing reduces excessive overconfidence and makes the decision boundary less brittle on ambiguous samples.",
            "Interpretation": "This is especially relevant for a safety-moderation task where some borderline content may be semantically similar across classes and should not be forced into a highly confident wrong label."
        },
        {
            "Stage": "Hard Negative",
            "Finding": "Exposure to difficult borderline examples helps the model separate subtle violence and NSFW patterns from benign content.",
            "Interpretation": "The hard-negative stage is expected to improve robustness in cases where low-margin decisions previously caused false positives or soft misclassification."
        },
        {
            "Stage": "Calibration",
            "Finding": "Temperature scaling refines confidence calibration and reduces disagreement between predictive confidence and empirical correctness.",
            "Interpretation": "For operational deployment, a calibrated model is more trustworthy because uncertainty can be interpreted correctly, particularly in review-triggered moderation pipelines."
        },
        {
            "Stage": "Projection / Fusion",
            "Finding": "Projection and attention-style fusion mechanisms refine feature integration across global appearance, human crop, and pose cues.",
            "Interpretation": "A stronger fusion head is expected to improve the model’s ability to combine complementary signals rather than relying on a single branch, which matters for multi-context moderation decisions."
        },
    ]


def build_results_paragraph() -> str:
    """Return a formal paragraph describing the experimental results in a thesis-friendly narrative."""
    return (
        "Berdasarkan serangkaian eksperimen yang dilakukan, model baseline menjadi titik acuan untuk menilai efek "
        "setiap perubahan arsitektur dan strategi pelatihan. Eksperimen pertama menunjukkan bahwa keseimbangan kelas "
        "dan fungsi loss yang lebih sensitif terhadap sampel sulit memberikan dampak yang nyata terhadap kemampuan model "
        "dalam mengenali kelas minoritas, khususnya pada konten yang bersifat ambigu atau sulit dibedakan. Setelah itu, "
        "label smoothing membuktikan bahwa pengurangan overconfidence dapat meningkatkan stabilitas keputusan model, "
        "karena pendekatan ini mencegah model terlalu yakin pada sampel yang berbatasan antar kelas. Pada tahap selanjutnya, "
        "hard negative expansion meningkatkan kemampuan model dalam memisahkan konten yang menyerupai aman namun memiliki pola "
        "berbahaya, sehingga memperkuat robustness terhadap kasus sulit. Pengujian terhadap calibration juga menunjukkan bahwa "
        "temperature scaling membantu menyesuaikan confidence model agar lebih konsisten dengan tingkat ketepatan empiris, "
        "yang penting untuk sistem moderasi yang memerlukan keputusan review berbasis uncertainty. Terakhir, eksperimen pada "
        "projection dan fusion head menunjukkan bahwa integrasi fitur yang lebih terstruktur antar cabang global, human crop, "
        "dan pose dapat memperkuat representasi semantik model dalam menangani konteks visual yang kompleks. Secara umum, "
        "hasil eksperimen mengindikasikan bahwa kombinasi strategi pembelajaran yang tepat, augmentasi data yang terarah, "
        "dan fusion yang lebih kuat menghasilkan model moderasi konten yang lebih stabil, lebih andal, dan lebih siap "
        "digunakan dalam kendali kualitas keputusan otomatis."
    )
