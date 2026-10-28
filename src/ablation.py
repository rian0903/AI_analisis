from __future__ import annotations

from typing import Any, Dict, List


def build_ablation_experiments() -> List[Dict[str, Any]]:
    """Return the required ablation stages for MobileNetV2 + multi-feature fusion research."""
    return [
        {"name": "E1", "global": True, "human": False, "pose": False, "spatial": False},
        {"name": "E2", "global": True, "human": True, "pose": False, "spatial": False},
        {"name": "E3", "global": True, "human": True, "pose": True, "spatial": False},
        {"name": "E4", "global": True, "human": True, "pose": True, "spatial": True},
    ]


def summarize_ablation_results(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Build a compact tabular summary for ablation comparisons."""
    summary: Dict[str, List[float | str]] = {"Model": [], "Accuracy": [], "Macro F1": []}
    for result in results:
        summary["Model"].append(result.get("name", ""))
        summary["Accuracy"].append(float(result.get("accuracy", 0.0)))
        summary["Macro F1"].append(float(result.get("macro_f1", 0.0)))
    return summary


def generate_ablation_table(results: List[Dict[str, Any]]) -> Dict[str, List[float | str]]:
    """Generate a research-ready experiment table matching the ablation plan in the thesis roadmap."""
    experiments = build_ablation_experiments()
    experiment_map = {exp["name"]: exp for exp in experiments}

    table: Dict[str, List[float | str]] = {
        "Model": [],
        "Global": [],
        "Human": [],
        "Pose": [],
        "Spatial": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "Macro F1": [],
        "Weighted F1": [],
    }

    for result in results:
        name = str(result.get("name", ""))
        config = experiment_map.get(name, {"global": False, "human": False, "pose": False, "spatial": False})
        table["Model"].append(name)
        table["Global"].append("✓" if config.get("global", False) else "—")
        table["Human"].append("✓" if config.get("human", False) else "—")
        table["Pose"].append("✓" if config.get("pose", False) else "—")
        table["Spatial"].append("✓" if config.get("spatial", False) else "—")
        table["Accuracy"].append(float(result.get("accuracy", 0.0)))
        table["Precision"].append(float(result.get("precision", 0.0)))
        table["Recall"].append(float(result.get("recall", 0.0)))
        table["Macro F1"].append(float(result.get("macro_f1", 0.0)))
        table["Weighted F1"].append(float(result.get("weighted_f1", 0.0)))

    return table
