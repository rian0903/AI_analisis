import math
from typing import Iterable, List, Optional

import numpy as np


def assess_decision(
    probabilities: Iterable[float],
    class_names: Optional[List[str]] = None,
    review_margin: float = 0.05,
    review_confidence: float = 0.60,
    entropy_ratio_threshold: float = 0.90,
) -> dict:
    """Convert softmax probabilities into a robust decision with uncertainty checks.

    The function exposes the top-1/top-2 confidence, margin, entropy and a final
    decision label. When the model is uncertain (small margin, weak top-1 score,
    or high entropy), it returns a REVIEW decision instead of making a direct
    class assignment.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 1:
        raise ValueError(f"Expected a 1D probability vector, got shape={probs.shape}")
    if probs.size == 0:
        raise ValueError("Probability vector cannot be empty")

    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be within [0, 1]")

    sum_prob = float(probs.sum())
    if not np.isclose(sum_prob, 1.0, atol=1e-6):
        probs = probs / sum_prob

    top_1_index = int(np.argmax(probs))
    top_1_confidence = float(probs[top_1_index])

    second_best = float(np.partition(probs, -2)[-2]) if probs.size > 1 else 0.0
    top_2_confidence = float(second_best)
    margin = float(top_1_confidence - top_2_confidence)

    eps = 1e-12
    entropy = float(-(probs * np.log(probs + eps)).sum())
    max_entropy = math.log(max(2, probs.size))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    review = bool(
        top_1_confidence < review_confidence
        or margin < review_margin
        or normalized_entropy >= entropy_ratio_threshold
    )

    names = class_names or [str(i) for i in range(len(probs))]
    decision_label = "REVIEW" if review else str(names[top_1_index])

    return {
        "top_1_index": top_1_index,
        "top_2_index": int(np.argsort(probs)[-2]) if probs.size > 1 else top_1_index,
        "top_1_confidence": round(top_1_confidence, 6),
        "top_2_confidence": round(top_2_confidence, 6),
        "margin": round(margin, 6),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "review": review,
        "decision": decision_label,
        "label": decision_label,
    }
