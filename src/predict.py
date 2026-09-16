"""
predict.py
==========
Inference module for the Multi-Feature Image Content Detection system.

Provides three entry points:

load_model()
    Load a trained multi-feature Keras model from disk.

predict_image()
    Predict content class for a single image file.

predict_folder()
    Batch-predict all images in a directory.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    LOW_CONFIDENCE_THRESHOLD,
    TOTAL_POSE_REL_DIM,
)
from src.human_analyzer import HumanAnalyzer
from src.utils import explain_activation, setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_default_predictor: Optional["ImagePredictor"] = None


def load_model(model_path: str = "models/final_model.keras") -> "ImagePredictor":
    global _default_predictor
    if _default_predictor is None or str(_default_predictor.model_path) != str(Path(model_path).resolve()):
        _default_predictor = ImagePredictor(model_path=model_path)
    return _default_predictor


def predict_image(
    image_path: Union[str, Path],
    model_path: str = "models/final_model.keras",
) -> Dict[str, Any]:
    predictor = load_model(model_path)
    return predictor.predict_image(image_path)


def predict_folder(
    folder_path: Union[str, Path],
    model_path: str = "models/final_model.keras",
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    predictor = load_model(model_path)
    return predictor.predict_folder(folder_path, recursive=recursive)


class ImagePredictor:
    """
    Object-oriented inference wrapper for the multi-feature content moderation model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/final_model.keras",
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.image_size = image_size

        self.human_analyzer = HumanAnalyzer(image_size=self.image_size)
        self._model: Optional[keras.Model] = None
        self._load_model_from_disk()

    def _load_model_from_disk(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Run 'python src/train.py' first to train and save the model."
            )
        logger.info("Loading model from %s...", self.model_path)
        self._model = keras.models.load_model(str(self.model_path))
        logger.info("Model loaded successfully.")

    def predict_image(
        self,
        image_path: Union[str, Path],
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.is_dir():
            raise IsADirectoryError(f"Expected a file, got directory: '{image_path}'. Use --folder instead.")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {image_path.suffix}. Supported: {SUPPORTED_EXTENSIONS}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not decode image file: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_array(img_rgb, source_name=str(image_path.resolve()))

    def predict_array(
        self,
        image_np: np.ndarray,
        source_name: str = "array_input",
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if image_np.ndim == 4:
            image_np = np.squeeze(image_np, axis=0)

        # Convert to float32 in [0, 1] for processing
        if image_np.dtype == np.uint8:
            img_float = image_np.astype(np.float32) / 255.0
        else:
            img_float = image_np.copy()
            if img_float.max() > 1.0:
                img_float /= 255.0

        # Resize to IMAGE_SIZE if needed
        if img_float.shape[:2] != self.image_size:
            img_float = cv2.resize(img_float, (self.image_size[1], self.image_size[0]))

        # Run HumanAnalyzer
        analysis = self.human_analyzer.analyze(img_float)

        # Prepare multi-inputs with batch dimension
        inputs_dict = {
            "input_global": np.expand_dims(img_float, axis=0),
            "input_human": np.expand_dims(analysis["human_crop"], axis=0),
            "input_pose": np.expand_dims(analysis["pose_rel_vector"], axis=0),
            "has_human": np.array([[analysis["has_human"]]], dtype=np.float32),
            "has_pose": np.array([[analysis["has_pose"]]], dtype=np.float32),
        }

        # Run forward pass
        probs_raw = self._model.predict(inputs_dict, verbose=0)[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        class_idx = int(np.argmax(probs_raw))
        predicted_class = self.class_names[class_idx]
        confidence = float(probs_raw[class_idx])

        probabilities_dict = {
            name: round(float(prob), 6)
            for name, prob in zip(self.class_names, probs_raw)
        }

        human_analysis_info = {
            "person_detected": bool(analysis["has_human"] == 1.0),
            "person_count": int(analysis["person_count"]),
            "pose_detected": bool(analysis["has_pose"] == 1.0),
        }

        low_confidence = bool(confidence < LOW_CONFIDENCE_THRESHOLD)

        explanations = {
            predicted_class: explain_activation(
                predicted_class, confidence, human_analysis_info
            )
        }

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities_dict,
            "human_analysis": human_analysis_info,
            "pipeline": analysis["pipeline"],
            "low_confidence": low_confidence,
            "inference_time_ms": round(elapsed_ms, 2),
            "file_path": source_name,
            "explanations": explanations,
            "explanation_summary": explanations[predicted_class],
            "input_tensors": inputs_dict,
        }

    def predict_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if recursive:
            image_paths = [p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            image_paths = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

        image_paths = sorted(image_paths)
        logger.info("Found %d image(s) in '%s'", len(image_paths), folder_path)

        results: List[Dict[str, Any]] = []
        for idx, img_path in enumerate(image_paths, start=1):
            try:
                res = self.predict_image(img_path)
                results.append(res)
                logger.info(
                    "[%d/%d] %s → %s (%.2f%%) [%s]",
                    idx, len(image_paths), img_path.name, res["prediction"], res["confidence"] * 100, res["pipeline"]
                )
            except Exception as exc:
                logger.warning("Failed to process %s: %s", img_path.name, exc)
                results.append({
                    "prediction": "Error",
                    "confidence": 0.0,
                    "probabilities": {n: 0.0 for n in self.class_names},
                    "human_analysis": {"person_detected": False, "person_count": 0, "pose_detected": False},
                    "pipeline": "error",
                    "low_confidence": True,
                    "inference_time_ms": 0.0,
                    "file_path": str(img_path.resolve()),
                    "error": str(exc),
                })
        return results

    @property
    def model(self) -> keras.Model:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model


def main() -> None:
    setup_logging(log_dir="logs", log_file="predict.log")
    import argparse

    parser = argparse.ArgumentParser(description="Predict content class from image(s).")
    parser.add_argument("--model_path", default="models/final_model.keras")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to single image file.")
    group.add_argument("--folder", help="Path to image folder.")

    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    predictor = ImagePredictor(model_path=args.model_path)

    if args.image:
        result = predictor.predict_image(args.image)
        # Omit input_tensors for readable JSON print
        print_res = {k: v for k, v in result.items() if k != "input_tensors"}
        print("\n=== Prediction Result ===")
        print(json.dumps(print_res, indent=2))
    elif args.folder:
        results = predictor.predict_folder(args.folder, recursive=args.recursive)
        clean_results = [{k: v for k, v in r.items() if k != "input_tensors"} for r in results]
        print(f"\n=== Batch Results ({len(clean_results)} images) ===")
        for r in clean_results:
            print(f"  {Path(r['file_path']).name:40s} → {r['prediction']:10s} ({r['confidence']*100:.2f}%) [{r['pipeline']}]")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(clean_results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
