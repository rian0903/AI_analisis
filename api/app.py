"""
app.py
======
Flask REST API for the Multi-Feature Image Content Detection System.

Endpoints
---------
GET  /health
GET  /classes
GET  /model_info
POST /predict
"""

import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES
from src.predict import ImagePredictor
from src.utils import setup_logging

MODEL_PATH = os.environ.get("MODEL_PATH", "models/final_model.keras")
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", 5000))
DEBUG = os.environ.get("API_DEBUG", "false").lower() == "true"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "gif"}

setup_logging(log_dir="logs", log_file="api.log")
logger = logging.getLogger(__name__)


def create_app(model_path: str = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    CORS(app, resources={r"/*": {"origins": "*"}})

    predictor_holder: Dict[str, Any] = {}

    def get_predictor() -> Optional[ImagePredictor]:
        if "instance" not in predictor_holder:
            try:
                predictor_holder["instance"] = ImagePredictor(model_path=model_path)
                logger.info("Predictor initialised from %s", model_path)
            except FileNotFoundError as exc:
                logger.error("Model not found: %s", exc)
                predictor_holder["instance"] = None
        return predictor_holder.get("instance")

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def make_error(message: str, status_code: int = 400) -> Response:
        logger.warning("API error [%d]: %s", status_code, message)
        return jsonify({"error": message}), status_code

    @app.get("/health")
    def health_check():
        p = get_predictor()
        return jsonify({
            "status": "ok",
            "model_loaded": p is not None,
            "model_path": model_path,
            "classes": CLASS_NAMES,
            "version": "2.0.0-multifeature",
        })

    @app.get("/classes")
    def get_classes():
        return jsonify({
            "classes": CLASS_NAMES,
            "descriptions": {
                "Safe": "Normal daily activities — no explicit or violent content.",
                "Violence": "Visual violence: blood, weapons, assault, physical conflict.",
                "NSFW": "Adult content: nudity, explicit sexual content.",
            },
        })

    @app.get("/model_info")
    def model_info():
        p = get_predictor()
        if p is None:
            return make_error("Model not loaded.", 503)

        try:
            model = p.model
            return jsonify({
                "model_path": str(p.model_path),
                "input_shapes": [str(i.shape) for i in model.inputs],
                "output_shape": str(model.output_shape),
                "total_params": model.count_params(),
                "classes": CLASS_NAMES,
            })
        except Exception as exc:
            return make_error(f"Could not retrieve model info: {exc}", 500)

    @app.post("/predict")
    def predict():
        start_time = time.perf_counter()

        p = get_predictor()
        if p is None:
            return make_error("Model is not available. Ensure model is trained.", 503)

        if "file" not in request.files:
            return make_error("No file field found in request. Use field name 'file'.")

        file = request.files["file"]
        if file.filename == "":
            return make_error("No file selected.")

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return make_error(f"File type not allowed. Supported: {ALLOWED_EXTENSIONS}")

        try:
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as exc:
            return make_error(f"Failed to decode image: {exc}")

        try:
            result = p.predict_array(img_np, source_name=filename)
        except Exception as exc:
            logger.exception("Prediction failed for %s", filename)
            return make_error(f"Prediction failed: {exc}", 500)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        response_payload = {
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100.0, 2),  # percentage 0-100 for API contract
            "probabilities": result["probabilities"],
            "human_analysis": result["human_analysis"],
            "pipeline": result["pipeline"],
            "low_confidence": result["low_confidence"],
            "explanations": result.get("explanations", {}),
            "explanation_summary": result.get("explanation_summary", ""),
            "filename": filename,
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.info("POST /predict | file=%s | class=%s | %.1fms", filename, result["prediction"], elapsed_ms)
        return jsonify(response_payload)

    @app.errorhandler(413)
    def request_too_large(error):
        return make_error(f"File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)} MB.", 413)

    return app


if __name__ == "__main__":
    app = create_app(model_path=MODEL_PATH)
    logger.info("Starting Flask API on http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG)
