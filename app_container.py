"""
app_container.py
================
Desktop Application Container for Multi-Feature Content Moderation AI System.
Wraps the modern HTML5/JS interface inside a native desktop application container (PyWebView / Edge WebView2)
and directly loads the latest trained deep learning model (final_model.keras).
"""

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import CLASS_NAMES, MODELS_DIR, IMAGE_SIZE
from src.predict import ImagePredictor
from src.utils import GradCAM, setup_logging

setup_logging(log_dir="logs", log_file="container_app.log")
logger = logging.getLogger("AppContainer")


def find_latest_model() -> Path:
    """Find the newest or best model file in models/ directory."""
    models_dir = BASE_DIR / "models"
    priority_names = ["final_model.keras", "best_model_phase2.keras", "best_model_phase1.keras"]

    for name in priority_names:
        p = models_dir / name
        if p.exists():
            return p

    # Fallback to any .keras file
    keras_files = list(models_dir.glob("*.keras"))
    if keras_files:
        # Sort by last modified
        keras_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return keras_files[0]

    return models_dir / "final_model.keras"


class ContainerApi:
    """
    Python Bridge API exposed to JavaScript in the Desktop Container.
    """

    def __init__(self, model_path: Optional[Path] = None, window: Optional[Any] = None) -> None:
        self._window = window
        self.current_model_path = model_path or find_latest_model()
        self.predictor: Optional[ImagePredictor] = None
        self.grad_cam: Optional[GradCAM] = None
        self._init_predictor()

    def set_window(self, window: Any) -> None:
        self._window = window

    def _init_predictor(self) -> None:
        try:
            logger.info("Initializing ImagePredictor from: %s", self.current_model_path)
            self.predictor = ImagePredictor(model_path=str(self.current_model_path))
            self.grad_cam = GradCAM(self.predictor.model)
            logger.info("Predictor and GradCAM initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize predictor: %s", exc)
            self.predictor = None
            self.grad_cam = None

    def get_available_models(self) -> List[Dict[str, Any]]:
        """List all available .keras model files in the models directory."""
        models_dir = BASE_DIR / "models"
        results = []
        if models_dir.exists():
            for f in sorted(models_dir.glob("*.keras")):
                results.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "is_active": (f.name == self.current_model_path.name),
                })
        return results

    def load_model(self, filename: str) -> Dict[str, Any]:
        """Switch or reload active model."""
        target_path = BASE_DIR / "models" / filename
        if not target_path.exists():
            return {"success": False, "error": f"Model file '{filename}' not found."}

        try:
            self.current_model_path = target_path
            self._init_predictor()
            if self.predictor is not None:
                return {
                    "success": True,
                    "model_name": filename,
                    "model_path": str(target_path),
                }
            return {"success": False, "error": "Model initialization failed."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_model_info(self) -> Dict[str, Any]:
        """Retrieve model metadata, parameter count, and tensor shapes."""
        if self.predictor is None or self.predictor.model is None:
            return {
                "model_path": str(self.current_model_path),
                "loaded": False,
                "error": "Model not loaded.",
            }

        try:
            model = self.predictor.model
            return {
                "model_path": str(self.current_model_path.relative_to(BASE_DIR)),
                "loaded": True,
                "input_shapes": [str(i.shape) for i in model.inputs],
                "output_shape": str(model.output_shape),
                "total_params": int(model.count_params()),
                "classes": CLASS_NAMES,
            }
        except Exception as exc:
            return {"loaded": False, "error": str(exc)}

    def open_file_dialog(self) -> Optional[str]:
        """Open native Windows file dialog to select an image."""
        if self._window is None:
            return None
        import webview
        file_types = ("Image Files (*.jpg;*.jpeg;*.png;*.webp;*.bmp)", "All files (*.*)")
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types
        )
        if result and len(result) > 0:
            return result[0]
        return None

    def open_folder_dialog(self) -> Optional[str]:
        """Open native Windows folder selection dialog."""
        if self._window is None:
            return None
        import webview
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if result and len(result) > 0:
            return result[0]
        return None

    def get_file_base64(self, file_path: str) -> str:
        """Convert a local file path to base64 data URL for preview."""
        p = Path(file_path)
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = p.suffix.lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["jpeg", "png", "webp", "bmp"] else "image/jpeg"
        return f"data:{mime};base64,{data}"

    def get_sample_image(self, class_name: str) -> Dict[str, Any]:
        """Retrieve a sample image from dataset or test folder."""
        search_dirs = [
            BASE_DIR / "dataset_large" / "test" / class_name,
            BASE_DIR / "dataset" / "test" / class_name,
            BASE_DIR / "raw_data" / class_name,
        ]

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for s_dir in search_dirs:
            if s_dir.exists():
                imgs = [p for p in s_dir.iterdir() if p.suffix.lower() in valid_exts]
                if imgs:
                    chosen = imgs[0]
                    return {
                        "name": chosen.name,
                        "path": str(chosen),
                        "base64": self.get_file_base64(str(chosen)),
                    }

        return {"name": "", "path": "", "base64": ""}

    def predict_file(self, file_path: str) -> Dict[str, Any]:
        """Run prediction on a local image path."""
        if self.predictor is None:
            return {"error": "Model is not loaded."}

        try:
            res = self.predictor.predict_image(file_path)
            # Store last input tensors for Grad-CAM
            self._last_input_tensors = res.get("input_tensors")
            # Remove raw tensors for clean JS JSON serialization
            clean_res = {k: v for k, v in res.items() if k != "input_tensors"}
            return clean_res
        except Exception as exc:
            logger.exception("Prediction failed for file: %s", file_path)
            return {"error": str(exc)}

    def predict_base64(self, base64_str: str, filename: str = "upload.jpg") -> Dict[str, Any]:
        """Run prediction on base64 encoded image data."""
        if self.predictor is None:
            return {"error": "Model is not loaded."}

        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32) / 255.0

            res = self.predictor.predict_array(img_np, source_name=filename)
            self._last_input_tensors = res.get("input_tensors")
            clean_res = {k: v for k, v in res.items() if k != "input_tensors"}
            return clean_res
        except Exception as exc:
            logger.exception("Prediction failed for base64 image: %s", exc)
            return {"error": str(exc)}

    def compute_gradcam(self, file_path_or_base64: str, class_name: str) -> Dict[str, str]:
        """Compute Grad-CAM heatmap and overlay images as base64 strings."""
        if self.predictor is None or self.grad_cam is None:
            return {}

        try:
            class_idx = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else 0

            # If we have stored input tensors from previous prediction
            if hasattr(self, "_last_input_tensors") and self._last_input_tensors is not None:
                input_tensors = self._last_input_tensors
            else:
                return {}

            heatmap = self.grad_cam.compute_heatmap(input_tensors, class_index=class_idx)
            orig_uint8 = np.uint8(np.squeeze(input_tensors["input_global"]) * 255)
            colorised_heatmap, overlay = self.grad_cam.overlay_heatmap(heatmap, orig_uint8)

            def to_b64(img_rgb: np.ndarray) -> str:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".jpg", img_bgr)
                b64_str = base64.b64encode(buffer).decode("utf-8")
                return f"data:image/jpeg;base64,{b64_str}"

            return {
                "original": to_b64(orig_uint8),
                "heatmap": to_b64(colorised_heatmap),
                "overlay": to_b64(overlay),
            }
        except Exception as exc:
            logger.exception("GradCAM calculation error: %s", exc)
            return {}

    def predict_batch(self, folder_path: str) -> List[Dict[str, Any]]:
        """Run batch prediction on all images inside a directory."""
        if self.predictor is None:
            return [{"error": "Model is not loaded."}]

        try:
            results = self.predictor.predict_folder(folder_path, recursive=False)
            clean_results = [{k: v for k, v in r.items() if k != "input_tensors"} for r in results]
            return clean_results
        except Exception as exc:
            logger.exception("Batch prediction error: %s", exc)
            return [{"error": str(exc)}]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Feature AI Content Moderation Desktop Container")
    parser.add_argument("--model", default=None, help="Path to .keras model file")
    parser.add_argument("--server", action="store_true", help="Run local HTTP server instead of GUI container")
    parser.add_argument("--port", type=int, default=8080, help="Port for server mode")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_latest_model()
    logger.info("Starting Desktop App Container with Model: %s", model_path)

    api = ContainerApi(model_path=model_path)
    html_path = BASE_DIR / "gui" / "index.html"

    if args.server:
        # Fallback HTTP server
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        os.chdir(str(BASE_DIR / "gui"))
        server = HTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler)
        print(f"Container UI Server running at http://127.0.0.1:{args.port}")
        server.serve_forever()
    else:
        try:
            import webview
            window = webview.create_window(
                title="🛡️ Multi-Feature AI Content Moderation Container",
                url=str(html_path.resolve()),
                js_api=api,
                width=1240,
                height=840,
                min_size=(980, 680),
                background_color="#0a0e17",
                text_select=True,
                easy_drag=False,
            )
            api.set_window(window)
            webview.start(debug=False)
        except Exception as exc:
            logger.error("Failed to start PyWebView window: %s", exc)
            print(f"\n[Warning] Could not launch native WebView container: {exc}")
            print("Launching fallback CustomTkinter Desktop App...")
            from desktop_app import launch_desktop_gui
            launch_desktop_gui(model_path=str(model_path))


if __name__ == "__main__":
    main()
