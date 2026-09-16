"""
utils.py
========
Shared utility functions and classes for the Multi-Feature Image Content Detection system.

Contents
--------
- setup_logging()         : Configure project-wide logging
- get_class_names()       : Return canonical class name list
- save_figure()           : Save matplotlib figures to disk
- preprocess_image()      : Load and preprocess a single image for inference
- GradCAM                 : Gradient-weighted Class Activation Mapping
- plot_gradcam()          : Visualise original image, heatmap, and overlay
- plot_sample_images()    : Display grid of sample images from a dataset split
- explain_activation()    : Semantic explanation generator based on learned features
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import CLASS_NAMES, IMAGE_SIZE, TOTAL_POSE_REL_DIM

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)

CLASS_COLORS: Dict[str, str] = {
    "Safe": "#2ecc71",       # green
    "Violence": "#e67e22",   # orange
    "NSFW": "#e74c3c",       # red
}


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "training.log",
    level: int = logging.INFO,
) -> None:
    """Configure project-wide logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    logger.info("Logging configured | log_file=%s", log_path)


def get_class_names() -> List[str]:
    return CLASS_NAMES.copy()


def get_class_color(class_name: str) -> str:
    return CLASS_COLORS.get(class_name, "#95a5a6")


def save_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 150,
    tight: bool = True,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if tight:
        fig.tight_layout()

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", save_path)


def preprocess_image(
    image_path: Union[str, Path],
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_pil_image(
    pil_image,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    img = pil_image.convert("RGB")
    img = img.resize((image_size[1], image_size[0]))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)


class GradCAM:
    """
    Grad-CAM for MobileNetV2 multi-input models.
    Target Conv2D layer in MobileNetV2 backbone: 'Conv_1'.
    """

    DEFAULT_CONV_LAYER = "Conv_1"

    def __init__(
        self,
        model: keras.Model,
        last_conv_layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self.DEFAULT_CONV_LAYER

        # Locate base MobileNetV2 model inside functional architecture
        self._base_model = None
        for layer in self.model.layers:
            if hasattr(layer, "layers") and isinstance(layer, keras.Model):
                self._base_model = layer
                break

        self._build_grad_model()

    def _build_grad_model(self) -> None:
        target_layer = None
        if self._base_model is not None:
            for layer in self._base_model.layers:
                if layer.name == self.last_conv_layer_name:
                    target_layer = layer
                    break

            if target_layer is None:
                for layer in reversed(self._base_model.layers):
                    if isinstance(layer, keras.layers.Conv2D):
                        target_layer = layer
                        self.last_conv_layer_name = layer.name
                        break

            self._base_sub_model = keras.Model(
                inputs=self._base_model.inputs,
                outputs=target_layer.output,
            )

            # Rebuild downstream classifier from Conv_1 to prediction output
            conv_in = keras.Input(shape=target_layer.output.shape[1:], name="grad_conv_in")
            human_in = keras.Input(shape=self.model.get_layer("input_human").output.shape[1:], name="grad_human_in")
            pose_in = keras.Input(shape=self.model.get_layer("input_pose").output.shape[1:], name="grad_pose_in")
            has_human_in = keras.Input(shape=(1,), name="grad_has_human_in")
            has_pose_in = keras.Input(shape=(1,), name="grad_has_pose_in")

            gap_g = self.model.get_layer("gap_global")(conv_in)
            human_raw = self._base_model(human_in, training=False)
            gap_h = self.model.get_layer("gap_human")(human_raw)
            gated_h = self.model.get_layer("gated_human")([gap_h, has_human_in])

            pose_dense = self.model.get_layer("pose_dense_64")(pose_in)
            pose_drop = self.model.get_layer("pose_dropout")(pose_dense, training=False)
            gated_p = self.model.get_layer("gated_pose")([pose_drop, has_pose_in])

            fused = self.model.get_layer("feature_fusion")([gap_g, gated_h, gated_p, has_human_in, has_pose_in])
            x = self.model.get_layer("dense_512")(fused)
            x = self.model.get_layer("bn_1")(x, training=False)
            x = self.model.get_layer("relu_1")(x)
            x = self.model.get_layer("dropout_1")(x, training=False)
            x = self.model.get_layer("dense_256")(x)
            x = self.model.get_layer("relu_2")(x)
            x = self.model.get_layer("dropout_2")(x, training=False)
            out_head = self.model.get_layer("output_softmax")(x)

            self._downstream_head = keras.Model(
                inputs=[conv_in, human_in, pose_in, has_human_in, has_pose_in],
                outputs=out_head,
            )
        else:
            for layer in self.model.layers:
                if layer.name == self.last_conv_layer_name:
                    target_layer = layer
                    break

            if target_layer is None:
                for layer in reversed(self.model.layers):
                    if isinstance(layer, keras.layers.Conv2D):
                        target_layer = layer
                        self.last_conv_layer_name = layer.name
                        break

            self._flat_grad_model = keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output],
            )
            self._base_sub_model = None
            self._downstream_head = None

    def compute_heatmap(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for global image input tensor."""
        if isinstance(inputs, dict):
            global_img = tf.cast(inputs["input_global"], tf.float32)
            input_dict = {
                "input_global": global_img,
                "input_human": tf.cast(inputs["input_human"], tf.float32),
                "input_pose": tf.cast(inputs["input_pose"], tf.float32),
                "has_human": tf.cast(inputs["has_human"], tf.float32),
                "has_pose": tf.cast(inputs["has_pose"], tf.float32),
            }
        else:
            global_img = tf.cast(inputs, tf.float32)
            if global_img.ndim == 3:
                global_img = tf.expand_dims(global_img, axis=0)
            empty_crop = tf.zeros_like(global_img)
            empty_pose = tf.zeros((global_img.shape[0], self.model.get_layer("input_pose").output.shape[1]), dtype=tf.float32)
            empty_flag = tf.zeros((global_img.shape[0], 1), dtype=tf.float32)

            input_dict = {
                "input_global": global_img,
                "input_human": empty_crop,
                "input_pose": empty_pose,
                "has_human": empty_flag,
                "has_pose": empty_flag,
            }

        with tf.GradientTape() as tape:
            if self._base_sub_model is not None and self._downstream_head is not None:
                conv_outputs = self._base_sub_model(input_dict["input_global"])
                tape.watch(conv_outputs)
                predictions = self._downstream_head([
                    conv_outputs,
                    input_dict["input_human"],
                    input_dict["input_pose"],
                    input_dict["has_human"],
                    input_dict["has_pose"],
                ])
            else:
                conv_outputs, predictions = self._flat_grad_model(input_dict, training=False)
                tape.watch(conv_outputs)

            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))

            class_score = predictions[:, class_index]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            # Fallback uniform activation if gradient is zero/None
            return np.ones((self.model.input_shape[0][1] // 32, self.model.input_shape[0][2] // 32), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap.numpy()

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap.astype(np.float32)

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colorised = cv2.applyColorMap(heatmap_uint8, colormap)
        colorised = cv2.cvtColor(colorised, cv2.COLOR_BGR2RGB)

        if original_image.dtype == np.float32:
            orig_uint8 = np.uint8(original_image * 255)
        else:
            orig_uint8 = original_image.copy()

        overlay = cv2.addWeighted(orig_uint8, 1 - alpha, colorised, alpha, 0)
        return colorised, overlay


def plot_gradcam(
    model: keras.Model,
    inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    class_index: Optional[int] = None,
    class_name: str = "Unknown",
    confidence: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    last_conv_layer: Optional[str] = None,
) -> plt.Figure:
    grad_cam = GradCAM(model, last_conv_layer_name=last_conv_layer)
    heatmap = grad_cam.compute_heatmap(inputs, class_index=class_index)

    if isinstance(inputs, dict):
        orig = np.squeeze(inputs["input_global"])
    else:
        orig = np.squeeze(inputs)

    orig_uint8 = np.uint8(orig * 255)
    colorised, overlay = grad_cam.overlay_heatmap(heatmap, orig_uint8)

    color = get_class_color(class_name)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (orig_uint8, "Original Image"),
        (colorised, "Grad-CAM Heatmap"),
        (overlay, "Overlay"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=13, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(
        f"Grad-CAM — Predicted: {class_name} ({confidence * 100:.1f}%)",
        color=color,
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    if save_path:
        save_figure(fig, save_path, tight=True)
    else:
        plt.tight_layout()

    return fig


def explain_activation(
    class_name: str,
    probability: float,
    human_analysis: Dict[str, Any],
) -> str:
    """
    Generate semantic explanation based on multi-feature model analysis.
    Uses human detection, pose structure, and global features without relying
    on hardcoded color heuristics.
    """
    person_detected = human_analysis.get("person_detected", False)
    person_count = human_analysis.get("person_count", 0)
    pose_detected = human_analysis.get("pose_detected", False)
    pipeline = human_analysis.get("pipeline", "global_fallback")

    if class_name == "Violence":
        if person_detected and person_count > 1:
            explanation = (
                f"Model menganalisis hubungan spasial dan struktur pose dari {person_count} orang. "
                f"Kombinasi interaksi fisik, jarak antar subjek, dan konteks visual mengindikasikan potensi perkelahian atau kekerasan."
            )
        elif person_detected:
            explanation = (
                f"Model mendeteksi manusia dan mengekstrak struktur gestur tubuh. "
                f"Vektor fitur penampilan manusia dan konteks global menunjukkan indikator kekerasan fisik atau bahaya terdeteksi."
            )
        else:
            explanation = (
                f"Model mendeteksi elemen objek sensitif atau adegan konteks global yang diasosiasikan dengan kekerasan "
                f"melalui ekstraksi fitur MobileNetV2 (analisis konteks tanpa bergantung pada aturan warna sederhana)."
            )

    elif class_name == "NSFW":
        if person_detected and pose_detected:
            explanation = (
                f"Model menganalisis siluet dan pose tubuh manusia terdeteksi. "
                f"Ekstraksi fitur penampilan area tubuh dan struktur pose tubuh mengindikasikan ketelanjangan atau konten bermuatan seksual."
            )
        elif person_detected:
            explanation = (
                f"Model mendeteksi subjek manusia dengan fitur penampilan tubuh yang kuat, "
                f"menunjukkan indikasi pakaian minim atau bentuk kontur sensitif."
            )
        else:
            explanation = (
                f"Model menganalisis pola visual global dari adegan yang diasosiasikan dengan konten dewasa/sensitif."
            )

    else:  # Safe
        if person_detected:
            explanation = (
                f"Model mendeteksi {person_count} subjek manusia dengan pose dan gestur tubuh netral. "
                f"Fitur penampilan tubuh dan konteks adegan menunjukkan aktivitas sehari-hari yang aman."
            )
        else:
            explanation = (
                f"Model menganalisis visual adegan global dan mengonfirmasi konten bersih dari indikator kekerasan maupun konten eksplisit."
            )

    return explanation
