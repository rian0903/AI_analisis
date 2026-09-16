"""
dataset.py
==========
Dataset loading, preprocessing, feature extraction, and augmentation pipeline for the
Multi-Feature Image Content Detection system.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    TOTAL_POSE_REL_DIM,
)
from src.human_analyzer import HumanAnalyzer

logger = logging.getLogger(__name__)


def build_augmentation_layer(seed: int = SEED) -> keras.Sequential:
    """
    Build a Keras Sequential augmentation pipeline applied to training images.

    Augmentations Applied
    ---------------------
    - RandomRotation    : ±10% rotation
    - RandomFlip        : horizontal and vertical
    - RandomZoom        : ±15% zoom
    - RandomContrast    : 20% contrast jitter
    - RandomBrightness  : 20% brightness jitter
    - RandomHue/Sat     : Color variations to prevent color shortcut bias
    - RandomGrayscale   : 20% chance to convert to grayscale to force shape learning
    """
    return keras.Sequential(
        [
            keras.layers.RandomRotation(factor=0.1, seed=seed),
            keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed),
            keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, seed=seed),
            keras.layers.RandomContrast(factor=0.2, seed=seed),
            keras.layers.RandomBrightness(factor=0.2, seed=seed),
            keras.layers.Lambda(
                lambda x: tf.image.random_hue(x, max_delta=0.08, seed=seed),
                name="random_hue",
            ),
            keras.layers.Lambda(
                lambda x: tf.image.random_saturation(x, lower=0.6, upper=1.4, seed=seed),
                name="random_saturation",
            ),
            keras.layers.Lambda(
                lambda x: tf.cond(
                    tf.random.uniform([], seed=seed) < 0.2,
                    lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x)),
                    lambda: x,
                ),
                name="random_grayscale",
            ),
        ],
        name="data_augmentation",
    )


class DatasetLoader:
    """
    Dataset loader producing multi-input tuples:
    ({"input_global": ..., "input_human": ..., "input_pose": ..., "has_human": ..., "has_pose": ...}, y)
    """

    def __init__(
        self,
        dataset_dir: str = "dataset_large",
        image_size: Tuple[int, int] = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        seed: int = SEED,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        self._train_labels: Optional[np.ndarray] = None
        self._class_names: List[str] = CLASS_NAMES
        self.human_analyzer = HumanAnalyzer(image_size=self.image_size)

        self._validate_directory_structure()
        logger.info(
            "DatasetLoader initialised | dir=%s | image_size=%s | batch=%d",
            self.dataset_dir,
            self.image_size,
            self.batch_size,
        )

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def _validate_directory_structure(self) -> None:
        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            split_path = self.dataset_dir / split
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Expected split directory not found: {split_path}\n"
                    "Please organize your dataset as:\n"
                    "  dataset/{train,validation,test}/{Safe,Violence,NSFW}/"
                )

    def _process_image_path(
        self, file_path_bytes: bytes, label_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Py-function worker loading image and performing human & pose feature analysis.
        """
        if hasattr(file_path_bytes, "numpy"):
            file_path_bytes = file_path_bytes.numpy()
        if isinstance(file_path_bytes, (bytes, np.bytes_)):
            path_str = file_path_bytes.decode("utf-8")
        elif isinstance(file_path_bytes, np.ndarray):
            item = file_path_bytes.item()
            path_str = item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        else:
            path_str = str(file_path_bytes)

        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            # Fallback for unreadable file
            img_rgb = np.zeros((*self.image_size, 3), dtype=np.float32)
            analysis = {
                "has_human": 0.0,
                "has_pose": 0.0,
                "human_crop": np.zeros((*self.image_size, 3), dtype=np.float32),
                "pose_rel_vector": np.zeros((TOTAL_POSE_REL_DIM,), dtype=np.float32),
            }
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (self.image_size[1], self.image_size[0]))
            img_rgb = img_rgb.astype(np.float32) / 255.0

            analysis = self.human_analyzer.analyze(img_rgb)

        one_hot = np.zeros((self.num_classes,), dtype=np.float32)
        one_hot[label_idx] = 1.0

        return (
            img_rgb,
            analysis["human_crop"],
            analysis["pose_rel_vector"],
            np.array([analysis["has_human"]], dtype=np.float32),
            np.array([analysis["has_pose"]], dtype=np.float32),
            one_hot,
        )

    def _load_split(
        self,
        split: str,
        shuffle: bool = False,
        augment: bool = False,
    ) -> tf.data.Dataset:
        """
        Load a single dataset split into a multi-input tf.data.Dataset.
        """
        split_dir = self.dataset_dir / split
        file_paths: List[str] = []
        labels: List[int] = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls_idx, cls_name in enumerate(self._class_names):
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        file_paths.append(str(p.resolve()))
                        labels.append(cls_idx)

        if split == "train":
            self._train_labels = np.array(labels, dtype=np.int32)

        file_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if shuffle:
            file_ds = file_ds.shuffle(buffer_size=len(file_paths), seed=self.seed)

        def map_fn(path, lbl):
            out = tf.py_function(
                func=self._process_image_path,
                inp=[path, lbl],
                Tout=[
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                ],
            )
            g, h, p, hh, hp, y = out
            g.set_shape((*self.image_size, 3))
            h.set_shape((*self.image_size, 3))
            p.set_shape((TOTAL_POSE_REL_DIM,))
            hh.set_shape((1,))
            hp.set_shape((1,))
            y.set_shape((self.num_classes,))

            inputs_dict = {
                "input_global": g,
                "input_human": h,
                "input_pose": p,
                "has_human": hh,
                "has_pose": hp,
            }
            return inputs_dict, y

        AUTOTUNE = tf.data.AUTOTUNE
        ds = file_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

        if augment:
            aug_layer = build_augmentation_layer(self.seed)

            def aug_fn(inputs, y):
                inputs["input_global"] = aug_layer(inputs["input_global"], training=True)
                inputs["input_human"] = aug_layer(inputs["input_human"], training=True)
                return inputs, y

            ds = ds.map(aug_fn, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        logger.info("Loaded split='%s' with %d samples.", split, len(file_paths))
        return ds

    def get_datasets(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Build and return train, validation, and test datasets.
        """
        train_ds = self._load_split("train", shuffle=True, augment=True)
        val_ds = self._load_split("validation", shuffle=False, augment=False)
        test_ds = self._load_split("test", shuffle=False, augment=False)
        return train_ds, val_ds, test_ds

    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute balanced class weights.
        """
        if self._train_labels is None:
            raise RuntimeError("Call get_datasets() first to populate labels.")

        unique_classes = np.unique(self._train_labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=self._train_labels,
        )
        class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
        logger.info("Computed class weights: %s", class_weight_dict)
        return class_weight_dict

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Count samples per class in training set.
        """
        if self._train_labels is None:
            raise RuntimeError("Call get_datasets() first.")

        distribution: Dict[str, int] = {}
        for idx, name in enumerate(self._class_names):
            distribution[name] = int(np.sum(self._train_labels == idx))

        logger.info("Class distribution: %s", distribution)
        return distribution
