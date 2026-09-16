r"""
model.py
========
MobileNetV2-based multi-feature content moderation model definition and factory utilities.

Architecture Overview
---------------------
               INPUT IMAGE
                    |
        +-----------+-----------+
        |                       |
        v                       v
 Global MobileNetV2       Person Detector & Crop
        |                       |
        |                Person Detected?
        |                  /          \
        |                YES           NO
        |                 |             |
        |                 v             |
        |            Human Crop         |
        |             /      \          |
        |            v        v         |
        |     MobileNetV2    Pose       |
        |     Human Feature  Estimator  |
        |            |        |         |
        |            +----+---+         |
        |                 |             |
        +-----------------+-------------+
                          |
                          v
                   FEATURE FUSION (2,626-dim)
                          |
                          v
                 Classification Head
                (Dense+BN+ReLU+Dropout)
                          |
                          v
               Safe / Violence / NSFW (Softmax)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    BASE_LEARNING_RATE,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
    FINE_TUNE_LAYERS,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    TOTAL_POSE_REL_DIM,
)

logger = logging.getLogger(__name__)


class ContentModerationModel:
    """
    Multi-feature fusion image content moderation model built on MobileNetV2.

    Combines:
    1. Global image MobileNetV2 features (1280-dim)
    2. Human appearance MobileNetV2 crop features (1280-dim)
    3. Normalized pose & multi-person relationship features (64-dim)
    4. Gating flags (has_human, has_pose)

    Produces mutually-exclusive Softmax probabilities across 3 classes:
    [Safe, Violence, NSFW].
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        pose_rel_dim: int = TOTAL_POSE_REL_DIM,
        dropout_1: float = DROPOUT_RATE_1,
        dropout_2: float = DROPOUT_RATE_2,
    ) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.pose_rel_dim = pose_rel_dim
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        self._model: Optional[keras.Model] = None
        self._base_model: Optional[keras.Model] = None

    def build_model(self) -> keras.Model:
        """
        Construct the Keras multi-input functional model with MobileNetV2 backbone.
        """
        logger.info("Building multi-feature MobileNetV2 model (input_shape=%s)...", self.input_shape)

        # ----------------------------------------------------------
        # 1. Shared MobileNetV2 backbone
        # ----------------------------------------------------------
        self._base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        self._base_model.trainable = False

        # ----------------------------------------------------------
        # 2. Define Model Inputs
        # ----------------------------------------------------------
        input_global = keras.Input(shape=self.input_shape, name="input_global")
        input_human = keras.Input(shape=self.input_shape, name="input_human")
        input_pose = keras.Input(shape=(self.pose_rel_dim,), name="input_pose")
        has_human = keras.Input(shape=(1,), name="has_human")
        has_pose = keras.Input(shape=(1,), name="has_pose")

        # ----------------------------------------------------------
        # 3. Branch Feature Extractions
        # ----------------------------------------------------------
        # Global Image Branch
        global_features = self._base_model(input_global, training=False)
        global_features = keras.layers.GlobalAveragePooling2D(name="gap_global")(global_features)

        # Human Appearance Branch (gated by has_human)
        human_features_raw = self._base_model(input_human, training=False)
        human_features_raw = keras.layers.GlobalAveragePooling2D(name="gap_human")(human_features_raw)
        human_features = keras.layers.Multiply(name="gated_human")(
            [human_features_raw, has_human]
        )

        # Pose & Relationship Branch (gated by has_pose)
        pose_dense = keras.layers.Dense(64, activation="relu", name="pose_dense_64")(input_pose)
        pose_dense = keras.layers.Dropout(0.2, name="pose_dropout")(pose_dense)
        pose_features = keras.layers.Multiply(name="gated_pose")([pose_dense, has_pose])

        # ----------------------------------------------------------
        # 4. Feature Fusion
        # ----------------------------------------------------------
        fused_features = keras.layers.Concatenate(name="feature_fusion")(
            [global_features, human_features, pose_features, has_human, has_pose]
        )

        # ----------------------------------------------------------
        # 5. Classification Head
        # ----------------------------------------------------------
        x = keras.layers.Dense(
            512, kernel_initializer="he_normal", name="dense_512"
        )(fused_features)
        x = keras.layers.BatchNormalization(name="bn_1")(x)
        x = keras.layers.Activation("relu", name="relu_1")(x)
        x = keras.layers.Dropout(self.dropout_1, name="dropout_1")(x)

        x = keras.layers.Dense(
            256, kernel_initializer="he_normal", name="dense_256"
        )(x)
        x = keras.layers.Activation("relu", name="relu_2")(x)
        x = keras.layers.Dropout(self.dropout_2, name="dropout_2")(x)

        outputs = keras.layers.Dense(
            self.num_classes,
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="output_softmax",
        )(x)

        # ----------------------------------------------------------
        # 6. Assemble Keras Model
        # ----------------------------------------------------------
        self._model = keras.Model(
            inputs=[input_global, input_human, input_pose, has_human, has_pose],
            outputs=outputs,
            name="MultiFeatureContentModerator",
        )

        logger.info(
            "Model built successfully | total_params=%s | trainable_params=%s",
            f"{self._model.count_params():,}",
            f"{sum(tf.size(v).numpy() for v in self._model.trainable_variables):,}",
        )
        return self._model

    def compile_model(
        self,
        learning_rate: float = BASE_LEARNING_RATE,
    ) -> None:
        """
        Compile model with CategoricalCrossentropy loss and Adam optimizer.
        """
        if self._model is None:
            raise RuntimeError("Call build_model() before compile_model().")

        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        logger.info("Model compiled with lr=%.2e", learning_rate)

    def freeze_base(self) -> None:
        """Freeze all layers in the MobileNetV2 backbone for Phase 1."""
        if self._base_model is None:
            raise RuntimeError("Call build_model() first.")
        self._base_model.trainable = False
        logger.info("Base model frozen (all layers non-trainable).")

    def unfreeze_last_n_layers(self, n: int = FINE_TUNE_LAYERS) -> None:
        """
        Unfreeze last n layers of MobileNetV2 backbone for Phase 2 fine-tuning.
        BatchNorm layers remain frozen.
        """
        if self._base_model is None:
            raise RuntimeError("Call build_model() first.")

        self._base_model.trainable = True
        total = len(self._base_model.layers)
        freeze_up_to = total - n

        for layer in self._base_model.layers[:freeze_up_to]:
            layer.trainable = False

        # Keep BatchNorm layers frozen
        for layer in self._base_model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        n_trainable = sum(1 for l in self._base_model.layers if l.trainable)
        logger.info(
            "Fine-tuning: unfrozen last %d layers | trainable_layers_in_base=%d/%d",
            n,
            n_trainable,
            total,
        )

    def get_model(self) -> keras.Model:
        """Return the compiled Keras model."""
        if self._model is None:
            raise RuntimeError("Call build_model() first.")
        return self._model

    def get_base_model(self) -> keras.Model:
        """Return the MobileNetV2 backbone."""
        if self._base_model is None:
            raise RuntimeError("Call build_model() first.")
        return self._base_model

    def summary(self) -> None:
        """Print model summary."""
        if self._model is None:
            raise RuntimeError("Call build_model() first.")
        self._model.summary()
