"""
train.py
========
Training pipeline for the Multi-Feature MobileNetV2 Content Moderation System.

Phase 1 — Transfer Learning
    The MobileNetV2 backbone is frozen. Only the multi-feature fusion head
    is trained. EarlyStopping monitors validation loss (patience=7).

Phase 2 — Fine-Tuning
    The last 30 layers of the MobileNetV2 backbone are unfrozen and
    re-trained at a lower learning rate (1e-5). EarlyStopping (patience=5).

Outputs
-------
- models/best_model_phase1.keras
- models/best_model_phase2.keras
- models/final_model.keras
- models/training_history.csv
- models/phase_comparison.csv
- models/accuracy_curve.png
- models/loss_curve.png
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_LEARNING_RATE,
    BATCH_SIZE,
    DATASET_DIR,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAYERS,
    FINE_TUNE_LEARNING_RATE,
    IMAGE_SIZE,
    LOGS_DIR,
    MODELS_DIR,
    SEED,
)
from src.dataset import DatasetLoader
from src.model import ContentModerationModel
from src.utils import save_figure, setup_logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates the complete two-phase training pipeline.
    """

    def __init__(
        self,
        dataset_dir: str = str(DATASET_DIR),
        models_dir: str = str(MODELS_DIR),
        logs_dir: str = str(LOGS_DIR),
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        fine_tune_epochs: int = FINE_TUNE_EPOCHS,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        base_lr: float = BASE_LEARNING_RATE,
        fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
        fine_tune_layers: int = FINE_TUNE_LAYERS,
        seed: int = SEED,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.image_size = image_size
        self.base_lr = base_lr
        self.fine_tune_lr = fine_tune_lr
        self.fine_tune_layers = fine_tune_layers
        self.seed = seed

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._loader: Optional[DatasetLoader] = None
        self._content_model: Optional[ContentModerationModel] = None
        self._history_phase1: Optional[keras.callbacks.History] = None
        self._history_phase2: Optional[keras.callbacks.History] = None

    def _build_callbacks(
        self,
        phase: int,
        checkpoint_name: str,
        patience_early: int = 7,
        patience_lr: int = 3,
    ) -> list:
        checkpoint_path = str(self.models_dir / checkpoint_name)
        tensorboard_dir = str(self.logs_dir / "fit" / f"phase{phase}")
        csv_logger_path = str(self.models_dir / f"history_phase{phase}.csv")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience_early,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience_lr,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            ),
            keras.callbacks.CSVLogger(
                filename=csv_logger_path,
                append=False,
            ),
        ]
        return callbacks

    def _phase1_train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: Dict[int, float],
    ) -> keras.callbacks.History:
        logger.info("=" * 60)
        logger.info("PHASE 1: Transfer Learning (frozen backbone)")
        logger.info("  LR=%.2e | Epochs=%d", self.base_lr, self.epochs)
        logger.info("=" * 60)

        callbacks = self._build_callbacks(
            phase=1,
            checkpoint_name="best_model_phase1.keras",
            patience_early=7,
            patience_lr=3,
        )

        history = self._content_model.get_model().fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        logger.info(
            "Phase 1 complete | best_val_acc=%.4f",
            max(history.history.get("val_accuracy", [0])),
        )
        return history

    def _phase2_finetune(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: Dict[int, float],
    ) -> keras.callbacks.History:
        logger.info("=" * 60)
        logger.info(
            "PHASE 2: Fine-Tuning (last %d backbone layers unfrozen)",
            self.fine_tune_layers,
        )
        logger.info("  LR=%.2e | Epochs=%d", self.fine_tune_lr, self.fine_tune_epochs)
        logger.info("=" * 60)

        self._content_model.unfreeze_last_n_layers(n=self.fine_tune_layers)
        self._content_model.compile_model(learning_rate=self.fine_tune_lr)

        callbacks = self._build_callbacks(
            phase=2,
            checkpoint_name="best_model_phase2.keras",
            patience_early=5,
            patience_lr=2,
        )

        history = self._content_model.get_model().fit(
            train_ds,
            epochs=self.fine_tune_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        logger.info(
            "Phase 2 complete | best_val_acc=%.4f",
            max(history.history.get("val_accuracy", [0])),
        )
        return history

    def _save_combined_history(
        self,
        history1: keras.callbacks.History,
        history2: keras.callbacks.History,
    ) -> None:
        records = []
        for epoch, metrics in enumerate(zip(*history1.history.values()), start=1):
            row: Dict[str, Any] = {"epoch": epoch, "phase": 1}
            for key, val in zip(history1.history.keys(), metrics):
                row[key] = val
            records.append(row)

        offset = len(history1.epoch)
        for epoch, metrics in enumerate(zip(*history2.history.values()), start=1):
            row = {"epoch": offset + epoch, "phase": 2}
            for key, val in zip(history2.history.keys(), metrics):
                row[key] = val
            records.append(row)

        df = pd.DataFrame(records)
        csv_path = self.models_dir / "training_history.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Training history saved to %s", csv_path)

    def _plot_training_curves(
        self,
        history1: keras.callbacks.History,
        history2: keras.callbacks.History,
    ) -> None:
        phase1_epochs = len(history1.epoch)

        def concat(key: str) -> list:
            return history1.history.get(key, []) + history2.history.get(key, [])

        train_acc = concat("accuracy")
        val_acc = concat("val_accuracy")
        train_loss = concat("loss")
        val_loss = concat("val_loss")
        epochs_range = range(1, len(train_acc) + 1)

        style_cfg = {
            "train_color": "#3498db",
            "val_color": "#e74c3c",
            "phase_line": "#f39c12",
            "bg": "#1a1a2e",
            "grid": "#2c2c54",
            "text": "white",
        }

        # Accuracy
        fig_acc, ax = plt.subplots(figsize=(12, 5))
        fig_acc.patch.set_facecolor(style_cfg["bg"])
        ax.set_facecolor(style_cfg["bg"])

        ax.plot(epochs_range, train_acc, color=style_cfg["train_color"], linewidth=2, label="Train Accuracy", marker="o", markersize=3)
        ax.plot(epochs_range, val_acc, color=style_cfg["val_color"], linewidth=2, label="Val Accuracy", marker="s", markersize=3, linestyle="--")
        ax.axvline(phase1_epochs, color=style_cfg["phase_line"], linestyle=":", linewidth=1.5, label="Fine-Tuning Start")

        ax.set_title("Training & Validation Accuracy", color=style_cfg["text"], fontsize=14)
        ax.set_xlabel("Epoch", color=style_cfg["text"])
        ax.set_ylabel("Accuracy", color=style_cfg["text"])
        ax.tick_params(colors=style_cfg["text"])
        ax.legend(facecolor=style_cfg["bg"], labelcolor=style_cfg["text"])
        ax.grid(color=style_cfg["grid"], linestyle="--", alpha=0.5)

        save_figure(fig_acc, self.models_dir / "accuracy_curve.png")

        # Loss
        fig_loss, ax = plt.subplots(figsize=(12, 5))
        fig_loss.patch.set_facecolor(style_cfg["bg"])
        ax.set_facecolor(style_cfg["bg"])

        ax.plot(epochs_range, train_loss, color=style_cfg["train_color"], linewidth=2, label="Train Loss", marker="o", markersize=3)
        ax.plot(epochs_range, val_loss, color=style_cfg["val_color"], linewidth=2, label="Val Loss", marker="s", markersize=3, linestyle="--")
        ax.axvline(phase1_epochs, color=style_cfg["phase_line"], linestyle=":", linewidth=1.5, label="Fine-Tuning Start")

        ax.set_title("Training & Validation Loss", color=style_cfg["text"], fontsize=14)
        ax.set_xlabel("Epoch", color=style_cfg["text"])
        ax.set_ylabel("Loss", color=style_cfg["text"])
        ax.tick_params(colors=style_cfg["text"])
        ax.legend(facecolor=style_cfg["bg"], labelcolor=style_cfg["text"])
        ax.grid(color=style_cfg["grid"], linestyle="--", alpha=0.5)

        save_figure(fig_loss, self.models_dir / "loss_curve.png")

    def _compare_phases(
        self,
        history1: keras.callbacks.History,
        history2: keras.callbacks.History,
    ) -> Dict[str, Dict[str, float]]:
        def best(history: keras.callbacks.History, metric: str) -> float:
            vals = history.history.get(metric, [0.0])
            if "loss" in metric:
                return float(min(vals))
            return float(max(vals))

        metrics = ["val_accuracy", "val_loss", "val_precision", "val_recall", "val_auc"]
        comparison = {
            "Phase 1 (Transfer Learning)": {m: best(history1, m) for m in metrics},
            "Phase 2 (Fine-Tuning)": {m: best(history2, m) for m in metrics},
        }

        comp_df = pd.DataFrame(comparison).T
        comp_path = self.models_dir / "phase_comparison.csv"
        comp_df.to_csv(comp_path)
        logger.info("Phase comparison saved to %s", comp_path)
        return comparison

    def run(self) -> None:
        logger.info("Starting multi-feature training pipeline...")
        logger.info("TensorFlow version: %s", tf.__version__)

        self._loader = DatasetLoader(
            dataset_dir=str(self.dataset_dir),
            image_size=self.image_size,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        train_ds, val_ds, _test_ds = self._loader.get_datasets()
        class_weights = self._loader.compute_class_weights()
        # Apply safety-weighted penalty for NSFW and Violence to completely eliminate false negatives
        class_weights[1] = round(class_weights.get(1, 1.0) * 1.25, 4)
        class_weights[2] = round(class_weights.get(2, 1.0) * 1.45, 4)
        logger.info("Optimized safety class weights: %s", class_weights)
        logger.info("Class distribution: %s", self._loader.get_class_distribution())

        self._content_model = ContentModerationModel()
        self._content_model.build_model()
        self._content_model.freeze_base()
        self._content_model.compile_model(learning_rate=self.base_lr)
        self._content_model.summary()

        self._history_phase1 = self._phase1_train(train_ds, val_ds, class_weights)
        self._history_phase2 = self._phase2_finetune(train_ds, val_ds, class_weights)

        self._save_combined_history(self._history_phase1, self._history_phase2)
        self._plot_training_curves(self._history_phase1, self._history_phase2)
        self._compare_phases(self._history_phase1, self._history_phase2)

        final_path = self.models_dir / "final_model.keras"
        self._content_model.get_model().save(str(final_path))
        logger.info("Final model saved to %s", final_path)


def main() -> None:
    setup_logging(log_dir="logs", log_file="training.log")
    import argparse

    parser = argparse.ArgumentParser(description="Train the Multi-Feature Content Detection model.")
    parser.add_argument("--dataset_dir", default="dataset_large")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fine_tune_epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--base_lr", type=float, default=BASE_LEARNING_RATE)
    parser.add_argument("--fine_tune_lr", type=float, default=FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--fine_tune_layers", type=int, default=FINE_TUNE_LAYERS)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()
    trainer = Trainer(
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        base_lr=args.base_lr,
        fine_tune_lr=args.fine_tune_lr,
        fine_tune_layers=args.fine_tune_layers,
        seed=args.seed,
    )
    trainer.run()


if __name__ == "__main__":
    main()
