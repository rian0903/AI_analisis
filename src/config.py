"""
config.py
=========
Centralized configuration module for the Image-Based Inappropriate Content Detection system.
Contains model hyperparameters, detector thresholds, dataset paths, and pipeline settings.
"""

from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = BASE_DIR / "dataset_large"
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Model Architecture & Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
NUM_CLASSES: int = 3
CLASS_NAMES: List[str] = ["Safe", "Violence", "NSFW"]

# Feature dimensions
GLOBAL_FEATURE_DIM: int = 1280
HUMAN_FEATURE_DIM: int = 1280
POSE_LANDMARK_POINTS: int = 12  # Key upper/lower body landmarks
POSE_FEATURE_DIM: int = POSE_LANDMARK_POINTS * 3  # (x, y, visibility)
MAX_PERSONS: int = 3
RELATION_FEATURE_DIM: int = MAX_PERSONS * (MAX_PERSONS - 1) * 3  # relative distance, overlap, dx
TOTAL_POSE_REL_DIM: int = POSE_FEATURE_DIM + RELATION_FEATURE_DIM

# Training
BATCH_SIZE: int = 32
EPOCHS: int = 12
FINE_TUNE_EPOCHS: int = 10
BASE_LEARNING_RATE: float = 1e-4
FINE_TUNE_LEARNING_RATE: float = 1e-5
FINE_TUNE_LAYERS: int = 30
DROPOUT_RATE_1: float = 0.4
DROPOUT_RATE_2: float = 0.3
SEED: int = 42

# ---------------------------------------------------------------------------
# Detector & Human Crop Configuration
# ---------------------------------------------------------------------------
PERSON_CONFIDENCE_THRESHOLD: float = 0.40
POSE_CONFIDENCE_THRESHOLD: float = 0.40
CROP_PADDING: float = 0.15  # 15% padding around detected human bounding box
LOW_CONFIDENCE_THRESHOLD: float = 0.60
