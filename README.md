# 🛡️ Multi-Feature Image Content Detection & Moderation

> A production-grade computer vision system for automatic content moderation using **MobileNetV2** multi-feature fusion (Global Image + Human Appearance + Pose Landmarks + Spatial Relationships) and **Grad-CAM** visual explainability.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)

---

## 📋 Table of Contents

1. [Overview & Research Objectives](#overview--research-objectives)
2. [Multi-Feature Architecture](#multi-feature-architecture)
3. [Fallback Pipeline Routing](#fallback-pipeline-routing)
4. [Classification Classes](#classification-classes)
5. [Project Directory Layout](#project-directory-layout)
6. [Installation & Dataset Setup](#installation--dataset-setup)
7. [Configuration Parameters](#configuration-parameters)
8. [Training Strategy](#training-strategy)
9. [Evaluation & Bias Testing](#evaluation--bias-testing)
10. [Sanity Testing](#sanity-testing)
11. [Inference & REST API](#inference--rest-api)
12. [Streamlit Web Interface](#streamlit-web-interface)
13. [Research & Methodological Limitations](#research--methodological-limitations)

---

## Overview & Research Objectives

This project implements a multi-feature computer vision pipeline that automatically classifies images into three mutually-exclusive categories:
- **Safe** (0)
- **Violence** (1)
- **NSFW** (2)

### Shortcut Learning Mitigation
Single-branch CNN classifiers frequently suffer from visual shortcut learning—for example, incorrectly classifying non-violent images with prominent red visual features (e.g. red clothing, red backgrounds, red vehicles, red flowers, or red ambient lighting) as `Violence`.

To solve this without replacing MobileNetV2 as the primary CNN backbone, this system extracts and fuses multiple feature representations:
1. **Global Image Branch**: MobileNetV2 feature extractor analyzing full image visual context (1280-dim).
2. **Person Detector & Padded Crop**: Extracts human bounding boxes with 15% padding and processes human region appearance with MobileNetV2 (1280-dim).
3. **Pose Landmark Estimator**: Extracts normalized body keypoint landmarks (36-dim) to capture human body posture.
4. **Multi-Person Spatial Relationships**: Computes pairwise distances, IoU bounding box overlaps, and relative offsets for up to 3 detected persons (18-dim).
5. **Color-Jitter & Grayscale Augmentations**: Prevents reliance on superficial color statistics by forcing shape and semantic feature learning.

---

## Multi-Feature Architecture

```
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
```

---

## Fallback Pipeline Routing

The inference system supports 3 robust fallback pipelines to handle diverse real-world images:

- **Case A (`global+human+pose`)**: Person(s) detected and pose landmarks extracted successfully. Uses the complete multi-feature fusion vector.
- **Case B (`global+human`)**: Person detected, but pose landmark estimation is unavailable. Gated feature fusion zeroes the pose vector and utilizes global + human appearance features.
- **Case C (`global_fallback`)**: No human detected in the image. Gated feature fusion zeroes human and pose vectors and classifies based on global MobileNetV2 features.

---

## Classification Classes

| Class | Label | Description |
|-------|-------|-------------|
| **Safe** | 0 | Normal daily activities — landscapes, family, sports, pets, food, objects |
| **Violence** | 1 | Visual violence, assault, weapons, explosions, physical fight scenes |
| **NSFW** | 2 | Nudity, exposed private body parts, explicit adult content |

---

## Project Directory Layout

```
skripsi/
├── src/
│   ├── config.py             # Centralized configuration & thresholds
│   ├── human_analyzer.py     # Person detection, human crop, pose & relationship features
│   ├── model.py              # Multi-feature MobileNetV2 Keras model
│   ├── dataset.py            # Multi-input data pipeline & color augmentations
│   ├── train.py              # Two-phase transfer learning & fine-tuning
│   ├── evaluate.py           # Comprehensive evaluation & error analysis export
│   ├── evaluate_bias.py      # Dedicated red-color shortcut bias testing
│   ├── predict.py            # Multi-pipeline inference engine
│   └── utils.py              # Grad-CAM, semantic explanations, logging
│
├── api/
│   └── app.py                # Flask REST API
│
├── models/                   # Saved models, evaluation reports, and error analysis
│   ├── final_model.keras
│   ├── best_model_phase1.keras
│   ├── best_model_phase2.keras
│   ├── evaluation_metrics.xlsx
│   ├── bias_evaluation_report.json
│   └── evaluation/errors/    # Misclassified test images categorized by error type
│
├── dataset_large/            # Organized training dataset
│   ├── train/{Safe,Violence,NSFW}/
│   ├── validation/{Safe,Violence,NSFW}/
│   └── test/{Safe,Violence,NSFW}/
│
├── setup_dataset.py          # Dataset download and split script
├── install_dataset_deps.py   # Dataset dependencies installation
├── test_pipeline.py          # Automated sanity test script
├── streamlit_app.py          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Installation & Dataset Setup

### 1. Environment Activation & Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize images into `dataset_large/` as follows:

```
dataset_large/
├── train/
│   ├── Safe/
│   ├── Violence/
│   └── NSFW/
├── validation/
│   ├── Safe/
│   ├── Violence/
│   └── NSFW/
└── test/
    ├── Safe/
    ├── Violence/
    └── NSFW/
```

Or run the automated setup scripts:
```bash
python install_dataset_deps.py
python setup_dataset.py
```

---

## Configuration Parameters

All pipeline constants are central in `src/config.py`:

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
FINE_TUNE_EPOCHS = 10
BASE_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 30
PERSON_CONFIDENCE_THRESHOLD = 0.40
POSE_CONFIDENCE_THRESHOLD = 0.40
MAX_PERSONS = 3
CROP_PADDING = 0.15          # 15% padding around detected human bounding box
LOW_CONFIDENCE_THRESHOLD = 0.60
```

---

## Training Strategy

```bash
python src/train.py --dataset_dir dataset_large --batch_size 32 --epochs 12 --fine_tune_epochs 10
```

### Phase 1 — Transfer Learning (Frozen Backbone)
- MobileNetV2 backbone is frozen (`trainable = False`).
- Trains multi-feature fusion head (`base_lr = 1e-4`, Adam optimizer, Categorical Crossentropy loss with 0.05 label smoothing).
- Callbacks: ModelCheckpoint (`val_accuracy`), EarlyStopping (`val_loss`, patience=7), ReduceLROnPlateau, TensorBoard.

### Phase 2 — Fine-Tuning (Last 30 Backbone Layers)
- Unfreezes the last 30 layers of the MobileNetV2 backbone while keeping BatchNormalization layers frozen (`fine_tune_lr = 1e-5`, EarlyStopping patience=5).

---

## Evaluation & Bias Testing

### Standard Performance Evaluation

```bash
python src/evaluate.py --model_path models/final_model.keras --dataset_dir dataset_large
```

Outputs:
- Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Confusion Matrix Heatmap (`models/confusion_matrix.png` & CSV)
- One-vs-Rest ROC Curves (`models/roc_curves.png`)
- Multi-sheet Excel workbook (`models/evaluation_metrics.xlsx`)
- Misclassified test images saved into `models/evaluation/errors/{Safe_as_Violence, Safe_as_NSFW, Violence_as_Safe, Violence_as_NSFW, NSFW_as_Safe, NSFW_as_Violence}/` with complete `.json` metadata.

### Shortcut & Bias Evaluation

```bash
python src/evaluate_bias.py --model_path models/final_model.keras --dataset_dir dataset_large
```

Evaluates performance across specific hard-negative test categories:
- **Red Safe Images** (red clothing, red backgrounds, red vehicles, red flowers, sports)
- **Red NSFW Images** (red ambient lighting with adult content)
- **Red Violence Images**
- **Non-Violent Hard Negatives** (hugging, dancing, swimming, sports)

---

## Sanity Testing

Verify system installation and multi-input model execution:

```bash
python test_pipeline.py
```

Expected output:
```
Step 1: Testing imports...
Imports OK.

Step 2: Testing HumanAnalyzer...
No Person Pipeline: global_fallback
Person Detected Pipeline: global+human+pose
Human Count: 1

Step 3: Building & Compiling MultiFeatureContentModerationModel...
Model Inputs: ['input_global', 'input_human', 'input_pose', 'has_human', 'has_pose']
Model Output: output_softmax/Softmax:0
Total Parameters: 3,672,067
Dummy Prediction Output: [0.33333334 0.33333334 0.33333334]
Sum of Probabilities: 1.0

All pipeline sanity tests PASSED successfully!
```

---

## Inference & REST API

### Python Inference API

```python
from src.predict import ImagePredictor

predictor = ImagePredictor(model_path="models/final_model.keras")
result = predictor.predict_image("sample.jpg")

print(result["prediction"])           # "NSFW"
print(result["confidence"])           # 0.9415
print(result["probabilities"])        # {"Safe": 0.0211, "Violence": 0.0374, "NSFW": 0.9415}
print(result["pipeline"])             # "global+human+pose"
print(result["human_analysis"])       # {"person_detected": True, "person_count": 1, "pose_detected": True}
print(result["low_confidence"])       # False
print(result["inference_time_ms"])    # 42.5
```

### Flask REST API

Start the REST API server:
```bash
python api/app.py
```

#### REST Endpoints

1. `GET /health`: Health status & loaded model info.
2. `GET /classes`: List of supported classes.
3. `GET /model_info`: Model parameter count and input/output shapes.
4. `POST /predict`: Classify image uploaded via multipart form data (`file=<image>`).

#### Example `POST /predict` Response:

```json
{
  "prediction": "NSFW",
  "confidence": 94.15,
  "probabilities": {
    "Safe": 0.0211,
    "Violence": 0.0374,
    "NSFW": 0.9415
  },
  "human_analysis": {
    "person_detected": true,
    "person_count": 1,
    "pose_detected": true
  },
  "pipeline": "global+human+pose",
  "low_confidence": false,
  "explanations": {
    "NSFW": "Model mendeteksi subjek manusia dengan fitur penampilan tubuh..."
  },
  "explanation_summary": "Model mendeteksi subjek manusia...",
  "filename": "uploaded_image.jpg",
  "processing_time_ms": 42.5
}
```

---

## Streamlit Web Interface

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

Features:
- Drag-and-drop image uploader
- Real-time classification badge (`Safe` 🟢, `Violence` 🟠, `NSFW` 🔴)
- Human analysis panel (Person count, pose detection status, active pipeline)
- Plotly probability distribution chart
- Grad-CAM heatmap explainability tabs (Original / Heatmap / Overlay)

---

## Research & Methodological Limitations

1. **Classification Output Classes**: Output classes remain strictly `Safe`, `Violence`, `NSFW` (3 classes). Visual hate speech or text-based moderation requires separate OCR/NLP processing.
2. **Pose Landmark Role**: Body landmarks and spatial relationships serve as input feature representations for neural classification, not hardcoded rule gates. Final classification decisions rely on learned multi-feature representations.
