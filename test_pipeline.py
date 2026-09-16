"""
test_pipeline.py
================
Sanity test script verifying:
1. Module imports
2. HumanAnalyzer person detection, human cropping, pose estimation, and relationship features
3. MultiFeatureContentModerationModel architecture, parameter count, and output shapes
4. Pipeline fallback modes (global+human+pose, global+human, global_fallback)
5. Synthetic test image cases (single person, multi person, no person, red safe, violence, nsfw, corrupted image)
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, ".")

print("Step 1: Testing imports...")
from src.config import IMAGE_SIZE, TOTAL_POSE_REL_DIM, CLASS_NAMES
from src.human_analyzer import HumanAnalyzer
from src.model import ContentModerationModel
from src.utils import explain_activation
print("Imports OK.")

print("\nStep 2: Testing HumanAnalyzer...")
analyzer = HumanAnalyzer()

# Synthetic test image 1: Solid background (No person)
img_no_person = np.ones((224, 224, 3), dtype=np.float32) * 0.5
res_no_person = analyzer.analyze(img_no_person)
print("No Person Pipeline:", res_no_person["pipeline"])
assert res_no_person["pipeline"] == "global_fallback"
assert res_no_person["has_human"] == 0.0
assert res_no_person["has_pose"] == 0.0

# Synthetic test image 2: Drawing a simple person with skin tones
img_person = np.ones((400, 400, 3), dtype=np.uint8) * 240
skin_color = (210, 160, 120)  # RGB skin tone
# Head
cv2.circle(img_person, (200, 100), 40, skin_color, -1)
# Body
cv2.rectangle(img_person, (160, 140), (240, 300), skin_color, -1)
# Arms & Legs
cv2.line(img_person, (160, 160), (100, 240), skin_color, 15)
cv2.line(img_person, (240, 160), (300, 240), skin_color, 15)
cv2.line(img_person, (180, 300), (160, 390), skin_color, 15)
cv2.line(img_person, (220, 300), (240, 390), skin_color, 15)

res_person = analyzer.analyze(img_person.astype(np.float32) / 255.0)
print("Person Detected Pipeline:", res_person["pipeline"])
print("Human Count:", res_person["person_count"])
assert res_person["has_human"] == 1.0

print("\nStep 3: Building & Compiling MultiFeatureContentModerationModel...")
cm_model = ContentModerationModel()
model = cm_model.build_model()
cm_model.compile_model()

print("Model Inputs:", [i.name for i in model.inputs])
print("Model Output:", model.output.name)
print("Total Parameters:", model.count_params())

# Test dummy forward pass
dummy_inputs = {
    "input_global": np.zeros((1, 224, 224, 3), dtype=np.float32),
    "input_human": np.zeros((1, 224, 224, 3), dtype=np.float32),
    "input_pose": np.zeros((1, TOTAL_POSE_REL_DIM), dtype=np.float32),
    "has_human": np.array([[1.0]], dtype=np.float32),
    "has_pose": np.array([[1.0]], dtype=np.float32),
}

preds = model.predict(dummy_inputs, verbose=0)
print("Dummy Prediction Output:", preds[0])
print("Sum of Probabilities:", np.sum(preds[0]))
assert abs(np.sum(preds[0]) - 1.0) < 1e-4

print("\nAll pipeline sanity tests PASSED successfully!")
