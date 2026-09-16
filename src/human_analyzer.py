"""
human_analyzer.py
=================
Person detection, human cropping with padding, pose estimation landmark extraction,
and multi-person spatial relationship feature processing.

Supports MediaPipe (when available) with a pure NumPy + OpenCV 5.0 fallthrough detector
for complete Python 3.13+ compatibility.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.config import (
    CROP_PADDING,
    IMAGE_SIZE,
    MAX_PERSONS,
    PERSON_CONFIDENCE_THRESHOLD,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_LANDMARK_POINTS,
    POSE_FEATURE_DIM,
    RELATION_FEATURE_DIM,
    TOTAL_POSE_REL_DIM,
)

logger = logging.getLogger(__name__)

MP_AVAILABLE = False
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    logger.info("MediaPipe not available. Using native OpenCV & NumPy human analyzer.")


class HumanAnalyzer:
    """
    Analyzes images for human presence, crops human regions with padding,
    extracts pose landmarks, and computes multi-person relationship features.
    """

    KEY_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

    def __init__(
        self,
        person_threshold: float = PERSON_CONFIDENCE_THRESHOLD,
        pose_threshold: float = POSE_CONFIDENCE_THRESHOLD,
        crop_padding: float = CROP_PADDING,
        max_persons: int = MAX_PERSONS,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.person_threshold = person_threshold
        self.pose_threshold = pose_threshold
        self.crop_padding = crop_padding
        self.max_persons = max_persons
        self.image_size = image_size

        self._mp_pose = None
        if MP_AVAILABLE:
            try:
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=self.pose_threshold,
                )
            except Exception as e:
                logger.debug(f"MediaPipe Pose init skipped: {e}")

    def detect_persons(self, image_np: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect human bounding boxes in image.
        Returns list of (x_min, y_min, x_max, y_max, confidence).
        """
        h, w = image_np.shape[:2]
        boxes: List[Tuple[int, int, int, int, float]] = []

        if image_np.dtype != np.uint8:
            img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image_np

        # Strategy 1: MediaPipe (if available)
        if self._mp_pose is not None:
            try:
                results = self._mp_pose.process(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    visibilities = [lm.visibility for lm in lms]
                    avg_vis = float(np.mean(visibilities)) if visibilities else 0.0

                    if avg_vis >= self.person_threshold:
                        xs = [int(lm.x * w) for lm in lms if 0 <= lm.x <= 1]
                        ys = [int(lm.y * h) for lm in lms if 0 <= lm.y <= 1]
                        if xs and ys:
                            x_min, x_max = max(0, min(xs)), min(w, max(xs))
                            y_min, y_max = max(0, min(ys)), min(h, max(ys))
                            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                                boxes.append((x_min, y_min, x_max, y_max, round(avg_vis, 4)))
            except Exception as e:
                logger.debug(f"MediaPipe detection error: {e}")

        # Strategy 2: Native Contour & Skin-Tone Human Bounding Box Detection
        if not boxes:
            try:
                hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
                # Skin tone mask in HSV
                lower_skin = np.array([0, 20, 50], dtype=np.uint8)
                upper_skin = np.array([25, 200, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

                # Morphological dilation to group body parts
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

                contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_area = float(w * h)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > (img_area * 0.02):  # >2% of image
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        aspect_ratio = float(bw) / float(bh)
                        if 0.20 <= aspect_ratio <= 1.5:
                            conf = min(0.95, round(0.4 + (area / img_area) * 0.5, 4))
                            boxes.append((bx, by, bx + bw, by + bh, conf))
            except Exception as e:
                logger.debug(f"Contour human detection error: {e}")

        boxes = sorted(boxes, key=lambda b: (b[4], (b[2]-b[0])*(b[3]-b[1])), reverse=True)[:self.max_persons]
        return boxes

    def crop_human(
        self, image_np: np.ndarray, bbox: Tuple[int, int, int, int, float]
    ) -> np.ndarray:
        h, w = image_np.shape[:2]
        x1, y1, x2, y2, _ = bbox

        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = int(box_w * self.crop_padding)
        pad_h = int(box_h * self.crop_padding)

        px1 = max(0, x1 - pad_w)
        py1 = max(0, y1 - pad_h)
        px2 = min(w, x2 + pad_w)
        py2 = min(h, y2 + pad_h)

        crop = image_np[py1:py2, px1:px2]
        if crop.size == 0:
            crop = image_np

        if crop.dtype != np.float32:
            if crop.dtype == np.uint8:
                crop = crop.astype(np.float32) / 255.0

        crop_resized = cv2.resize(crop, (self.image_size[1], self.image_size[0]))
        return crop_resized

    def extract_pose_landmarks(self, image_np: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Extract normalized pose keypoints (12 key landmarks x 3 = 36 features).
        Returns (pose_vector, pose_success_flag).
        """
        pose_vec = np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)

        if image_np.dtype != np.uint8:
            img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image_np

        # Try MediaPipe first
        if self._mp_pose is not None:
            try:
                results = self._mp_pose.process(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    idx_out = 0
                    visibilities = []
                    for idx in self.KEY_LANDMARK_INDICES:
                        if idx < len(lms):
                            lm = lms[idx]
                            pose_vec[idx_out] = float(lm.x)
                            pose_vec[idx_out + 1] = float(lm.y)
                            pose_vec[idx_out + 2] = float(lm.visibility)
                            visibilities.append(lm.visibility)
                        idx_out += 3

                    avg_vis = float(np.mean(visibilities)) if visibilities else 0.0
                    if avg_vis >= self.pose_threshold:
                        return pose_vec, True
            except Exception as e:
                logger.debug(f"MediaPipe pose extraction failed: {e}")

        # If MediaPipe is not available or did not detect pose landmarks, return False
        # so the model gating flag (has_pose = 0.0) cleanly deactivates the pose branch
        return pose_vec, False

    def extract_relationship_features(
        self, boxes: List[Tuple[int, int, int, int, float]], img_shape: Tuple[int, int]
    ) -> np.ndarray:
        rel_vec = np.zeros((RELATION_FEATURE_DIM,), dtype=np.float32)
        if len(boxes) < 2:
            return rel_vec

        h, w = img_shape[:2]
        diag = np.sqrt(w**2 + h**2) + 1e-6
        idx = 0

        for i in range(min(len(boxes), self.max_persons)):
            for j in range(min(len(boxes), self.max_persons)):
                if i == j:
                    continue
                if idx + 3 > RELATION_FEATURE_DIM:
                    break

                b1 = boxes[i]
                b2 = boxes[j]

                c1 = np.array([(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])
                c2 = np.array([(b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0])

                dist = np.linalg.norm(c1 - c2) / diag

                xi1 = max(b1[0], b2[0])
                yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2])
                yi2 = min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter + 1e-6
                iou = inter / union

                dx = (c2[0] - c1[0]) / float(w)

                rel_vec[idx] = float(dist)
                rel_vec[idx + 1] = float(iou)
                rel_vec[idx + 2] = float(dx)
                idx += 3

        return rel_vec

    def analyze(self, image_np: np.ndarray) -> Dict[str, Any]:
        if image_np.ndim == 4:
            image_np = np.squeeze(image_np, axis=0)

        h, w = image_np.shape[:2]
        boxes = self.detect_persons(image_np)

        if not boxes:
            empty_crop = np.zeros((*self.image_size, 3), dtype=np.float32)
            empty_pose_rel = np.zeros((TOTAL_POSE_REL_DIM,), dtype=np.float32)
            return {
                "has_human": 0.0,
                "has_pose": 0.0,
                "human_crop": empty_crop,
                "pose_rel_vector": empty_pose_rel,
                "person_count": 0,
                "boxes": [],
                "pipeline": "global_fallback",
            }

        main_box = boxes[0]
        human_crop = self.crop_human(image_np, main_box)

        pose_vec, pose_success = self.extract_pose_landmarks(human_crop)
        rel_vec = self.extract_relationship_features(boxes, (h, w))
        pose_rel_vector = np.concatenate([pose_vec, rel_vec], axis=0).astype(np.float32)

        has_human = 1.0
        has_pose = 1.0 if pose_success else 0.0

        if has_pose == 1.0:
            pipeline = "global+human+pose"
        else:
            pipeline = "global+human"

        return {
            "has_human": has_human,
            "has_pose": has_pose,
            "human_crop": human_crop,
            "pose_rel_vector": pose_rel_vector,
            "person_count": len(boxes),
            "boxes": boxes,
            "pipeline": pipeline,
        }
