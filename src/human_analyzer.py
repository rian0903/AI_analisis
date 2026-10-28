"""Person detection, human cropping, pose landmarks, and spatial features."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    mp = None
    logger.warning("MediaPipe is unavailable; human and pose branches will be disabled.")


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

        self._pose_landmarker = None
        model_path = Path(__file__).resolve().parent.parent / "assets" / "pose_landmarker_lite.task"
        if MP_AVAILABLE and model_path.is_file():
            try:
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=self.max_persons,
                    min_pose_detection_confidence=self.person_threshold,
                    min_pose_presence_confidence=self.person_threshold,
                    min_tracking_confidence=self.pose_threshold,
                )
                self._pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            except Exception as e:
                logger.warning("MediaPipe Pose Landmarker init failed: %s", e)
        elif MP_AVAILABLE:
            logger.warning("Pose task asset not found at %s; human branches will be disabled.", model_path)

    @staticmethod
    def _as_uint8_rgb(image_np: np.ndarray) -> np.ndarray:
        if image_np.dtype == np.uint8:
            return np.ascontiguousarray(image_np)
        return np.ascontiguousarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))

    def _detect_pose_landmarks(self, image_np: np.ndarray) -> List[List[Any]]:
        if self._pose_landmarker is None:
            return []
        try:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=self._as_uint8_rgb(image_np),
            )
            return self._pose_landmarker.detect(image).pose_landmarks
        except Exception as e:
            logger.debug("Pose detection failed: %s", e)
            return []

    def _pose_detections(
        self, poses: List[List[Any]], image_shape: Tuple[int, ...]
    ) -> List[Tuple[Tuple[int, int, int, int, float], List[Any]]]:
        h, w = image_shape[:2]
        boxes = []
        for landmarks in poses:
            visible = [lm for lm in landmarks if float(lm.visibility) >= self.person_threshold]
            if len(visible) < 4:
                continue
            xs = [int(np.clip(lm.x * w, 0, w - 1)) for lm in visible]
            ys = [int(np.clip(lm.y * h, 0, h - 1)) for lm in visible]
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            if x2 - x1 <= 10 or y2 - y1 <= 10:
                continue
            confidence = float(np.mean([lm.visibility for lm in visible]))
            boxes.append(((x1, y1, x2, y2, confidence), landmarks))
        return sorted(boxes, key=lambda item: item[0][4], reverse=True)[:self.max_persons]

    def detect_persons(self, image_np: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect human bounding boxes in image.
        Returns list of (x_min, y_min, x_max, y_max, confidence).
        """
        poses = self._detect_pose_landmarks(image_np)
        return [box for box, _ in self._pose_detections(poses, image_np.shape)]

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
        poses = self._detect_pose_landmarks(image_np)
        if not poses:
            return np.zeros((POSE_FEATURE_DIM,), dtype=np.float32), False
        return self._pose_vector(poses[0]), True

    def _pose_vector(self, landmarks: List[Any]) -> np.ndarray:
        pose_vec = np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)
        for output_idx, landmark_idx in enumerate(self.KEY_LANDMARK_INDICES):
            if landmark_idx >= len(landmarks):
                continue
            landmark = landmarks[landmark_idx]
            offset = output_idx * 3
            pose_vec[offset:offset + 3] = (landmark.x, landmark.y, landmark.visibility)
        return pose_vec

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
        poses = self._detect_pose_landmarks(image_np)
        detections = self._pose_detections(poses, image_np.shape)
        boxes = [box for box, _ in detections]

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

        pose_vec = self._pose_vector(detections[0][1])
        pose_success = bool(np.mean(pose_vec[2::3]) >= self.pose_threshold)
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
