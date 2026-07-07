from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from src.config import (
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_SCALE,
)
from src.nms import non_max_suppression
from src.utils import draw_boxes, resize_with_aspect_ratio


@lru_cache(maxsize=1)
def get_hog_detector() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people(
    image,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    scale: float = DEFAULT_SCALE,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect people using OpenCV pretrained HOG + Linear SVM."""
    image_rgb = np.asarray(image)
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image is empty.")
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if resize_width and resize_width > 0 and image_rgb.shape[1] > resize_width:
        image_rgb = resize_with_aspect_ratio(image_rgb, width=resize_width)

    hog = get_hog_detector()
    rects, weights = hog.detectMultiScale(
        image_rgb,
        float(hit_threshold),
        (8, 8),
        (8, 8),
        float(scale),
        0,
        False,
    )

    boxes: list[list[int]] = []
    for x, y, w, h in rects:
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    scores = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return image_rgb.copy(), np.empty((0, 4), dtype=np.int32), scores

    boxes_array, scores_array = non_max_suppression(boxes, scores, nms_threshold)
    if len(scores_array) > 0:
        order = scores_array.argsort()[::-1]
        if max_detections and max_detections > 0:
            order = order[: int(max_detections)]
        boxes_array = boxes_array[order]
        scores_array = scores_array[order]

    result = draw_boxes(image_rgb, boxes_array, scores_array)
    return result, boxes_array, scores_array
