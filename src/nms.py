from __future__ import annotations

import numpy as np


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - inter_area
    return inter_area / np.maximum(union, 1e-8)


def non_max_suppression(
    boxes,
    scores,
    iou_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0 or scores.size == 0:
        return np.empty((0, 4), dtype=np.int32), np.asarray([], dtype=np.float32)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return boxes[keep].astype(np.int32), scores[keep]
