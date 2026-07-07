from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp_filename(prefix: str, ext: str = ".jpg", suffix: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ext if ext.startswith(".") else f".{ext}"
    parts = [prefix, timestamp]
    if suffix:
        parts.append(safe_stem(suffix))
    return "_".join(parts) + ext


def safe_stem(value: str | Path) -> str:
    stem = Path(value).stem if isinstance(value, (str, Path)) else str(value)
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    return cleaned.strip("_") or "file"


def resize_with_aspect_ratio(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None and width > 0:
        scale = width / float(image_width)
        size = (int(width), max(1, int(round(image_height * scale))))
    elif height is not None and height > 0:
        scale = height / float(image_height)
        size = (max(1, int(round(image_width * scale))), int(height))
    else:
        return image
    return cv2.resize(image, size, interpolation=interpolation)


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, image_rgb: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if image_rgb.ndim == 3 else image_rgb
    if not cv2.imwrite(str(output_path), image_bgr):
        raise IOError(f"Could not save image: {output_path}")
    return output_path


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: Sequence[Sequence[int | float]],
    scores: Sequence[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    output = image_rgb.copy()
    height, width = output.shape[:2]

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        label = "Person"
        if scores is not None and index < len(scores):
            label = f"Person: {float(scores[index]):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        text_y1 = max(0, y1 - text_height - baseline - 4)
        text_y2 = text_y1 + text_height + baseline + 4
        text_x2 = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(output, (x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, text_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    return output


def draw_fps(image_rgb: np.ndarray, fps: float) -> np.ndarray:
    output = image_rgb.copy()
    cv2.putText(
        output,
        f"FPS: {fps:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def is_video_readable(path: str | Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        cap.release()
