from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DEFAULT_WEBCAM_FRAME_SKIP,
    DEFAULT_WEBCAM_HIT_THRESHOLD,
    DEFAULT_WEBCAM_MAX_DETECTIONS,
    DEFAULT_WEBCAM_NMS_THRESHOLD,
    DEFAULT_WEBCAM_RESIZE_WIDTH,
    DEFAULT_WEBCAM_SCALE,
)
from src.detector import detect_people  # noqa: E402
from src.utils import draw_boxes, draw_fps, resize_with_aspect_ratio  # noqa: E402


DEFAULT_CAMERA_INDEX = 0
WINDOW_NAME = "HOG + SVM Light Demo - Webcam"


def run_webcam(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    resize_width: int = DEFAULT_WEBCAM_RESIZE_WIDTH,
    hit_threshold: float = DEFAULT_WEBCAM_HIT_THRESHOLD,
    scale: float = DEFAULT_WEBCAM_SCALE,
    nms_threshold: float = DEFAULT_WEBCAM_NMS_THRESHOLD,
    frame_skip: int = DEFAULT_WEBCAM_FRAME_SKIP,
    max_detections: int = DEFAULT_WEBCAM_MAX_DETECTIONS,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Không mở được webcam index {camera_index}. "
            "Hãy kiểm tra camera, quyền truy cập hoặc thử --camera-index khác."
        )

    frame_skip = max(1, int(frame_skip))
    resize_width = int(resize_width)
    frame_index = 0
    last_boxes = []
    last_scores = []
    fps = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Không đọc được frame từ webcam.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and frame_rgb.shape[1] > resize_width:
                frame_rgb = resize_with_aspect_ratio(frame_rgb, width=resize_width)

            if frame_index % frame_skip == 0:
                result_rgb, last_boxes, last_scores = detect_people(
                    frame_rgb,
                    resize_width=0,
                    hit_threshold=hit_threshold,
                    scale=scale,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections,
                )
            else:
                result_rgb = draw_boxes(frame_rgb, last_boxes, last_scores)

            elapsed = max(time.perf_counter() - loop_start, 1e-8)
            current_fps = 1.0 / elapsed
            fps = current_fps if fps <= 0 else (0.85 * fps + 0.15 * current_fps)
            result_rgb = draw_fps(result_rgb, fps)

            cv2.imshow(WINDOW_NAME, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
            frame_index += 1

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime webcam people detection using OpenCV HOG + SVM."
    )
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--resize-width", type=int, default=DEFAULT_WEBCAM_RESIZE_WIDTH)
    parser.add_argument("--hit-threshold", type=float, default=DEFAULT_WEBCAM_HIT_THRESHOLD)
    parser.add_argument("--scale", type=float, default=DEFAULT_WEBCAM_SCALE)
    parser.add_argument("--nms", type=float, default=DEFAULT_WEBCAM_NMS_THRESHOLD)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_WEBCAM_FRAME_SKIP)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_WEBCAM_MAX_DETECTIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_webcam(
            camera_index=args.camera_index,
            resize_width=args.resize_width,
            hit_threshold=args.hit_threshold,
            scale=args.scale,
            nms_threshold=args.nms,
            frame_skip=args.frame_skip,
            max_detections=args.max_detections,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
