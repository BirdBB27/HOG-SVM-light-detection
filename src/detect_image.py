from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_SCALE,
    OUTPUT_IMAGES_DIR,
    ensure_project_structure,
)
from src.detector import detect_people  # noqa: E402
from src.utils import ensure_dir, get_timestamp_filename, load_image, save_image  # noqa: E402


def detect_image_file(
    image_path: str | Path,
    output_path: str | Path | None = None,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    scale: float = DEFAULT_SCALE,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> tuple[Path, int, float]:
    ensure_project_structure()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    start_time = time.perf_counter()
    result, boxes, _ = detect_people(
        image,
        resize_width=resize_width,
        hit_threshold=hit_threshold,
        scale=scale,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
    )
    elapsed = time.perf_counter() - start_time

    if output_path is None:
        output_path = OUTPUT_IMAGES_DIR / get_timestamp_filename(
            "detect_image", ".jpg", suffix=image_path.stem
        )
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    save_image(output_path, result)
    return output_path, len(boxes), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in an image using OpenCV HOG + SVM.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path.")
    parser.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH)
    parser.add_argument("--hit-threshold", type=float, default=DEFAULT_HIT_THRESHOLD)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path, count, elapsed = detect_image_file(
            args.image,
            output_path=args.output,
            resize_width=args.resize_width,
            hit_threshold=args.hit_threshold,
            scale=args.scale,
            nms_threshold=args.nms,
            max_detections=args.max_detections,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        sys.exit(1)

    print(f"People detected: {count}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Saved result image: {output_path}")


if __name__ == "__main__":
    main()
