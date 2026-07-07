from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.compress_video import compress_video  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_COMPRESS_FPS,
    DEFAULT_COMPRESS_WIDTH,
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MAX_DURATION,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_SCALE,
    DEFAULT_VIDEO_RESIZE_WIDTH,
    DEFAULT_FRAME_SKIP,
    OUTPUT_VIDEOS_DIR,
    ensure_project_structure,
)
from src.detector import detect_people  # noqa: E402
from src.utils import (  # noqa: E402
    draw_boxes,
    draw_fps,
    ensure_dir,
    get_timestamp_filename,
    is_video_readable,
    resize_with_aspect_ratio,
)


def create_avi_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    if path.exists():
        path.unlink()
    for codec in ("MJPG", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not create video writer: {path}")


def convert_avi_to_mp4(temp_avi: Path, final_mp4: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"ffmpeg not found. Keeping AVI fallback: {temp_avi}")
        return temp_avi
    if final_mp4.exists():
        final_mp4.unlink()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(temp_avi),
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-vf",
        "format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and final_mp4.exists() and final_mp4.stat().st_size > 0:
        if is_video_readable(final_mp4):
            return final_mp4

    print("ffmpeg mp4 conversion failed. Keeping AVI fallback.")
    if result.stderr:
        print(result.stderr.strip().splitlines()[-1])
    return temp_avi


def compressed_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_compressed.mp4")


def detect_video_file(
    video_path: str | Path,
    output_path: str | Path | None = None,
    auto_compress: bool = False,
    resize_width: int = DEFAULT_VIDEO_RESIZE_WIDTH,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    scale: float = DEFAULT_SCALE,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    compressed_width: int = DEFAULT_COMPRESS_WIDTH,
    compressed_fps: int = DEFAULT_COMPRESS_FPS,
    max_duration: int = DEFAULT_MAX_DURATION,
) -> tuple[Path, int, float]:
    ensure_project_structure()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if auto_compress:
        result = compress_video(
            video_path,
            compressed_path_for(video_path),
            resize_width=compressed_width,
            fps=compressed_fps,
            max_duration=max_duration,
        )
        video_path = result.output_path
        print(f"Using compressed video: {video_path}")
        print(f"Compressed size: {result.input_mb:.2f} MB -> {result.output_mb:.2f} MB")

    if output_path is None:
        output_path = OUTPUT_VIDEOS_DIR / get_timestamp_filename(
            "detect_video", ".mp4", suffix=video_path.stem
        )
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    temp_avi = output_path.with_name(f"{output_path.stem}_temp.avi")

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    writer_size: tuple[int, int] | None = None
    processed_frames = 0
    frame_index = 0
    last_boxes = []
    last_scores = []
    last_fps = 0.0
    frame_skip = max(1, int(frame_skip))
    start = time.perf_counter()

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps != source_fps:
            source_fps = 25.0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if resize_width and resize_width > 0 and frame_rgb.shape[1] > resize_width:
                frame_rgb = resize_with_aspect_ratio(frame_rgb, width=resize_width)

            if writer is None:
                height, width = frame_rgb.shape[:2]
                writer_size = (width, height)
                writer, codec = create_avi_writer(temp_avi, source_fps, writer_size)
                print(f"Writing temporary AVI: {temp_avi} ({codec}, {width}x{height})")

            if writer_size is None:
                raise RuntimeError("Video writer was not initialized.")
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != writer_size:
                frame_rgb = cv2.resize(frame_rgb, writer_size, interpolation=cv2.INTER_AREA)

            frame_start = time.perf_counter()
            if frame_index % frame_skip == 0:
                result_rgb, last_boxes, last_scores = detect_people(
                    frame_rgb,
                    resize_width=0,
                    hit_threshold=hit_threshold,
                    scale=scale,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections,
                )
                elapsed = max(time.perf_counter() - frame_start, 1e-8)
                last_fps = 1.0 / elapsed
            else:
                result_rgb = draw_boxes(frame_rgb, last_boxes, last_scores)

            result_rgb = draw_fps(result_rgb, last_fps)
            writer.write(cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
            processed_frames += 1
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start
    if processed_frames == 0:
        raise RuntimeError(f"No frames were read from: {video_path}")
    if not temp_avi.exists() or temp_avi.stat().st_size == 0 or not is_video_readable(temp_avi):
        raise RuntimeError(f"Temporary AVI is not readable: {temp_avi}")

    if output_path.suffix.lower() == ".avi":
        final_output = temp_avi
        if temp_avi != output_path:
            if output_path.exists():
                output_path.unlink()
            temp_avi.replace(output_path)
            final_output = output_path
    else:
        final_output = convert_avi_to_mp4(temp_avi, output_path)

    if not final_output.exists() or final_output.stat().st_size == 0:
        raise RuntimeError(f"Output video was not created: {final_output}")
    return final_output, processed_frames, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in video using OpenCV HOG + SVM.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--auto-compress", action="store_true")
    parser.add_argument("--resize-width", type=int, default=DEFAULT_VIDEO_RESIZE_WIDTH)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--hit-threshold", type=float, default=DEFAULT_HIT_THRESHOLD)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS)
    parser.add_argument("--compressed-width", type=int, default=DEFAULT_COMPRESS_WIDTH)
    parser.add_argument("--compressed-fps", type=int, default=DEFAULT_COMPRESS_FPS)
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path, frame_count, elapsed = detect_video_file(
            args.video,
            output_path=args.output,
            auto_compress=args.auto_compress,
            resize_width=args.resize_width,
            frame_skip=args.frame_skip,
            hit_threshold=args.hit_threshold,
            scale=args.scale,
            nms_threshold=args.nms,
            max_detections=args.max_detections,
            compressed_width=args.compressed_width,
            compressed_fps=args.compressed_fps,
            max_duration=args.max_duration,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        sys.exit(1)

    print(f"Saved result video: {output_path}")
    print(f"Processed frames: {frame_count}")
    print(f"Processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
