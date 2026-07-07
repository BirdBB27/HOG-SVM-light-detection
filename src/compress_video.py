from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DEFAULT_COMPRESS_FPS,
    DEFAULT_COMPRESS_WIDTH,
    DEFAULT_CRF,
    DEFAULT_MAX_DURATION,
)
from src.utils import ensure_dir, is_video_readable, resize_with_aspect_ratio  # noqa: E402


@dataclass
class CompressionResult:
    input_path: Path
    output_path: Path
    input_mb: float
    output_mb: float
    original_fps: float
    target_fps: float
    frames_written: int
    processing_time: float
    used_ffmpeg: bool


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def get_video_info(video_path: str | Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return float(fps), frame_count
    finally:
        cap.release()


def run_ffmpeg(
    input_path: Path,
    output_path: Path,
    resize_width: int,
    fps: int,
    max_duration: int,
    crf: int,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    if output_path.exists():
        output_path.unlink()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale={resize_width}:-2,fps={fps}",
        "-t",
        str(max_duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print("ffmpeg compression failed:")
            print(result.stderr.strip().splitlines()[-1])
        return False
    return output_path.exists() and output_path.stat().st_size > 0 and is_video_readable(output_path)


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    codecs = ("mp4v", "avc1") if path.suffix.lower() == ".mp4" else ("MJPG", "XVID")
    for codec in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def write_with_opencv(
    input_path: Path,
    output_path: Path,
    resize_width: int,
    fps: int,
    max_duration: int,
) -> tuple[Path, int]:
    def write_to(path: Path) -> int:
        cap = cv2.VideoCapture(str(input_path))
        writer = None
        frames_written = 0
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {input_path}")
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0 or source_fps != source_fps:
                source_fps = 25.0
            target_interval = 1.0 / fps
            next_sample_time = 0.0
            frame_index = 0
            max_frames = int(max_duration * fps)
            if path.exists():
                path.unlink()

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                current_time = frame_index / source_fps
                frame_index += 1
                if current_time > max_duration:
                    break
                if current_time + 1e-9 < next_sample_time:
                    continue
                next_sample_time += target_interval

                if resize_width > 0 and frame_bgr.shape[1] > resize_width:
                    frame_bgr = resize_with_aspect_ratio(frame_bgr, width=resize_width)
                if writer is None:
                    height, width = frame_bgr.shape[:2]
                    writer, codec = create_writer(path, fps, (width, height))
                    if writer is None:
                        raise RuntimeError(f"Could not create video writer: {path}")
                    print(f"OpenCV writer: {path} ({codec}, {width}x{height}, {fps} FPS)")
                writer.write(frame_bgr)
                frames_written += 1
                if frames_written >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if frames_written == 0:
            raise RuntimeError(f"No frames written from: {input_path}")
        if not path.exists() or path.stat().st_size == 0 or not is_video_readable(path):
            raise RuntimeError(f"Output video is not readable: {path}")
        return frames_written

    try:
        frames = write_to(output_path)
        return output_path, frames
    except RuntimeError:
        if output_path.suffix.lower() != ".mp4":
            raise
        fallback = output_path.with_suffix(".avi")
        frames = write_to(fallback)
        return fallback, frames


def estimate_frames(source_fps: float, frame_count: int, fps: int, max_duration: int) -> int:
    if frame_count <= 0 or source_fps <= 0:
        return 0
    duration = min(frame_count / source_fps, max_duration)
    return int(round(duration * fps))


def compress_video(
    input_path: str | Path,
    output_path: str | Path,
    resize_width: int = DEFAULT_COMPRESS_WIDTH,
    fps: int = DEFAULT_COMPRESS_FPS,
    max_duration: int = DEFAULT_MAX_DURATION,
    crf: int = DEFAULT_CRF,
) -> CompressionResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must be different from --input.")
    if resize_width <= 0 or fps <= 0 or max_duration <= 0:
        raise ValueError("resize_width, fps, and max_duration must be greater than 0.")

    ensure_dir(output_path.parent)
    source_fps, frame_count = get_video_info(input_path)
    start = time.perf_counter()
    used_ffmpeg = run_ffmpeg(input_path, output_path, resize_width, fps, max_duration, crf)
    if used_ffmpeg:
        real_output = output_path
        frames_written = estimate_frames(source_fps, frame_count, fps, max_duration)
    else:
        print("ffmpeg not found or failed. Falling back to OpenCV.")
        real_output, frames_written = write_with_opencv(
            input_path, output_path, resize_width, fps, max_duration
        )
    elapsed = time.perf_counter() - start
    return CompressionResult(
        input_path=input_path,
        output_path=real_output,
        input_mb=file_size_mb(input_path),
        output_mb=file_size_mb(real_output),
        original_fps=source_fps,
        target_fps=float(fps),
        frames_written=frames_written,
        processing_time=elapsed,
        used_ffmpeg=used_ffmpeg,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress a video for faster HOG detection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=DEFAULT_COMPRESS_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_COMPRESS_FPS)
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = compress_video(
            args.input,
            args.output,
            resize_width=args.resize_width,
            fps=args.fps,
            max_duration=args.max_duration,
            crf=args.crf,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    print(f"Input video: {result.input_path}")
    print(f"Input size: {result.input_mb:.2f} MB")
    print(f"Output video: {result.output_path}")
    print(f"Output size: {result.output_mb:.2f} MB")
    print(f"Original FPS: {result.original_fps:.2f}")
    print(f"New FPS: {result.target_fps:.2f}")
    print(f"Frames written: {result.frames_written}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Encoder: {'ffmpeg' if result.used_ffmpeg else 'OpenCV'}")


if __name__ == "__main__":
    main()
