from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compress_video import compress_video  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_COMPRESS_FPS,
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MAX_DURATION,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_SCALE,
    DEFAULT_VIDEO_RESIZE_WIDTH,
    DEFAULT_WEBCAM_HIT_THRESHOLD,
    DEFAULT_WEBCAM_MAX_DETECTIONS,
    DEFAULT_WEBCAM_NMS_THRESHOLD,
    DEFAULT_WEBCAM_RESIZE_WIDTH,
    DEFAULT_WEBCAM_SCALE,
    DEMO_IMAGES_DIR,
    DEMO_VIDEOS_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_VIDEOS_DIR,
    ensure_project_structure,
)
from src.detect_video import detect_video_file  # noqa: E402
from src.detector import detect_people  # noqa: E402
from src.utils import ensure_dir, get_timestamp_filename, safe_stem, save_image  # noqa: E402


def prepare_folders() -> None:
    ensure_project_structure()
    for folder in [DEMO_IMAGES_DIR, DEMO_VIDEOS_DIR, OUTPUT_IMAGES_DIR, OUTPUT_VIDEOS_DIR]:
        ensure_dir(folder)


def path_from_gradio(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if isinstance(uploaded_file, dict):
        for key in ("path", "name", "video"):
            value = uploaded_file.get(key)
            if value:
                return Path(value)
    name = getattr(uploaded_file, "name", None)
    if name:
        return Path(name)
    return None


def copy_uploaded_file(uploaded_file, output_dir: Path, prefix: str, default_ext: str) -> Path:
    source = path_from_gradio(uploaded_file)
    if source is None:
        raise ValueError("No uploaded file.")
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")

    extension = source.suffix.lower() or default_ext
    suffix = f"{safe_stem(source)}_{uuid.uuid4().hex[:8]}"
    destination = output_dir / get_timestamp_filename(prefix, extension, suffix=suffix)
    ensure_dir(destination.parent)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def image_demo(
    image: np.ndarray | None,
    resize_width: int,
    hit_threshold: float,
    scale: float,
    nms_threshold: float,
    max_detections: int,
):
    if image is None:
        return None, "Hay upload mot anh."

    start = time.perf_counter()
    try:
        result, boxes, _ = detect_people(
            image,
            resize_width=int(resize_width),
            hit_threshold=float(hit_threshold),
            scale=float(scale),
            nms_threshold=float(nms_threshold),
            max_detections=int(max_detections),
        )
        output_path = OUTPUT_IMAGES_DIR / get_timestamp_filename("app_image", ".jpg")
        save_image(output_path, result)
        elapsed = time.perf_counter() - start
        message = (
            f"People detected: {len(boxes)}\n"
            f"Processing time: {elapsed:.2f} seconds\n"
            f"Output: {output_path}"
        )
        return result, message
    except Exception as exc:
        return None, str(exc)


def video_demo(
    video,
    auto_compress: bool,
    resize_width: int,
    frame_skip: int,
    hit_threshold: float,
    scale: float,
    nms_threshold: float,
    max_detections: int,
):
    if video is None:
        return None, "Hay upload mot video."

    start = time.perf_counter()
    try:
        input_path = copy_uploaded_file(video, DEMO_VIDEOS_DIR, "upload_video", ".mp4")
        detect_input = input_path
        compress_text = ""

        if bool(auto_compress):
            compressed_path = DEMO_VIDEOS_DIR / get_timestamp_filename(
                "compressed", ".mp4", suffix=input_path.stem
            )
            compression = compress_video(
                input_path,
                compressed_path,
                resize_width=int(resize_width),
                fps=DEFAULT_COMPRESS_FPS,
                max_duration=DEFAULT_MAX_DURATION,
            )
            detect_input = compression.output_path
            compress_text = (
                f"Compressed: {compression.input_mb:.2f} MB -> {compression.output_mb:.2f} MB\n"
            )

        output_path = OUTPUT_VIDEOS_DIR / get_timestamp_filename(
            "app_video", ".mp4", suffix=detect_input.stem
        )
        result_path, frame_count, detect_time = detect_video_file(
            detect_input,
            output_path=output_path,
            auto_compress=False,
            resize_width=int(resize_width),
            frame_skip=int(frame_skip),
            hit_threshold=float(hit_threshold),
            scale=float(scale),
            nms_threshold=float(nms_threshold),
            max_detections=int(max_detections),
        )
        elapsed = time.perf_counter() - start
        message = (
            compress_text
            + f"Processed frames: {frame_count}\n"
            + f"Detection time: {detect_time:.2f} seconds\n"
            + f"Total time: {elapsed:.2f} seconds\n"
            + f"Output: {result_path}"
        )
        return str(result_path), message
    except Exception as exc:
        return None, str(exc)


def realtime_webcam_demo(
    frame: np.ndarray | None,
    resize_width: int,
    hit_threshold: float,
    scale: float,
    nms_threshold: float,
    max_detections: int,
):
    if frame is None:
        return None
    try:
        result, _, _ = detect_people(
            frame,
            resize_width=int(resize_width),
            hit_threshold=float(hit_threshold),
            scale=float(scale),
            nms_threshold=float(nms_threshold),
            max_detections=int(max_detections),
        )
        return result
    except Exception:
        return frame


prepare_folders()


with gr.Blocks(title="HOG + SVM Light Demo") as demo:
    gr.Markdown("# HOG + SVM Light Demo")
    gr.Markdown(
        "Demo nhe dung OpenCV pretrained HOG + Linear SVM people detector, "
        "sliding window va Non-Maximum Suppression."
    )

    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload image")
            image_output = gr.Image(type="numpy", label="Detected image")
        with gr.Row():
            image_resize = gr.Slider(320, 1280, value=DEFAULT_RESIZE_WIDTH, step=80, label="Resize width")
            image_hit = gr.Slider(-1.0, 3.0, value=DEFAULT_HIT_THRESHOLD, step=0.1, label="Hit threshold")
            image_scale = gr.Slider(1.01, 1.5, value=DEFAULT_SCALE, step=0.01, label="Scale")
            image_nms = gr.Slider(0.05, 0.8, value=DEFAULT_NMS_THRESHOLD, step=0.05, label="NMS")
            image_max = gr.Slider(1, 30, value=DEFAULT_MAX_DETECTIONS, step=1, label="Max detections")
        image_button = gr.Button("Detect Image")
        image_message = gr.Textbox(label="Result", lines=4, interactive=False)
        image_button.click(
            image_demo,
            inputs=[image_input, image_resize, image_hit, image_scale, image_nms, image_max],
            outputs=[image_output, image_message],
        )

    with gr.Tab("Video"):
        video_input = gr.Video(label="Upload video")
        with gr.Row():
            video_compress = gr.Checkbox(value=True, label="Auto compress")
            video_resize = gr.Slider(
                320,
                1280,
                value=DEFAULT_VIDEO_RESIZE_WIDTH,
                step=80,
                label="Resize width",
            )
            video_skip = gr.Slider(1, 10, value=3, step=1, label="Frame skip")
        with gr.Row():
            video_hit = gr.Slider(-1.0, 3.0, value=DEFAULT_HIT_THRESHOLD, step=0.1, label="Hit threshold")
            video_scale = gr.Slider(1.01, 1.5, value=DEFAULT_SCALE, step=0.01, label="Scale")
            video_nms = gr.Slider(0.05, 0.8, value=DEFAULT_NMS_THRESHOLD, step=0.05, label="NMS")
            video_max = gr.Slider(1, 30, value=DEFAULT_MAX_DETECTIONS, step=1, label="Max detections")
        video_button = gr.Button("Detect Video")
        video_output = gr.Video(label="Detected video")
        video_message = gr.Textbox(label="Result", lines=5, interactive=False)
        video_button.click(
            video_demo,
            inputs=[
                video_input,
                video_compress,
                video_resize,
                video_skip,
                video_hit,
                video_scale,
                video_nms,
                video_max,
            ],
            outputs=[video_output, video_message],
        )

    with gr.Tab("Realtime Webcam"):
        with gr.Row():
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Webcam input",
            )
            webcam_output = gr.Image(type="numpy", label="Detected webcam frame")
        with gr.Row():
            webcam_resize = gr.Slider(
                320,
                960,
                value=DEFAULT_WEBCAM_RESIZE_WIDTH,
                step=80,
                label="Resize width",
            )
            webcam_hit = gr.Slider(
                -1.0,
                3.0,
                value=DEFAULT_WEBCAM_HIT_THRESHOLD,
                step=0.1,
                label="Hit threshold",
            )
            webcam_scale = gr.Slider(
                1.01,
                1.5,
                value=DEFAULT_WEBCAM_SCALE,
                step=0.01,
                label="Scale",
            )
            webcam_nms = gr.Slider(
                0.05,
                0.8,
                value=DEFAULT_WEBCAM_NMS_THRESHOLD,
                step=0.05,
                label="NMS",
            )
            webcam_max = gr.Slider(
                1,
                20,
                value=DEFAULT_WEBCAM_MAX_DETECTIONS,
                step=1,
                label="Max detections",
            )
        webcam_input.stream(
            realtime_webcam_demo,
            inputs=[
                webcam_input,
                webcam_resize,
                webcam_hit,
                webcam_scale,
                webcam_nms,
                webcam_max,
            ],
            outputs=webcam_output,
            show_progress="hidden",
            queue=False,
            stream_every=0.5,
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
