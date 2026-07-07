# HOG-SVM Light Detection

Đây là bản demo nhẹ để phát hiện người bằng HOG + Linear SVM pretrained của OpenCV. Project không train model, không dùng YOLO, CNN hay deep learning.

Pipeline:

```text
Image/Video -> HOG -> Linear SVM -> NMS -> Bounding Box
```

## Cài đặt

```bash
cd HOG-SVM-light-detection
pip install -r requirements.txt
```


## Detect ảnh

Đặt ảnh demo vào `demo/images/`, ví dụ `demo/images/test.jpg`, rồi chạy:

```bash
python src/detect_image.py --image demo/images/test.jpg
```

Tùy chỉnh tham số:

```bash
python src/detect_image.py --image demo/images/test.jpg --resize-width 640 --hit-threshold 0.0 --scale 1.05 --nms 0.35 --max-detections 10
```

Output nằm trong `outputs/images/`.

## Nén video

```bash
python src/compress_video.py --input demo/videos/test.mp4 --output demo/videos/test_compressed.mp4
```

Mặc định video được giảm về width `640`, FPS `10`, tối đa `20` giây. Script ưu tiên `ffmpeg`; nếu không có sẽ fallback bằng OpenCV.

## Detect video

Đặt video demo vào `demo/videos/`, ví dụ `demo/videos/test.mp4`, rồi chạy:

```bash
python src/detect_video.py --video demo/videos/test.mp4 --auto-compress --frame-skip 3
```

Output nằm trong `outputs/videos/`. Script sẽ tạo MP4 H.264 nếu máy có `ffmpeg`; nếu MP4 lỗi sẽ giữ file AVI fallback.

## Chạy web app

```bash
python app.py
```

Mở link local Gradio, thường là:

```text
http://127.0.0.1:7861
```

Web app có 2 tab:

- `Image`: upload ảnh và chỉnh resize width, hit threshold, scale, NMS, max detections.
- `Video`: upload video, bật/tắt auto compress và chỉnh resize width, frame skip, hit threshold, scale, NMS, max detections.
- `Realtime Webcam`: chạy webcam streaming trong Gradio nếu trình duyệt hỗ trợ.

## Webcam realtime

Cách ổn định nhất là chạy webcam realtime bằng OpenCV:

```bash
python src/detect_webcam.py --resize-width 480 --frame-skip 3 --hit-threshold 0.5 --scale 1.1
```

Nhấn phím `q` để thoát cửa sổ webcam.

Cách chạy qua web:

```bash
python app.py
```

Sau đó vào tab `Realtime Webcam`. Nếu webcam streaming trong trình duyệt bị lag hoặc không mở được camera, dùng cách OpenCV ở trên.

Gợi ý:

- Nếu chậm: `resize-width 320`, `frame-skip 5`, `scale 1.2`.
- Nếu nhiều box sai: tăng `hit-threshold`, giảm `max-detections`.
- Nếu không detect được: giảm `hit-threshold` về `0` hoặc `-0.5`.

## Gợi ý chỉnh tham số

- Nhiều box sai riêng lẻ: tăng `hit-threshold`, giảm `max-detections`.
- Nhiều box chồng quanh cùng một người: giảm `NMS`.
- Không detect được: giảm `hit-threshold`.
- Video chậm: bật `auto-compress`, giảm `resize width`, tăng `frame-skip`.

## Cấu trúc

```text
HOG-SVM-light-detection/
|-- app.py
|-- requirements.txt
|-- README.md
|-- DOC_HIEU_DO_AN_HOG_SVM.md
|-- GIAI_THICH_THUAT_TOAN_VA_THAM_SO.md
|-- HUONG_DAN_SU_DUNG_VA_GIAI_THICH.md
|-- demo/
|   |-- images/
|   `-- videos/
|-- outputs/
|   |-- images/
|   `-- videos/
`-- src/
    |-- config.py
    |-- detector.py
    |-- nms.py
    |-- detect_image.py
    |-- detect_video.py
    |-- detect_webcam.py
    |-- compress_video.py
    `-- utils.py
```
