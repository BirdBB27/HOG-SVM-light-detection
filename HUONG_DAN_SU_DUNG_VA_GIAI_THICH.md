# Hướng Dẫn Sử Dụng Và Giải Thích Project

Project: `HOG-SVM-light-detection`

Đây là bản demo nhẹ phát hiện người đi bộ bằng HOG + Linear SVM pretrained của OpenCV. Project không dùng YOLO, CNN hay deep learning, và cũng không cần train model.

## 1. Mục tiêu

Project dùng để minh họa pipeline xử lý ảnh truyền thống:

```text
Ảnh/Video/Webcam -> HOG feature -> Linear SVM -> NMS -> Bounding Box
```

Hệ thống hỗ trợ:

- Detect người trên ảnh.
- Detect người trên video.
- Nén video trước khi detect để chạy nhanh hơn.
- Detect webcam realtime bằng OpenCV.
- Demo web bằng Gradio gồm Image, Video và Realtime Webcam.

## 2. Cài đặt môi trường

Mở terminal tại thư mục chứa project, sau đó chạy:

```bash
cd HOG-SVM-light-detection
pip install -r requirements.txt
```

Các thư viện chính:

- `opencv-python`: đọc ảnh/video, HOG detector, webcam, vẽ bounding box.
- `numpy`: xử lý mảng ảnh.
- `gradio`: giao diện web demo.
- `tqdm`: hỗ trợ progress nếu cần mở rộng.

## 3. Cấu trúc thư mục

```text
HOG-SVM-light-detection/
|-- app.py
|-- requirements.txt
|-- README.md
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

## 4. Giải thích thuật toán

### Image Gradient

Ảnh có thể được hiểu thông qua sự thay đổi cường độ sáng giữa các pixel. Gradient mô tả hướng và độ mạnh của sự thay đổi này. Biên người, vai, đầu, chân tay thường tạo ra gradient rõ.

### HOG

HOG là viết tắt của Histogram of Oriented Gradients. Ý tưởng chính:

1. Chia ảnh thành các ô nhỏ.
2. Tính gradient tại từng pixel.
3. Gom hướng gradient thành histogram.
4. Ghép các histogram lại thành vector đặc trưng.

HOG phù hợp cho phát hiện người vì hình dáng người có cấu trúc biên khá ổn định.

### Linear SVM

SVM là bộ phân loại tuyến tính. Trong project này, SVM không tự train lại mà dùng bộ pretrained có sẵn của OpenCV:

```python
cv2.HOGDescriptor_getDefaultPeopleDetector()
```

Bộ detector này đã được huấn luyện để nhận diện người trong cửa sổ HOG.

### Sliding Window Và Scale Pyramid

Detector quét nhiều vùng trên ảnh bằng cửa sổ trượt. Tham số `scale` quyết định mức thay đổi kích thước ảnh giữa các tầng pyramid. Scale nhỏ quét kỹ hơn nhưng chậm hơn.

### NMS

NMS là Non-Maximum Suppression. Khi detector trả về nhiều box chồng lên cùng một người, NMS giữ box có score cao hơn và loại các box trùng lặp.

## 5. Chạy detect ảnh

Ảnh demo đặt trong:

```text
demo/images/test.jpg
```

Chạy:

```bash
python src/detect_image.py --image demo/images/test.jpg
```

Chạy với tham số:

```bash
python src/detect_image.py --image demo/images/test.jpg --resize-width 640 --hit-threshold 0.0 --scale 1.05 --nms 0.35 --max-detections 10
```

Output nằm trong:

```text
outputs/images/
```

## 6. Chạy detect video

Video demo đặt trong:

```text
demo/videos/test.mp4
```

Chạy:

```bash
python src/detect_video.py --video demo/videos/test.mp4 --auto-compress --frame-skip 3
```

Output nằm trong:

```text
outputs/videos/
```

Nếu video lớn hoặc máy yếu, nên bật `--auto-compress`.

## 7. Nén video trước khi detect

```bash
python src/compress_video.py --input demo/videos/test.mp4 --output demo/videos/test_compressed.mp4
```

Mặc định:

- Width: `640`
- FPS: `10`
- Thời lượng tối đa: `20` giây

Script ưu tiên `ffmpeg`. Nếu máy không có `ffmpeg`, code fallback bằng OpenCV.

## 8. Chạy webcam realtime bằng OpenCV

Đây là cách ổn định nhất:

```bash
python src/detect_webcam.py --resize-width 480 --frame-skip 3 --hit-threshold 0.5 --scale 1.1
```

Thoát webcam bằng phím:

```text
q
```

Nếu webcam không mở được, kiểm tra quyền camera hoặc thử:

```bash
python src/detect_webcam.py --camera-index 1
```

## 9. Chạy web app Gradio

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:7861
```

Web app có 3 tab:

- `Image`: upload ảnh và detect.
- `Video`: upload video, có tùy chọn auto compress.
- `Realtime Webcam`: webcam streaming trong trình duyệt nếu máy và browser hỗ trợ.

Nếu webcam web bị lag hoặc không mở được camera, dùng script OpenCV:

```bash
python src/detect_webcam.py
```

## 10. Ý nghĩa các tham số

### `resize-width`

Giảm chiều rộng ảnh/video trước khi detect. Width nhỏ chạy nhanh hơn nhưng có thể mất chi tiết.

Gợi ý:

- Ảnh/video bình thường: `640`
- Webcam realtime: `480`
- Máy yếu: `320`

### `hit-threshold`

Ngưỡng score của HOG + SVM.

- Tăng lên: ít box sai hơn, nhưng có thể bỏ sót người.
- Giảm xuống: dễ detect hơn, nhưng có thể nhiều box sai hơn.

Gợi ý:

- Ảnh/video: `0.0`
- Webcam: `0.5`
- Nếu không detect được: thử `0` hoặc `-0.5`
- Nếu nhiều box sai: thử `1.0`

### `scale`

Tỷ lệ thay đổi giữa các tầng pyramid.

- Nhỏ hơn: quét kỹ hơn, chậm hơn.
- Lớn hơn: nhanh hơn, có thể bỏ sót.

Gợi ý:

- Ảnh/video: `1.05`
- Webcam nhanh: `1.1` hoặc `1.2`

### `nms`

Ngưỡng loại box chồng lặp.

- Nhỏ hơn: loại mạnh hơn.
- Lớn hơn: giữ nhiều box hơn.

Gợi ý:

- `0.25` đến `0.35`

### `frame-skip`

Chỉ dùng cho video/webcam. Nếu `frame-skip=3`, hệ thống detect mỗi 3 frame, các frame giữa dùng lại box gần nhất.

Gợi ý:

- Video/webcam bình thường: `3`
- Máy yếu: `5`

### `max-detections`

Số box tối đa giữ lại sau NMS.

- Ảnh ít người: `3` hoặc `5`
- Cảnh nhiều người: `10`

## 11. Gợi ý xử lý lỗi

### Không cài được thư viện

Chạy lại:

```bash
pip install -r requirements.txt
```

### Không mở được video output

Nếu máy có `ffmpeg`, output MP4 sẽ dễ xem hơn. Nếu không có, chương trình có thể tạo file `.avi` fallback trong `outputs/videos/`.

### Detect quá nhiều box sai

Thử:

```text
tăng hit-threshold
tăng hoặc giảm nms tùy mức chồng box
giảm max-detections
```

### Không detect được người

Thử:

```text
giảm hit-threshold về 0 hoặc -0.5
tăng resize-width
giảm scale về 1.05
```

### Video hoặc webcam chậm

Thử:

```text
bật auto-compress cho video
resize-width = 320 hoặc 480
frame-skip = 5
scale = 1.2
```

## 12. Điểm cần nhấn mạnh khi thuyết trình

- Project dùng phương pháp xử lý ảnh truyền thống, không dùng deep learning.
- Detector là HOG + Linear SVM pretrained của OpenCV.
- NMS giúp giảm box bị trùng.
- Nén video và frame-skip giúp chạy nhanh hơn.
- Bản demo phù hợp minh họa thuật toán, không nhằm đạt độ chính xác như YOLO/CNN hiện đại.

## 13. Lệnh kiểm tra nhanh

```bash
python -m compileall .
python src/detect_image.py --image demo/images/test.jpg
python src/detect_video.py --video demo/videos/test.mp4 --auto-compress --frame-skip 3
python app.py
```
