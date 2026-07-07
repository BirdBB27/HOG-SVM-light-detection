# Giải Thích Thuật Toán Và Tham Số

Project: `HOG-SVM-light-detection`

Đây là bản demo phát hiện người đi bộ bằng phương pháp xử lý ảnh truyền thống: HOG + Linear SVM pretrained của OpenCV. Project không dùng YOLO, CNN hay deep learning.

## 1. Pipeline tổng quát

Pipeline chính:

```text
Ảnh / Video / Webcam
-> resize để chạy nhanh hơn
-> HOG feature extraction
-> Linear SVM pretrained của OpenCV
-> lọc theo hit-threshold
-> Non-Maximum Suppression
-> vẽ bounding box + score
```

Trong code, pipeline này nằm chủ yếu ở:

```text
src/detector.py
src/nms.py
src/detect_image.py
src/detect_video.py
src/detect_webcam.py
app.py
```

## 2. HOG là gì?

HOG là viết tắt của `Histogram of Oriented Gradients`, nghĩa là histogram hướng gradient.

Ý tưởng:

1. Ảnh được chuyển thành thông tin biên/cạnh thông qua gradient.
2. Mỗi pixel có độ mạnh gradient và hướng gradient.
3. Ảnh được chia thành các cell nhỏ.
4. Trong mỗi cell, hướng gradient được gom vào histogram.
5. Các histogram được chuẩn hóa theo block để giảm ảnh hưởng ánh sáng.
6. Toàn bộ histogram được ghép thành vector đặc trưng HOG.

HOG phù hợp với phát hiện người vì hình dáng người có cấu trúc biên khá rõ: đầu, vai, thân, chân, tay.

## 3. Các tham số HOG mặc định của OpenCV

Trong project, detector được tạo bằng:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

`cv2.HOGDescriptor()` dùng cấu hình HOG mặc định thường gặp cho pedestrian detection:

| Tham số | Giá trị thường dùng | Ý nghĩa |
|---|---:|---|
| `winSize` | `64 x 128` | Kích thước cửa sổ phát hiện người |
| `blockSize` | `16 x 16` | Mỗi block gồm nhiều cell để chuẩn hóa |
| `blockStride` | `8 x 8` | Bước trượt của block |
| `cellSize` | `8 x 8` | Kích thước cell để gom histogram gradient |
| `nbins` | `9` | Số bin hướng gradient |
| `histogramNormType` | `L2-Hys` | Cách chuẩn hóa histogram |

Với cửa sổ `64 x 128`:

- Số cell theo chiều ngang: `64 / 8 = 8`
- Số cell theo chiều dọc: `128 / 8 = 16`
- Mỗi block có `2 x 2` cell
- Số block ngang: `8 - 2 + 1 = 7`
- Số block dọc: `16 - 2 + 1 = 15`
- Tổng số block: `7 x 15 = 105`
- Mỗi block có `2 x 2 x 9 = 36` giá trị
- Vector HOG có `105 x 36 = 3780` đặc trưng

Bộ SVM pretrained của OpenCV đi kèm vector trọng số tương ứng với đặc trưng HOG này.

## 4. SVM hoạt động như thế nào?

SVM là `Support Vector Machine`. Trong project này, SVM là bộ phân loại tuyến tính.

Mỗi cửa sổ ảnh sau khi trích xuất HOG sẽ có vector đặc trưng:

```text
x = [hog_1, hog_2, ..., hog_3780]
```

Linear SVM tính điểm:

```text
score = w · x + b
```

Trong đó:

- `x`: vector HOG của cửa sổ ảnh
- `w`: trọng số SVM đã được OpenCV huấn luyện sẵn
- `b`: bias
- `score`: độ tin cậy tương đối của cửa sổ

Nếu `score` vượt qua ngưỡng `hit-threshold`, cửa sổ đó được xem là có khả năng chứa người.

## 5. Vì sao project không cần train model?

Project dùng detector pretrained có sẵn của OpenCV:

```python
cv2.HOGDescriptor_getDefaultPeopleDetector()
```

Điều này có nghĩa:

- Không cần chuẩn bị dataset.
- Không cần train SVM.
- Chạy nhanh hơn và dễ demo hơn.
- Độ chính xác không bằng các mô hình deep learning hiện đại như YOLO/CNN.

Đây là bản demo nhẹ để minh họa thuật toán HOG + SVM truyền thống.

## 6. `Person: ???` trên bounding box là gì?

Khi detect, OpenCV trả về:

```python
rects, weights = hog.detectMultiScale(...)
```

Trong đó:

- `rects`: danh sách bounding box
- `weights`: score tương ứng với từng box

Project vẽ label dạng:

```text
Person: 0.87
Person: 1.42
Person: 2.15
```

Con số sau `Person:` là score của Linear SVM đối với box đó.

Quan trọng:

- Đây không phải phần trăm xác suất.
- Đây không phải confidence kiểu YOLO.
- Đây gần với decision score / margin của SVM.
- Score càng cao thì detector càng tự tin hơn.
- Score thấp nhưng vẫn dương có thể là người hoặc false positive.

Ví dụ:

```text
Person: 2.30
```

Nghĩa là cửa sổ đó có điểm SVM khá cao, thường đáng tin hơn:

```text
Person: 0.15
```

Nhưng score không đảm bảo đúng tuyệt đối, vì HOG + SVM vẫn có thể nhầm với cột, biển báo, thân cây, bóng người hoặc vật có cạnh giống dáng người.

## 7. `hit-threshold` là gì?

`hit-threshold` là ngưỡng lọc score SVM.

Trong code:

```python
hog.detectMultiScale(
    image,
    hitThreshold=hit_threshold,
    ...
)
```

Ý nghĩa:

```text
score >= hit-threshold -> giữ box
score < hit-threshold  -> loại box
```

Cách chỉnh:

| Tình huống | Cách chỉnh |
|---|---|
| Nhiều box sai | Tăng `hit-threshold` |
| Không detect được người | Giảm `hit-threshold` |
| Webcam nhiễu, nhiều false positive | Dùng `0.5`, `1.0` hoặc cao hơn |
| Ảnh khó, người nhỏ hoặc mờ | Dùng `0`, `-0.5` |

Giá trị gợi ý:

```text
Ảnh/video thường: 0.0
Webcam realtime: 0.5
Muốn nhạy hơn: -0.5
Muốn ít box sai hơn: 1.0
```

## 8. `scale` là gì?

`scale` là tỉ lệ thu nhỏ ảnh giữa các tầng pyramid trong `detectMultiScale`.

Vì người trong ảnh có thể lớn hoặc nhỏ, detector cần quét ảnh ở nhiều kích thước khác nhau.

Ví dụ:

```text
scale = 1.05
```

Nghĩa là mỗi tầng pyramid thay đổi kích thước khoảng 5%.

So sánh:

| `scale` | Tốc độ | Khả năng detect |
|---:|---|---|
| `1.01` | Rất chậm | Quét rất kỹ |
| `1.05` | Chậm vừa | Thường tốt cho ảnh/video |
| `1.10` | Nhanh hơn | Phù hợp webcam |
| `1.20` | Nhanh | Có thể bỏ sót người |

Gợi ý:

```text
Ảnh/video: 1.05
Webcam: 1.1
Máy yếu: 1.2
```

## 9. NMS là gì?

NMS là `Non-Maximum Suppression`.

Khi detector quét sliding window, một người có thể sinh ra nhiều bounding box chồng lên nhau. NMS giữ box tốt nhất và loại các box trùng.

Project tự implement NMS trong:

```text
src/nms.py
```

NMS dùng IoU:

```text
IoU = diện tích giao nhau / diện tích hợp nhau
```

Nếu IoU giữa hai box lớn hơn ngưỡng `nms`, box có score thấp hơn bị loại.

Ý nghĩa `nms`:

| Giá trị NMS | Hiệu ứng |
|---:|---|
| Thấp, ví dụ `0.15` | Loại box chồng rất mạnh |
| Trung bình, ví dụ `0.25` - `0.35` | Cân bằng |
| Cao, ví dụ `0.6` | Giữ lại nhiều box hơn |

Gợi ý:

```text
Nhiều box chồng quanh cùng một người -> giảm NMS
Box người bị loại quá nhiều -> tăng NMS
```

Lưu ý: nếu box sai nằm riêng lẻ, NMS không loại được. Khi đó nên tăng `hit-threshold` hoặc giảm `max-detections`.

## 10. `max-detections` là gì?

`max-detections` giới hạn số bounding box cuối cùng sau NMS.

Ví dụ:

```text
max-detections = 5
```

Nghĩa là chỉ giữ tối đa 5 box có score cao nhất.

Cách chỉnh:

| Tình huống | Gợi ý |
|---|---|
| Ảnh chỉ có 1 người | `1` hoặc `3` |
| Cảnh có vài người | `5` |
| Cảnh đông người | `10` hoặc hơn |
| Nhiều box sai | Giảm `max-detections` |

## 11. `resize-width` là gì?

`resize-width` giảm chiều rộng ảnh/frame trước khi detect.

Ví dụ:

```text
resize-width = 480
```

Nếu ảnh gốc rộng 1920 pixel, ảnh sẽ được resize về rộng 480 pixel và giữ nguyên tỉ lệ.

Ảnh nhỏ hơn giúp chạy nhanh hơn vì sliding window phải quét ít pixel hơn.

So sánh:

| `resize-width` | Tốc độ | Độ chi tiết |
|---:|---|---|
| `320` | Rất nhanh | Dễ mất người nhỏ |
| `480` | Nhanh | Tốt cho webcam |
| `640` | Cân bằng | Tốt cho demo |
| `960` trở lên | Chậm | Giữ nhiều chi tiết hơn |

## 12. `frame-skip` là gì?

`frame-skip` dùng cho video và webcam.

Nếu:

```text
frame-skip = 3
```

Thì chương trình chỉ detect mỗi 3 frame. Các frame ở giữa dùng lại bounding box gần nhất.

Ưu điểm:

- Tăng tốc đáng kể.
- Phù hợp realtime webcam.

Nhược điểm:

- Box có thể hơi trễ nếu người di chuyển nhanh.
- Không phải tracking thật sự, chỉ là dùng lại kết quả gần nhất.

Gợi ý:

```text
Máy mạnh: 1 hoặc 2
Bình thường: 3
Máy yếu: 5
```

## 13. `winStride` và `padding`

Trong `src/detector.py`, detector gọi:

```python
hog.detectMultiScale(
    image_rgb,
    float(hit_threshold),
    (8, 8),
    (8, 8),
    float(scale),
    0,
    False,
)
```

Các tham số positional tương ứng:

```text
hitThreshold
winStride
padding
scale
finalThreshold
useMeanshiftGrouping
```

Trong đó:

- `winStride = (8, 8)`: bước trượt cửa sổ HOG. Bước nhỏ quét kỹ hơn nhưng chậm hơn.
- `padding = (8, 8)`: thêm vùng đệm quanh cửa sổ khi tính HOG.
- `finalThreshold = 0`: project tự xử lý NMS, không phụ thuộc grouping mặc định.
- `useMeanshiftGrouping = False`: không dùng mean-shift grouping của OpenCV.

## 14. Vì sao output video cần nén?

HOG + SVM dùng sliding window nên khá tốn CPU với video lớn.

Project có `src/compress_video.py` để:

- Giảm width.
- Giảm FPS.
- Cắt video tối đa một số giây.
- Xuất MP4 nhẹ hơn.

Mặc định:

```text
width = 640
fps = 10
max-duration = 20 giây
crf = 28
```

Nếu có `ffmpeg`, project dùng `ffmpeg` để tạo MP4 H.264 dễ xem trong VS Code/browser. Nếu không có, project fallback bằng OpenCV.

## 15. Gradio app hoạt động như thế nào?

File:

```text
app.py
```

Web app gồm 3 tab:

1. `Image`: upload ảnh, gọi `detect_people()`, trả ảnh đã vẽ box.
2. `Video`: upload video, có thể nén trước, sau đó detect từng frame.
3. `Realtime Webcam`: dùng webcam streaming của Gradio, detect từng frame trực tiếp.

Tab webcam web không lưu từng frame ra ổ đĩa để tránh sinh quá nhiều file.

## 16. Khi nào nên dùng OpenCV webcam thay vì Gradio webcam?

Nên dùng:

```bash
python src/detect_webcam.py
```

khi:

- Browser không mở được camera.
- Gradio streaming bị lag.
- Muốn FPS ổn định hơn.
- Muốn demo realtime trực tiếp trên máy.

Gradio webcam tiện để demo trên web, nhưng OpenCV webcam thường ổn định hơn.

## 17. Bảng chỉnh tham số nhanh

| Vấn đề | Cách chỉnh |
|---|---|
| Quá nhiều box sai | Tăng `hit-threshold`, giảm `max-detections` |
| Nhiều box chồng lên cùng một người | Giảm `nms` |
| Không detect được người | Giảm `hit-threshold` về `0` hoặc `-0.5` |
| Video/webcam chậm | Giảm `resize-width`, tăng `frame-skip`, tăng `scale` |
| Người nhỏ bị bỏ sót | Tăng `resize-width`, giảm `scale` |
| Box bị giật trong webcam | Giảm `frame-skip` |

## 18. Hạn chế của HOG + SVM

HOG + SVM là phương pháp truyền thống nên có một số hạn chế:

- Dễ bỏ sót người bị che khuất.
- Dễ nhầm vật có hình dạng giống người.
- Kém ổn định với góc chụp lạ.
- Không mạnh bằng YOLO/CNN trong môi trường phức tạp.
- Chạy chậm hơn nếu ảnh/video quá lớn.

Tuy nhiên, phương pháp này rất phù hợp để học xử lý ảnh vì pipeline rõ ràng, dễ giải thích và không phụ thuộc deep learning.

## 19. Lệnh demo khuyến nghị khi nộp bài

Detect ảnh:

```bash
python src/detect_image.py --image demo/images/test.jpg --resize-width 640 --hit-threshold 0.0 --scale 1.05 --nms 0.35 --max-detections 10
```

Detect video:

```bash
python src/detect_video.py --video demo/videos/test.mp4 --auto-compress --frame-skip 3 --resize-width 640
```

Web app:

```bash
python app.py
```

Webcam OpenCV:

```bash
python src/detect_webcam.py --resize-width 480 --frame-skip 3 --hit-threshold 0.5 --scale 1.1
```
