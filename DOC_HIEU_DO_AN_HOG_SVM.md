# Tài Liệu Đọc Hiểu Đồ Án HOG + SVM Pedestrian Detection

Tài liệu này dùng để đọc hiểu đồ án từ lúc bắt đầu đến phiên bản hiện tại, tập trung vào lý thuyết, cách project hoạt động, nội dung slide thuyết trình và kịch bản demo. Mục tiêu là giúp nhóm nắm được bản chất của đồ án, không chỉ học thuộc lệnh chạy.

Nguồn tham chiếu chính:

- Báo cáo PDF: `2302700129-2302700121.pdf`.
- Slide thuyết trình: `BaoCaoCuoiKy_HOG_SVM_AcademicMinimal_Light.pptx`.
- Project demo hiện tại: `HOG-SVM-light-detection`.

---

## 1. Ý Tưởng Một Câu

Đồ án xây dựng hệ thống phát hiện người đi bộ bằng pipeline thị giác máy tính cổ điển:

```text
Ảnh / Video / Webcam
-> Image Gradient
-> HOG feature
-> Linear SVM
-> Non-Maximum Suppression
-> Bounding Box + Person score
```

Điểm quan trọng nhất: project không dùng YOLO, không dùng CNN, không dùng deep learning. Tất cả quyết định phát hiện đều có thể giải thích bằng gradient, histogram hướng cạnh, SVM tuyến tính và lọc box trùng lặp.

---

## 2. Quá Trình Phát Triển Từ Đầu Đến Hiện Tại

### 2.1. Giai đoạn đầu: bản đầy đủ tự train HOG + SVM

Lúc đầu project được xây theo hướng đầy đủ:

```text
HOG_SVM_Pedestrian_Detection
```

Bản này có các bước:

1. Chuẩn bị dataset positive/negative.
2. Hỗ trợ INRIA Person Dataset.
3. Crop người từ annotation.
4. Sinh negative samples.
5. Trích HOG bằng `skimage.feature.hog`.
6. Train `LinearSVC`.
7. Lưu `hog_svm_model.pkl` và `scaler.pkl`.
8. Evaluate accuracy, precision, recall, F1-score.
9. Detect ảnh, video, webcam bằng sliding window + SVM tự train.

Giá trị học thuật của bản này là nó thể hiện trọn vẹn quá trình huấn luyện mô hình: từ dữ liệu, đặc trưng, chuẩn hóa, train SVM, đánh giá đến ứng dụng detect.

### 2.2. Vấn đề gặp phải

Khi đưa vào demo, bản tự train có một số khó khăn:

- Cần dataset đủ lớn và đúng định dạng.
- Training và sliding window có thể chậm.
- Kết quả phụ thuộc nhiều vào chất lượng crop positive/negative.
- Video input lớn làm detect rất chậm.
- Cần một bản gọn, dễ chạy, dễ trình bày ổn định.

### 2.3. Giai đoạn hiện tại: bản light demo

Vì mục tiêu nộp bài và demo trực quan, nhóm tạo bản:

```text
HOG-SVM-light-detection
```

Bản hiện tại:

- Không train model.
- Dùng `cv2.HOGDescriptor()`.
- Dùng `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
- Vẫn là HOG + Linear SVM, nhưng SVM là pretrained của OpenCV.
- Hỗ trợ ảnh, video, webcam local và web app Gradio.
- Có auto-compress video, frame-skip, NMS và output MP4 để xem được.

Cần nói rõ khi thuyết trình:

```text
Bản demo hiện tại dùng HOG + Linear SVM pretrained của OpenCV.
Nó không phải deep learning.
Nó vẫn dùng pipeline HOG + SVM truyền thống, chỉ không tự train lại SVM trong bản demo light.
```

---

## 3. Bài Toán Cần Giải Quyết

Bài toán pedestrian detection hỏi:

```text
Trong ảnh hoặc frame video, vùng nào có khả năng chứa người đi bộ?
```

Output mong muốn:

- Bounding box quanh người.
- Label dạng `Person: score`.
- Ảnh/video/webcam có vẽ khung phát hiện.

Khó khăn:

- Người có nhiều kích thước khác nhau trong ảnh.
- Ánh sáng, màu quần áo, nền ảnh thay đổi.
- Người có thể bị che khuất, đi xa, nhỏ, hoặc đứng trong nền nhiều cạnh.
- Sliding window có thể sinh nhiều box quanh cùng một người.
- Video có nhiều frame nên xử lý chậm.

---

## 4. Lý Thuyết 1: Image Gradient

### 4.1. Gradient là gì?

Gradient mô tả sự thay đổi cường độ sáng trong ảnh. Với ảnh xám \(I(x,y)\):

```text
Gx = partial I / partial x
Gy = partial I / partial y
```

Trong code hoặc lý thuyết, \(G_x\) và \(G_y\) có thể tính bằng Sobel.

Từ đó tính:

```text
magnitude = sqrt(Gx^2 + Gy^2)
orientation = atan2(Gy, Gx)
```

Ý nghĩa:

- `magnitude`: cạnh ở điểm đó mạnh hay yếu.
- `orientation`: cạnh đó theo hướng nào.

### 4.2. Vì sao gradient quan trọng trong bài toán người đi bộ?

Người có hình dáng đặc trưng:

- Đầu.
- Vai.
- Thân.
- Tay.
- Chân.

Những thành phần này tạo ra biên/cạnh trong ảnh. Màu áo quần có thể thay đổi, ánh sáng có thể thay đổi, nhưng cấu trúc biên của dáng người tương đối ổn định hơn. Vì vậy, thay vì dùng trực tiếp pixel màu, HOG dùng gradient để mô tả hình dáng.

Nói ngắn gọn khi thuyết trình:

```text
Gradient giúp biến ảnh thành thông tin về cạnh và hướng cạnh. HOG dựa trên gradient để mô tả hình dáng người.
```

---

## 5. Lý Thuyết 2: HOG

### 5.1. HOG là gì?

HOG là viết tắt của:

```text
Histogram of Oriented Gradients
```

HOG mô tả phân bố hướng gradient trong các vùng nhỏ của ảnh.

Thay vì hỏi:

```text
Pixel này màu gì?
```

HOG hỏi:

```text
Trong vùng nhỏ này, các cạnh chủ yếu theo hướng nào?
```

### 5.2. Các bước tính HOG

Với một cửa sổ ảnh:

1. Chuyển về grayscale.
2. Tính gradient \(G_x\), \(G_y\).
3. Tính magnitude và orientation.
4. Chia ảnh thành các cell, thường là `8 x 8`.
5. Trong mỗi cell, gom hướng gradient vào histogram.
6. Gom nhiều cell thành block, thường là `2 x 2` cell.
7. Chuẩn hóa block để giảm ảnh hưởng ánh sáng.
8. Nối các histogram đã chuẩn hóa thành vector HOG.

### 5.3. Tham số HOG quan trọng

OpenCV HOG people detector thường dùng:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `winSize` | `64 x 128` | Cửa sổ phát hiện người |
| `cellSize` | `8 x 8` | Vùng nhỏ để lập histogram |
| `blockSize` | `16 x 16` | Block gồm `2 x 2` cell |
| `blockStride` | `8 x 8` | Bước trượt block |
| `nbins` | `9` | Số bin hướng gradient |

### 5.4. Vì sao vector HOG có 3780 chiều?

Với cửa sổ `64 x 128`:

```text
Số cell ngang = 64 / 8 = 8
Số cell dọc   = 128 / 8 = 16
```

Mỗi block gồm `2 x 2` cell:

```text
Số block ngang = 8 - 2 + 1 = 7
Số block dọc   = 16 - 2 + 1 = 15
Tổng block     = 7 x 15 = 105
```

Mỗi block có:

```text
2 x 2 x 9 = 36 giá trị
```

Độ dài vector HOG:

```text
105 x 36 = 3780
```

Đây là con số nên nhớ khi bị hỏi về HOG.

### 5.5. Chuẩn hóa block để làm gì?

Ánh sáng có thể làm magnitude gradient thay đổi. Nếu không chuẩn hóa, cùng một dáng người nhưng ánh sáng khác nhau có thể tạo vector khác nhau mạnh.

Chuẩn hóa block giúp HOG tập trung vào:

```text
phân bố hướng cạnh
```

hơn là:

```text
độ sáng tuyệt đối
```

Công thức chuẩn hóa L2 dạng tổng quát:

```text
v_norm = v / sqrt(||v||^2 + epsilon^2)
```

Cần nhấn mạnh:

```text
HOG bền vững hơn pixel gốc với thay đổi ánh sáng cục bộ, nhưng vẫn không mạnh bằng deep learning khi tư thế người phức tạp.
```

---

## 6. Lý Thuyết 3: Linear SVM

### 6.1. SVM làm gì trong project?

HOG biến mỗi cửa sổ ảnh thành vector đặc trưng:

```text
x = [hog_1, hog_2, ..., hog_3780]
```

SVM nhận vector này và tính điểm:

```text
score = w^T x + b
```

Trong đó:

- `x`: vector HOG của cửa sổ ảnh.
- `w`: vector trọng số SVM.
- `b`: bias.
- `score`: decision score.

Nếu:

```text
score > hit-threshold
```

thì cửa sổ đó được xem là ứng viên có người.

### 6.2. Score `Person: ???` là gì?

Khi output hiện:

```text
Person: 1.14
```

thì `1.14` là decision score của Linear SVM:

```text
score = w^T x + b
```

Nó không phải:

- Xác suất.
- Phần trăm tin cậy.
- Confidence của YOLO.

Nó là giá trị cho biết vector HOG nằm về phía lớp `person` mạnh hay yếu so với biên quyết định.

Cách nói khi thuyết trình:

```text
Score càng cao thì cửa sổ càng giống người theo mô hình SVM. Tuy nhiên score không đảm bảo đúng tuyệt đối vì nền ảnh có thể có cấu trúc cạnh giống người.
```

### 6.3. Score nằm trong khoảng nào?

Về lý thuyết:

```text
score có thể từ âm vô cực đến dương vô cực
```

Trong thực tế với OpenCV HOG people detector, score hay gặp trong khoảng:

```text
-1 đến 3 hoặc cao hơn
```

Cách hiểu:

| Score | Diễn giải gần đúng |
|---:|---|
| `< 0` | Yếu, thường không chắc |
| `gần 0` | Sát biên quyết định |
| `0.5 - 1.0` | Có khả năng là người nhưng chưa mạnh |
| `> 1.0` | Khá chắc |
| `> 2.0` | Mạnh hơn, thường là dáng người rõ |

Nếu `hit-threshold = 0.5`, chỉ giữ box có:

```text
score > 0.5
```

---

## 7. Lý Thuyết 4: Sliding Window Và Scale Pyramid

### 7.1. Vì sao cần sliding window?

SVM chỉ phân loại một cửa sổ có kích thước chuẩn. Nhưng trong ảnh gốc, người có thể nằm ở bất kỳ vị trí nào.

Vì vậy detector phải quét ảnh bằng nhiều cửa sổ:

```text
W1, W2, W3, ..., Wk
```

Mỗi cửa sổ:

```text
Wk -> HOG -> SVM score
```

### 7.2. Vì sao cần scale pyramid?

Người trong ảnh có thể:

- Gần camera: to.
- Xa camera: nhỏ.

Detector cần quét nhiều kích thước. Scale pyramid làm việc này bằng cách thay đổi kích thước ảnh qua nhiều tầng.

Tham số `scale` quyết định khoảng cách giữa các tầng.

| `scale` | Tác động |
|---:|---|
| `1.05` | Quét kỹ, chậm hơn, ít bỏ sót hơn |
| `1.1` | Cân bằng, phù hợp video/webcam |
| `1.2` | Nhanh hơn, có thể bỏ sót |

Nói ngắn gọn:

```text
scale nhỏ thì detector nhìn kỹ hơn, nhưng mất thời gian hơn.
```

---

## 8. Lý Thuyết 5: IoU Và NMS

### 8.1. Vì sao có nhiều bounding box quanh một người?

Sliding window quét rất nhiều cửa sổ gần nhau. Nếu một người nằm trong ảnh, nhiều cửa sổ gần vị trí người đều có HOG giống người và đều vượt ngưỡng SVM.

Kết quả là:

```text
Một người có thể bị bao quanh bởi nhiều box.
```

Đây là lý do slide kết quả ảnh tĩnh có hình:

- Trước lọc: nhiều bounding box.
- Sau lọc: một bounding box chính.

### 8.2. IoU là gì?

IoU là Intersection over Union:

```text
IoU = Area(B1 giao B2) / Area(B1 hợp B2)
```

Nếu hai box trùng nhau nhiều, IoU cao.

### 8.3. NMS làm gì?

NMS là Non-Maximum Suppression.

Ý tưởng:

1. Sắp xếp box theo score giảm dần.
2. Giữ box có score cao nhất.
3. Loại các box còn lại nếu trùng lặp quá nhiều với box đã giữ.
4. Lặp lại đến khi hết box.

Trong project:

```text
src/nms.py
```

tự cài đặt IoU và NMS.

Cần nhớ:

- `nms` thấp: lọc mạnh, box gọn hơn.
- `nms` cao: giữ nhiều box hơn, hữu ích nếu nhiều người đứng gần nhau.

---

## 9. Project Hiện Tại Hoạt Động Như Thế Nào?

### 9.1. File quan trọng

| File | Vai trò |
|---|---|
| `app.py` | Web app Gradio cho image, video, realtime webcam |
| `src/detector.py` | Lõi phát hiện HOG + SVM |
| `src/nms.py` | IoU và Non-Maximum Suppression |
| `src/detect_image.py` | Detect ảnh bằng command line |
| `src/detect_video.py` | Detect video, auto-compress, frame-skip, output MP4 |
| `src/detect_webcam.py` | Webcam realtime bằng OpenCV |
| `src/compress_video.py` | Nén video bằng ffmpeg hoặc OpenCV fallback |
| `src/utils.py` | Resize, vẽ box, vẽ FPS, lưu ảnh/video |
| `src/config.py` | Giá trị mặc định và đường dẫn |

### 9.2. Lõi detect trong `detector.py`

Code tạo detector:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

Hàm chính:

```python
detect_people(
    image,
    resize_width,
    hit_threshold,
    scale,
    nms_threshold,
    max_detections
)
```

Luồng xử lý:

1. Kiểm tra ảnh input.
2. Chuyển về RGB/uint8 nếu cần.
3. Resize giữ tỉ lệ.
4. Gọi `hog.detectMultiScale`.
5. Đổi `(x, y, w, h)` thành `[x1, y1, x2, y2]`.
6. Gọi NMS.
7. Sắp xếp theo score.
8. Giữ tối đa `max_detections`.
9. Vẽ box và label `Person: score`.

### 9.3. Vì sao dùng `resize_width`?

Ảnh/video lớn làm sliding window quét rất nhiều cửa sổ. Resize giúp:

- Giảm số pixel.
- Giảm số cửa sổ cần quét.
- Tăng tốc video/webcam.

Đổi lại:

- Nếu resize quá nhỏ, người nhỏ có thể mất chi tiết.

### 9.4. Vì sao video cần auto-compress?

Video có:

- Nhiều frame.
- Độ phân giải cao.
- FPS cao.

Nên HOG + SVM rất chậm nếu xử lý trực tiếp.

`compress_video.py` giảm:

- Width về 640 hoặc 480.
- FPS về 10.
- Thời lượng về 20 giây.

Nếu có ffmpeg:

```text
mp4 H.264, yuv420p
```

giúp xem được trong VS Code/browser.

### 9.5. Vì sao cần `frame-skip`?

Nếu `frame-skip = 3`:

```text
Chỉ detect mỗi 3 frame.
Frame giữa dùng lại box gần nhất.
```

Tác dụng:

- Nhanh hơn.
- Webcam mượt hơn.

Hạn chế:

- Nếu người di chuyển nhanh, box có thể chậm cập nhật.

---

## 10. Kết Nối Với Slide Thuyết Trình

Slide có 15 trang. Khi thuyết trình, nên nắm mạch như sau.

### Slide 1: Tiêu đề

Cần nói:

```text
Đồ án phát hiện người đi bộ bằng HOG + SVM, tập trung vào pipeline thị giác máy tính cổ điển, không dùng deep learning.
```

### Slide 2: Nội dung trình bày

Đây là agenda:

1. Giới thiệu.
2. Phân công.
3. Lý thuyết.
4. Thiết kế hệ thống.
5. Tham số.
6. Kết quả.
7. Đánh giá.
8. Hạn chế.
9. Kết luận.

### Slide 3: Giới thiệu và mục tiêu

Cần nói được:

- Pedestrian detection là bài toán nền tảng.
- HOG + SVM có giá trị học thuật vì dễ giải thích.
- Project demo được ảnh, video, webcam, web.

### Slide 4: Phân công

Theo slide:

- Dũng: lập trình, triển khai, app, image/video/webcam, demo.
- Kiệt: cơ sở toán, gradient, HOG, SVM, IoU/NMS, báo cáo lý thuyết.

Khi bị hỏi, mỗi thành viên nên bảo vệ đúng phần mình, nhưng cả hai vẫn nên nắm pipeline chung.

### Slide 5: Gradient

Cần nhớ công thức:

```text
Gx = partial I / partial x
Gy = partial I / partial y
m = sqrt(Gx^2 + Gy^2)
theta = atan2(Gy, Gx)
```

Ý chính:

```text
Gradient biến ảnh thành thông tin cạnh. Dáng người được nhận qua cấu trúc cạnh.
```

### Slide 6: HOG và Linear SVM

Cần nhớ:

```text
HOG vector = 3780 chiều
SVM score = w^T x + b
```

Nói rõ:

```text
Person score không phải xác suất.
```

### Slide 7: Thiết kế hệ thống

Pipeline trên slide:

```text
I -> {Wk} -> phi(Wk) -> f(phi(Wk)) -> {Bj}_NMS
```

Diễn giải:

- `I`: ảnh input.
- `Wk`: các cửa sổ sliding window.
- `phi(Wk)`: vector HOG.
- `f(phi(Wk))`: SVM score.
- `{Bj}_NMS`: box cuối sau NMS.

### Slide 8: Phân tích tham số

Đây là slide rất dễ bị hỏi.

Cần nắm:

- `hit-threshold`: ngưỡng score SVM.
- `scale`: mức quét pyramid.
- `nms`: lọc box trùng.
- `resize-width`: tốc độ so với chi tiết.
- `frame-skip`: tốc độ video/webcam.

### Slide 9: Kết quả ảnh tĩnh

Cần nói:

```text
Nhiều box quanh một người là hiện tượng bình thường của sliding window.
NMS giúp lọc lại còn một box chính.
```

Đây là điểm báo cáo trung thực: không chỉ show kết quả đẹp, mà show cả hiện tượng cần xử lý.

### Slide 10: Ảnh hưởng resize-width

Cần nói:

- Width tăng: giữ chi tiết, nhưng chậm hơn.
- Width giảm: nhanh hơn, nhưng dễ mất người nhỏ.

### Slide 11: Video và webcam

Cần nói:

- Video có nhiều người, score khác nhau theo độ rõ dáng người.
- Webcam có thể score âm nếu hạ `hit-threshold` thấp.
- Score âm không phải lỗi, mà do ngưỡng phát hiện được hạ xuống.

### Slide 12: Đánh giá và thảo luận

Ưu điểm:

- Dễ giải thích.
- Không cần GPU.
- Một lõi dùng cho ảnh/video/webcam.
- Tham số rõ nghĩa.

Hạn chế:

- Nhạy với che khuất/tư thế lạ.
- False positive với nền nhiều cạnh.
- Chậm với video lớn.
- Cần NMS để xử lý nhiều box.

### Slide 13: Hướng cải tiến

Cần nói:

- Tự train SVM riêng để so sánh.
- Đánh giá precision/recall/F1.
- Bổ sung dataset đầy đủ.
- Thêm tracking để ổn định video.

### Slide 14: Kết luận

Kết luận nên nói:

```text
Đồ án đã minh họa được một pipeline thị giác máy tính cổ điển: gradient tạo thông tin cạnh, HOG gom cạnh thành đặc trưng hình dạng, SVM biến đặc trưng thành score, NMS biến nhiều candidate thành box cuối.
```

### Slide 15: Cảm ơn

Chuẩn bị cho phần hỏi đáp.

---

## 11. Kịch Bản Demo Khuyến Nghị

### 11.1. Demo web app

Chạy:

```bash
python app.py
```

Mở:

```text
http://127.0.0.1:7861
```

Thứ tự demo:

1. Tab Image.
2. Upload `demo/images/test.jpg`.
3. Chạy detect với tham số mặc định.
4. Giải thích `Person: score`.
5. Chỉnh `max-detections` để thấy box ít/hơn.
6. Tab Video.
7. Upload `demo/videos/test.mp4`.
8. Bật auto-compress.
9. Chạy detect.
10. Nói về frame-skip và output MP4.
11. Tab Realtime Webcam nếu máy cho phép.

### 11.2. Demo command line

Detect ảnh:

```bash
python src/detect_image.py --image demo/images/test.jpg --resize-width 640 --hit-threshold 0.0 --scale 1.05 --nms 0.35 --max-detections 10
```

Detect video:

```bash
python src/detect_video.py --video demo/videos/test.mp4 --auto-compress --frame-skip 3 --resize-width 640
```

Webcam local:

```bash
python src/detect_webcam.py --resize-width 480 --frame-skip 3 --hit-threshold 0.5 --scale 1.1
```

### 11.3. Khi demo bị chậm

Nói và làm:

```text
Giảm resize-width về 480 hoặc 320.
Tăng frame-skip lên 5.
Tăng scale lên 1.2.
Bật auto-compress cho video.
```

### 11.4. Khi nhiều box sai

Nói và làm:

```text
Tăng hit-threshold.
Giảm max-detections.
Điều chỉnh nms để lọc box trùng.
```

### 11.5. Khi không detect được người

Nói và làm:

```text
Giảm hit-threshold về 0 hoặc -0.5.
Tăng resize-width.
Giảm scale về 1.05 để quét kỹ hơn.
```

---

## 12. Các Tham Số Cần Thuộc

| Tham số | Nói ngắn gọn | Tăng lên thì sao | Giảm xuống thì sao |
|---|---|---|---|
| `hit-threshold` | Ngưỡng score SVM | Ít box sai hơn, dễ bỏ sót | Nhạy hơn, dễ sai hơn |
| `scale` | Bước pyramid | Nhanh hơn, dễ bỏ sót | Kỹ hơn, chậm hơn |
| `nms` | Ngưỡng IoU lọc box | Giữ nhiều box hơn | Lọc mạnh hơn |
| `resize-width` | Độ rộng xử lý | Rõ hơn, chậm hơn | Nhanh hơn, mất chi tiết |
| `frame-skip` | Detect mỗi N frame | Nhanh hơn, box trễ hơn | Mượt/chính xác hơn, chậm hơn |
| `max-detections` | Số box tối đa | Giữ được nhiều người | Gọn hơn, có thể mất người |

---

## 13. Câu Hỏi Vấn Đáp Dễ Gặp

### Câu 1: Tại sao không dùng YOLO?

Trả lời:

```text
Mục tiêu môn học là hiểu pipeline xử lý ảnh truyền thống. HOG + SVM cho phép giải thích từng bước: gradient, histogram, SVM, NMS. YOLO/CNN mạnh hơn nhưng là deep learning và không dùng theo yêu cầu đồ án.
```

### Câu 2: HOG khác gì so với dùng pixel gốc?

Trả lời:

```text
Pixel gốc phụ thuộc màu và ánh sáng. HOG dựa trên hướng gradient nên tập trung vào hình dáng và cấu trúc biên, phù hợp hơn với phát hiện người.
```

### Câu 3: Vì sao HOG vector có 3780 chiều?

Trả lời:

```text
Ảnh 64x128 chia cell 8x8 tạo 8x16 cell. Block 2x2 cell nên có 7x15 block. Mỗi block có 2x2x9 = 36 giá trị. Tổng 7x15x36 = 3780.
```

### Câu 4: `Person: 1.14` có phải 114% không?

Trả lời:

```text
Không. Đó là decision score của Linear SVM: score = w^T x + b. Nó không phải xác suất. Score càng cao thì cửa sổ càng giống lớp người theo SVM.
```

### Câu 5: Vì sao có score âm mà vẫn có box?

Trả lời:

```text
Vì hit-threshold có thể được đặt âm, ví dụ -0.7. Khi ngưỡng thấp hơn score âm, detector vẫn giữ box. Điều này tăng độ nhạy nhưng dễ false positive hơn.
```

### Câu 6: Tại sao một người có nhiều box?

Trả lời:

```text
Sliding window quét nhiều vị trí và scale gần nhau. Nhiều cửa sổ cùng bao quanh một người và cùng vượt ngưỡng SVM. Vì vậy cần NMS để giữ box tốt nhất.
```

### Câu 7: NMS dựa vào đâu để loại box?

Trả lời:

```text
NMS dùng IoU. Nếu hai box chồng lấp quá ngưỡng, box có score thấp hơn bị loại.
```

### Câu 8: Tại sao video chậm?

Trả lời:

```text
Video có nhiều frame. Mỗi frame lại cần sliding window và scale pyramid, nên chi phí lớn. Project giảm bằng auto-compress, resize-width và frame-skip.
```

### Câu 9: Bản hiện tại có tự train SVM không?

Trả lời:

```text
Bản demo light hiện tại không tự train. Nó dùng pretrained HOG + SVM people detector của OpenCV. Tuy nhiên pipeline vẫn là HOG + Linear SVM. Bản phát triển trước đó đã có hướng tự train với dataset INRIA.
```

### Câu 10: Hạn chế lớn nhất của HOG + SVM là gì?

Trả lời:

```text
HOG + SVM phụ thuộc vào hình dáng biên tương đối chuẩn, nên kém với người bị che khuất, tư thế lạ, nền nhiều cạnh, hoặc video độ phân giải lớn. Độ chính xác không bằng deep learning hiện đại.
```

---

## 14. Cách Nói Về Phân Công

Theo slide:

### Nguyễn Lê Trí Dũng

Tập trung nói:

- Xây dựng code project.
- `detector.py`, `nms.py`, `utils.py`.
- Luồng detect ảnh/video/webcam.
- Auto-compress, frame-skip, xuất MP4.
- Gradio app.
- Demo và tinh chỉnh tham số.

### Đỗ Anh Kiệt

Tập trung nói:

- Gradient: Sobel, magnitude, orientation.
- HOG: cell, block, bin, chuẩn hóa, vector 3780.
- SVM: score, margin, soft-margin.
- IoU và NMS.
- Giải thích tham số và trả lời vấn đáp lý thuyết.

Nhưng cả hai nên nắm:

```text
Gradient -> HOG -> SVM -> NMS -> Bounding Box
```

---

## 15. Bản Chất Của Demo Ảnh, Video, Webcam

### 15.1. Ảnh tĩnh

Ảnh tĩnh dùng để giải thích:

- Box quanh người.
- Score.
- Nhiều box trước/sau NMS.
- Ảnh hưởng của `hit-threshold`, `nms`, `max-detections`.

### 15.2. Video

Video dùng để giải thích:

- Detect trên từng frame.
- Nhiều người ở khoảng cách khác nhau.
- Score khác nhau theo độ rõ của hình dáng.
- Auto-compress để giảm kích thước.
- Frame-skip để tăng tốc.

### 15.3. Webcam

Webcam dùng để giải thích:

- Realtime yêu cầu độ trễ thấp.
- Nên dùng `resize-width=480`, `frame-skip=3`.
- Gradio webcam có thể tiện nhưng OpenCV webcam ổn định hơn.

---

## 16. Điều Cần Nhấn Mạnh Khi Thuyết Trình

1. Đồ án không dùng deep learning.
2. HOG + SVM là pipeline có thể giải thích được.
3. Gradient là nền tảng để HOG mô tả hình dáng.
4. HOG vector của OpenCV people detector có 3780 đặc trưng.
5. SVM score là \(w^T x + b\), không phải xác suất.
6. Sliding window sinh nhiều candidate, NMS lọc thành box cuối.
7. Video chậm là do số frame và số cửa sổ quét lớn.
8. Auto-compress và frame-skip là tối ưu thực tế cho demo.
9. Bản hiện tại dùng pretrained detector của OpenCV để demo nhẹ và ổn định.
10. Hạn chế của HOG + SVM là một phần của báo cáo, không nên che giấu.

---

## 17. Tóm Tắt Siêu Ngắn Để Học Thuộc

```text
Project phát hiện người đi bộ bằng HOG + Linear SVM pretrained của OpenCV.
Ảnh được resize, detector quét sliding window qua nhiều scale.
Mỗi cửa sổ được trích HOG: gradient -> cell -> block -> histogram -> vector 3780 chiều.
Linear SVM tính score = w^T x + b.
Nếu score vượt hit-threshold thì cửa sổ là candidate người.
Nhiều candidate trùng nhau được lọc bằng NMS dựa trên IoU.
Kết quả vẽ bounding box Person: score.
Ảnh/video/webcam dùng chung detector.py.
Video được tăng tốc bằng auto-compress và frame-skip.
```

---

## 18. Checklist Trước Khi Nộp/Thuyết Trình

- Chạy được `python app.py`.
- Tab Image detect được ảnh mẫu.
- Tab Video detect được video, output xem được.
- Webcam local biết lệnh chạy nếu Gradio webcam lag.
- Nhớ công thức gradient, HOG 3780, SVM score, IoU.
- Giải thích được `Person: score`.
- Giải thích được vì sao nhiều box và NMS xử lý ra sao.
- Giải thích được vì sao project không dùng deep learning.
- Giải thích được hạn chế và hướng cải tiến.

