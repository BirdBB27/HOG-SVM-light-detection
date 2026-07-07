# Doc Hieu Do An HOG + SVM Pedestrian Detection

Tai lieu nay dung de doc hieu do an tu luc bat dau den phien ban hien tai, tap trung vao ly thuyet, cach project hoat dong, noi dung slide thuyet trinh va kich ban demo. Muc tieu la giup nhom nam duoc ban chat, khong chi hoc thuoc lenh chay.

Nguon tham chieu chinh:

- Bao cao PDF: `2302700129-2302700121.pdf`.
- Slide thuyet trinh: `BaoCaoCuoiKy_HOG_SVM_AcademicMinimal_Light.pptx`.
- Project demo hien tai: `HOG-SVM-light-detection`.

---

## 1. Y tuong mot cau

Do an xay dung he thong phat hien nguoi di bo bang pipeline thi giac may tinh co dien:

```text
Image / Video / Webcam
-> Image Gradient
-> HOG feature
-> Linear SVM
-> Non-Maximum Suppression
-> Bounding Box + Person score
```

Diem quan trong nhat: project khong dung YOLO, khong dung CNN, khong dung deep learning. Tat ca quyet dinh phat hien deu co the giai thich bang gradient, histogram huong canh, SVM tuyen tinh va loc box trung lap.

---

## 2. Qua trinh phat trien tu dau den hien tai

### 2.1. Giai doan dau: ban day du tu train HOG + SVM

Luc dau project duoc xay theo huong day du:

```text
HOG_SVM_Pedestrian_Detection
```

Ban nay co cac buoc:

1. Chuan bi dataset positive/negative.
2. Ho tro INRIA Person Dataset.
3. Crop nguoi tu annotation.
4. Sinh negative samples.
5. Trich HOG bang `skimage.feature.hog`.
6. Train `LinearSVC`.
7. Luu `hog_svm_model.pkl` va `scaler.pkl`.
8. Evaluate accuracy, precision, recall, F1-score.
9. Detect anh, video, webcam bang sliding window + SVM tu train.

Gia tri hoc thuat cua ban nay la no the hien tron ven qua trinh huan luyen mo hinh: tu du lieu, dac trung, chuan hoa, train SVM, danh gia.

### 2.2. Van de gap phai

Khi dua vao demo, ban tu train co mot so kho khan:

- Can dataset du lon va dung dinh dang.
- Training va sliding window co the cham.
- Ket qua phu thuoc nhieu vao chat luong crop positive/negative.
- Video input lon lam detect rat cham.
- Can mot ban gon, de chay, de trinh bay on dinh.

### 2.3. Giai doan hien tai: ban light demo

Vi muc tieu nop bai va demo truc quan, nhom tao ban:

```text
HOG-SVM-light-detection
```

Ban hien tai:

- Khong train model.
- Dung `cv2.HOGDescriptor()`.
- Dung `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
- Van la HOG + Linear SVM, nhung SVM la pretrained cua OpenCV.
- Ho tro anh, video, webcam local va web app Gradio.
- Co auto-compress video, frame-skip, NMS va output MP4 de xem duoc.

Can noi ro khi thuyet trinh:

```text
Ban demo hien tai dung HOG + Linear SVM pretrained cua OpenCV.
No khong phai deep learning.
No van dung pipeline HOG + SVM truyen thong, chi khong tu train lai SVM trong ban demo light.
```

---

## 3. Bai toan can giai quyet

Bai toan pedestrian detection hoi:

```text
Trong anh hoac frame video, vung nao co kha nang chua nguoi di bo?
```

Output mong muon:

- Bounding box quanh nguoi.
- Label dang `Person: score`.
- Anh/video/webcam co ve khung phat hien.

Kho khan:

- Nguoi co nhieu kich thuoc khac nhau trong anh.
- Anh sang, mau quan ao, nen anh thay doi.
- Nguoi co the bi che khuat, di xa, nho, hoac dung trong nen nhieu canh.
- Sliding window co the sinh nhieu box quanh cung mot nguoi.
- Video co nhieu frame nen xu ly cham.

---

## 4. Ly thuyet 1: Image Gradient

### 4.1. Gradient la gi?

Gradient mo ta su thay doi cuong do sang trong anh. Voi anh xam \(I(x,y)\):

```text
Gx = partial I / partial x
Gy = partial I / partial y
```

Trong code hoac ly thuyet, \(G_x\) va \(G_y\) co the tinh bang Sobel.

Tu do tinh:

```text
magnitude = sqrt(Gx^2 + Gy^2)
orientation = atan2(Gy, Gx)
```

Y nghia:

- `magnitude`: canh o diem do manh hay yeu.
- `orientation`: canh do theo huong nao.

### 4.2. Vi sao gradient quan trong trong bai toan nguoi di bo?

Nguoi co hinh dang dac trung:

- Dau.
- Vai.
- Than.
- Tay.
- Chan.

Nhung thanh phan nay tao ra bien/canh trong anh. Mau ao quan co the thay doi, anh sang co the thay doi, nhung cau truc bien cua dang nguoi tuong doi on dinh hon. Vi vay, thay vi dung truc tiep pixel mau, HOG dung gradient de mo ta hinh dang.

Noi ngan gon khi thuyet trinh:

```text
Gradient giup bien anh thanh thong tin ve canh va huong canh. HOG dua tren gradient de mo ta hinh dang nguoi.
```

---

## 5. Ly thuyet 2: HOG

### 5.1. HOG la gi?

HOG la viet tat cua:

```text
Histogram of Oriented Gradients
```

HOG mo ta phan bo huong gradient trong cac vung nho cua anh.

Thay vi hoi:

```text
Pixel nay mau gi?
```

HOG hoi:

```text
Trong vung nho nay, cac canh chu yeu theo huong nao?
```

### 5.2. Cac buoc tinh HOG

Voi mot cua so anh:

1. Chuyen ve grayscale.
2. Tinh gradient \(G_x\), \(G_y\).
3. Tinh magnitude va orientation.
4. Chia anh thanh cac cell, thuong la `8 x 8`.
5. Trong moi cell, gom huong gradient vao histogram.
6. Gom nhieu cell thanh block, thuong la `2 x 2` cell.
7. Chuan hoa block de giam anh huong anh sang.
8. Noi cac histogram da chuan hoa thanh vector HOG.

### 5.3. Tham so HOG quan trong

OpenCV HOG people detector thuong dung:

| Tham so | Gia tri | Y nghia |
|---|---:|---|
| `winSize` | `64 x 128` | Cua so phat hien nguoi |
| `cellSize` | `8 x 8` | Vung nho de lap histogram |
| `blockSize` | `16 x 16` | Block gom `2 x 2` cell |
| `blockStride` | `8 x 8` | Buoc truot block |
| `nbins` | `9` | So bin huong gradient |

### 5.4. Vi sao vector HOG co 3780 chieu?

Voi cua so `64 x 128`:

```text
So cell ngang = 64 / 8 = 8
So cell doc   = 128 / 8 = 16
```

Moi block gom `2 x 2` cell:

```text
So block ngang = 8 - 2 + 1 = 7
So block doc   = 16 - 2 + 1 = 15
Tong block     = 7 x 15 = 105
```

Moi block co:

```text
2 x 2 x 9 = 36 gia tri
```

Do dai vector HOG:

```text
105 x 36 = 3780
```

Day la con so nen nho khi bi hoi ve HOG.

### 5.5. Chuan hoa block de lam gi?

Anh sang co the lam magnitude gradient thay doi. Neu khong chuan hoa, cung mot dang nguoi nhung anh sang khac nhau co the tao vector khac nhau manh.

Chuan hoa block giup HOG tap trung vao:

```text
phan bo huong canh
```

hon la:

```text
do sang tuyet doi
```

Cong thuc chuan hoa L2 dang tong quat:

```text
v_norm = v / sqrt(||v||^2 + epsilon^2)
```

Can nhan manh:

```text
HOG ben vung hon pixel goc voi thay doi anh sang cuc bo, nhung van khong manh bang deep learning khi tu the nguoi phuc tap.
```

---

## 6. Ly thuyet 3: Linear SVM

### 6.1. SVM lam gi trong project?

HOG bien moi cua so anh thanh vector dac trung:

```text
x = [hog_1, hog_2, ..., hog_3780]
```

SVM nhan vector nay va tinh diem:

```text
score = w^T x + b
```

Trong do:

- `x`: vector HOG cua cua so anh.
- `w`: vector trong so SVM.
- `b`: bias.
- `score`: decision score.

Neu:

```text
score > hit-threshold
```

thi cua so do duoc xem la ung vien co nguoi.

### 6.2. Score `Person: ???` la gi?

Khi output hien:

```text
Person: 1.14
```

thi `1.14` la decision score cua Linear SVM:

```text
score = w^T x + b
```

No khong phai:

- Xac suat.
- Phan tram tin cay.
- Confidence cua YOLO.

No la gia tri cho biet vector HOG nam ve phia lop `person` manh hay yeu so voi bien quyet dinh.

Cach noi khi thuyet trinh:

```text
Score cang cao thi cua so cang giong nguoi theo mo hinh SVM. Tuy nhien score khong dam bao dung tuyet doi vi nen anh co the co cau truc canh giong nguoi.
```

### 6.3. Score nam trong khoang nao?

Ve ly thuyet:

```text
score co the tu -vo cuc den +vo cuc
```

Trong thuc te voi OpenCV HOG people detector, score hay gap trong khoang:

```text
-1 den 3 hoac cao hon
```

Cach hieu:

| Score | Dien giai gan dung |
|---:|---|
| `< 0` | Yeu, thuong khong chac |
| `gan 0` | Sat bien quyet dinh |
| `0.5 - 1.0` | Co kha nang la nguoi nhung chua manh |
| `> 1.0` | Kha chac |
| `> 2.0` | Manh hon, thuong la dang nguoi ro |

Neu `hit-threshold = 0.5`, chi giu box co:

```text
score > 0.5
```

---

## 7. Ly thuyet 4: Sliding Window va Scale Pyramid

### 7.1. Vi sao can sliding window?

SVM chi phan loai mot cua so co kich thuoc chuan. Nhung trong anh goc, nguoi co the nam o bat ky vi tri nao.

Vi vay detector phai quet anh bang nhieu cua so:

```text
W1, W2, W3, ..., Wk
```

Moi cua so:

```text
Wk -> HOG -> SVM score
```

### 7.2. Vi sao can scale pyramid?

Nguoi trong anh co the:

- Gan camera: to.
- Xa camera: nho.

Detector can quet nhieu kich thuoc. Scale pyramid lam viec nay bang cach thay doi kich thuoc anh qua nhieu tang.

Tham so `scale` quyet dinh khoang cach giua cac tang.

| `scale` | Tac dong |
|---:|---|
| `1.05` | Quet ky, cham hon, it bo sot hon |
| `1.1` | Can bang, phu hop video/webcam |
| `1.2` | Nhanh hon, co the bo sot |

Noi ngan gon:

```text
scale nho thi detector nhin ky hon, nhung mat thoi gian hon.
```

---

## 8. Ly thuyet 5: IoU va NMS

### 8.1. Vi sao co nhieu bounding box quanh mot nguoi?

Sliding window quet rat nhieu cua so gan nhau. Neu mot nguoi nam trong anh, nhieu cua so gan vi tri nguoi deu co HOG giong nguoi va deu vuot nguong SVM.

Ket qua la:

```text
Mot nguoi co the bi bao quanh boi nhieu box.
```

Day la ly do slide ket qua anh tinh co hinh:

- Truoc loc: nhieu bounding box.
- Sau loc: mot bounding box chinh.

### 8.2. IoU la gi?

IoU la Intersection over Union:

```text
IoU = Area(B1 giao B2) / Area(B1 hop B2)
```

Neu hai box trung nhau nhieu, IoU cao.

### 8.3. NMS lam gi?

NMS la Non-Maximum Suppression.

Y tuong:

1. Sap xep box theo score giam dan.
2. Giu box co score cao nhat.
3. Loai cac box con lai neu trung lap qua nhieu voi box da giu.
4. Lap lai den khi het box.

Trong project:

```text
src/nms.py
```

tu cai dat IoU va NMS.

Can nho:

- `nms` thap: loc manh, box gon hon.
- `nms` cao: giu nhieu box hon, huu ich neu nhieu nguoi dung gan nhau.

---

## 9. Project hien tai hoat dong nhu the nao?

### 9.1. File quan trong

| File | Vai tro |
|---|---|
| `app.py` | Web app Gradio cho image, video, realtime webcam |
| `src/detector.py` | Loi phat hien HOG + SVM |
| `src/nms.py` | IoU va Non-Maximum Suppression |
| `src/detect_image.py` | Detect anh bang command line |
| `src/detect_video.py` | Detect video, auto-compress, frame-skip, output MP4 |
| `src/detect_webcam.py` | Webcam realtime bang OpenCV |
| `src/compress_video.py` | Nen video bang ffmpeg hoac OpenCV fallback |
| `src/utils.py` | Resize, ve box, ve FPS, luu anh/video |
| `src/config.py` | Gia tri mac dinh va duong dan |

### 9.2. Loi detect trong `detector.py`

Code tao detector:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

Ham chinh:

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

Luon xu ly:

1. Kiem tra anh input.
2. Chuyen ve RGB/uint8 neu can.
3. Resize giu ti le.
4. Goi `hog.detectMultiScale`.
5. Doi `(x, y, w, h)` thanh `[x1, y1, x2, y2]`.
6. Goi NMS.
7. Sap xep theo score.
8. Giu toi da `max_detections`.
9. Ve box va label `Person: score`.

### 9.3. Vi sao dung `resize_width`?

Anh/video lon lam sliding window quet rat nhieu cua so. Resize giup:

- Giam so pixel.
- Giam so cua so can quet.
- Tang toc video/webcam.

Doi lai:

- Neu resize qua nho, nguoi nho co the mat chi tiet.

### 9.4. Vi sao video can auto-compress?

Video co:

- Nhieu frame.
- Do phan giai cao.
- FPS cao.

Nen HOG + SVM rat cham neu xu ly truc tiep.

`compress_video.py` giam:

- Width ve 640 hoac 480.
- FPS ve 10.
- Thoi luong ve 20 giay.

Neu co ffmpeg:

```text
mp4 H.264, yuv420p
```

giup xem duoc trong VS Code/browser.

### 9.5. Vi sao can `frame-skip`?

Neu `frame-skip = 3`:

```text
Chi detect moi 3 frame.
Frame giua dung lai box gan nhat.
```

Tac dung:

- Nhanh hon.
- Webcam muot hon.

Han che:

- Neu nguoi di chuyen nhanh, box co the cham cap nhat.

---

## 10. Ket noi voi slide thuyet trinh

Slide co 15 trang. Khi thuyet trinh, nen nam mach nhu sau.

### Slide 1: Tieu de

Can noi:

```text
Do an phat hien nguoi di bo bang HOG + SVM, tap trung vao pipeline thi giac may tinh co dien, khong dung deep learning.
```

### Slide 2: Noi dung trinh bay

Day la agenda:

1. Gioi thieu.
2. Phan cong.
3. Ly thuyet.
4. Thiet ke he thong.
5. Tham so.
6. Ket qua.
7. Danh gia.
8. Han che.
9. Ket luan.

### Slide 3: Gioi thieu va muc tieu

Can noi duoc:

- Pedestrian detection la bai toan nen tang.
- HOG + SVM co gia tri hoc thuat vi de giai thich.
- Project demo duoc anh, video, webcam, web.

### Slide 4: Phan cong

Theo slide:

- Dung: lap trinh, trien khai, app, image/video/webcam, demo.
- Kiet: co so toan, gradient, HOG, SVM, IoU/NMS, bao cao ly thuyet.

Khi bi hoi, moi thanh vien nen bao ve dung phan minh, nhung ca hai van nen nam pipeline chung.

### Slide 5: Gradient

Can nho cong thuc:

```text
Gx = partial I / partial x
Gy = partial I / partial y
m = sqrt(Gx^2 + Gy^2)
theta = atan2(Gy, Gx)
```

Y chinh:

```text
Gradient bien anh thanh thong tin canh. Dang nguoi duoc nhan qua cau truc canh.
```

### Slide 6: HOG va Linear SVM

Can nho:

```text
HOG vector = 3780 chieu
SVM score = w^T x + b
```

Noi ro:

```text
Person score khong phai xac suat.
```

### Slide 7: Thiet ke he thong

Pipeline tren slide:

```text
I -> {Wk} -> phi(Wk) -> f(phi(Wk)) -> {Bj}_NMS
```

Dien giai:

- `I`: anh input.
- `Wk`: cac cua so sliding window.
- `phi(Wk)`: vector HOG.
- `f(phi(Wk))`: SVM score.
- `{Bj}_NMS`: box cuoi sau NMS.

### Slide 8: Phan tich tham so

Day la slide rat de bi hoi.

Can nam:

- `hit-threshold`: nguong score SVM.
- `scale`: muc quet pyramid.
- `nms`: loc box trung.
- `resize-width`: toc do vs chi tiet.
- `frame-skip`: toc do video/webcam.

### Slide 9: Ket qua anh tinh

Can noi:

```text
Nhieu box quanh mot nguoi la hien tuong binh thuong cua sliding window.
NMS giup loc lai con mot box chinh.
```

Day la diem bao cao trung thuc: khong chi show ket qua dep, ma show ca hien tuong can xu ly.

### Slide 10: Anh huong resize-width

Can noi:

- Width tang: giu chi tiet, nhung cham hon.
- Width giam: nhanh hon, nhung de mat nguoi nho.

### Slide 11: Video va webcam

Can noi:

- Video co nhieu nguoi, score khac nhau theo do ro dang nguoi.
- Webcam co the score am neu ha `hit-threshold` thap.
- Score am khong phai loi, ma do ngưong phat hien duoc ha xuong.

### Slide 12: Danh gia va thao luan

Uu diem:

- De giai thich.
- Khong can GPU.
- Mot loi dung cho anh/video/webcam.
- Tham so ro nghia.

Han che:

- Nhay voi che khuat/tu the la.
- False positive voi nen nhieu canh.
- Cham voi video lon.
- Can NMS de xu ly nhieu box.

### Slide 13: Huong cai tien

Can noi:

- Tu train SVM rieng de so sanh.
- Danh gia precision/recall/F1.
- Bo sung dataset day du.
- Them tracking de on dinh video.

### Slide 14: Ket luan

Ket luan nen noi:

```text
Do an da minh hoa duoc mot pipeline thi giac may tinh co dien: gradient tao thong tin canh, HOG gom canh thanh dac trung hinh dang, SVM bien dac trung thanh score, NMS bien nhieu candidate thanh box cuoi.
```

### Slide 15: Cam on

Chuan bi cho phan hoi dap.

---

## 11. Kich ban demo khuyen nghi

### 11.1. Demo web app

Chay:

```bash
python app.py
```

Mo:

```text
http://127.0.0.1:7861
```

Thu tu demo:

1. Tab Image.
2. Upload `demo/images/test.jpg`.
3. Chay detect voi tham so mac dinh.
4. Giai thich `Person: score`.
5. Chinh `max-detections` de thay box it/hon.
6. Tab Video.
7. Upload `demo/videos/test.mp4`.
8. Bat auto-compress.
9. Chay detect.
10. Noi ve frame-skip va output MP4.
11. Tab Realtime Webcam neu may cho phep.

### 11.2. Demo command line

Detect anh:

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

### 11.3. Khi demo bi cham

Noi va lam:

```text
Giam resize-width ve 480 hoac 320.
Tang frame-skip len 5.
Tang scale len 1.2.
Bat auto-compress cho video.
```

### 11.4. Khi nhieu box sai

Noi va lam:

```text
Tang hit-threshold.
Giam max-detections.
Dieu chinh nms de loc box trung.
```

### 11.5. Khi khong detect duoc nguoi

Noi va lam:

```text
Giam hit-threshold ve 0 hoac -0.5.
Tang resize-width.
Giam scale ve 1.05 de quet ky hon.
```

---

## 12. Cac tham so can thuoc

| Tham so | Noi ngan gon | Tang len thi sao | Giam xuong thi sao |
|---|---|---|---|
| `hit-threshold` | Nguong score SVM | It box sai hon, de bo sot | Nhay hon, de sai hon |
| `scale` | Buoc pyramid | Nhanh hon, de bo sot | Ky hon, cham hon |
| `nms` | Nguong IoU loc box | Giu nhieu box hon | Loc manh hon |
| `resize-width` | Do rong xu ly | Ro hon, cham hon | Nhanh hon, mat chi tiet |
| `frame-skip` | Detect moi N frame | Nhanh hon, box tre hon | Muot/chinh xac hon, cham hon |
| `max-detections` | So box toi da | Giu duoc nhieu nguoi | Gon hon, co the mat nguoi |

---

## 13. Cau hoi van dap de gap

### Cau 1: Tai sao khong dung YOLO?

Tra loi:

```text
Muc tieu mon hoc la hieu pipeline xu ly anh truyen thong. HOG + SVM cho phep giai thich tung buoc: gradient, histogram, SVM, NMS. YOLO/CNN manh hon nhung la deep learning va khong dung theo yeu cau do an.
```

### Cau 2: HOG khac gi so voi dung pixel goc?

Tra loi:

```text
Pixel goc phu thuoc mau va anh sang. HOG dua tren huong gradient nen tap trung vao hinh dang va cau truc bien, phu hop hon voi phat hien nguoi.
```

### Cau 3: Vi sao HOG vector co 3780 chieu?

Tra loi:

```text
Anh 64x128 chia cell 8x8 tao 8x16 cell. Block 2x2 cell nen co 7x15 block. Moi block co 2x2x9 = 36 gia tri. Tong 7x15x36 = 3780.
```

### Cau 4: `Person: 1.14` co phai 114% khong?

Tra loi:

```text
Khong. Do la decision score cua Linear SVM: score = w^T x + b. No khong phai xac suat. Score cang cao thi cua so cang giong lop nguoi theo SVM.
```

### Cau 5: Vi sao co score am ma van co box?

Tra loi:

```text
Vi hit-threshold co the duoc dat am, vi du -0.7. Khi nguong thap hon score am, detector van giu box. Dieu nay tang do nhay nhung de false positive hon.
```

### Cau 6: Tai sao mot nguoi co nhieu box?

Tra loi:

```text
Sliding window quet nhieu vi tri va scale gan nhau. Nhieu cua so cung bao quanh mot nguoi va cung vuot nguong SVM. Vi vay can NMS de giu box tot nhat.
```

### Cau 7: NMS dua vao dau de loai box?

Tra loi:

```text
NMS dung IoU. Neu hai box chong lap qua nguong, box co score thap hon bi loai.
```

### Cau 8: Tai sao video cham?

Tra loi:

```text
Video co nhieu frame. Moi frame lai can sliding window va scale pyramid, nen chi phi lon. Project giam bang auto-compress, resize-width va frame-skip.
```

### Cau 9: Ban hien tai co tu train SVM khong?

Tra loi:

```text
Ban demo light hien tai khong tu train. No dung pretrained HOG + SVM people detector cua OpenCV. Tuy nhien pipeline van la HOG + Linear SVM. Ban phat trien truoc do da co huong tu train voi dataset INRIA.
```

### Cau 10: Han che lon nhat cua HOG + SVM la gi?

Tra loi:

```text
HOG + SVM phu thuoc vao hinh dang bien tuong doi chuan, nen kem voi nguoi bi che khuat, tu the la, nen nhieu canh, hoac video do phan giai lon. Do chinh xac khong bang deep learning hien dai.
```

---

## 14. Cach noi ve phan cong

Theo slide:

### Nguyen Le Tri Dung

Tap trung noi:

- Xay dung code project.
- `detector.py`, `nms.py`, `utils.py`.
- Luong detect anh/video/webcam.
- Auto-compress, frame-skip, xuat MP4.
- Gradio app.
- Demo va tinh chinh tham so.

### Do Anh Kiet

Tap trung noi:

- Gradient: Sobel, magnitude, orientation.
- HOG: cell, block, bin, chuan hoa, vector 3780.
- SVM: score, margin, soft-margin.
- IoU va NMS.
- Giai thich tham so va tra loi van dap ly thuyet.

Nhung ca hai nen nam:

```text
Gradient -> HOG -> SVM -> NMS -> Bounding Box
```

---

## 15. Ban chat cua demo anh, video, webcam

### 15.1. Anh tinh

Anh tinh dung de giai thich:

- Box quanh nguoi.
- Score.
- Nhieu box truoc/sau NMS.
- Anh huong cua `hit-threshold`, `nms`, `max-detections`.

### 15.2. Video

Video dung de giai thich:

- Detect tren tung frame.
- Nhieu nguoi o khoang cach khac nhau.
- Score khac nhau theo do ro cua hinh dang.
- Auto-compress de giam kich thuoc.
- Frame-skip de tang toc.

### 15.3. Webcam

Webcam dung de giai thich:

- Realtime yeu cau do tre thap.
- Nen dung `resize-width=480`, `frame-skip=3`.
- Gradio webcam co the tien nhung OpenCV webcam on dinh hon.

---

## 16. Dieu can nhan manh khi thuyet trinh

1. Do an khong dung deep learning.
2. HOG + SVM la pipeline co the giai thich duoc.
3. Gradient la nen tang de HOG mo ta hinh dang.
4. HOG vector cua OpenCV people detector co 3780 dac trung.
5. SVM score la \(w^T x + b\), khong phai xac suat.
6. Sliding window sinh nhieu candidate, NMS loc thanh box cuoi.
7. Video cham la do so frame va so cua so quet lon.
8. Auto-compress va frame-skip la toi uu thuc te cho demo.
9. Ban hien tai dung pretrained detector cua OpenCV de demo nhe va on dinh.
10. Han che cua HOG + SVM la mot phan cua bao cao, khong nen che giau.

---

## 17. Tom tat sieu ngan de hoc thuoc

```text
Project phat hien nguoi di bo bang HOG + Linear SVM pretrained cua OpenCV.
Anh duoc resize, detector quet sliding window qua nhieu scale.
Moi cua so duoc trich HOG: gradient -> cell -> block -> histogram -> vector 3780 chieu.
Linear SVM tinh score = w^T x + b.
Neu score vuot hit-threshold thi cua so la candidate nguoi.
Nhieu candidate trung nhau duoc loc bang NMS dua tren IoU.
Ket qua ve bounding box Person: score.
Anh/video/webcam dung chung detector.py.
Video duoc tang toc bang auto-compress va frame-skip.
```

---

## 18. Checklist truoc khi nop/thuyet trinh

- Chay duoc `python app.py`.
- Tab Image detect duoc anh mau.
- Tab Video detect duoc video, output xem duoc.
- Webcam local biet lenh chay neu Gradio webcam lag.
- Nho cong thuc gradient, HOG 3780, SVM score, IoU.
- Giai thich duoc `Person: score`.
- Giai thich duoc vi sao nhieu box va NMS xu ly ra sao.
- Giai thich duoc vi sao project khong dung deep learning.
- Giai thich duoc han che va huong cai tien.

