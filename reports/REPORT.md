# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
   "class_id": 468,   "class_name": "cab",   "rank": 1,   "score": 0.510915,   "taxonomy_name": "ImageNet-1K"
  
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Cả bức ảnh, Tại vì record đưa ra các nhãn là nhũng lớp mô hình dự đoán cho ảnh kèm điểm số và thứ hạng
   
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách lớp đi kèm mô hình, không tự nghĩ thêm
   
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   vì 3 cai này bổ sung cho nhau: class_id là mã để máy đối chiếu ổn định, kể cả class_name được đổi tên hoặc dịch; class_name dễ đọc và tiện hiển thị; taxonomy_name cho bieetsmax đó thuộc lớp nào, mang nghĩa nào.
   
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline cần quy định một quy tắc ưu tiên thống nhất: chọn lớp thể hiện chủ thể chính hoặc nội dung nổi bật nhất của ảnh, không liệt kê mọi thứ nhìn thấy.
Nên nói rõ cách xác định “nổi bật”, chẳng hạn ưu tiên chủ thể được đặt ở trung tâm, chiếm nhiều diện tích hoặc là trọng tâm hành động. Nếu có lớp mô tả toàn cảnh phù hợp thì dùng lớp đó; nếu không, chọn vật thể chính. Cũng cần quy định cách xử lý khi hai chủ thể nổi bật ngang nhau — ví dụ chọn chủ thể ở tiền cảnh — và khi ảnh không đủ rõ để gán lớp. Tránh để người gán nhãn tự suy đoán ý định hay bối cảnh không nhìn thấy trong ảnh.
   
6. Vì sao không được lấy `score` của model làm ground truth?
Model score là một tín hiệu đánh giá, không phải sự thật. Ground truth phải đến từ nguồn độc lập và có thể kiểm chứng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
 "class_name": "person", "score": 0.912625,"bbox_xyxy": [385.33,69.24,498.92,348.92], "bbox_width": 113.58, "bbox_height": 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   góc trên-trái ở (385.33, 69.24), góc dưới-phải ở (498.92, 348.92), rộng khoảng 114px, cao khoảng 280px
   
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   threshold = 0.2 có 17 vật thể
   threshold = 0.6 có 6 vật thể
   
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Khi hạ ngưỡng, reviewer cần kiểm tra thêm các hộp mới được giữ lại: xác nhận chúng có đúng vật thể không, lớp và vị trí hộp có chính xác không, rồi loại các phát hiện sai hoặc trùng. Hạ ngưỡng giúp giữ được nhiều vật thể hơn nhưng cũng có thể thêm nhiều hộp sai.
   
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Quy tắc box: Với mỗi vật thể được gán nhãn, vẽ một hộp chữ nhật ôm sát phần nhìn thấy của vật thể, không bao gồm nền xung quanh; đặt góc trên-trái và dưới-phải theo tọa độ pixel của ảnh, đồng thời gán đúng lớp vật thể.
   
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ở mép trái ảnh, có thể thấy một phần người—bàn tay và cánh tay bị ảnh cắt mất.
Nếu vật thể bị mép ảnh cắt, vẽ hộp quanh phần vật thể còn nhìn thấy và để hộp chạm mép ảnh. Đây là tình huống phổ biến, có thể giải quyết bằng một quy tắc áp dụng cho nhiều ảnh; chỉ cần escalation nếu không thể xác định đó là vật thể gì hoặc cách gán nhãn còn mơ hồ.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
   `instance_id`= "kitchen-001" ; `class_name`="person" ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   so với box, polygon cho thêm thông tin về đường viền hình dạng thức tế của vật thể bằng nhiều tọa độ
  
4. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   dùng để phân biệt các đối tượng với nhau;
   Không phải class ID: không xác định loại đối tượng (ví dụ: người, xe, chó), không phải tracking ID: Không theo dõi cùng một đối tượng qua nhiều frame/video.
   Mục đích: Phân biệt từng instance riêng biệt, kể cả khi chúng cùng class.
5. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Quy tắc vẽ biên mask: Với mỗi vật thể, vẽ polygon theo sát đường ranh giữa vật thể nhìn thấy và nền; chỉ bao phần nhìn thấy, không gộp vật thể khác vào mask, và không mở rộng mask qua vùng bị che khuất hoặc nằm ngoài mép ảnh.
   
6. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Ở khu vực bàn bếp phía dưới bên trái, các cái bát chạm và che khuất một phần nhau, nên ranh giới giữa chúng không rõ.
Mình sẽ dùng guideline nếu đã có quy tắc chung về vật thể chạm hoặc bị che—chẳng hạn, vẽ mask riêng theo phần nhìn thấy của từng bát. Nếu chưa rõ có nên gán nhãn từng chiếc bát khi ranh giới không phân biệt được, nên escalate lên Lab Coach để thống nhất cách xử lý trước khi áp dụng nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh | Traffic có nhiều loại xe; điểm "cab": 0,51 trong JSON chỉ là dự đoán | Chọn nhãn theo guideline | Nhãn có khớp nội dung chính không |
| Phát hiện vật thể | Một hộp và nhãn cho mỗi vật | Vật nhỏ/chồng lấp; hộp có thể thiếu hoặc sai | Vẽ hộp sát vật, chọn đúng lớp | Bỏ sót, trùng, hộp và lớp |
| Instance segmentation | Một mask và nhãn cho mỗi vật thể | Biên mờ, vật che nhau; polygon có thể lấn nền | Tách từng vật, vẽ phần nhìn thấy | Mask có bám biên và tách đúng instance không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Giảng viên, lab coach, gửi mail.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
