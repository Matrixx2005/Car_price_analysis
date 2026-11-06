# Hướng dẫn Tuần 3: tôi đã xử lý từng vấn đề như thế nào

Tài liệu cho bạn (sinh viên) để **hiểu, chạy lại và tự trình bày** tuần 3. Kết quả chính thức nằm ở `reports/week3_report.md`; tài liệu này giải thích *vì sao* làm như vậy và những gì đã gặp phải.

---

## 1. Tuần 3 làm gì

Tuần 2 cho biết nên dùng cách mã hóa/chuẩn hóa nào. Tuần 3 làm ba việc: (1) **tinh chỉnh** siêu tham số của 4 mô hình, (2) quyết định hai việc còn treo (**xe giá cực cao**, **gom danh mục hiếm** của `model_version`), (3) **chốt cấu hình rồi đánh giá trên tập test đúng một lần**, kèm khoảng tin cậy và phân tích sai số.

Tất cả lựa chọn chỉ dựa trên cross-validation trên tập **train**. Tập test chỉ được đọc ở bước cuối.

## 2. Cách chạy

```powershell
python -m src.week3_experiments --stage all   # 4 giai đoạn CV trên train (tune, extremes, grouping, combine); chạy tiếp được
python -m src.week3_select                    # đóng băng cấu hình -> reports/week3_frozen_config.json
python -m src.final_eval                      # đánh giá test, CHỈ MỘT LẦN (~1,3 giờ trên máy 16 nhân)
python -m src.week3_analysis                  # bảng/biểu đồ CV + phân tích phần dư test
python -m src.reporting_week3                 # sinh reports/week3_report.md (tự kiểm chứng phát biểu)
python -m pytest -v                           # 66 test (25 tuần 1 + 25 tuần 2 + 16 tuần 3)
```

**Quan trọng:** đừng chạy lại `final_eval` sau khi đã có file `reports/tables/week3_test_evaluation_done.json` (chương trình từ chối). Muốn phân tích thêm, dùng dự đoán đã lưu `data/processed/week3_test_predictions.parquet`. Xem kết quả: `reports/week3_report.md`, `notebooks/04_week3_results.ipynb`, `reports/figures/week3_*.png`.

---

## 3. Cách tôi xử lý từng vấn đề

### Vấn đề 1: "Làm sao tập test thật sự chưa bị dùng để chọn gì?"
Đây là điểm quan trọng nhất của tuần 3. Tôi cưỡng chế bằng code chứ không chỉ bằng lời hứa:
- Cấu hình được **đóng băng** (`week3_frozen_config.json`) cùng mã băm của file train và file kết quả CV.
- `final_eval.py` kiểm tra lại các mã băm; đổi bất cứ thứ gì sau khi đóng băng thì **từ chối chạy**.
- `final_eval.py` là module **duy nhất** đọc `test.parquet`; sau khi chạy ghi file đánh dấu và từ chối lần chạy thứ hai.
- Có test quét toàn bộ mã nguồn để chắc chắn không module nào khác đọc tập test.

### Vấn đề 2: "Cắt xe giá cực cao mà không rò rỉ"
Ngưỡng cắt (phân vị giá) tính **chỉ từ phần train của từng fold**, và chỉ cắt các dòng huấn luyện; dòng đánh giá không bao giờ bị cắt (`drop_extreme_prices`). Báo cáo thêm MAE riêng cho xe dưới/từ 100.000 EUR để thấy cái giá phải trả: cắt làm sai số trên xe đắt tăng ở cả 8/8 cặp (mô hình, mức cắt).

### Vấn đề 3: "Gom danh mục hiếm của `model_version`"
Làm bằng `RareCategoryGrouper` nằm **trong Pipeline**: đếm tần suất trên fold train, danh mục hiếm hoặc chưa từng thấy gộp vào một nhãn chung. Thử ngưỡng 10/50/200 với One-hot và Target Encoding. Kết quả: gom mạnh (≥ 50) luôn làm tệ hơn; chỉ KNN hưởng lợi (Target Encoding, −3,4%).

### Vấn đề 4: "Chọn tốt nhất trong nhiều ứng viên dễ chạy theo nhiễu"
Với mọi thành phần tùy chọn tôi dùng quy tắc: **chỉ thay cấu hình đơn giản hơn khi MAE trung bình thấp hơn VÀ thắng ít nhất 4/5 fold**. Ví dụ Ridge nhận cắt phân vị 99,9% (4/5 fold) nhưng các lựa chọn thắng 2–3 fold bị bỏ. Siêu tham số thì chọn MAE CV thấp nhất, và báo cáo ghi rõ cách chọn này có thể hơi lạc quan (nên mới cần tập test).

### Vấn đề 5: "SVR quá chậm" (vấn đề lớn nhất của tuần này)
- Thực tế gặp: một cấu hình SVR cực đoan (C=300, ε=0,02, log giá) chạy hơn một giờ chưa xong một fold. Tôi **hủy và ghi lại** (`week3_skipped_configs.json`), không âm thầm bỏ qua.
- Nghiêm trọng hơn: SVR có độ phức tạp ~O(n²). Cấu hình SVR tốt nhất theo CV (MAE 6.818) mất ~32 phút/fold trên 16.000 dòng, tức ước tính ~12 giờ khi huấn luyện lại trên 75.000 dòng.
- Giải pháp: thêm **ràng buộc chi phí** (thời gian huấn luyện ước tính trên toàn train ≤ 30 phút) vào quy tắc chọn. Cấu hình quá đắt không đủ điều kiện được chọn, nhưng được đóng băng làm **mốc tham khảo** (huấn luyện trên cùng 20k dòng như CV) để biết ta đã bỏ lỡ gì.

### Vấn đề 6: "Khoảng tin cậy cho kết quả test"
Các tin cùng một xe không độc lập, nên bootstrap **theo nhóm xe** (lấy mẫu lại cả nhóm, 1.000 lần) cho khoảng tin cậy 95% của MAE/R² và cho hiệu MAE giữa các cặp mô hình. Nhờ đó biết khác biệt nào có ý nghĩa: ví dụ SVR và Random Forest đã tinh chỉnh chênh nhau không có ý nghĩa.

### Vấn đề 7: "Đọc kết quả test không hợp ý mình"
Kết quả test **không hoàn toàn khớp CV** và tôi nêu thẳng:
- Random Forest tinh chỉnh (MAE 7.061) tệ hơn cấu hình mặc định (6.557), khoảng tin cậy của hiệu không chứa 0. Giải thích hợp lý (chưa kiểm chứng): siêu tham số chọn trên 16.000 dòng không còn tối ưu trên 75.000 dòng.
- Tôi **không** chọn lại Random Forest mặc định chỉ vì nó tốt nhất trên test: làm vậy là dùng test để chọn mô hình. Báo cáo ghi đó là quan sát sau khi dùng test.
- Báo cáo có Phụ lục A tự kiểm chứng 12 phát biểu; 4 phát biểu "kỳ vọng" (ví dụ mô hình chọn bằng CV cũng tốt nhất trên test) **không đúng** và được hiển thị như vậy.

### Vấn đề 8: "R² chỉ 0,5 nghĩa là mô hình tệ?"
Không. Một dòng nhãn lỗi (BMW 316 giá 9,5 triệu EUR, đã gắn cờ từ tuần 1) và vài xe hiếm chi phối RMSE: 5 dự đoán sai nhất gây ~93% tổng sai số bình phương. Bỏ đúng một dòng đó, R² của Random Forest lên 0,87. Vì vậy chỉ số chính là MAE/MedAPE. Phân tích độ nhạy này được ghi rõ là mô tả sau khi đánh giá, không dùng để chọn lại mô hình.

### Các lỗi/sự cố tôi gặp và cách xử lý

| Gặp phải | Xử lý |
|---|---|
| Lệnh dừng toàn bộ tiến trình python bị từ chối (quá rộng) | Chỉ dừng đúng 5 worker của lần chạy tôi khởi tạo (theo PID); tiến trình chính ghi nốt bộ đệm trong `finally` nên không mất kết quả đã xong |
| Chưa tính đến chi phí khi chọn cấu hình; bảng "tinh chỉnh so với mặc định" ban đầu lấy cấu hình không xét chi phí, lệch với cấu hình thật được chọn | Thêm ràng buộc chi phí, sửa bảng để dùng cấu hình khả thi và hiển thị riêng cấu hình tốt nhất nhưng quá đắt |
| `choose_final` lập chỉ mục sai (`KeyError`) khi đóng băng | Sửa lại cách lấy dòng cấu hình; chưa ghi gì ra file khi lỗi |
| Test bảo vệ tập test phát hiện chuỗi "test.parquet" trong một docstring | Sửa lời văn để quy tắc quét mã nguồn không có ngoại lệ |
| Lần chạy `final_eval` đầu bị tắt giữa chừng (máy dừng), chưa có kết quả | Chạy lại vì chưa có file đánh dấu (tập test chưa được dùng để quyết định gì); ghi quy trình này vào `TIEN_DO_TUAN3.md` |
| Ridge chạy lại lệch cỡ 1e-4 tương đối so với tuần 2 | Do bộ giải lặp (`tol=1e-4`) trên ma trận thưa; ghi vào mục giới hạn |

---

## 4. Kết quả chính (đọc nhanh; chi tiết ở `reports/week3_report.md`)

Trên tập test (MAE, EUR; CV là trên tập con 20k của train):

| Mô hình (cấu hình tinh chỉnh) | MAE CV | MAE test | So với mặc định tuần 2 |
|---|---|---|---|
| SVR | 7.977 | 6.862 | tốt hơn (mặc định 8.123) |
| Random Forest (được chọn bằng CV) | 7.351 | 7.061 | **tệ hơn** (mặc định 6.557) |
| KNN | 9.154 | 7.587 | tốt hơn (mặc định 7.928) |
| Ridge | 9.441 | 8.997 | tốt hơn nhiều (mặc định 15.589) |
| Mô hình ngây thơ (trung vị) | | 33.581 | |

Năm điều để nhớ:
1. Mọi mô hình tinh chỉnh giảm sai số > 70% so với mô hình ngây thơ.
2. Ridge hưởng lợi nhiều nhất (log giá + α): MAE test giảm ~42%.
3. Tinh chỉnh trên tập con nhỏ không chuyển được sang dữ liệu lớn với Random Forest.
4. SVR tốt hơn trên test nhưng quá đắt để tinh chỉnh kỹ (O(n²)).
5. R² thấp do nhãn lỗi; MedAPE khoảng 8–9% ở các nhóm giá thông thường; xe từ 160.000 EUR bị dự đoán thấp hơn ~23.700 EUR.

---

## 5. Câu hỏi giáo viên có thể hỏi và cách trả lời

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao biết em không dùng test để chọn mô hình? | Cấu hình được đóng băng cùng mã băm trước khi đọc test; module duy nhất đọc test từ chối chạy lần hai; có test quét mã nguồn. |
| Mô hình em chọn không phải cái tốt nhất trên test? | Đúng, và em báo cáo trung thực. Em không chọn lại theo test vì như vậy test thành tập chọn. Khác biệt giữa các mô hình dẫn đầu nằm trong khoảng bất định (bootstrap theo nhóm xe). |
| Tinh chỉnh có ích không? | Rất ích với Ridge, SVR, KNN; không ích (thậm chí hại) với Random Forest vì siêu tham số chọn trên 16.000 dòng không còn tối ưu ở 75.000 dòng. |
| Sao R² thấp? | Vài nhãn lỗi chi phối (một xe BMW 316 giá 9,5 triệu EUR); bỏ một dòng đó R² lên 0,87. Em dùng MAE/MedAPE làm chỉ số chính. |
| Sao không dùng cấu hình SVR tốt nhất? | Ước tính ~12 giờ để huấn luyện lại trên toàn train; em đặt ràng buộc chi phí và vẫn đánh giá cấu hình đó làm mốc tham khảo trên 20k dòng. |
| Hạn chế? | Tìm kiếm ngẫu nhiên hữu hạn, tinh chỉnh trên tập con, dữ liệu thiên về xe cao cấp châu Âu, giá chào bán, chưa dùng mô hình boosting; cần dữ liệu mới để kết luận chắc chắn về Random Forest. |

---

## 6. Danh sách file của tuần 3

| File | Vai trò |
|---|---|
| `src/transformers.py` | `RareCategoryGrouper`, `drop_extreme_prices` |
| `src/week3_experiments.py` | Không gian tìm kiếm, 4 giai đoạn CV, quy tắc quyết định, ràng buộc chi phí, chạy tiếp được |
| `src/week3_select.py` | Đóng băng cấu hình, mốc tham khảo |
| `src/final_eval.py` | Đánh giá test một lần (khóa, mã băm, bootstrap theo nhóm) |
| `src/week3_analysis.py`, `src/reporting_week3.py` | Bảng, biểu đồ, báo cáo tự kiểm chứng |
| `tests/test_week3.py` | 16 test (transformer, tìm kiếm, quy tắc quyết định, chi phí, bảo vệ test, bootstrap) |
| `reports/week3_report.md`, `reports/tables/week3_*.csv`, `reports/figures/week3_*.png` | Báo cáo, bảng, biểu đồ |
| `reports/week3_frozen_config.json` | Cấu hình đóng băng |
| `notebooks/04_week3_results.ipynb` | Notebook trình bày kết quả |
