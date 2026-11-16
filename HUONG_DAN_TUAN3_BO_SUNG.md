# Hướng dẫn phần bổ sung Tuần 3: tôi đã xử lý từng vấn đề như thế nào

Phần này làm **bước 2 và bước 3** của kế hoạch cải thiện sau đợt đánh giá dự án: (2) làm `final_eval` không mất kết quả khi bị gián đoạn; (3) bổ sung boosting, chẩn đoán overfit, độ quan trọng đặc trưng và đường cong học. Báo cáo kết quả: `reports/week3_extra_report.md`; notebook: `notebooks/05_week3_extra_results.ipynb`.

---

## 1. Nguyên tắc quyết định mọi việc: tập test chỉ được xem một lần

Tập test đã được đánh giá một lần với 4 mô hình đã đóng băng. Nếu bây giờ thêm boosting vào bảng test thì đó là **lần xem thứ hai**, và test trở thành tập chọn mô hình (rò rỉ thông tin). Vì vậy:

- Mọi phân tích mới chỉ dùng **tập train** (cross-validation).
- **Boosting chưa được đánh giá trên test** (chủ ý). Báo cáo ghi rõ điều này; để so sánh công bằng trên test cần tập giữ lại mới.
- Số liệu test của báo cáo chính tuần 3 **không đổi**.
- Code mới nằm ở file riêng (`src/week3_extra*.py`); có test quét mã nguồn để đảm bảo chúng không đọc tập test.

---

## 2. Bước 2: `final_eval` ghi từng mô hình ngay khi xong

**Vấn đề:** lần chạy đánh giá test đầu tiên mất 77 phút và chỉ ghi kết quả ở cuối. Một lần bị tắt giữa chừng đã làm mất toàn bộ (bạn đã gặp), và chúng tôi từng không thể dừng một worker chậm mà không mất hết.

**Cách xử lý (`src/final_eval.py`):**
- Mỗi mô hình khi huấn luyện xong sẽ **ghi dự đoán ngay** vào `data/processed/week3_final_partial/` (ghi vào file tạm rồi đổi tên, nên không bao giờ có file dở dang).
- Chạy lại cùng lệnh thì chỉ **tính tiếp các mô hình còn thiếu**. Đây vẫn là *một* lần đánh giá được tiếp tục, không phải lần xem thứ hai: chưa có số liệu nào tính từ nhãn test cho tới khi đủ mọi mô hình.
- Chống dữ liệu cũ: mỗi file mang mã băm của cấu hình đóng băng và danh sách dòng test; nếu khác thì **từ chối** và yêu cầu xóa thư mục tạm.
- Có lệnh xem tiến độ: `python -m src.final_eval --status`.
- Thư mục tạm tự xóa sau khi hoàn tất.
- Tên file test chỉ xuất hiện đúng một chỗ trong mã (một hằng số, một hàm đọc); có test kiểm tra điều này và test mô phỏng sự cố (mô hình thứ 7 bị lỗi: các mô hình xong trước vẫn còn, chạy lại chỉ tính phần thiếu).
- **Không chạy lại `final_eval`** vì đã có file đánh dấu `week3_test_evaluation_done.json`. Cơ chế mới chỉ được kiểm tra bằng test với dữ liệu giả.

---

## 3. Bước 3: những gì đã thêm và cách chạy

```powershell
python -m src.week3_extra --part all --jobs 14     # boosting, chẩn đoán, đường cong học (~1 giờ trên máy 16 nhân)
python -m src.week3_extra_analysis                 # bảng + biểu đồ
python -m src.reporting_week3_extra                # reports/week3_extra_report.md (tự kiểm chứng phát biểu)
python -m pytest -q                                # test
```
Mọi phần đều **chạy tiếp được** (kết quả ghi từng tác vụ ra CSV, có chữ ký chống trộn). Có thể chạy từng phần: `--part hgb`, `--part diagnostics`, `--part curves`.

### 3.1. Boosting (HistGradientBoosting) tinh chỉnh trên toàn train
- Boosting **rất nhanh** (vài giây đến vài phút mỗi lần) so với Random Forest (cả nghìn giây), nên được tinh chỉnh trên **toàn bộ 75.457 dòng** bằng GroupKFold 5 fold. Điều này khắc phục luôn điểm yếu "tinh chỉnh trên tập con 20k" của các mô hình khác.
- Quy trình 3 bước: chọn bộ mã hóa × log giá (mặc định) → tìm kiếm ngẫu nhiên 60 bộ siêu tham số → thử `model_version`; mỗi quyết định tùy chọn chỉ được nhận nếu thắng ≥ 4/5 fold.
- Boosting **không** được thêm vào danh sách `MODELS` để lưới tuần 2/3 và chữ ký kết quả cũ không bị đổi.
- Kết quả: MAE CV **5.880 EUR** (R² 0,83), tốt hơn mặc định của chính nó (7.240) và Random Forest mặc định (6.276), thắng 5/5 fold; chọn Target Encoding + log giá, không dùng `model_version`.
- **Lưu ý trung thực:** so với các mô hình ở tham số mặc định thì không cùng mức nỗ lực tinh chỉnh; chọn tốt nhất trong 60 cấu hình làm MAE hơi lạc quan; chưa có số liệu test.

### 3.2. Chẩn đoán overfit (train so với validation)
Mỗi mô hình được huấn luyện lại trên từng fold, đo trên chính phần huấn luyện và trên validation. **Điều cần hiểu:** khoảng cách lớn không tự nó là lý do loại mô hình. Các mô hình cây có khoảng cách 1,6–2,0 nhưng vẫn có MAE validation thấp nhất; SVR có khoảng cách nhỏ nhất (1,12) vì nó **khớp train kém** (underfitting), không phải vì tổng quát tốt. KNN mặc định có MAE train ≈ 0 (mỗi điểm là láng giềng của chính nó) nên tỷ lệ vô nghĩa và không được vẽ.

### 3.3. Độ quan trọng đặc trưng (permutation importance)
- Xáo trộn cột **gốc** (trước mã hóa) giữa các dòng validation và đo mức tăng MAE; vì hãng/mẫu và năm/tuổi mang thông tin chồng nhau, thêm hai phép xáo trộn **cả nhóm** ("định danh xe", "tuổi xe").
- Kết quả: `power_kw` là cột đơn lẻ quan trọng nhất ở nhiều mô hình (đặc biệt Random Forest, ~29–35%), rồi `mileage_km`, `brand`, `model`, tuổi/năm sản xuất; các cột như `body_color`, `seller_type`, `drive_train`, `model_version` gần như không đóng góp. Nhóm định danh xe và nhóm tuổi xe đều làm MAE tăng hàng chục nghìn EUR khi bị xáo trộn.
- Cẩn thận khi diễn giải: đó là mô hình *dựa vào* cột nào, không phải quan hệ nhân quả; cột tương quan chia sẻ độ quan trọng.

### 3.4. Đường cong học: kiểm chứng vì sao Random Forest tinh chỉnh thua mặc định
Báo cáo chính đưa ra giả thuyết (chưa kiểm chứng): siêu tham số chọn trên 16.000 dòng không còn tối ưu ở 75.000 dòng. Đường cong học kiểm tra trực tiếp bằng cách huấn luyện trên 15%, 30%, 50%, 100% dữ liệu của mỗi fold:

| Dữ liệu huấn luyện | RF mặc định | RF tinh chỉnh | Ai tốt hơn |
|---|---|---|---|
| 15% (9.000 dòng) | 8.358 | 8.069 | tinh chỉnh (3/3 fold) |
| 30% (18.000 dòng) | 7.242 | 7.435 | mặc định |
| 50% (30.000 dòng) | 7.014 | 7.438 | mặc định |
| 100% (60.000 dòng) | 6.276 | 6.888 | mặc định (5/5 fold) |

→ **Giả thuyết được ủng hộ bằng dữ liệu**: tinh chỉnh tốt ở dữ liệu nhỏ nhưng khoảng cách đảo chiều và tăng dần khi có thêm dữ liệu.

---

## 4. Các vấn đề tôi gặp và cách xử lý

| Gặp phải | Xử lý |
|---|---|
| **Máy chạy rất chậm**: các worker chỉ dùng ~10% CPU, mất hơn một giờ chưa xong tác vụ nào ở lần chạy đường cong học đầu | Nguyên nhân chính là **quá tải luồng** (OpenMP của boosting × nhiều worker). Đặt mỗi worker một luồng (`OMP_NUM_THREADS=1`) làm CPU đạt ~98%; đã đưa vào code (`src/week3_extra.py` tự đặt biến này khi chạy) |
| Vẫn bị kìm tốc độ sau vài phút khi chạy nền lâu | Máy giới hạn CPU cho tiến trình nền chạy lâu (chỉ dùng ~2 nhân). Giải pháp: **thu gọn thiết kế** đường cong học (bỏ mức 75%, Random Forest dùng 3/5 fold ở mức nhỏ, dùng lại điểm 100% của Random Forest mặc định từ xác nhận tuần 2: cùng mô hình, cùng fold). Giảm từ 150 xuống 98 tác vụ mà vẫn kiểm chứng được giả thuyết |
| Tôi mất hơn một giờ chạy thiết kế ban đầu trước khi thu gọn | Ghi lại như một bài học: kiểm tra tốc độ thực tế sớm hơn thay vì chờ |
| Dừng tiến trình đang chạy | Chỉ dừng đúng **cây tiến trình do tôi khởi chạy** (theo PID), không dừng diện rộng (đã từng bị từ chối) |
| Test mới bắt lỗi thật: bộ phân tích đọc cột `medape_valid` trong khi kết quả lưu tên `medape_pct_valid` | Sửa bộ phân tích; test cũng đổi cho khớp |
| joblib báo lỗi khó hiểu khi một tác vụ ném lỗi ở chế độ tuần tự có `verbose` | Chỉ bật `verbose` khi chạy song song; lỗi gốc được giữ lại |
| Test bảo vệ tập test bắt được cách viết cũ (tên file test xuất hiện ở hai chỗ) | Gom thành một hằng số duy nhất `TEST_PATH` |
| Báo cáo nháp nhận xét sai (Ridge có khoảng cách overfit nhỏ nhất; thực tế là SVR) và câu về đảo thứ hạng bị lẫn dấu | Sinh nhận xét **từ số liệu** thay vì viết tay; thêm Phụ lục A tự kiểm chứng. 3 phát biểu "kỳ vọng" của tôi (RF khớp train < nửa MAE validation; Ridge overfit ít nhất; nhóm định danh quan trọng hơn mọi cột đơn lẻ) **không đúng** và được hiển thị như vậy |

---

## 5. Câu hỏi giáo viên có thể hỏi

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao không đưa boosting vào bảng test? | Tập test đã được đánh giá một lần; đưa thêm mô hình sau đó là xem lần hai, biến test thành tập chọn. Em chỉ so sánh bằng CV trên train và nói rõ boosting chưa có số liệu test. |
| Boosting có tốt hơn Random Forest không? | Trong CV trên toàn train, boosting tinh chỉnh có MAE 5.880 so với 6.276 của Random Forest mặc định, thắng 5/5 fold. Lưu ý boosting được tinh chỉnh còn Random Forest ở mặc định. |
| Mô hình có overfit không? | Các mô hình cây có khoảng cách train-validation khoảng 1,6–2,0 nhưng vẫn có MAE validation thấp nhất; em dùng MAE validation chứ không dùng khoảng cách để kết luận. |
| Biến nào quyết định giá? | Công suất (`power_kw`), nhóm định danh xe (hãng + mẫu), tuổi/năm sản xuất và số km; em đo bằng permutation importance trên validation, kèm cảnh báo về cột tương quan. |
| Vì sao Random Forest tinh chỉnh thua? | Em đã kiểm chứng bằng đường cong học: tinh chỉnh tốt hơn ở 9.000 dòng nhưng thua ở 18.000 dòng trở lên, khoảng cách tăng dần tới 60.000 dòng. |

---

## 6. Danh sách file

| File | Vai trò |
|---|---|
| `src/final_eval.py` (sửa) | Ghi từng mô hình khi xong, chạy tiếp, chống dữ liệu cũ |
| `src/pipelines.py` (sửa) | Thêm boosting (không thuộc `MODELS`) |
| `src/week3_experiments.py` (sửa nhỏ) | Dùng chung `run_configs`, `estimator_from_cfg` |
| `src/week3_extra.py` | Boosting trên toàn train, chẩn đoán, đường cong học |
| `src/week3_extra_analysis.py`, `src/reporting_week3_extra.py` | Bảng, biểu đồ, báo cáo tự kiểm chứng |
| `tests/test_week3_extra.py` + thêm trong `tests/test_week3.py` | Test mới |
| `reports/week3_extra_report.md`, `reports/week3_hgb_best.json`, `reports/tables/week3_hgb_*`, `week3_diag_*`, `week3_importance_summary.csv`, `week3_learning_curves*.csv`, `week3_rf_crossover.csv` | Kết quả |
| `notebooks/05_week3_extra_results.ipynb` | Notebook trình bày |
