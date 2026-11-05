# Tiến độ Tuần 3 (đã lưu tạm, chưa hoàn tất)

_Lưu lúc tạm dừng. Đọc file này để làm tiếp._

## Đã xong
- Code: `src/transformers.py`, `src/week3_experiments.py`, `src/week3_select.py`, `src/final_eval.py`, `src/week3_analysis.py`, `src/reporting_week3.py`, `tests/test_week3.py` (16 test), `build_week3_notebook.py`.
- Thí nghiệm CV trên train (4 giai đoạn: tune, extremes, grouping, combine): **xong**, kết quả ở `reports/tables/week3_fold_results.csv` (cùng chữ ký `week3_signature.json`).
- Cấu hình **đã đóng băng**: `reports/week3_frozen_config.json` (mô hình được chọn bằng CV: Random Forest, MAE CV ≈ 7.351).
  - KNN: Target + `model_version` (9.154); Ridge: One-hot + `model_version`, cắt phân vị 99,9%, log giá (9.441); SVR: C=1, ε=0,1, γ=0,012, log giá (7.977); Random Forest: max_depth=15, max_features=0,5, 200 cây (7.351).
  - 1 cấu hình SVR cực đoan bị bỏ (xem `week3_skipped_configs.json`); 2 mốc tham khảo "tốt nhất theo CV nhưng quá đắt" (SVR 6.818, RF 7.221) được đánh giá trên tập con 20k.

## Đang chạy / chưa xong
- `python -m src.final_eval` (đánh giá cuối trên test, **chỉ được chạy một lần**) đã được khởi chạy nền và còn vài mô hình SVR đang huấn luyện. Khi xong nó sẽ tạo:
  `reports/tables/week3_test_evaluation_done.json` (file đánh dấu), `week3_test_metrics.csv`, `week3_test_bootstrap_ci.csv`, `data/processed/week3_test_predictions.parquet`.
- Kiểm tra: `Test-Path reports\tables\week3_test_evaluation_done.json`.
  - Nếu **có**: xong, làm các bước "Làm tiếp" bên dưới.
  - Nếu **chưa có và không còn tiến trình `src.final_eval`** (bị tắt giữa chừng): chạy lại `python -m src.final_eval` (được phép vì chưa có file đánh dấu; mất khoảng 35 phút). Đừng chạy khi tiến trình cũ còn sống.
  - Nếu **đã có** file đánh dấu: KHÔNG chạy lại (chương trình sẽ từ chối); dùng dự đoán đã lưu.

## Làm tiếp (theo thứ tự)
```powershell
python -m src.week3_analysis        # bảng/biểu đồ CV + phân tích phần dư test
python -m src.reporting_week3       # sinh reports/week3_report.md (tự kiểm chứng phát biểu)
python build_week3_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_week3_results.ipynb
python -m pytest -q                 # kỳ vọng tất cả đạt
```
Rồi cần viết: `HUONG_DAN_TUAN3.md` (cách xử lý từng vấn đề, như tuần 2), cập nhật `README.md` (mục Tuần 3), cập nhật `reports/tables/pytest_summary.txt`, rồi commit/push.

## Điểm cần nhớ khi viết báo cáo/hướng dẫn
- Quy tắc chọn: thành phần tùy chọn chỉ được nhận nếu MAE thấp hơn **và** thắng ≥ 4/5 fold; siêu tham số bị ràng buộc chi phí (ước tính thời gian huấn luyện trên toàn train ≤ 30 phút).
- Chi phí SVR rất lớn (O(n²)); log giá làm SVR chậm hơn nhiều lần.
- Ridge: log giá giảm MAE CV từ 22.681 xuống ~9.500.
- Lỗi/vấn đề đã gặp: một cấu hình SVR chạy > 1 giờ không xong một fold (đã hủy, ghi lại); lệnh dừng tiến trình diện rộng bị từ chối nên chỉ dừng đúng 5 worker của lần chạy; `choose_final` có lỗi lập chỉ mục (đã sửa); `tune_vs_default` ban đầu không xét chi phí (đã sửa); test bảo vệ tập test bắt được chuỗi "test.parquet" trong docstring (đã sửa).
- Không chạy lại `python -m src.week3_experiments` với tham số khác khi chưa xóa file kết quả và chữ ký (chương trình sẽ từ chối).
