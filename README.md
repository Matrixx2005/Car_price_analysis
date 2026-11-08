# Dự đoán giá ô tô cũ trên dữ liệu có nhiều thuộc tính phân loại

Dự án môn Học máy – dự đoán giá chào bán ô tô cũ (hồi quy), so sánh các phương pháp mã hóa và chuẩn hóa dữ liệu.
Dataset: AutoScout24 Car Listings 2025 – Zenodo [17643343](https://zenodo.org/records/17643343), DOI `10.5281/zenodo.17643343` (giấy phép MIT theo metadata Zenodo).

Tiến độ:
- **Tuần 1:** dữ liệu, EDA, cardinality, schema, làm sạch, chia train/test theo nhóm xe (không có mô hình).
- **Tuần 2:** lưới Encoding × Scaling × Mô hình (KNN, Ridge, SVR, Random Forest) bằng GroupKFold trên train; phân tích bổ sung sai số train–validation, baseline theo fold, đường cong học tập, phần dư và độ quan trọng đặc trưng (`src/week2_extra.py`).
- **Tuần 3:** tinh chỉnh, đóng băng cấu hình, đánh giá cuối một lần trên test.

## Bắt đầu nhanh (người mới clone repo)

Yêu cầu: Python 3.13 (3.12 trở lên), Git, khoảng 2 GB ổ đĩa trống, kết nối mạng (tải 548 MB dữ liệu từ Zenodo, một lần).

```powershell
git clone https://github.com/Matrixx2005/Car_price_analysis.git
cd Car_price_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py                  # tải dữ liệu, làm sạch, chia train/test, EDA, báo cáo tuần 1 (~2-5 phút)
python -m pytest -q             # kỳ vọng: tất cả đạt
python demo.py                  # in các bảng số liệu chính
```
Linux/macOS: thay `.\.venv\Scripts\Activate.ps1` bằng `source .venv/bin/activate`.

**Đã có sẵn trong repo (không cần chạy lại):** mọi bảng, biểu đồ, báo cáo của tuần 1-3 (`reports/`), notebook đã chạy (`notebooks/`), cấu hình đóng băng và dự đoán test của tuần 3.
**Được tạo khi chạy `python main.py`:** dữ liệu gốc, `data/interim/`, `data/processed/train.parquet` và `test.parquet` (không đưa lên GitHub vì lớn hoặc chứa VIN).

Sau `python main.py` bạn có thể chạy các lệnh của tuần 2-3 bên dưới (cần `train.parquet`). Lưu ý:
- Các lệnh thí nghiệm có "chữ ký" chống trộn kết quả; nếu báo chữ ký khác (ví dụ khác phiên bản thư viện), xóa file kết quả và file chữ ký tương ứng rồi chạy lại, hoặc cứ dùng kết quả có sẵn.
- **Không chạy lại `python -m src.final_eval`**: tập test đã được đánh giá một lần (file `reports/tables/week3_test_evaluation_done.json`), chương trình sẽ từ chối.
- Tái lập đúng phiên bản thư viện (Windows): `pip install -r requirements-lock.txt`.

## Chạy lại toàn bộ từ dataset gốc (PowerShell, Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py                 # tải + kiểm MD5 + profile + clean + split + EDA + báo cáo
.\.venv\Scripts\python -m pytest -v            # unit test
# (tuỳ chọn) dựng lại notebook EDA đã chạy sẵn:
.\.venv\Scripts\python build_notebook.py
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb
```

`main.py` chỉ điều phối; logic nằm trong `src/`. Chạy lại cho cùng kết quả (seed 42, đã kiểm tra manifest chia dữ liệu giống hệt giữa hai lần chạy).

## Tuần 2: Encoding × Scaling × Mô hình (chạy trên tập train, không dùng test)

```powershell
python -m src.experiments              # lưới 112 cấu hình x 5 fold (~10 phút); chạy tiếp được nếu bị dừng
python -m src.experiments --confirm    # xác nhận trên toàn bộ train (~16-20 phút); chạy tiếp được
python -m src.week2_analysis           # bảng tổng hợp + biểu đồ
python -m src.reporting_week2          # sinh reports/week2_report.md (tự kiểm chứng 14 phát biểu)
python -m src.week2_extra              # phân tích bổ sung (~10-20 phút, 12 lõi); --quick để chạy thử, --plots để vẽ lại hình
python -m pytest -v                    # 70 test
```
`src.week2_extra` chạy lại đúng cấu hình tốt nhất của mỗi mô hình trên cùng tập con 20.001 dòng và cùng 5 GroupKFold, ghi thêm:
sai số trên phần train và phần validation từng fold (nhận diện quá khớp/chưa khớp), baseline trung vị theo fold,
đường cong học tập (20–100 % phần train của fold, lấy nguyên nhóm xe), dự đoán ngoài fold (biểu đồ phần dư) và
permutation importance theo cột gốc. Kết quả: `reports/tables/week2_extra_*.csv`, `reports/figures/week2_extra_*.png`,
`reports/week2_extra_meta.json` (có kiểm tra MAE validation khớp với `week2_fold_results.csv`). Không đọc tập test.
Mỗi lần chạy có "chữ ký" (dữ liệu, số fold, seed, siêu tham số, phiên bản sklearn); đổi những thứ đó thì chương trình từ chối ghi tiếp vào kết quả cũ (xóa file kết quả và file chữ ký tương ứng để chạy lại từ đầu).
Xem `HUONG_DAN_TUAN2.md` (cách xử lý từng vấn đề và các lỗi đã sửa), `reports/week2_report.md` (kết quả) và `notebooks/03_week2_results.ipynb`.
Để tái lập đúng phiên bản thư viện: `pip install -r requirements-lock.txt`.

## Tuần 3: tinh chỉnh, đóng băng cấu hình, đánh giá cuối trên test

```powershell
python -m src.week3_experiments --stage all   # 4 giai đoạn CV trên train (chạy tiếp được)
python -m src.week3_select                    # đóng băng cấu hình
python -m src.final_eval                      # đánh giá test, CHỈ MỘT LẦN (chương trình từ chối lần hai)
python -m src.week3_analysis                  # bảng/biểu đồ + phân tích phần dư
python -m src.reporting_week3                 # sinh reports/week3_report.md
python -m pytest -v                           # 70 test
```
Mọi lựa chọn chỉ dựa trên CV trên train; tập test chỉ được đọc bởi `src/final_eval.py`. Xem `HUONG_DAN_TUAN3.md` và `reports/week3_report.md`.

## Lưu ý: không sửa file trong `data/raw/`
Đừng mở CSV gốc bằng Excel/editor rồi lưu: kích thước/MD5 đổi, `download_dataset` sẽ báo `Existing file is NOT valid` và `main.py` dừng. Cách xử lý: đổi tên file hỏng (vd. thêm đuôi `.modified_backup`, đã được `.gitignore` chặn) rồi chạy lại `python -m src.download_dataset` để tải bản chuẩn.

## Tải dữ liệu thủ công (nếu mạng lỗi)
Tải `autoscout24_dataset_20251108.csv` (548.610.318 byte, MD5 `b23a122cc51baf7de39f449193ff0d28`) từ trang Zenodo ở trên, đặt vào `data/raw/`. File CSV lớn không được commit (xem `.gitignore`).

## Cấu trúc

| Thư mục/tệp | Nội dung |
|---|---|
| `src/download_dataset.py` | Tải streaming, `.part`, retry giới hạn, kiểm MD5, không ghi đè file hợp lệ |
| `src/load_data.py`, `src/profiling.py` | Đọc CSV, profile → `reports/dataset_profile.md` |
| `src/cleaning.py`, `src/data_validation.py` | Lọc xe cũ/EUR/giá hợp lệ, chuẩn hoá, log từng bước; kiểm tra dữ liệu |
| `src/adapters/` | `schema.py` (common schema), `autoscout24_adapter.py` (sau này thêm `vietnam_adapter.py`) |
| `src/feature_engineering.py` | `vehicle_age` (năm snapshot 2025), `make_model`, audit feature/leakage |
| `src/split_data.py` | GroupShuffleSplit theo nhóm xe (VIN ∪ dấu vân tay kỹ thuật), seed 42, 80/20 |
| `src/eda.py`, `src/reporting.py` | Bảng/biểu đồ và báo cáo Markdown sinh từ số liệu thật |
| `data/interim/` | Dữ liệu trung gian (parquet; chứa VIN nên không commit) |
| `data/processed/` | `train.parquet`, `test.parquet`, `split_manifest.csv`, `split_metadata.json` |
| `reports/` | `week1_report.md` (báo cáo chính), `cleaning_log.md`, `feature_selection.md`, `leakage_risks.md`, `data_dictionary.md`, `dataset_profile.md`, `validation_report.md`, `stats.json`, `figures/`, `tables/` |
| `notebooks/01_exploratory_data_analysis.ipynb` | EDA đầy đủ (đã chạy, có output) |

## Lineage
Hash raw, số dòng từng bước, quy tắc, phiên bản thư viện, seed, ngày xử lý: `data/raw/SOURCE.json`, `reports/stats.json`, `reports/tables/scope_funnel.csv`, `data/processed/split_metadata.json`.

## Quy tắc chống leakage (tuần 2 phải tuân thủ)
Imputer/scaler/encoder chỉ `fit` trên train (trong Pipeline/CV); không dùng test để chọn feature hay hyperparameter; không đưa `price_*`, VIN, id, tên người bán vào X. Chi tiết: `reports/leakage_risks.md`.
