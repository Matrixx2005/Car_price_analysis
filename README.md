# Dự đoán giá ô tô cũ trên dữ liệu có nhiều thuộc tính phân loại

Dự án môn Học máy – **Tuần 1: dữ liệu, EDA, cardinality, schema, cleaning, train/test split an toàn.**
Dataset: AutoScout24 Car Listings 2025 – Zenodo [17643343](https://zenodo.org/records/17643343), DOI `10.5281/zenodo.17643343` (giấy phép MIT theo metadata Zenodo).

Tuần 1 **không** có encoding, scaling hay mô hình hồi quy (để dành tuần 2–3).

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
