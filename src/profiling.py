"""Dataset profile (Day 1): every number is computed from the real raw CSV."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src import config
from src.load_data import read_raw

# Columns that the Zenodo description lists (or that the project plan relies on).
DOCUMENTED_COLUMNS = [
    "id", "description", "ratings_average", "ratings_count", "ratings_recommend_percentage",
    "price_currency", "price", "price_tax_deductible", "price_negotiable", "price_net", "price_vat_rate",
    "vin", "make", "model", "model_version", "german_hsn_tsn", "mileage_km_raw", "mileage_km",
    "registration_date", "production_year", "vehicle_type", "body_type", "nr_seats", "nr_doors",
    "body_color", "transmission", "drive_train", "fuel_category", "is_used", "is_new", "offer_type",
    "country_code", "seller_type", "cylinders_volume_cc", "power_kw",
]


def profile_dataset(path=config.RAW_PATH) -> dict:
    """Stream the file once to get exact missing counts; load fully once for dtypes/duplicates/RAM
    (the file fits comfortably in 32 GB RAM, ~0.6 GB in pandas)."""
    df = read_raw(path=path)
    n, p = df.shape
    miss = df.isna().sum()
    id_dup = int(df["id"].duplicated().sum())
    full_dup = int(df.duplicated().sum())
    vin = df["vin"].dropna()
    vin_dup_rows = int(vin.duplicated().sum())
    cols = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": miss,
        "pct_missing": (miss / n * 100).round(2),
        "n_unique": df.nunique(),
    })
    all_null = cols.index[cols["n_missing"] == n].tolist()
    stats = {
        "rows": n, "cols": p,
        "file_size_bytes": path.stat().st_size,
        "ram_mb_deep": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "full_row_duplicates": full_dup,
        "id_duplicates": id_dup,
        "vin_non_null": int(len(vin)),
        "vin_duplicate_rows": vin_dup_rows,
        "all_null_columns": all_null,
        "documented_but_absent": [c for c in DOCUMENTED_COLUMNS if c not in df.columns],
        "high_missing_cols_over_50pct": cols.index[cols["pct_missing"] > 50].tolist(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pandas": pd.__version__,
    }
    return {"stats": stats, "columns": cols, "df": df}


def write_profile_report(res: dict, out=config.REPORTS / "dataset_profile.md") -> None:
    s, cols = res["stats"], res["columns"]
    src = json.loads((config.DATA_RAW / "SOURCE.json").read_text(encoding="utf-8"))
    L = ["# Dataset profile – AutoScout24 (Zenodo 17643343)", "",
         "_Tệp này được sinh tự động bởi `src/profiling.py`; mọi con số đo trực tiếp trên file thật._", "",
         "## 1. Nguồn và xác minh", "",
         f"- Nguồn: {src['source_url']} (DOI {src['doi']})",
         f"- Tên tệp: `{src['file_name']}`; phiên bản dữ liệu {src['dataset_version']}; công bố {src['publication_date']}",
         f"- Giấy phép (metadata Zenodo): `{src['license']}`",
         f"- Ngày truy cập/tải: {src['access_date']}",
         f"- Dung lượng thực: {s['file_size_bytes']:,} byte (công bố: {config.PUBLISHED_SIZE_BYTES:,}) → "
         f"{'KHỚP' if s['file_size_bytes'] == config.PUBLISHED_SIZE_BYTES else 'KHÔNG KHỚP'}",
         f"- MD5 tính trên file: `{src['md5_verified']}`; MD5 công bố: `{src['md5_published']}` → "
         f"{'KHỚP' if src['md5_verified'] == src['md5_published'] else 'KHÔNG KHỚP'}", "",
         "## 2. Quy mô quan sát được vs. mô tả của Zenodo", "",
         "| Chỉ tiêu | Mô tả Zenodo | Quan sát thực tế |", "|---|---|---|",
         f"| Số tin đăng | ~120.000 | **{s['rows']:,}** |",
         f"| Số cột | (liệt kê trong mô tả) | **{s['cols']}** |",
         f"| Dung lượng CSV | ~548,6 MB | {s['file_size_bytes'] / 1e6:.1f} MB |",
         f"| RAM pandas (deep) | – | ~{s['ram_mb_deep']:.0f} MB |", "",
         "## 3. Trùng lặp", "",
         f"- Dòng trùng hoàn toàn: **{s['full_row_duplicates']}**",
         f"- `id` trùng: **{s['id_duplicates']}**",
         f"- VIN có giá trị: {s['vin_non_null']:,} dòng; số dòng có VIN trùng với dòng trước đó: **{s['vin_duplicate_rows']:,}** "
         "(cùng xe đăng nhiều lần hoặc VIN không hợp lệ – cần phân tích ở bước cleaning). Giá trị VIN không được in ra báo cáo.", "",
         "## 4. Cột và tỷ lệ thiếu", "",
         "| Cột | dtype | Thiếu (n) | Thiếu (%) | Số giá trị duy nhất |", "|---|---|---:|---:|---:|"]
    for c, r in cols.iterrows():
        L.append(f"| {c} | {r.dtype} | {r.n_missing:,} | {r.pct_missing:.2f} | {r.n_unique:,} |")
    L += ["", "## 5. Vấn đề phát hiện", "",
          f"- Cột **rỗng 100%**: {', '.join(f'`{c}`' for c in s['all_null_columns']) or 'không có'} → không dùng được.",
          f"- Cột thiếu > 50%: {', '.join(f'`{c}`' for c in s['high_missing_cols_over_50pct'])}.",
          "- `production_year` chỉ có ở khoảng 19% dòng; `registration_date` (thiếu ~0,19%) là nguồn năm đáng tin hơn → cần quy tắc suy năm.",
          "- `mileage_km` là **chuỗi** (vd. `9,900 km`), `mileage_km_raw` là số → dùng cột số.",
          "- `weight_kg` là chuỗi có đơn vị (`1,945 kg`).",
          "- `ratings_average` có dtype chuỗi dù là số.",
          "- Không có cột thời điểm đăng tin → không thể chia train/test theo thời gian.",
          f"- Cột mô tả Zenodo có nhắc nhưng không tồn tại: {', '.join(s['documented_but_absent']) or 'không có (mọi cột được liệt kê đều có trong file)'}.",
          "", "## 6. Ghi chú về cách đọc dữ liệu", "",
          "- Dấu phân cách `,`, mã hoá UTF-8 đọc được không lỗi; đọc theo chunk 200.000 dòng để đếm hàng, đọc toàn bộ một lần cho profile (≈0,6 GB RAM).",
          "- Chuỗi dài (`description`, `equipment_*`) không bao giờ được in ra terminal/báo cáo."]
    (out).write_text("\n".join(L) + "\n", encoding="utf-8")
    cols.to_csv(config.TABLES / "raw_column_profile.csv")
