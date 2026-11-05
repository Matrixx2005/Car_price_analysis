"""Demo for presenting week 1: prints the key tables from real outputs (run `python main.py` first).

    python demo.py            # all sections
    python demo.py 1          # only section 1 (raw data), 2 = cleaning, 3 = target, 4 = categorical, 5 = split
"""
import json
import sys

import pandas as pd

from src import config
from src.load_data import peek_csv, truncate_long_strings

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 40,
              "display.float_format", lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
T = config.TABLES
stats = json.loads((config.REPORTS / "stats.json").read_text(encoding="utf-8"))


def title(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def s1():
    title("1. DỮ LIỆU GỐC (AutoScout24, Zenodo 17643343)")
    src = json.loads((config.DATA_RAW / "SOURCE.json").read_text(encoding="utf-8"))
    raw = stats["raw"]
    print(f"File: {src['file_name']} | {src['size_bytes']:,} byte | MD5 khớp: {src['md5_verified'] == src['md5_published']}")
    print(f"Số dòng x số cột: {raw['rows']:,} x {raw['cols']} | dòng trùng toàn bộ: {raw['full_row_duplicates']} | id trùng: {raw['id_duplicates']}")
    print("Cột rỗng 100%:", raw["all_null_columns"])
    print("\n5 dòng đầu (một số cột, chuỗi dài đã cắt):")
    head = peek_csv(nrows=5)["head"]
    print(truncate_long_strings(head[["id", "make", "model", "price", "price_currency", "mileage_km_raw",
                                      "registration_date", "fuel_category", "transmission", "country_code", "is_used", "is_new"]]))
    print("\n10 cột thiếu nhiều nhất:")
    print(pd.read_csv(T / "raw_column_profile.csv", index_col=0).sort_values("pct_missing", ascending=False).head(10))


def s2():
    title("2. LÀM SẠCH: PHỄU LỌC XE CŨ")
    print(pd.read_csv(T / "scope_funnel.csv")[["step", "rows_before", "rows_after", "rows_removed", "reason"]].to_string(index=False))
    print("\nGiá trị kỹ thuật vô lý đặt thành NaN (giữ dòng):")
    print(pd.read_csv(T / "implausible_values_log.csv").to_string(index=False))
    print("\nKiểm tra validation sau cleaning (mức ERROR/WARNING có số dòng > 0):")
    v = pd.read_csv(T / "validation_after_cleaning.csv")
    print(v[(v["severity"] != "INFO") & (v["count"] > 0)].to_string(index=False) or "Không có")


def s3():
    title("3. BIẾN MỤC TIÊU: GIÁ (EUR) SAU LÀM SẠCH")
    print(pd.read_csv(T / "price_statistics.csv").to_string(index=False))
    print("\nSố xe và giá trung vị theo quốc gia:")
    print(pd.read_csv(T / "country_summary.csv")[["country_code", "n", "share_pct", "median_price"]].to_string(index=False))


def s4():
    title("4. BIẾN PHÂN LOẠI: CARDINALITY (điểm trọng tâm của đề tài)")
    c = pd.read_csv(T / "categorical_cardinality.csv")
    print(c[["column", "n_unique", "level", "pct_missing", "n_cat_count_eq1", "n_cat_count_lt10", "pct_rows_in_cats_lt10"]].to_string(index=False))
    print("\nTop 10 hãng xe:")
    print(pd.read_csv(T / "brand_top20.csv").head(10).to_string(index=False))
    print("\nTỷ lệ dòng TEST có danh mục chưa từng thấy ở TRAIN:")
    print(pd.read_csv(T / "unseen_categories_test_vs_train.csv").to_string(index=False))


def s5():
    title("5. CHIA TRAIN/TEST CHỐNG RÒ RỈ")
    sp, rl = stats["split"], stats["random_split_leak"]
    print(f"Phương pháp: {sp['method']} | seed {sp['seed']}")
    print(f"Train {sp['n_train']:,} | Test {sp['n_test']:,} ({sp['test_fraction'] * 100:.2f}% test)")
    print(f"Nhóm xe có nhiều tin đăng: {sp['n_groups_with_multiple_listings']:,} nhóm ({sp['listings_in_multi_listing_groups']:,} tin)")
    print(f"Nhóm xuất hiện ở cả train và test: {sp['group_overlap_train_test']}  (chia ngẫu nhiên thường sẽ để {rl['test_rows_with_same_group_in_train']:,} dòng test có xe trùng ở train)")
    tr = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    print("\n5 dòng đầu của train.parquet (X + y = price):")
    print(tr.drop(columns=["group_id"]).head(5))
    print("\nCác cột:", list(tr.columns))


if __name__ == "__main__":
    sections = {"1": s1, "2": s2, "3": s3, "4": s4, "5": s5}
    for k in (sys.argv[1:] or sections):
        sections[k]()
