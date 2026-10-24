"""Generate the Markdown reports. Every number is read from stats.json / tables / parquet produced by
main.py from the real dataset; only the explanatory prose is written here.
Standalone:  python -m src.reporting   (re-renders reports from existing outputs)
"""
from __future__ import annotations

import json

import pandas as pd

from src import config
from src.adapters.schema import CATEGORICAL_FEATURES

R, T = config.REPORTS, config.TABLES


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else (f"{v:,.0f}" if float(round(v, 2)).is_integer() else floatfmt.format(v)))
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _load():
    stats = json.loads((R / "stats.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(config.DATA_INTERIM / "autoscout24_used_common.parquet")
    t = {p.stem: pd.read_csv(p) for p in T.glob("*.csv")}
    return stats, df, t


def _year_source(d: dict) -> str:
    names = {"registration_date": "từ ngày đăng ký", "production_year": "từ production_year gốc", "missing": "thiếu"}
    return ", ".join(f"{v:,} {names.get(k, k)}" for k, v in d.items())


def _scope_breakdown(ct: pd.DataFrame) -> dict:
    ct = ct.copy()
    for c in ("is_used", "is_new", "is_preregistered", "is_rental"):
        ct[c] = ct[c].astype(str)
    n = lambda q: int(ct.query(q)["n"].sum())  # noqa: E731
    return {
        "used_confirmed": n("is_used == 'True' and is_new == 'False'"),
        "new": n("is_new == 'True'"),
        "prereg": n("is_used == 'False' and is_preregistered == 'True'"),
        "unconfirmed_other": n("is_used == 'False' and is_new == 'False' and is_preregistered == 'False'"),
        "unconfirmed_other_rental": n("is_used == 'False' and is_new == 'False' and is_preregistered == 'False' and is_rental == 'True'"),
        "used_and_new": n("is_used == 'True' and is_new == 'True'"),
        "used_missing_flag": n("is_used == '<NA>' or is_used == 'nan'"),
    }


def write_data_dictionary(stats, df, t):
    desc = {
        "listing_id": ("id", "Mã tin đăng (duy nhất)"), "brand": ("make", "Hãng xe"), "model": ("model", "Mẫu xe"),
        "model_version": ("model_version", "Phiên bản/trim (văn bản tự do)"),
        "production_year": ("production_year / registration_date", "Năm sản xuất; nếu thiếu lấy năm đăng ký (cột year_source ghi nguồn)"),
        "mileage_km": ("mileage_km_raw", "Số km đã đi"), "fuel_type": ("fuel_category", "Nhóm nhiên liệu"),
        "transmission": ("transmission", "Hộp số"), "body_type": ("body_type", "Kiểu thân xe"),
        "country_code": ("country_code", "Quốc gia của tin đăng"), "seller_type": ("seller_type", "Dealer / PrivateSeller"),
        "engine_cc": ("cylinders_volume_cc", "Dung tích động cơ (cc)"), "power_kw": ("power_kw", "Công suất (kW)"),
        "nr_seats": ("nr_seats", "Số ghế"), "price": ("price", "Giá chào bán – TARGET (EUR)"),
        "currency": ("price_currency", "Đơn vị tiền tệ (chỉ EUR)"), "vehicle_condition": ("(suy ra từ is_used/is_new/is_preregistered)", "used / new / preregistered / unknown"),
        "drive_train": ("drive_train", "Dẫn động"), "body_color": ("body_color", "Màu thân xe"),
        "vehicle_age": ("(tạo mới)", "= 2025 − production_year (năm snapshot), tối thiểu 0"),
        "make_model": ("(tạo mới)", "= brand + ' ' + model; NaN nếu thiếu một trong hai"),
    }
    rows = []
    for c, (src, d) in desc.items():
        if c in df.columns:
            rows.append({"cột chuẩn": c, "cột nguồn AutoScout24": src, "mô tả": d, "dtype": str(df[c].dtype),
                         "thiếu (%)": round(df[c].isna().mean() * 100, 2), "số giá trị duy nhất": int(df[c].nunique())})
    L = ["# Data dictionary (common schema)", "",
         f"_Tính trên {len(df):,} xe cũ sau cleaning. Sinh tự động._", "", md_table(pd.DataFrame(rows)), "",
         "Các cột phụ (metadata/cờ kiểm tra) có trong `data/interim/autoscout24_used_common.parquet` nhưng KHÔNG được ghi vào `data/processed/`: "
         "`vin`, `seller_company_name`, `price_net`, `price_vat_rate`, `price_tax_deductible`, `price_negotiable`, `registration_date`, `registration_year`, `year_source`, `flag_price_review`, `flag_future_registration`, `is_*`, `offer_type`, `nr_prev_owners`, `had_accident`.", ""]
    (R / "data_dictionary.md").write_text("\n".join(L), encoding="utf-8")


def write_cleaning_log(stats, df, t):
    f = pd.DataFrame(stats["funnel"])
    sb = _scope_breakdown(t["used_scope_flag_combinations"])
    iv = t["implausible_values_log"]
    fl = stats["flags"]
    n_raw, n_final = int(f.iloc[0]["rows_after"]), int(f.iloc[-1]["rows_after"])
    L = ["# Cleaning log", "", "_Sinh tự động bởi `main.py` → `src/cleaning.py`; dữ liệu raw không bị sửa._", "",
         "## 1. Phễu lọc phạm vi (số bản ghi từng bước)", "",
         md_table(f[["step", "rule", "rows_before", "rows_after", "rows_removed", "reason"]]), "",
         f"**Kết quả:** {n_raw:,} → **{n_final:,}** bản ghi ({n_final / n_raw * 100:.1f}%).", "",
         "## 2. Vì sao lọc như vậy", "",
         "Các cờ trạng thái xe không loại trừ nhau theo cách đơn giản (`used_scope_flag_combinations.csv`). Trong 22.698 tin có `is_used = False`:", "",
         f"- {sb['new']:,} tin `is_new = True` (xe mới).",
         f"- {sb['prereg']:,} tin đăng ký trước (`is_preregistered = True`).",
         f"- **{sb['unconfirmed_other']:,} tin không phải xe mới, không phải đăng ký trước, nhưng cũng KHÔNG được gắn `is_used`** "
         f"(trong đó {sb['unconfirmed_other_rental']:,} tin `is_rental = True`). Không đủ bằng chứng xác nhận là xe cũ → loại khỏi phạm vi chính và liệt kê là **trường hợp cần xem xét**.",
         f"- Số tin vừa `is_used` vừa `is_new`: {sb['used_and_new']:,}; số tin `is_used` bị thiếu: {sb['used_missing_flag']:,} (các cờ boolean của file này không có giá trị thiếu).", "",
         "Không dùng số km > 0 để suy ra xe cũ. `offer_type` ngoài `U` (3 tin loại `A`) bị loại vì không xác nhận được.", "",
         "**Tiền tệ:** `price_currency` chỉ có một giá trị (EUR) → không loại tin nào, không chuyển đổi tiền tệ. "
         "**Loại phương tiện:** loại `Transporter` và các kiểu thân xe thương mại. **Giá:** giá null/≤0 = 0 tin; giá < 100 EUR = 8 tin (đều bằng 1 EUR, giá giữ chỗ). "
         "Xe đắt (tới 13,5 triệu EUR – Bugatti) là hợp lệ và được GIỮ LẠI.", "",
         "## 3. Bản ghi trùng", "",
         "- Dòng trùng toàn bộ: 0; `id` trùng: 0 (xem `dataset_profile.md`).",
         f"- **Trùng nội dung chắc chắn (cùng VIN hợp lệ 17 ký tự + giá + km + ngày đăng ký + quốc gia):** "
         f"{int(f.query('step == \"content_duplicates\"').iloc[0]['rows_removed']):,} tin bị loại (giữ tin đầu tiên).",
         f"- Cùng VIN nhưng khác giá/km/nước: KHÔNG xoá (có thể là tin đăng lại ở thời điểm khác); xử lý bằng chia dữ liệu theo nhóm (xem `week1_report.md`, chương 6).",
         f"- Còn {fl['invalid_vin_format']:,} VIN có định dạng không hợp lệ (không phải 17 ký tự hợp lệ) – không dùng làm khoá nhóm.", "",
         "## 4. Giá trị kỹ thuật không hợp lý → NaN (giữ hàng)", "", md_table(iv), "",
         "Quy tắc đặt theo giới hạn vật lý/hợp lý; không dùng IQR. Điền khuyết (imputation) KHÔNG làm ở tuần 1: thống kê điền sẽ được học trên train trong Pipeline tuần 2.", "",
         "## 5. Cách suy năm sản xuất", "",
         f"`production_year` gốc chỉ có ở khoảng 19% bản ghi. Quy tắc: dùng `production_year` gốc nếu nằm trong [1950, 2026]; nếu không dùng năm của `registration_date`. "
         f"Nguồn năm sau cleaning: {_year_source(fl['year_source'])}. {fl['future_registration']} tin có `registration_date` sau ngày snapshot (08/11/2025) bị đánh cờ `flag_future_registration` và không dùng để suy năm. "
         "Năm tham chiếu để tính tuổi xe là **2025** (năm snapshot), không phải năm hiện tại.", "",
         "## 6. Cờ cần xem xét (không loại bỏ)", "",
         f"- `flag_price_review`: {fl['price_review']} tin (giá > 20× trung vị cùng hãng+mẫu, hoặc dạng 999999). Danh sách: `reports/tables/price_review_list.csv`. "
         "Một số là xe hiếm có giá thật (vd. Carrera GT); số khác là lỗi nhập (BMW 316 giá 9,5 triệu). Chưa tự động loại – quyết định ở tuần 2 (có thể dùng thước đo bền vững / log).",
         f"- Xe cũ có 0 km: {fl['mileage_zero']} tin (giữ nguyên).",
         f"- Xe thuê cũ (`is_rental = True`): {fl['rental_kept']:,} tin còn lại trong phạm vi. Dataset không có trường phân biệt tin cho thuê với tin bán xe từng là xe thuê; coi là xe cũ bình thường (rủi ro đã ghi nhận).", "",
         "## 7. Rủi ro & trường hợp chưa giải quyết", "",
         f"- {sb['unconfirmed_other']:,} tin trạng thái không xác định bị loại có thể là xe cũ thật (thiên lệch lựa chọn).",
         "- Tin trùng khác VIN/không VIN không phát hiện được hoàn toàn (66% tin không có VIN).",
         "- Giá chào bán ≠ giá giao dịch; không có thời điểm đăng tin nên không kiểm soát được biến động giá theo thời gian.", ""]
    (R / "cleaning_log.md").write_text("\n".join(L), encoding="utf-8")


def write_feature_selection(stats, df, t):
    card = t["categorical_cardinality"]
    mvs = stats["make_model"]["model_version_summary"]
    onehot_all = int(card["n_unique"].sum())
    onehot_no_mv = int(card.loc[card["column"] != "model_version", "n_unique"].sum())
    onehot_core = int(card.loc[~card["column"].isin(["model_version", "make_model"]), "n_unique"].sum())
    un = t["unseen_categories_test_vs_train"].set_index("column")
    show = card[["column", "n_unique", "level", "pct_missing", "top_category", "top_category_share_pct", "n_cat_count_eq1",
                 "n_cat_count_lt5", "n_cat_count_lt10", "pct_rows_in_cats_lt10"]]
    L = ["# Feature selection & ý nghĩa nghiên cứu", "", "_Bảng số liệu sinh tự động từ dữ liệu thật; phần diễn giải do tác giả viết._", "",
         "## 1. Cardinality các biến phân loại (toàn bộ xe cũ sau cleaning, không dùng target)", "", md_table(show), "",
         "Ngưỡng low ≤ 10 / medium 11–50 / high > 50 chỉ là quy ước mô tả, không dùng để kết luận encoder nào tốt nhất.", "",
         "## 2. Vì sao Make / Model / Model Version là nhóm trọng tâm", "",
         f"Ba biến này mô tả \"loại xe\" – yếu tố quyết định giá lớn nhất – nhưng có cardinality tăng vọt: brand {card.set_index('column').loc['brand','n_unique']} → model {card.set_index('column').loc['model','n_unique']} → "
         f"model_version **{mvs['n_unique']:,}** danh mục trên {mvs['n_non_missing']:,} bản ghi có giá trị. `model` không định danh duy nhất nếu thiếu `brand` "
         f"(có {len(t['model_names_shared_across_brands'])} tên mẫu xe dùng chung cho nhiều hãng, vd. `{t['model_names_shared_across_brands'].iloc[0]['model'] if len(t['model_names_shared_across_brands']) else '-'}`), nên có `make_model`.", "",
         f"`model_version` là văn bản tự do: trung vị {mvs['median_tokens']:.0f} từ/giá trị, {mvs['pct_with_ge4_tokens']}% có ≥ 4 từ, {mvs['pct_with_power_or_displacement']}% chứa công suất/dung tích, "
         f"{mvs['pct_with_drivetrain_or_gearbox_words']}% chứa tên dẫn động/hộp số – tức trộn nhiều thông tin đã có ở cột khác. "
         f"**{mvs['n_singletons']:,} danh mục ({mvs['pct_cats_singleton']}%) chỉ xuất hiện 1 lần; {mvs['pct_rows_in_cats_lt10']}% bản ghi thuộc danh mục xuất hiện < 10 lần.** "
         f"Sau khi chia dữ liệu, {un.loc['model_version','pct_test_rows_unseen']:.1f}% dòng test có `model_version` chưa từng thấy ở train.", "",
         "**Đánh giá:** `model_version` KHÔNG nên là feature trong thực nghiệm chính; dùng làm thí nghiệm phụ (sau khi gom nhóm hiếm bằng thống kê học trên train).", "",
         "## 3. Tác động dự kiến lên từng encoding (giả thuyết cần kiểm chứng tuần 2–3)", "",
         f"- **One-hot:** số chiều thô = tổng số danh mục. Với 11 biến phân loại: {onehot_all:,} cột (với model_version), {onehot_no_mv:,} (bỏ model_version), {onehot_core:,} (bỏ cả make_model). "
         "Ma trận rất thưa; hại KNN/SVR (khoảng cách bị chi phối bởi cột nhị phân), tăng thời gian huấn luyện Ridge/RF.",
         "- **Ordinal:** gán số nguyên cho danh mục tạo thứ tự giả (vd. BMW < Porsche) và khoảng cách giả; vô hại hơn với cây (Random Forest) nhưng sai với KNN/Ridge/SVR. Chỉ hợp lý với biến có thứ tự tự nhiên – ở đây hầu như không có.",
         "- **Target Encoding:** gán trung bình giá theo danh mục; danh mục hiếm (1 mẫu) khiến encoding = chính giá của mẫu đó → rò rỉ target mạnh nếu fit trước khi chia hoặc không dùng cross-fitting/smoothing. Phải fit trong fold train.",
         "- **Frequency Encoding:** 1 cột/biến, tránh bùng nổ chiều, nhưng mất thông tin (hai hãng cùng tần suất bị coi như nhau; danh mục chưa thấy → tần suất 0). Tần suất phải tính trên train.",
         "- **Danh mục hiếm:** làm cả bốn phương pháp kém ổn định; cần chiến lược gom 'other' (học trên train).", "",
         "## 4. Quyết định về feature set", "",
         "Feature set đầy đủ (`src/adapters/schema.py`): " + ", ".join(f"`{c}`" for c in CATEGORICAL_FEATURES) +
         "; số: `production_year`, `vehicle_age`, `mileage_km`, `power_kw`, `engine_cc`, `nr_seats`. "
         "Không loại biến nào để giảm chiều trước khi nghiên cứu encoding (theo yêu cầu đề tài). Lưu ý đa cộng tuyến: `vehicle_age = 2025 − production_year`; `make_model` chứa `brand`+`model`.", "",
         "**Đề xuất cho tuần 2:**", "",
         "1. Nhóm chính: `brand`, `model`/`make_model`, `fuel_type`, `transmission`, `body_type`, `country_code`, `seller_type`, `drive_train`, `body_color` + các biến số.",
         "2. Nhóm thí nghiệm riêng về cardinality: `model_version` (sau khi gom hiếm theo train).",
         "3. Không dùng đồng thời `brand`, `model` và `make_model` trong cùng một mô hình tuyến tính nếu chưa giải thích tác động lên số chiều.",
         "4. Cần pipeline xử lý thiếu: `drive_train` ~24%, `body_color` ~12%, `transmission` ~0,8%, model ~0,75%.", "",
         "## 5. Bảng audit feature", "",
         md_table(t["feature_audit"][["column", "role", "dtype", "pct_missing", "n_unique", "available_at_prediction", "leakage_risk", "reason"]]), ""]
    (R / "feature_selection.md").write_text("\n".join(L), encoding="utf-8")


def write_leakage(stats, df, t):
    sp, rl = stats["split"], stats["random_split_leak"]
    audit = t["feature_audit"]
    risky = audit[audit["leakage_risk"].isin(["high", "medium"])][["column", "role", "leakage_risk", "reason"]]
    L = ["# Rủi ro rò rỉ dữ liệu (data leakage)", "",
         "## 1. Biến bị loại / hạn chế vì liên quan đến giá hoặc định danh", "", md_table(risky), "",
         "## 2. Rò rỉ do cùng một xe xuất hiện ở train và test", "",
         f"- {sp['n_groups_with_multiple_listings']:,} nhóm xe có nhiều tin đăng (chung VIN hợp lệ hoặc cùng 'dấu vân tay' kỹ thuật), gồm {sp['listings_in_multi_listing_groups']:,} tin; nhóm lớn nhất {sp['largest_group']} tin.",
         f"- Nếu dùng random split thường (seed 42), **{rl['test_rows_with_same_group_in_train']:,} / {rl['n_test']:,} dòng test** có tin cùng nhóm nằm ở train. "
         f"Với GroupShuffleSplit: {rl['group_split_same_metric']} dòng.",
         "- Hạn chế: 66% tin không có VIN; nhóm theo dấu vân tay chỉ bắt được tin trùng thông số – tin trùng thật nhưng khác km/ngày đăng ký vẫn có thể lọt qua.", "",
         "## 3. Quy tắc bắt buộc cho tuần 2", "",
         "- Mọi imputer, scaler, encoder (đặc biệt Target/Frequency) chỉ `fit` trên train (trong `Pipeline`/cross-validation); không fit trên toàn bộ dữ liệu.",
         "- Không dùng test để chọn feature, hyperparameter hay quy tắc cleaning phụ thuộc giá.",
         "- Các bảng quan hệ feature–giá trong EDA được tính trên train; phân phối giá/đếm danh mục (không dùng target để chọn biến) tính trên toàn bộ dữ liệu sạch.",
         "- Feature engineering ở tuần 1 chỉ gồm phép biến đổi từng dòng (không học thống kê): `vehicle_age`, `make_model`.",
         "- Cờ `flag_price_review` được dựng từ quan hệ giá–mẫu xe trên toàn dữ liệu nên chỉ để sàng lỗi dữ liệu; **không dùng làm feature** và nếu dùng để loại dòng ở tuần 2 phải áp dụng thống nhất trước khi chia hoặc chỉ dựa trên train.", ""]
    (R / "leakage_risks.md").write_text("\n".join(L), encoding="utf-8")


def write_week1_report(stats, df, t):
    raw, f = stats["raw"], pd.DataFrame(stats["funnel"])
    sp, rl, fl, ps = stats["split"], stats["random_split_leak"], stats["flags"], stats["price_stats"]
    card = t["categorical_cardinality"].set_index("column")
    mvs = stats["make_model"]["model_version_summary"]
    ctry = t["country_summary"]
    de = ctry.loc[ctry["country_code"] == "DE", "n"].iloc[0]
    corr = stats["train_corr_with_price"]
    tests = (T / "pytest_summary.txt").read_text(encoding="utf-8-sig").strip() if (T / "pytest_summary.txt").exists() else "chưa chạy – chạy `python -m pytest -v`"
    n_final = stats["n_clean"]
    L = [
        "# Báo cáo Tuần 1 – Dự đoán giá ô tô cũ trên dữ liệu có nhiều thuộc tính phân loại", "",
        "_Mọi số liệu trong báo cáo được sinh từ dataset thật bởi `python main.py` (xem `reports/stats.json`)._", "",
        "## Chương 1. Tổng quan", "",
        "**Đề tài.** Dự đoán giá chào bán (EUR) của ô tô cũ từ các thuộc tính mô tả xe, tập trung đánh giá ảnh hưởng của *mã hóa biến phân loại* (One-hot, Ordinal, Target, Frequency) "
        "và *chuẩn hóa dữ liệu* (No scaling, Standard, MinMax, Robust, log) lên bốn mô hình hồi quy (KNN, Ridge, SVR, Random Forest). Tuần 1 chỉ chuẩn bị dữ liệu: làm sạch, EDA, phân tích cardinality, schema, chia train/test an toàn – không có encoding/scaling/mô hình.", "",
        "**Bài toán:** học có giám sát, hồi quy, dữ liệu bảng. X = thuộc tính xe lúc đăng bán; y = `price` (EUR), là *giá chào bán* chứ không phải giá giao dịch. "
        "**Vai trò biến phân loại:** hầu hết thuộc tính mô tả xe (hãng, mẫu, phiên bản, nhiên liệu…) là phân loại, một số có hàng nghìn giá trị → cách mã hóa quyết định số chiều, mức rò rỉ và chất lượng mô hình.", "",
        "Định hướng dài hạn: pipeline dùng *common schema* và *adapter* để sau này thay AutoScout24 bằng dữ liệu Việt Nam (chỉ cần viết `vietnam_adapter.py`).", "",
        "## Chương 2. Dataset", "",
        f"- **Nguồn:** AutoScout24 Car Listings Dataset (snapshot 08/11/2025), Zenodo record 17643343, DOI 10.5281/zenodo.17643343, v1.0.0, công bố 18/11/2025, giấy phép MIT theo metadata Zenodo. Tệp `autoscout24_dataset_20251108.csv`, 548.610.318 byte, MD5 `{stats['lineage']['raw_md5']}` (đã tính lại trên file và khớp với giá trị công bố).",
        f"- **Quy mô thực tế:** **{raw['rows']:,} dòng × {raw['cols']} cột** (Zenodo ghi \"~120K\"). RAM pandas ≈ {raw['ram_mb_deep']:.0f} MB.",
        f"- **Cột rỗng 100%:** {', '.join(raw['all_null_columns'])}. Cột thiếu > 50%: {len(raw['high_missing_cols_over_50pct'])} cột (gồm `production_year`, `vin`, `price_net`).",
        "- **Giới hạn:** (1) chỉ 25 hãng, chủ yếu BMW/Porsche/Mercedes/Audi – thiên về xe cao cấp, không đại diện thị trường phổ thông; (2) không có thời điểm đăng tin; "
        "(3) giá chào bán; (4) 8 thị trường châu Âu; (5) một số cờ trạng thái xe không nhất quán (Chương 4); (6) dữ liệu gốc do bên thứ ba tổng hợp, có lỗi nhập liệu (giá, công suất, số ghế).", "",
        "## Chương 3. Khám phá dữ liệu", "",
        f"**Target (sau cleaning, {n_final:,} xe):** trung vị {ps['p50']:,.0f} EUR, trung bình {ps['mean']:,.0f}, p5 {ps['p5']:,.0f}, p95 {ps['p95']:,.0f}, p99.9 {ps['p99.9']:,.0f}, tối đa {ps['max']:,.0f}. "
        f"Độ lệch (skew) = **{ps['skew']:.1f}** trên thang EUR nhưng chỉ **{ps['skew_log1p']:.2f}** sau log1p → phân phối lệch phải rất mạnh, log-transform là ứng viên hợp lý cho thí nghiệm tuần 2 (cần kiểm chứng, đặc biệt khi đánh giá RMSE trên thang gốc). "
        "Xe cực đắt (Bugatti…) là hợp lệ và được giữ lại; sẽ ảnh hưởng mạnh RMSE.", "",
        "![](figures/price_histograms.png)", "",
        f"**Thiếu/trùng:** raw có 0 dòng trùng toàn bộ, 0 `id` trùng, nhưng {raw['vin_duplicate_rows']:,} dòng có VIN lặp. Sau cleaning còn thiếu: drive_train {card.loc['drive_train','pct_missing']:.1f}%, body_color {card.loc['body_color','pct_missing']:.1f}%, transmission {card.loc['transmission','pct_missing']:.2f}%.", "",
        f"**Hãng & mẫu xe:** {stats['make_model']['brand_n']} hãng (không có biến thể chính tả trùng nhau theo casefold), {stats['make_model']['model_n']} mẫu, {card.loc['make_model','n_unique']} cặp hãng-mẫu. "
        f"Hãng lớn nhất: {card.loc['brand','top_category']} ({card.loc['brand','top_category_share_pct']}%).", "",
        "![](figures/categorical_counts.png)", "",
        f"**Thuộc tính số (tương quan Spearman với giá, train):** power_kw {corr['power_kw']:+.2f}, mileage_km {corr['mileage_km']:+.2f}, production_year {corr['production_year']:+.2f}, engine_cc {corr['engine_cc']:+.2f}, nr_seats {corr['nr_seats']:+.2f}. "
        "Đây chỉ là mô tả đơn biến; không dùng để kết luận biến nào hữu ích hay loại biến.", "",
        "![](figures/price_vs_year_mileage.png)", "![](figures/price_relationships.png)", "",
        "Lưu ý khi đọc: giá trung vị theo nhóm đều kèm số quan sát n; nhóm n nhỏ (vd. LPG, CNG, Luxembourg) không đủ cơ sở kết luận. Giá theo quốc gia bị nhiễu bởi cơ cấu hãng/xe khác nhau giữa các nước.", "",
        "## Chương 4. Tiền xử lý", "",
        "**Phễu lọc (chi tiết trong `cleaning_log.md`):**", "", md_table(f[["step", "rows_before", "rows_after", "rows_removed"]]), "",
        f"**Quy tắc xe cũ:** `is_used = True` và `is_new = False`, `offer_type = U`. {_scope_breakdown(t['used_scope_flag_combinations'])['unconfirmed_other']:,} tin không phải xe mới/đăng ký trước nhưng không có cờ `is_used` bị loại vì không xác nhận được (trường hợp cần xem xét).", "",
        "**Tiền tệ:** toàn bộ EUR (kiểm tra bằng code), không chuyển đổi. **Phạm vi thị trường:** ", "",
        md_table(ctry[["country_code", "n", "share_pct", "median_price", "p10", "p90"]]), "",
        f"→ **Chọn phương án A (tất cả thị trường hợp lệ, giữ `country_code` làm feature)** vì: mẫu lớn hơn ({n_final:,} vs. {de:,} nếu chỉ lấy Đức), `country_code` thêm một biến phân loại (cardinality thấp) cho nghiên cứu encoding, "
        f"và không thị trường nào áp đảo hoàn toàn (Đức {de / n_final * 100:.1f}%). Nhược điểm: khác biệt thuế/chính sách giữa nước, Luxembourg ({int(ctry.loc[ctry.country_code == 'LU', 'n'].iloc[0])} tin) và {int(ctry['n'].iloc[-1])} tin thiếu quốc gia là nhóm nhỏ. "
        "Không loại/gộp quốc gia nào. Phương án B (chỉ Đức) có thể chạy như thí nghiệm phụ bằng cách lọc `country_code == 'DE'`.", "",
        f"**Feature engineering:** `vehicle_age = 2025 − production_year` (năm snapshot); `make_model`. Năm sản xuất (nguồn): {_year_source(fl['year_source'])}. "
        "**Feature selection & leakage:** xem `feature_selection.md` và `leakage_risks.md`; `price_net`, `price_vat_rate`, `price_tax_deductible`, `price_negotiable`, VIN, id, mô tả, tên người bán bị loại khỏi feature chính.", "",
        "## Chương 5. Phân tích cardinality", "",
        md_table(t["categorical_cardinality"][["column", "n_unique", "level", "pct_missing", "n_cat_count_eq1", "n_cat_count_lt5", "n_cat_count_lt10", "pct_rows_in_cats_lt10"]]), "",
        "![](figures/cardinality.png)", "",
        f"**Phát hiện chính:** `model_version` có {mvs['n_unique']:,} danh mục, {mvs['pct_cats_singleton']}% chỉ xuất hiện một lần, {mvs['pct_rows_in_cats_lt10']}% bản ghi thuộc danh mục < 10 lần, "
        f"{t['categorical_cardinality'].set_index('column').loc['model_version','n_case_variant_groups']} nhóm khác nhau chỉ ở chữ hoa/thường. `model`/`make_model` có ~600 danh mục (≈ {card.loc['model','pct_rows_in_cats_lt10']}% bản ghi ở danh mục hiếm < 10). "
        "Các biến còn lại có ≤ 25 danh mục. Dự kiến: One-hot với model_version tạo ~71 nghìn cột; Ordinal tạo thứ tự giả; Target Encoding rủi ro rò rỉ nặng ở danh mục hiếm; Frequency Encoding gọn nhưng mất thông tin. Đây là *giả thuyết*, kiểm chứng ở tuần 2–3. Chi tiết: `feature_selection.md`.", "",
        "## Chương 6. Chuẩn bị cho Machine Learning", "",
        "**Common schema** (`src/adapters/schema.py`) + `autoscout24_adapter.py` (make→brand, fuel_category→fuel_type, cylinders_volume_cc→engine_cc, price_currency→currency, id→listing_id). "
        "**X và y:** y = `price`; X gồm 11 biến phân loại và 6 biến số (`select_xy` chặn target lọt vào X).", "",
        f"**Chia train/test:** không có cột thời gian → không chia theo thời gian. Nhưng cùng một xe có nhiều tin (VIN/dấu vân tay) → dùng **GroupShuffleSplit** (test 20%, seed 42). "
        f"Kết quả: train {sp['n_train']:,} / test {sp['n_test']:,} ({sp['test_fraction']*100:.2f}% test), {sp['n_groups']:,} nhóm, {sp['n_groups_with_multiple_listings']:,} nhóm nhiều tin ({sp['listings_in_multi_listing_groups']:,} tin), "
        f"giao nhóm train∩test = {sp['group_overlap_train_test']}. Random split thường sẽ để {rl['test_rows_with_same_group_in_train']:,} dòng test có tin cùng nhóm trong train. "
        "Ưu: không rò rỉ giữa các tin cùng xe; nhược: nhóm bắt được chưa đầy đủ (66% tin không VIN), tỷ lệ 80/20 chỉ xấp xỉ.", "",
        "**Rò rỉ:** không imputer/scaler/encoder nào được fit trong tuần 1 (có unit test kiểm tra); bảng quan hệ feature–giá tính trên train. Tỷ lệ dòng test có danh mục chưa thấy ở train: "
        + ", ".join(f"{r.column} {r.pct_test_rows_unseen:.2f}%" for r in t["unseen_categories_test_vs_train"].query("pct_test_rows_unseen > 0").itertuples()) + ".", "",
        "**Thiết kế Pipeline tuần 2:** `ColumnTransformer(imputer → encoder → scaler)` + mô hình, bọc trong `Pipeline`; đánh giá bằng GroupKFold trên train, test chỉ dùng một lần ở cuối; "
        "target/frequency encoding fit trong từng fold; báo cáo RMSE, MAE, R², số chiều, thời gian huấn luyện.", "",
        "## Chương 7. Kết quả đạt được", "",
        "**Tệp chính:** `src/` (config, download_dataset, load_data, profiling, data_validation, cleaning, feature_engineering, split_data, eda, reporting, adapters/), `tests/`, `main.py`, "
        "`notebooks/01_exploratory_data_analysis.ipynb`, `reports/*.md`, `reports/figures/`, `reports/tables/`, `data/interim/*.parquet`, `data/processed/{train,test}.parquet`, `split_manifest.csv`, `split_metadata.json`.", "",
        f"**Kiểm thử:** {tests}", "",
        "**Vấn đề chưa giải quyết:**", "",
        f"- {_scope_breakdown(t['used_scope_flag_combinations'])['unconfirmed_other']:,} tin trạng thái xe không xác định bị loại – có thể gây thiên lệch.",
        f"- {fl['price_review']} tin giá nghi lỗi chỉ gắn cờ (`price_review_list.csv`), chưa loại.",
        "- Tin trùng không VIN chỉ được phát hiện gần đúng.",
        "- Dữ liệu thiên về xe cao cấp châu Âu; chưa đại diện cho thị trường Việt Nam.",
        "- Chưa kiểm tra tính nhất quán giữa `model_version` và các cột kỹ thuật.", "",
        "**Kế hoạch tuần 2:** xây Pipeline encoding × scaling; baseline; xử lý thiếu trong pipeline; thí nghiệm log-target; gom danh mục hiếm học trên train; đánh giá bằng GroupKFold.", "",
        "## Phụ lục – Khả năng tái lập", "",
        f"Phiên bản: Python {stats['lineage']['python']}, pandas {stats['lineage']['pandas']}, numpy {stats['lineage']['numpy']}, scikit-learn {stats['lineage']['sklearn']}. Seed {stats['lineage']['seed']}, test_size {stats['lineage']['test_size']}, snapshot {stats['lineage']['snapshot_date']}. Chạy lại: xem README.", ""]
    (R / "week1_report.md").write_text("\n".join(L), encoding="utf-8")


def write_all():
    stats, df, t = _load()
    write_data_dictionary(stats, df, t)
    write_cleaning_log(stats, df, t)
    write_feature_selection(stats, df, t)
    write_leakage(stats, df, t)
    write_week1_report(stats, df, t)
    print("Reports written to", R)


if __name__ == "__main__":
    write_all()
