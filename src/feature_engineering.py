"""Feature engineering on the COMMON schema. Stateless row-wise transforms only
(no statistic is learned from data here), so applying it before the split cannot leak."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.adapters.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def make_make_model(brand: pd.Series, model: pd.Series) -> pd.Series:
    """'<brand> <model>'. NaN if either part is missing, so 'Model X' of two brands never merges."""
    both = brand.notna() & model.notna()
    out = pd.Series(np.nan, index=brand.index, dtype="object")
    out[both] = brand[both].astype(str) + " " + model[both].astype(str)
    return out


def add_vehicle_age(df: pd.DataFrame, reference_year: int = config.SNAPSHOT_YEAR) -> pd.DataFrame:
    """Age at the data-collection snapshot (2025), NOT at today's year. Negative ages (model year
    2026 listed in 2025) are clipped to 0."""
    out = df.copy()
    out["vehicle_age"] = (reference_year - out["production_year"]).clip(lower=0)
    return out


def add_features(df: pd.DataFrame, reference_year: int = config.SNAPSHOT_YEAR) -> pd.DataFrame:
    out = add_vehicle_age(df, reference_year)
    out["make_model"] = make_make_model(out["brand"], out["model"])
    return out


def select_xy(df: pd.DataFrame, features=FEATURES, target: str = TARGET):
    """X contains only declared features; the target can never be one of them."""
    if target in features:
        raise ValueError("target column must not be in the feature list")
    present = [c for c in features if c in df.columns]
    return df[present].copy(), df[target].copy()


# ---- feature registry: why each field is kept / excluded --------------------------------------
# (column, role, available at prediction time, leakage risk, reason)
REGISTRY = [
    ("price", "target", "no", "-", "Giá chào bán EUR – biến mục tiêu y."),
    ("brand", "feature", "yes", "low", "Hãng xe; biến phân loại cardinality thấp–trung bình."),
    ("model", "feature", "yes", "low", "Mẫu xe; cardinality cao; chỉ định danh đầy đủ khi đi cùng brand."),
    ("model_version", "feature", "yes", "low", "Phiên bản/trim; cardinality rất cao, nhiều danh mục hiếm – dùng làm thí nghiệm encoding trọng tâm."),
    ("make_model", "feature", "yes", "low", "Tổ hợp brand+model (khử nhầm lẫn model trùng tên giữa các hãng). Trùng thông tin với brand & model → chỉ đưa vào thí nghiệm có kiểm soát số chiều."),
    ("fuel_type", "feature", "yes", "low", "Loại nhiên liệu."),
    ("transmission", "feature", "yes", "low", "Hộp số."),
    ("body_type", "feature", "yes", "low", "Kiểu thân xe."),
    ("country_code", "feature", "yes", "low", "Thị trường; giá/thuế khác nhau giữa các nước."),
    ("seller_type", "feature", "yes", "medium", "Dealer/Private: biết khi đăng tin, nhưng liên quan chiến lược định giá của người bán."),
    ("drive_train", "feature", "yes", "low", "Dẫn động; thiếu ~23% (cần impute ở Pipeline)."),
    ("body_color", "feature", "yes", "low", "Màu thân xe; thiếu ~10%."),
    ("production_year", "feature", "yes", "low", "Năm sản xuất (production_year, nếu thiếu thì lấy năm đăng ký)."),
    ("vehicle_age", "feature", "yes", "low", "= 2025 − production_year; thay thế tuyến tính của production_year – KHÔNG dùng đồng thời với production_year trong mô hình tuyến tính nếu muốn tránh đa cộng tuyến."),
    ("mileage_km", "feature", "yes", "low", "Số km đã đi."),
    ("power_kw", "feature", "yes", "low", "Công suất kW."),
    ("engine_cc", "feature", "yes", "low", "Dung tích xy-lanh; xe điện = 0/thiếu."),
    ("nr_seats", "feature", "yes", "low", "Số ghế."),
    ("price_net", "excluded", "no", "high", "Giá chưa VAT – dẫn xuất trực tiếp từ price (price/(1+VAT)). Dùng sẽ làm lộ target."),
    ("price_vat_rate", "excluded", "no", "high", "Thuế suất VAT, chỉ có khi có price_net (thiếu >74%); gắn với cách tính giá."),
    ("price_tax_deductible", "excluded", "unclear", "medium", "Cờ khấu trừ VAT: thuộc cấu trúc giá của tin đăng, không phải đặc điểm xe."),
    ("price_negotiable", "excluded", "unclear", "medium", "Cờ thương lượng: hành vi định giá của người bán."),
    ("vin", "excluded", "yes", "high", "Định danh xe duy nhất (nhận dạng được); chỉ dùng để tạo nhóm chia dữ liệu, không là feature."),
    ("listing_id", "metadata", "yes", "high", "Mã tin đăng – định danh, không có ý nghĩa dự đoán."),
    ("currency", "metadata", "yes", "-", "Điều kiện lọc (chỉ EUR)."),
    ("vehicle_condition", "metadata", "yes", "-", "Điều kiện lọc (chỉ 'used')."),
    ("description", "excluded", "yes", "medium", "Văn bản tự do; có thể chứa giá/thông tin người bán; chưa thuộc phạm vi tuần 1."),
    ("seller_company_name", "excluded", "yes", "medium", "Tên người bán cụ thể – dễ học thuộc người bán (nhận dạng); chỉ dùng cho khóa nhóm."),
    ("ratings_average", "excluded", "unclear", "medium", "Điểm đánh giá của người bán, thiếu 35%; không phải thuộc tính xe."),
    ("registration_date", "excluded", "yes", "low", "Đã được phản ánh qua production_year; giữ lại để suy luận năm."),
    ("equipment_*", "excluded", "yes", "low", "Văn bản liệt kê trang bị – cần xử lý NLP, ngoài phạm vi tuần 1."),
    ("zip/city/street/latitude/longitude", "excluded", "yes", "medium", "Vị trí chi tiết người bán; nguy cơ nhận dạng và học thuộc cửa hàng."),
]


def feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature facts computed from the data: dtype, missing %, cardinality (+ registry columns)."""
    rows = []
    for name, role, avail, leak, reason in REGISTRY:
        if name in df.columns:
            s = df[name]
            rows.append(dict(column=name, role=role, dtype=str(s.dtype), pct_missing=round(s.isna().mean() * 100, 2),
                             n_unique=int(s.nunique()), available_at_prediction=avail, leakage_risk=leak, reason=reason))
        else:
            rows.append(dict(column=name, role=role, dtype="(không có trong bảng sạch)", pct_missing=np.nan,
                             n_unique=np.nan, available_at_prediction=avail, leakage_risk=leak, reason=reason))
    return pd.DataFrame(rows)


assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES) == set(FEATURES)
