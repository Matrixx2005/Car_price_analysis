# Data dictionary (common schema)

_Tính trên 94,266 xe cũ sau cleaning. Sinh tự động._

| cột chuẩn | cột nguồn AutoScout24 | mô tả | dtype | thiếu (%) | số giá trị duy nhất |
|---|---|---|---|---|---|
| listing_id | id | Mã tin đăng (duy nhất) | str | 0 | 94,266 |
| brand | make | Hãng xe | str | 0 | 25 |
| model | model | Mẫu xe | str | 0.75 | 602 |
| model_version | model_version | Phiên bản/trim (văn bản tự do) | str | 0.77 | 71,015 |
| production_year | production_year / registration_date | Năm sản xuất; nếu thiếu lấy năm đăng ký (cột year_source ghi nguồn) | float64 | 0.02 | 58 |
| mileage_km | mileage_km_raw | Số km đã đi | float64 | 0 | 37,310 |
| fuel_type | fuel_category | Nhóm nhiên liệu | str | 0.06 | 9 |
| transmission | transmission | Hộp số | str | 0.81 | 3 |
| body_type | body_type | Kiểu thân xe | str | 0 | 8 |
| country_code | country_code | Quốc gia của tin đăng | str | 0.01 | 8 |
| seller_type | seller_type | Dealer / PrivateSeller | str | 0.01 | 2 |
| engine_cc | cylinders_volume_cc | Dung tích động cơ (cc) | float64 | 9.42 | 453 |
| power_kw | power_kw | Công suất (kW) | float64 | 1.30 | 512 |
| nr_seats | nr_seats | Số ghế | float64 | 3.65 | 9 |
| price | price | Giá chào bán – TARGET (EUR) | float64 | 0 | 12,073 |
| currency | price_currency | Đơn vị tiền tệ (chỉ EUR) | str | 0 | 1 |
| vehicle_condition | (suy ra từ is_used/is_new/is_preregistered) | used / new / preregistered / unknown | str | 0 | 1 |
| drive_train | drive_train | Dẫn động | str | 24.18 | 3 |
| body_color | body_color | Màu thân xe | str | 12.05 | 14 |
| vehicle_age | (tạo mới) | = 2025 − production_year (năm snapshot), tối thiểu 0 | float64 | 0.02 | 57 |
| make_model | (tạo mới) | = brand + ' ' + model; NaN nếu thiếu một trong hai | str | 0.75 | 603 |

Các cột phụ (metadata/cờ kiểm tra) có trong `data/interim/autoscout24_used_common.parquet` nhưng KHÔNG được ghi vào `data/processed/`: `vin`, `seller_company_name`, `price_net`, `price_vat_rate`, `price_tax_deductible`, `price_negotiable`, `registration_date`, `registration_year`, `year_source`, `flag_price_review`, `flag_future_registration`, `is_*`, `offer_type`, `nr_prev_owners`, `had_accident`.
