# Dataset profile – AutoScout24 (Zenodo 17643343)

_Tệp này được sinh tự động bởi `src/profiling.py`; mọi con số đo trực tiếp trên file thật._

## 1. Nguồn và xác minh

- Nguồn: https://zenodo.org/records/17643343 (DOI 10.5281/zenodo.17643343)
- Tên tệp: `autoscout24_dataset_20251108.csv`; phiên bản dữ liệu 1.0.0; công bố 2025-11-18
- Giấy phép (metadata Zenodo): `mit-license`
- Ngày truy cập/tải: 2026-10-24
- Dung lượng thực: 548,610,318 byte (công bố: 548,610,318) → KHỚP
- MD5 tính trên file: `b23a122cc51baf7de39f449193ff0d28`; MD5 công bố: `b23a122cc51baf7de39f449193ff0d28` → KHỚP

## 2. Quy mô quan sát được vs. mô tả của Zenodo

| Chỉ tiêu | Mô tả Zenodo | Quan sát thực tế |
|---|---|---|
| Số tin đăng | ~120.000 | **118,382** |
| Số cột | (liệt kê trong mô tả) | **75** |
| Dung lượng CSV | ~548,6 MB | 548.6 MB |
| RAM pandas (deep) | – | ~586 MB |

## 3. Trùng lặp

- Dòng trùng hoàn toàn: **0**
- `id` trùng: **0**
- VIN có giá trị: 40,222 dòng; số dòng có VIN trùng với dòng trước đó: **2,644** (cùng xe đăng nhiều lần hoặc VIN không hợp lệ – cần phân tích ở bước cleaning). Giá trị VIN không được in ra báo cáo.

## 4. Cột và tỷ lệ thiếu

| Cột | dtype | Thiếu (n) | Thiếu (%) | Số giá trị duy nhất |
|---|---|---:|---:|---:|
| id | object | 0 | 0.00 | 118,382 |
| description | object | 4,392 | 3.71 | 107,516 |
| ratings_average | object | 41,954 | 35.44 | 42 |
| ratings_count | object | 41,954 | 35.44 | 737 |
| ratings_recommend_percentage | object | 41,954 | 35.44 | 63 |
| price_currency | object | 0 | 0.00 | 1 |
| price | object | 0 | 0.00 | 16,476 |
| price_tax_deductible | object | 0 | 0.00 | 2 |
| price_negotiable | object | 0 | 0.00 | 2 |
| price_net | object | 84,366 | 71.27 | 14,414 |
| price_vat_rate | object | 88,003 | 74.34 | 10 |
| vin | object | 78,160 | 66.02 | 37,578 |
| make | object | 0 | 0.00 | 25 |
| model | object | 1,402 | 1.18 | 627 |
| model_version | object | 838 | 0.71 | 84,171 |
| german_hsn_tsn | object | 87,571 | 73.97 | 3,073 |
| mileage_km_raw | object | 644 | 0.54 | 39,809 |
| mileage_km | object | 644 | 0.54 | 39,809 |
| registration_date | object | 221 | 0.19 | 419 |
| production_year | object | 96,110 | 81.19 | 34 |
| vehicle_type | object | 0 | 0.00 | 2 |
| body_type | object | 0 | 0.00 | 18 |
| nr_seats | object | 4,906 | 4.14 | 17 |
| nr_doors | object | 1,453 | 1.23 | 7 |
| body_color | object | 11,911 | 10.06 | 14 |
| paint_type | object | 26,047 | 22.00 | 2 |
| body_color_original | object | 44,332 | 37.45 | 11,203 |
| upholstery | object | 26,745 | 22.59 | 6 |
| upholstery_color | object | 32,485 | 27.44 | 11 |
| power_kw | object | 1,612 | 1.36 | 532 |
| power_hp | object | 1,612 | 1.36 | 532 |
| transmission | object | 816 | 0.69 | 3 |
| gears | object | 43,248 | 36.53 | 19 |
| drive_train | object | 27,753 | 23.44 | 3 |
| cylinders | object | 24,858 | 21.00 | 17 |
| cylinders_volume_cc | object | 12,286 | 10.38 | 473 |
| weight_kg | object | 24,086 | 20.35 | 1,637 |
| has_particle_filter | object | 0 | 0.00 | 2 |
| fuel_category | object | 73 | 0.06 | 9 |
| primary_fuel | object | 58,409 | 49.34 | 14 |
| electric_range_km | object | 105,637 | 89.23 | 581 |
| electric_range_city_km | object | 117,743 | 99.46 | 300 |
| fuel_cons_comb_l100_km | object | 76,538 | 64.65 | 240 |
| fuel_cons_city_l100_km | object | 118,382 | 100.00 | 0 |
| fuel_cons_highway_l100_km | object | 118,382 | 100.00 | 0 |
| co2_emission_grper_km | object | 93,571 | 79.04 | 381 |
| fuel_cons_comb_l100_wltp_km | object | 88,112 | 74.43 | 235 |
| fuel_cons_electric_comb_l100_wltp_km | object | 113,456 | 95.84 | 311 |
| co2_emission_grper_wltp_km | object | 71,927 | 60.76 | 506 |
| equipment_comfort | object | 0 | 0.00 | 74,157 |
| equipment_entertainment | object | 0 | 0.00 | 6,687 |
| equipment_extra | object | 0 | 0.00 | 48,495 |
| equipment_safety | object | 0 | 0.00 | 53,289 |
| is_used | object | 0 | 0.00 | 2 |
| is_new | object | 0 | 0.00 | 2 |
| is_preregistered | object | 0 | 0.00 | 2 |
| had_accident | object | 0 | 0.00 | 2 |
| has_full_service_history | object | 0 | 0.00 | 2 |
| non_smoking | object | 0 | 0.00 | 2 |
| nr_prev_owners | object | 53,568 | 45.25 | 16 |
| is_rental | object | 0 | 0.00 | 2 |
| envir_standard | object | 41,261 | 34.85 | 11 |
| original_market | object | 72,766 | 61.47 | 33 |
| offer_type | object | 0 | 0.00 | 3 |
| country_code | object | 15 | 0.01 | 8 |
| zip | object | 19 | 0.02 | 13,889 |
| city | object | 108 | 0.09 | 16,704 |
| street | object | 20,328 | 17.17 | 17,476 |
| latitude | object | 389 | 0.33 | 27,334 |
| longitude | object | 389 | 0.33 | 27,497 |
| seller_is_dealer | object | 0 | 0.00 | 2 |
| seller_type | object | 15 | 0.01 | 2 |
| seller_company_name | object | 19,817 | 16.74 | 17,141 |
| has_warranty | object | 118,382 | 100.00 | 0 |
| warranty | object | 118,382 | 100.00 | 0 |

## 5. Vấn đề phát hiện

- Cột **rỗng 100%**: `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km`, `has_warranty`, `warranty` → không dùng được.
- Cột thiếu > 50%: `price_net`, `price_vat_rate`, `vin`, `german_hsn_tsn`, `production_year`, `electric_range_km`, `electric_range_city_km`, `fuel_cons_comb_l100_km`, `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km`, `co2_emission_grper_km`, `fuel_cons_comb_l100_wltp_km`, `fuel_cons_electric_comb_l100_wltp_km`, `co2_emission_grper_wltp_km`, `original_market`, `has_warranty`, `warranty`.
- `production_year` chỉ có ở khoảng 19% dòng; `registration_date` (thiếu ~0,19%) là nguồn năm đáng tin hơn → cần quy tắc suy năm.
- `mileage_km` là **chuỗi** (vd. `9,900 km`), `mileage_km_raw` là số → dùng cột số.
- `weight_kg` là chuỗi có đơn vị (`1,945 kg`).
- `ratings_average` có dtype chuỗi dù là số.
- Không có cột thời điểm đăng tin → không thể chia train/test theo thời gian.
- Cột mô tả Zenodo có nhắc nhưng không tồn tại: không có (mọi cột được liệt kê đều có trong file).

## 6. Ghi chú về cách đọc dữ liệu

- Dấu phân cách `,`, mã hoá UTF-8 đọc được không lỗi; đọc theo chunk 200.000 dòng để đếm hàng, đọc toàn bộ một lần cho profile (≈0,6 GB RAM).
- Chuỗi dài (`description`, `equipment_*`) không bao giờ được in ra terminal/báo cáo.
