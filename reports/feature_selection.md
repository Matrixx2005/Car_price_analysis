# Feature selection & ý nghĩa nghiên cứu

_Bảng số liệu sinh tự động từ dữ liệu thật; phần diễn giải do tác giả viết._

## 1. Cardinality các biến phân loại (toàn bộ xe cũ sau cleaning, không dùng target)

| column | n_unique | level | pct_missing | top_category | top_category_share_pct | n_cat_count_eq1 | n_cat_count_lt5 | n_cat_count_lt10 | pct_rows_in_cats_lt10 |
|---|---|---|---|---|---|---|---|---|---|
| model_version | 71,015 | high (>50) | 0.77 | E-Hybrid | 0.14 | 63,854 | 69,713 | 70,560 | 88.44 |
| make_model | 603 | high (>50) | 0.75 | Porsche Cayenne | 5.66 | 92 | 181 | 249 | 0.83 |
| model | 602 | high (>50) | 0.75 | Cayenne | 5.66 | 91 | 180 | 248 | 0.83 |
| brand | 25 | medium (11-50) | 0 | BMW | 38.68 | 0 | 0 | 1 | 0.01 |
| body_color | 14 | medium (11-50) | 12.05 | Black | 34.37 | 0 | 0 | 0 | 0 |
| fuel_type | 9 | low (<=10) | 0.06 | Gasoline | 46.66 | 0 | 0 | 1 | 0.01 |
| body_type | 8 | low (<=10) | 0 | Off-Road/Pick-up | 37.45 | 0 | 0 | 0 | 0 |
| country_code | 8 | low (<=10) | 0.01 | DE | 34.50 | 0 | 0 | 0 | 0 |
| transmission | 3 | low (<=10) | 0.81 | Automatic | 79.17 | 0 | 0 | 0 | 0 |
| drive_train | 3 | low (<=10) | 24.18 | 4WD | 55.18 | 0 | 0 | 0 | 0 |
| seller_type | 2 | low (<=10) | 0.01 | Dealer | 79.03 | 0 | 0 | 0 | 0 |

Ngưỡng low ≤ 10 / medium 11–50 / high > 50 chỉ là quy ước mô tả, không dùng để kết luận encoder nào tốt nhất.

## 2. Vì sao Make / Model / Model Version là nhóm trọng tâm

Ba biến này mô tả "loại xe" – yếu tố quyết định giá lớn nhất – nhưng có cardinality tăng vọt: brand 25 → model 602 → model_version **71,015** danh mục trên 93,538 bản ghi có giá trị. `model` không định danh duy nhất nếu thiếu `brand` (có 1 tên mẫu xe dùng chung cho nhiều hãng, vd. `440`), nên có `make_model`.

`model_version` là văn bản tự do: trung vị 5 từ/giá trị, 72.54% có ≥ 4 từ, 28.38% chứa công suất/dung tích, 28.19% chứa tên dẫn động/hộp số – tức trộn nhiều thông tin đã có ở cột khác. **63,854 danh mục (89.92%) chỉ xuất hiện 1 lần; 89.13% bản ghi thuộc danh mục xuất hiện < 10 lần.** Sau khi chia dữ liệu, 72.2% dòng test có `model_version` chưa từng thấy ở train.

**Đánh giá:** `model_version` KHÔNG nên là feature trong thực nghiệm chính; dùng làm thí nghiệm phụ (sau khi gom nhóm hiếm bằng thống kê học trên train).

## 3. Tác động dự kiến lên từng encoding (giả thuyết cần kiểm chứng tuần 2–3)

- **One-hot:** số chiều thô = tổng số danh mục. Với 11 biến phân loại: 72,292 cột (với model_version), 1,277 (bỏ model_version), 674 (bỏ cả make_model). Ma trận rất thưa; hại KNN/SVR (khoảng cách bị chi phối bởi cột nhị phân), tăng thời gian huấn luyện Ridge/RF.
- **Ordinal:** gán số nguyên cho danh mục tạo thứ tự giả (vd. BMW < Porsche) và khoảng cách giả; vô hại hơn với cây (Random Forest) nhưng sai với KNN/Ridge/SVR. Chỉ hợp lý với biến có thứ tự tự nhiên – ở đây hầu như không có.
- **Target Encoding:** gán trung bình giá theo danh mục; danh mục hiếm (1 mẫu) khiến encoding = chính giá của mẫu đó → rò rỉ target mạnh nếu fit trước khi chia hoặc không dùng cross-fitting/smoothing. Phải fit trong fold train.
- **Frequency Encoding:** 1 cột/biến, tránh bùng nổ chiều, nhưng mất thông tin (hai hãng cùng tần suất bị coi như nhau; danh mục chưa thấy → tần suất 0). Tần suất phải tính trên train.
- **Danh mục hiếm:** làm cả bốn phương pháp kém ổn định; cần chiến lược gom 'other' (học trên train).

## 4. Quyết định về feature set

Feature set đầy đủ (`src/adapters/schema.py`): `brand`, `model`, `model_version`, `make_model`, `fuel_type`, `transmission`, `body_type`, `country_code`, `seller_type`, `drive_train`, `body_color`; số: `production_year`, `vehicle_age`, `mileage_km`, `power_kw`, `engine_cc`, `nr_seats`. Không loại biến nào để giảm chiều trước khi nghiên cứu encoding (theo yêu cầu đề tài). Lưu ý đa cộng tuyến: `vehicle_age = 2025 − production_year`; `make_model` chứa `brand`+`model`.

**Đề xuất cho tuần 2:**

1. Nhóm chính: `brand`, `model`/`make_model`, `fuel_type`, `transmission`, `body_type`, `country_code`, `seller_type`, `drive_train`, `body_color` + các biến số.
2. Nhóm thí nghiệm riêng về cardinality: `model_version` (sau khi gom hiếm theo train).
3. Không dùng đồng thời `brand`, `model` và `make_model` trong cùng một mô hình tuyến tính nếu chưa giải thích tác động lên số chiều.
4. Cần pipeline xử lý thiếu: `drive_train` ~24%, `body_color` ~12%, `transmission` ~0,8%, model ~0,75%.

## 5. Bảng audit feature

| column | role | dtype | pct_missing | n_unique | available_at_prediction | leakage_risk | reason |
|---|---|---|---|---|---|---|---|
| price | target | float64 | 0 | 12,073 | no | - | Giá chào bán EUR – biến mục tiêu y. |
| brand | feature | object | 0 | 25 | yes | low | Hãng xe; biến phân loại cardinality thấp–trung bình. |
| model | feature | object | 0.75 | 602 | yes | low | Mẫu xe; cardinality cao; chỉ định danh đầy đủ khi đi cùng brand. |
| model_version | feature | object | 0.77 | 71,015 | yes | low | Phiên bản/trim; cardinality rất cao, nhiều danh mục hiếm – dùng làm thí nghiệm encoding trọng tâm. |
| make_model | feature | object | 0.75 | 603 | yes | low | Tổ hợp brand+model (khử nhầm lẫn model trùng tên giữa các hãng). Trùng thông tin với brand & model → chỉ đưa vào thí nghiệm có kiểm soát số chiều. |
| fuel_type | feature | object | 0.06 | 9 | yes | low | Loại nhiên liệu. |
| transmission | feature | object | 0.81 | 3 | yes | low | Hộp số. |
| body_type | feature | object | 0 | 8 | yes | low | Kiểu thân xe. |
| country_code | feature | str | 0.01 | 8 | yes | low | Thị trường; giá/thuế khác nhau giữa các nước. |
| seller_type | feature | object | 0.01 | 2 | yes | medium | Dealer/Private: biết khi đăng tin, nhưng liên quan chiến lược định giá của người bán. |
| drive_train | feature | object | 24.18 | 3 | yes | low | Dẫn động; thiếu ~23% (cần impute ở Pipeline). |
| body_color | feature | object | 12.05 | 14 | yes | low | Màu thân xe; thiếu ~10%. |
| production_year | feature | float64 | 0.02 | 58 | yes | low | Năm sản xuất (production_year, nếu thiếu thì lấy năm đăng ký). |
| vehicle_age | feature | float64 | 0.02 | 57 | yes | low | = 2025 − production_year; thay thế tuyến tính của production_year – KHÔNG dùng đồng thời với production_year trong mô hình tuyến tính nếu muốn tránh đa cộng tuyến. |
| mileage_km | feature | float64 | 0 | 37,310 | yes | low | Số km đã đi. |
| power_kw | feature | float64 | 1.30 | 512 | yes | low | Công suất kW. |
| engine_cc | feature | float64 | 9.42 | 453 | yes | low | Dung tích xy-lanh; xe điện = 0/thiếu. |
| nr_seats | feature | float64 | 3.65 | 9 | yes | low | Số ghế. |
| price_net | excluded | float64 | 80.80 | 9,509 | no | high | Giá chưa VAT – dẫn xuất trực tiếp từ price (price/(1+VAT)). Dùng sẽ làm lộ target. |
| price_vat_rate | excluded | float64 | 80.91 | 10 | no | high | Thuế suất VAT, chỉ có khi có price_net (thiếu >74%); gắn với cách tính giá. |
| price_tax_deductible | excluded | boolean | 0 | 2 | unclear | medium | Cờ khấu trừ VAT: thuộc cấu trúc giá của tin đăng, không phải đặc điểm xe. |
| price_negotiable | excluded | boolean | 0 | 2 | unclear | medium | Cờ thương lượng: hành vi định giá của người bán. |
| vin | excluded | str | 70.93 | 26,345 | yes | high | Định danh xe duy nhất (nhận dạng được); chỉ dùng để tạo nhóm chia dữ liệu, không là feature. |
| listing_id | metadata | object | 0 | 94,266 | yes | high | Mã tin đăng – định danh, không có ý nghĩa dự đoán. |
| currency | metadata | str | 0 | 1 | yes | - | Điều kiện lọc (chỉ EUR). |
| vehicle_condition | metadata | str | 0 | 1 | yes | - | Điều kiện lọc (chỉ 'used'). |
| description | excluded | (không có trong bảng sạch) |  |  | yes | medium | Văn bản tự do; có thể chứa giá/thông tin người bán; chưa thuộc phạm vi tuần 1. |
| seller_company_name | excluded | object | 20.98 | 16,523 | yes | medium | Tên người bán cụ thể – dễ học thuộc người bán (nhận dạng); chỉ dùng cho khóa nhóm. |
| ratings_average | excluded | (không có trong bảng sạch) |  |  | unclear | medium | Điểm đánh giá của người bán, thiếu 35%; không phải thuộc tính xe. |
| registration_date | excluded | object | 0 | 391 | yes | low | Đã được phản ánh qua production_year; giữ lại để suy luận năm. |
| equipment_* | excluded | (không có trong bảng sạch) |  |  | yes | low | Văn bản liệt kê trang bị – cần xử lý NLP, ngoài phạm vi tuần 1. |
| zip/city/street/latitude/longitude | excluded | (không có trong bảng sạch) |  |  | yes | medium | Vị trí chi tiết người bán; nguy cơ nhận dạng và học thuộc cửa hàng. |
