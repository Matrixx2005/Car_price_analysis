# Cleaning log

_Sinh tự động bởi `main.py` → `src/cleaning.py`; dữ liệu raw không bị sửa._

## 1. Phễu lọc phạm vi (số bản ghi từng bước)

| step | rule | rows_before | rows_after | rows_removed | reason |
|---|---|---|---|---|---|
| raw | start (after type/whitespace/missing-token normalisation only) | 118,382 | 118,382 | 0 |  |
| used_only | is_used == True and is_new == False (NaN counts as unconfirmed) | 118,382 | 95,684 | 22,698 | new cars, pre-registered/demo cars and records without a confirmed used flag |
| offer_type | offer_type == 'U' (or missing) | 95,684 | 95,681 | 3 | offer types other than 'U' (used) are unconfirmed |
| vehicle_type | vehicle_type == 'Car' | 95,681 | 95,600 | 81 | non-passenger vehicles (e.g. Transporter) |
| commercial_body_type | body_type not in ['box', 'flatbed van', 'flatbed+tarpaulin', 'panel van', 'transporter', 'van-high roof'] | 95,600 | 95,407 | 193 | commercial body types listed under vehicle_type=Car |
| currency | price_currency == 'EUR' | 95,407 | 95,407 | 0 | other/missing currency (no conversion performed) |
| price_not_null | price is numeric and not null | 95,407 | 95,407 | 0 | unparseable/missing target |
| price_positive | price > 0 | 95,407 | 95,407 | 0 | zero/negative price |
| price_min | price >= 100 EUR | 95,407 | 95,399 | 8 | placeholder prices (e.g. price = 1) |
| content_duplicates | drop repeated (valid VIN, price, mileage, registration_date, country) | 95,399 | 94,266 | 1,133 | identical listings re-posted under a different id |

**Kết quả:** 118,382 → **94,266** bản ghi (79.6%).

## 2. Vì sao lọc như vậy

Các cờ trạng thái xe không loại trừ nhau theo cách đơn giản (`used_scope_flag_combinations.csv`). Trong 22.698 tin có `is_used = False`:

- 4,252 tin `is_new = True` (xe mới).
- 3,702 tin đăng ký trước (`is_preregistered = True`).
- **14,744 tin không phải xe mới, không phải đăng ký trước, nhưng cũng KHÔNG được gắn `is_used`** (trong đó 3,627 tin `is_rental = True`). Không đủ bằng chứng xác nhận là xe cũ → loại khỏi phạm vi chính và liệt kê là **trường hợp cần xem xét**.
- Số tin vừa `is_used` vừa `is_new`: 0; số tin `is_used` bị thiếu: 0 (các cờ boolean của file này không có giá trị thiếu).

Không dùng số km > 0 để suy ra xe cũ. `offer_type` ngoài `U` (3 tin loại `A`) bị loại vì không xác nhận được.

**Tiền tệ:** `price_currency` chỉ có một giá trị (EUR) → không loại tin nào, không chuyển đổi tiền tệ. **Loại phương tiện:** loại `Transporter` và các kiểu thân xe thương mại. **Giá:** giá null/≤0 = 0 tin; giá < 100 EUR = 8 tin (đều bằng 1 EUR, giá giữ chỗ). Xe đắt (tới 13,5 triệu EUR – Bugatti) là hợp lệ và được GIỮ LẠI.

## 3. Bản ghi trùng

- Dòng trùng toàn bộ: 0; `id` trùng: 0 (xem `dataset_profile.md`).
- **Trùng nội dung chắc chắn (cùng VIN hợp lệ 17 ký tự + giá + km + ngày đăng ký + quốc gia):** 1,133 tin bị loại (giữ tin đầu tiên).
- Cùng VIN nhưng khác giá/km/nước: KHÔNG xoá (có thể là tin đăng lại ở thời điểm khác); xử lý bằng chia dữ liệu theo nhóm (xem `week1_report.md`, chương 6).
- Còn 387 VIN có định dạng không hợp lệ (không phải 17 ký tự hợp lệ) – không dùng làm khoá nhóm.

## 4. Giá trị kỹ thuật không hợp lý → NaN (giữ hàng)

| column | rule | n_set_to_nan |
|---|---|---|
| nr_seats | seats outside (1, 9) | 7 |
| power_kw | power_kw outside (10, 1500) | 33 |
| cylinders_volume_cc | engine_cc > 10000 or <= 0 for non-electric (electric keeps 0) | 5 |
| mileage_km | mileage < 0 or > 1,000,000 km | 3 |

Quy tắc đặt theo giới hạn vật lý/hợp lý; không dùng IQR. Điền khuyết (imputation) KHÔNG làm ở tuần 1: thống kê điền sẽ được học trên train trong Pipeline tuần 2.

## 5. Cách suy năm sản xuất

`production_year` gốc chỉ có ở khoảng 19% bản ghi. Quy tắc: dùng `production_year` gốc nếu nằm trong [1950, 2026]; nếu không dùng năm của `registration_date`. Nguồn năm sau cleaning: 82,793 từ ngày đăng ký, 11,453 từ production_year gốc, 20 thiếu. 21 tin có `registration_date` sau ngày snapshot (08/11/2025) bị đánh cờ `flag_future_registration` và không dùng để suy năm. Năm tham chiếu để tính tuổi xe là **2025** (năm snapshot), không phải năm hiện tại.

## 6. Cờ cần xem xét (không loại bỏ)

- `flag_price_review`: 6 tin (giá > 20× trung vị cùng hãng+mẫu, hoặc dạng 999999). Danh sách: `reports/tables/price_review_list.csv`. Một số là xe hiếm có giá thật (vd. Carrera GT); số khác là lỗi nhập (BMW 316 giá 9,5 triệu). Chưa tự động loại – quyết định ở tuần 2 (có thể dùng thước đo bền vững / log).
- Xe cũ có 0 km: 14 tin (giữ nguyên).
- Xe thuê cũ (`is_rental = True`): 717 tin còn lại trong phạm vi. Dataset không có trường phân biệt tin cho thuê với tin bán xe từng là xe thuê; coi là xe cũ bình thường (rủi ro đã ghi nhận).

## 7. Rủi ro & trường hợp chưa giải quyết

- 14,744 tin trạng thái không xác định bị loại có thể là xe cũ thật (thiên lệch lựa chọn).
- Tin trùng khác VIN/không VIN không phát hiện được hoàn toàn (66% tin không có VIN).
- Giá chào bán ≠ giá giao dịch; không có thời điểm đăng tin nên không kiểm soát được biến động giá theo thời gian.
