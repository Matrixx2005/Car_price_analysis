# Rủi ro rò rỉ dữ liệu (data leakage)

## 1. Biến bị loại / hạn chế vì liên quan đến giá hoặc định danh

| column | role | leakage_risk | reason |
|---|---|---|---|
| seller_type | feature | medium | Dealer/Private: biết khi đăng tin, nhưng liên quan chiến lược định giá của người bán. |
| price_net | excluded | high | Giá chưa VAT – dẫn xuất trực tiếp từ price (price/(1+VAT)). Dùng sẽ làm lộ target. |
| price_vat_rate | excluded | high | Thuế suất VAT, chỉ có khi có price_net (thiếu >74%); gắn với cách tính giá. |
| price_tax_deductible | excluded | medium | Cờ khấu trừ VAT: thuộc cấu trúc giá của tin đăng, không phải đặc điểm xe. |
| price_negotiable | excluded | medium | Cờ thương lượng: hành vi định giá của người bán. |
| vin | excluded | high | Định danh xe duy nhất (nhận dạng được); chỉ dùng để tạo nhóm chia dữ liệu, không là feature. |
| listing_id | metadata | high | Mã tin đăng – định danh, không có ý nghĩa dự đoán. |
| description | excluded | medium | Văn bản tự do; có thể chứa giá/thông tin người bán; chưa thuộc phạm vi tuần 1. |
| seller_company_name | excluded | medium | Tên người bán cụ thể – dễ học thuộc người bán (nhận dạng); chỉ dùng cho khóa nhóm. |
| ratings_average | excluded | medium | Điểm đánh giá của người bán, thiếu 35%; không phải thuộc tính xe. |
| zip/city/street/latitude/longitude | excluded | medium | Vị trí chi tiết người bán; nguy cơ nhận dạng và học thuộc cửa hàng. |

## 2. Rò rỉ do cùng một xe xuất hiện ở train và test

- 1,866 nhóm xe có nhiều tin đăng (chung VIN hợp lệ hoặc cùng 'dấu vân tay' kỹ thuật), gồm 4,869 tin; nhóm lớn nhất 56 tin.
- Nếu dùng random split thường (seed 42), **860 / 18,854 dòng test** có tin cùng nhóm nằm ở train. Với GroupShuffleSplit: 0 dòng.
- Hạn chế: 66% tin không có VIN; nhóm theo dấu vân tay chỉ bắt được tin trùng thông số – tin trùng thật nhưng khác km/ngày đăng ký vẫn có thể lọt qua.

## 3. Quy tắc bắt buộc cho tuần 2

- Mọi imputer, scaler, encoder (đặc biệt Target/Frequency) chỉ `fit` trên train (trong `Pipeline`/cross-validation); không fit trên toàn bộ dữ liệu.
- Không dùng test để chọn feature, hyperparameter hay quy tắc cleaning phụ thuộc giá.
- Các bảng quan hệ feature–giá trong EDA được tính trên train; phân phối giá/đếm danh mục (không dùng target để chọn biến) tính trên toàn bộ dữ liệu sạch.
- Feature engineering ở tuần 1 chỉ gồm phép biến đổi từng dòng (không học thống kê): `vehicle_age`, `make_model`.
- Cờ `flag_price_review` được dựng từ quan hệ giá–mẫu xe trên toàn dữ liệu nên chỉ để sàng lỗi dữ liệu; **không dùng làm feature** và nếu dùng để loại dòng ở tuần 2 phải áp dụng thống nhất trước khi chia hoặc chỉ dựa trên train.
