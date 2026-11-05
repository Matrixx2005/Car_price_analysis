# Hướng dẫn Tuần 2: tôi đã xử lý từng vấn đề như thế nào

Tài liệu này dành cho bạn (sinh viên) để **hiểu, chạy lại và tự trình bày** phần tuần 2. Báo cáo kết quả chính thức nằm ở `reports/week2_report.md`; tài liệu này giải thích *vì sao* làm như vậy.

---

## 1. Tuần 2 làm gì (một đoạn)

Tuần 1 đã cho ra `train.parquet` (75.457 xe) và `test.parquet` (18.809 xe). Tuần 2 trả lời câu hỏi trung tâm của đề tài: **cách mã hóa biến phân loại và cách chuẩn hóa dữ liệu ảnh hưởng thế nào đến độ chính xác của bốn mô hình hồi quy.** Tôi xây một "máy chạy thí nghiệm" thử 4 encoder × 4 scaler × 4 mô hình (64 cấu hình), cộng thêm 4 thí nghiệm phụ (log của giá, log của biến đầu vào, thêm `model_version`, xác nhận trên toàn bộ train), đánh giá bằng cross-validation trên tập train, và **không đụng tới tập test**.

## 2. Cách chạy

Trong PowerShell, tại `F:\Car_price_analysis` (có `(.venv)` ở đầu dòng thì gõ `python ...` là đủ):

```powershell
python -m src.experiments              # lưới chính: 560 lần huấn luyện (112 cấu hình × 5 fold), khoảng 10 phút (có thể dừng và chạy tiếp)
python -m src.week2_analysis           # sinh bảng tổng hợp + biểu đồ (không cần chạy trước --confirm)
python -m src.experiments --confirm    # xác nhận trên toàn bộ train (chậm hơn, 16-20 phút trên máy 16 nhân; có thể dừng và chạy tiếp)
python -m src.week2_analysis           # chạy lại để thêm bảng xác nhận
python -m src.reporting_week2          # sinh reports/week2_report.md
python -m pytest -v                    # 50 test (25 của tuần 1 + 25 của tuần 2)
python -m src.experiments --quick      # chạy thử nhanh 5 giây để kiểm tra môi trường
```

Kết quả thô từng fold nằm ở `reports/tables/week2_fold_results.csv`. Nếu bị dừng giữa chừng, chạy lại đúng lệnh cũ: chương trình bỏ qua phần đã xong. Mỗi lần chạy có một "chữ ký" (`week2_signature.json`, `week2_confirm_signature.json`) ghi dữ liệu, số fold, seed, siêu tham số và phiên bản sklearn; nếu bạn đổi những thứ đó, chương trình **từ chối** ghi tiếp vào kết quả cũ để không trộn lẫn. Khi đó hãy xóa cả file kết quả (`week2_fold_results.csv` hoặc `week2_confirm_fold_results.csv`) lẫn file chữ ký tương ứng rồi chạy lại. Đừng tự xóa hoặc sửa một file kết quả giữa chừng nếu không định chạy lại toàn bộ.

Xem kết quả: `reports/week2_report.md`, notebook `notebooks/03_week2_results.ipynb`, ảnh trong `reports/figures/week2_*.png`.

---

## 3. Cách tôi xử lý từng vấn đề

### Vấn đề 1: "Làm sao thí nghiệm công bằng và không rò rỉ dữ liệu?"

**Nguy cơ:** nếu fit scaler, encoder (đặc biệt Target Encoding) trên toàn bộ dữ liệu rồi mới chia, thông tin của tập đánh giá lọt vào lúc huấn luyện và kết quả đẹp một cách giả tạo.

**Cách xử lý:**
- Mọi bước "học từ dữ liệu" (điền thiếu, encoder, scaler) là một bước **bên trong `Pipeline`** (`src/pipelines.py`). Khi cross-validation gọi `fit`, chúng chỉ học trên phần train của fold đó.
- Chia fold bằng **GroupKFold theo `group_id`** (nhóm xe từ tuần 1), để cùng một xe không nằm ở cả hai phía.
- Target Encoding dùng `TargetEncoder` của sklearn: nó tự "cross-fit" (mã hóa mỗi dòng train bằng thống kê của các dòng khác) và làm mượt, nên dòng train không bị mã hóa bằng chính giá của nó.
- Không đọc `test.parquet` ở bất kỳ file nào của tuần 2. Có unit test quét mã nguồn để chắc chắn điều đó.
- Có test chứng minh: thống kê scaler đúng bằng thống kê của phần train (dù tôi cố tình làm bẩn dữ liệu validation), danh mục lạ không làm chương trình lỗi.

**Còn một hạn chế tôi ghi thẳng trong báo cáo:** cross-fitting bên trong `TargetEncoder` chia ngẫu nhiên chứ không theo nhóm xe, nên các tin trùng xe trong cùng fold train có thể làm mã hóa của train hơi lạc quan. Chỉ ảnh hưởng phần train, không chạm vào fold đánh giá.

### Vấn đề 2: "SVR quá chậm trên 75 nghìn dòng"

**Nguyên nhân:** SVR kernel RBF có độ phức tạp xấp xỉ O(n²). Tôi đo thử: 16 nghìn dòng đã mất cỡ một phút mỗi lần, và lưới có 80+ lần huấn luyện SVR.

**Cách xử lý:**
- Lưới chính chạy trên **tập con 20.000 dòng** của train (`make_dev_sample`). Lấy theo **nhóm xe nguyên vẹn** (không cắt một xe làm đôi), seed 42, nên tái lập được.
- **Mọi** mô hình đều chạy trên cùng tập con này để so sánh công bằng.
- Sau đó chạy lại 4 encoder trên **toàn bộ train** cho KNN, Ridge, Random Forest để kiểm tra kết luận có giữ nguyên khi tăng dữ liệu (bước `--confirm`). SVR được bỏ qua ở bước này và ghi rõ là hạn chế.

### Vấn đề 3: "Giá lệch cực mạnh nên RMSE/R² dao động"

**Nguyên nhân:** vài xe giá hàng triệu EUR (Bugatti) làm RMSE và R² nhảy mạnh giữa các fold. Tôi thấy MAE giữa các fold vẫn lệch ~1.000 EUR trong khi nhiều khác biệt giữa các cấu hình chỉ cỡ vài trăm EUR.

**Cách xử lý:**
- Chỉ số chính là **MAE** và **MedAPE** (trung vị sai số phần trăm). RMSE và R² vẫn báo cáo đầy đủ (đề bài yêu cầu) nhưng không dùng để kết luận.
- Không chỉ nhìn "trung bình", mà làm **so sánh cặp theo fold**: với cùng một fold, encoder A tốt hơn encoder B hay không? Báo cáo đếm số fold trong 5 fold. Nếu A thua B ở 5/5 fold thì khác biệt là nhất quán, không phải nhiễu.
- Không đưa p-value vì 5 fold quá ít cho một kiểm định chính thức; báo cáo nói thẳng điều này.

### Vấn đề 4: "Scaler áp dụng lên cột nào?"

Đây là quyết định thiết kế ảnh hưởng trực tiếp tới kết quả nên tôi chốt và ghi rõ:
- Biến số: điền thiếu bằng trung vị, rồi scale.
- Ordinal, Target, Frequency tạo ra cột số liên tục, nên **được scale như biến số**.
- Cột One-hot (0/1) **không scale**: scale cột nhị phân hiếm sẽ làm các giá trị bị phóng đại thành số lớn bất thường và không đúng mục đích.
- Với "None", các cột giữ nguyên giá trị gốc.
Quyết định này được kiểm tra bằng unit test.

### Vấn đề 5: "Danh mục lạ và danh mục hiếm"

- **One-hot:** `min_frequency=10`: danh mục xuất hiện < 10 lần trong fold train được gom vào một cột "infrequent"; danh mục chưa từng thấy cũng rơi vào cột này. Nếu không gom, `model_version` sẽ tạo hàng chục nghìn cột.
- **Ordinal:** danh mục lạ = −1. Thứ tự là theo bảng chữ cái, tức là **thứ tự giả**; đây chính là điều tuần 1 đã cảnh báo, và kết quả tuần 2 cho thấy nó có hại.
- **Frequency:** tự viết `FrequencyEncoder` (`src/encoders.py`) vì sklearn không có; danh mục lạ có tần suất 0 (đúng với nghĩa "chưa thấy trong train").
- **Target:** danh mục lạ được thay bằng trung bình toàn cục (hành vi của `TargetEncoder`).

### Vấn đề 6: "Dữ liệu thiếu"

Điền thiếu nằm trong `Pipeline` nên chỉ học trên train: biến số dùng trung vị kèm cột "cờ báo thiếu" (ví dụ xe điện không có dung tích xi-lanh: việc thiếu là thông tin hữu ích); biến phân loại coi "missing" là một danh mục riêng.

### Vấn đề 7: "SVR cần chuẩn hóa target, nhưng đó có phải log transformation không?"

Không. Tham số C và ε của SVR tính theo đơn vị của y, nên với giá hàng chục nghìn EUR thì SVR hầu như không học được nếu để nguyên. Tôi chuẩn hóa y *bên trong định nghĩa mô hình SVR* (một phép biến đổi tuyến tính, đổi ngược chính xác) để SVR hoạt động đúng. **Log transformation** gồm hai thí nghiệm riêng: (a) áp `log1p` lên **giá**, huấn luyện, rồi `expm1` để đổi ngược, mọi chỉ số vẫn tính trên thang EUR để so sánh công bằng với giá gốc; (b) áp `log1p` lên ba **biến đầu vào** lệch phải (`mileage_km`, `power_kw`, `engine_cc`) trước khi điền thiếu và scale. Phép `log1p` không học gì từ dữ liệu nên không gây rò rỉ.

### Vấn đề 8: "Thí nghiệm phụ về model_version"

Tuần 1 phát hiện `model_version` có 71 nghìn danh mục. Thay vì để nó phá lưới chính, tôi tách thành thí nghiệm riêng (thêm biến này vào 4 encoder × 4 mô hình, scaler Standard) và so sánh cặp với lưới chính. Kết quả đáng chú ý: One-hot gần như không đổi (vì đã gom danh mục hiếm) trong khi Ordinal tệ đi rõ rệt.

### Các lỗi/cảnh báo tôi gặp và cách xử lý

| Gặp phải | Xử lý |
|---|---|
| sklearn 1.9 cảnh báo `TargetEncoder(random_state=...)` đã lỗi thời | Đổi sang truyền `cv=KFold(shuffle=True, random_state=42)` theo hướng dẫn mới; chạy lại và kiểm tra kết quả giống hệt lần trước |
| Hai unit test của chính tôi lỗi: một test giả định sai cấu trúc `ColumnTransformer` (`too many values to unpack`), một test tôi tính nhẩm sai kết quả mong đợi (1,8 chứ không phải 2,0) | Cả hai là lỗi của test chứ không phải của pipeline; đã sửa test và kiểm tra lại, tất cả test tuần 2 đạt |
| `TargetEncoder(cv=KFold(...))` chỉ chạy với scikit-learn ≥ 1.9 | Ghim `scikit-learn>=1.9` trong `requirements.txt` |
| Khi chạy `--confirm`, các tác vụ Random Forest trên 60.000 dòng với One-hot mất rất lâu | Giữ nguyên thiết kế (đã dùng song song), chấp nhận chờ, và ghi chú thời gian trong tài liệu |
| Chạy lưới dài có thể bị dừng | Ghi kết quả từng tác vụ ra CSV, chạy lại sẽ tiếp tục, không làm lại từ đầu |

---

## 4. Kết quả chính (đọc nhanh)

Số liệu trung bình 5 fold, tập con 20.000 dòng, giá gốc, MAE đơn vị EUR (chi tiết ở `reports/week2_report.md`):

| Mô hình | Cấu hình tốt nhất | MAE | Nhận xét |
|---|---|---|---|
| Random Forest | One-hot, scaler nào cũng như nhau | ≈ 7.370 | Tốt nhất; không phụ thuộc scaler |
| SVR | One-hot + Standard | ≈ 9.445 | Rất nhạy với scaling |
| KNN | One-hot + Standard | ≈ 9.622 | Rất nhạy với scaling |
| Ridge | One-hot + MinMax | ≈ 22.681 | Kém nhất với giá gốc; log giá cải thiện mạnh |

**Năm phát hiện để nhớ:**
1. **Scaling quyết định KNN và SVR** (giảm MAE khoảng 33–40% so với không scale), nhưng gần như không ảnh hưởng Ridge và hoàn toàn không ảnh hưởng Random Forest (cây chỉ so sánh ngưỡng).
2. **One-hot thắng Ordinal và Frequency** ở KNN, SVR, Random Forest, nhất quán ở 5/5 fold. Trên toàn bộ train, One-hot là encoder tốt nhất của cả KNN, Ridge và Random Forest. Cái giá: 282 cột (400 trên toàn train) so với 21 cột, và Random Forest chạy lâu hơn nhiều lần (khoảng 11–20 lần tùy lần đo; con số cụ thể trong báo cáo thay đổi theo tải của máy) trong khi MAE chỉ tốt hơn 3,4%.
   **Chú ý với Ridge:** trên tập con 20k các encoder gần như ngang nhau, nhưng trên toàn bộ train One-hot giảm MAE khoảng 32% (15.741 so với 23.124 EUR). Đây là lý do cần bước xác nhận: kết luận từ tập con không luôn đúng khi có nhiều dữ liệu hơn.
3. **Log giá** giúp Ridge và SVR rất nhiều nhưng có thể làm RMSE của KNN/Random Forest tệ hơn; **log biến đầu vào** chỉ giúp Ridge (MAE giảm 4–10%), làm KNN và SVR tệ hơn 3–11% và không ảnh hưởng Random Forest: hiệu quả phụ thuộc mô hình và chỉ số.
4. **model_version:** One-hot không đổi vì gom hiếm; Ordinal tệ hơn 8–12% ở KNN, SVR, Random Forest.
5. **Target Encoding** không vượt One-hot trong lưới chính, nhưng hữu ích khi KNN/SVR không scale (vì tự nhiên có thang đo cùng đơn vị với giá).

---

## 5. Câu hỏi giáo viên có thể hỏi và cách trả lời

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao không dùng toàn bộ train cho mọi thí nghiệm? | SVR có độ phức tạp O(n²). Em chạy lưới trên 20k dòng cho công bằng, rồi xác nhận lại trên toàn bộ train cho ba mô hình còn lại. |
| Làm sao biết em không rò rỉ dữ liệu? | Mọi bước học từ dữ liệu nằm trong Pipeline, chia fold theo nhóm xe, không đọc tập test; có unit test kiểm tra từng điểm. |
| Tại sao dùng MAE thay vì RMSE? | Giá lệch cực mạnh, vài xe triệu EUR làm RMSE/R² dao động giữa các fold. Em vẫn báo cáo RMSE nhưng kết luận dựa trên MAE, MedAPE và so sánh cặp theo fold. |
| Kết quả này có chắc chắn không? | Chênh lệch giữa các cấu hình hàng đầu nhỏ hơn độ lệch chuẩn giữa các fold nên em không kết luận một cấu hình duy nhất. Kết luận vững là theo nhóm: Random Forest tốt nhất, hiệu ứng scaling mạnh với KNN/SVR. |
| Sao Ridge kém như vậy? | Em chưa tinh chỉnh α, và quan hệ giá–đặc trưng phi tuyến với đuôi giá rất dài. Log giá cải thiện Ridge mạnh (MAE giảm gần một nửa), và khi dùng toàn bộ train với One-hot thì MAE Ridge giảm từ khoảng 23 nghìn xuống 15,7 nghìn EUR. Tuần 3 sẽ tinh chỉnh. |
| Kết luận trên tập con 20k có đúng khi dùng toàn bộ dữ liệu không? | Em đã kiểm tra: encoder tốt nhất của KNN, Ridge, Random Forest không đổi, nhưng thứ hạng các encoder còn lại đổi một phần (7/12 dòng giữ nguyên hạng) và hiệu ứng với Ridge rõ hơn nhiều trên toàn train. SVR chưa được xác nhận vì quá chậm. |
| Em đã dùng tập test chưa? | Chưa. Tập test chỉ dùng một lần ở tuần 3 với cấu hình đã chốt. |
| Hạn chế của nghiên cứu? | Siêu tham số cố định, 5 fold, chạy trên tập con 20k, dữ liệu thiên về xe cao cấp châu Âu, chưa tinh chỉnh gom danh mục hiếm cho các encoder khác One-hot. |

---

## 6. Danh sách file của tuần 2

| File | Vai trò |
|---|---|
| `src/encoders.py` | `FrequencyEncoder` (tự cài đặt) |
| `src/pipelines.py` | Dựng Pipeline: encoder × scaler × mô hình × log giá / log biến đầu vào |
| `src/experiments.py` | Chạy lưới thí nghiệm song song, ghi kết quả từng fold, chạy tiếp được, kiểm tra chữ ký để không trộn kết quả |
| `src/week2_analysis.py` | Gộp kết quả, so sánh cặp theo fold, vẽ biểu đồ |
| `src/reporting_week2.py` | Sinh `reports/week2_report.md` từ các bảng số liệu và tự kiểm chứng 14 phát biểu (Phụ lục A của báo cáo) |
| `tests/test_week2_pipelines.py` | 25 test: lưới, rò rỉ, encoder, scaler, log giá/log đầu vào, metric, so sánh cặp, chạy tiếp, chữ ký, tái lập, phát biểu của báo cáo |
| `reports/week2_report.md` | Báo cáo kết quả tuần 2 |
| `reports/tables/week2_*.csv` | Mọi bảng số liệu (thô theo fold và tổng hợp) |
| `reports/figures/week2_*.png` | Heatmap, thời gian/số chiều, hiệu ứng scaler/log/model_version |
| `notebooks/03_week2_results.ipynb` | Notebook trình bày kết quả |
| `notebooks/02_colab_baseline_train.ipynb` | Baseline nhỏ chạy trên Colab (làm từ trước) |
| `requirements-lock.txt` | Phiên bản chính xác của mọi thư viện đã dùng (`pip install -r requirements-lock.txt` để tái lập) |

---

## 7. Rà soát cuối: lỗi và thiếu sót đã phát hiện và sửa

Sau khi tuần 2 chạy xong, tôi rà soát lại toàn bộ code bằng phân tích tĩnh (`pyflakes`), đọc lại từng module và chạy lại thử nghiệm. Đã phát hiện và sửa:

| # | Phần | Vấn đề | Đã sửa |
|---|---|---|---|
| 1 | `experiments.py` | `--confirm` bắt buộc phải có `week2_summary_all.csv` (sản phẩm của bước phân tích); thiếu file thì báo lỗi khó hiểu | Tính scaler tốt nhất trực tiếp từ kết quả từng fold; thiếu kết quả chính thì báo rõ phải chạy lưới chính trước |
| 2 | `experiments.py` | Chạy tiếp (resume) không kiểm tra điều kiện: đổi dữ liệu, số fold, siêu tham số hoặc phiên bản sklearn thì kết quả cũ và mới bị **trộn lẫn mà không ai biết** | Thêm "chữ ký" cho mỗi lần chạy, từ chối ghi tiếp nếu chữ ký khác, kèm hướng dẫn xóa để chạy lại |
| 3 | `experiments.py` | Nếu một tác vụ lỗi giữa chừng, tối đa 7 kết quả đã xong nhưng chưa ghi bị mất; chạy khi không còn tác vụ nào và chưa có file kết quả thì lỗi | Ghi nốt kết quả đang chờ trong `finally`; xử lý trường hợp không có tác vụ |
| 4 | `week2_analysis.py` | Nếu file xác nhận chỉ chạy được một phần, vẫn tạo bảng xác nhận cụt rồi làm báo cáo hỏng (`KeyError`) | Chỉ tạo bảng khi đủ 12 cấu hình × 5 fold, nếu thiếu thì cảnh báo và xóa bảng cũ; có test |
| 5 | `reporting_week2.py` | Nhiều con số và phát biểu **gõ cứng** ("400 cột", "0/5 fold", "20/20", "20k"…) không được kiểm tra với dữ liệu; chạy lại ra kết quả khác thì báo cáo có thể nói sai. Có mã chết (biến không dùng) | Mọi con số lấy từ bảng; 14 phát biểu định tính được máy tự kiểm chứng và liệt kê ở Phụ lục A; nếu có phát biểu sai, báo cáo hiện cảnh báo đầu trang và chương trình in `[CLAIM FAILED]` |
| 6 | Thiếu so với đề bài | "Log transformation như thí nghiệm riêng" mới làm log của **giá**, chưa làm log của **biến đầu vào** | Thêm thí nghiệm `log_features` (16 cấu hình × 5 fold), bảng so sánh cặp, biểu đồ, mục 7 của báo cáo, phát biểu kiểm chứng |
| 7 | `encoders.py` | `FrequencyEncoder` đặt tên cột vô nghĩa (`x0_freq`…) | Dùng tên cột thật (`brand_freq`…); có test |
| 8 | Thiếu | Hình tổng hợp mô hình tốt nhất; file khóa phiên bản thư viện; test cho chạy tiếp/chữ ký | Thêm `week2_best_models.png`, `requirements-lock.txt` và 6 test mới |
| 9 | Chất lượng mã | Import thừa, f-string không có placeholder, biến không dùng (`eda.py`, `reporting.py`, `main.py`, test) | Dọn sạch; `pyflakes` không còn cảnh báo |
| 10 | Diễn giải | Một vài "giải thích" trong báo cáo được viết như sự thật dù chưa kiểm chứng | Ghi rõ "giải thích hợp lý, chưa kiểm chứng riêng"; thêm vào mục giới hạn |

**Một lỗi do chính bản sửa đầu tiên của tôi gây ra (đã phát hiện nhờ chạy thật và đã khắc phục):** khi sửa mục 1, tôi định nghĩa "scaler tốt nhất" là scaler có MAE trung bình tốt nhất trên các encoder, trong khi định nghĩa gốc (dùng cho bước xác nhận và cho báo cáo) là scaler của **cấu hình tốt nhất** của từng mô hình. Kết quả là `--confirm` tưởng cần chạy 20 tác vụ mới và đã ghi 32 dòng thừa vào `week2_confirm_fold_results.csv`. Tôi đã dừng tiến trình, sao lưu rồi gỡ đúng 32 dòng thừa (còn 60 dòng đúng như báo cáo), khôi phục định nghĩa gốc, và thêm test hồi quy để lỗi này không lặp lại. Số liệu sai số trong báo cáo không bị ảnh hưởng.

**Kiểm tra sau khi sửa:** chạy lại `--confirm`, lưới chính và `--quick` đều cho "0 tác vụ cần chạy" (đúng kỳ vọng), phân tích và báo cáo được sinh lại, 14/14 phát biểu được xác nhận, tất cả test đạt.
