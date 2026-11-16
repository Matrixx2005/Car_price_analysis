# Hướng dẫn chạy dự án bằng Jupyter Notebook

## 1. Trả lời nhanh

**Có, toàn bộ dự án chạy được trong Jupyter**, qua notebook `notebooks/00_run_everything.ipynb` (mỗi ô chạy một bước). Nhưng "chậm trên máy local" là do **một số bước tính toán nặng**, không phải do chạy bằng terminal hay notebook. Chạy bằng Jupyter không làm nhanh hơn. Cách tiết kiệm thời gian nhất là **không chạy lại những gì đã có kết quả sẵn** (đã nằm trong repo).

| Bước | Thời gian (máy 16 nhân) | Có cần chạy lại không? |
|---|---|---|
| Tuần 1: `main.py` (tải 548 MB + làm sạch + chia dữ liệu) | 2-5 phút | **Có**, chạy một lần để tạo `train.parquet`/`test.parquet` |
| Kiểm thử `pytest` | < 1 phút | Nên chạy |
| `demo.py`, sinh bảng/biểu đồ/báo cáo từ kết quả có sẵn | < 1 phút | Nên chạy |
| Tuần 2 lưới thí nghiệm (`src.experiments`) | ~10 phút | Không cần (có sẵn) |
| Tuần 2 xác nhận toàn train (`--confirm`) | 16-20 phút | Không cần (có sẵn) |
| Tuần 3 CV 4 giai đoạn (`src.week3_experiments`) | 1-2 giờ | Không cần (có sẵn) |
| Tuần 3 đánh giá test (`src.final_eval`) | ~1,3 giờ | **TUYỆT ĐỐI KHÔNG** (đã đánh giá một lần, chương trình từ chối) |

Máy ít nhân hơn (4-8 nhân) sẽ lâu hơn tỉ lệ thuận với số nhân; Colab miễn phí chỉ có 2 nhân nên **không nên** chạy các bước nặng ở đó.

## 2. Cách A: VS Code (khuyến nghị, đang dùng)

1. Cài extension **Jupyter** và **Python** của VS Code (nếu chưa có).
2. Mở thư mục `F:\Car_price_analysis`, mở `notebooks/00_run_everything.ipynb`.
3. Góc trên bên phải bấm **Select Kernel → Python Environments → `.venv`** (đường dẫn `.venv\Scripts\python.exe`).
   - Không thấy `.venv`? Chạy trong terminal: `.\.venv\Scripts\python -m ipykernel install --user --name car-price --display-name "Car price (.venv)"` rồi chọn kernel `Car price (.venv)`.
4. Chạy lần lượt từng ô (**Shift+Enter**) từ trên xuống:
   - Ô 1 (chuẩn bị) → Ô 2 (công tắc `RUN_HEAVY = False`, **giữ nguyên False**) → Bước 1 `main.py` → Bước 2 pytest → Bước 3 demo → Bước 4 sinh báo cáo → Bước 6 xem báo cáo.
5. Kết quả mong đợi: bước 1 kết thúc bằng `Done. n_clean = 94266 | train/test = 75457 / 18809`; bước 2 báo `81 passed`.

Các notebook khác (chỉ **hiển thị** kết quả, đã chạy sẵn, mở ra xem được ngay; không đọc tập test):

| Notebook | Nội dung |
|---|---|
| `00_run_everything.ipynb` | Chạy toàn bộ dự án từng bước (phần này) |
| `01_exploratory_data_analysis.ipynb` | EDA tuần 1 |
| `02_colab_baseline_train.ipynb` | Baseline nhỏ chạy trên Colab (~2 phút), tải `train.parquet` lên |
| `03_week2_results.ipynb` | Kết quả tuần 2 |
| `04_week3_results.ipynb` | Kết quả tuần 3 |

## 3. Cách B: Jupyter Lab trong trình duyệt

```powershell
cd F:\Car_price_analysis
.\.venv\Scripts\Activate.ps1
python -m jupyter lab
```
Trình duyệt tự mở; vào thư mục `notebooks/`, mở `00_run_everything.ipynb`, kernel mặc định `Python 3` (chính là `.venv` vì đã mở từ môi trường ảo).

## 4. Cách C: Google Colab (chỉ nên dùng cho bước nhẹ)

Sau khi repo đã có trên GitHub, mở:
`https://colab.research.google.com/github/Matrixx2005/Car_price_analysis/blob/main/notebooks/00_run_everything.ipynb`

- Ô 1 tự clone repo và cài thư viện. Giữ `RUN_HEAVY = False`.
- **Mỗi phiên Colab mới phải tải lại 548 MB dữ liệu** (phiên bị xóa sau khi đóng), `main.py` mất khoảng 5-10 phút.
- Chỉ nên chạy bước 1-4. Các bước nặng trên 2 nhân sẽ mất nhiều giờ và Colab có thể ngắt phiên.
- Muốn thử mô hình nhanh trên Colab: dùng `02_colab_baseline_train.ipynb` (tải lên `data/processed/train.parquet`).

## 5. Nếu thật sự cần chạy lại các bước nặng

1. **Chạy trong terminal, không chạy trong ô notebook.** Ô notebook bị chặn tới khi xong và dễ chết kernel; terminal chạy nền được. Ví dụ (PowerShell, mở ở thư mục dự án, đã `Activate`):
   ```powershell
   Start-Process python -ArgumentList "-u","-m","src.experiments" -RedirectStandardOutput exp.log -RedirectStandardError exp.err -WindowStyle Hidden
   Get-Content exp.log -Tail 3        # xem tiến độ
   ```
2. Tắt chế độ ngủ của máy trong lúc chạy (Settings → System → Power → Screen and sleep → Never), nhớ cắm sạc.
3. Các lệnh đều **chạy tiếp được**: nếu bị dừng, chạy lại đúng lệnh, phần đã xong được bỏ qua.
4. Thứ tự: `python -m src.experiments` → `python -m src.experiments --confirm` → `python -m src.week3_experiments --stage all` → `python -m src.week3_select`. **Không** chạy `src.final_eval`.
5. Nếu bị báo "different settings" (chữ ký khác, ví dụ khác phiên bản sklearn): xóa file kết quả và file chữ ký tương ứng trong `reports/tables/` rồi chạy lại:
   - tuần 2 lưới: `week2_fold_results.csv` + `week2_signature.json`;
   - tuần 2 xác nhận: `week2_confirm_fold_results.csv` + `week2_confirm_signature.json`;
   - tuần 3: `week3_fold_results.csv` + `week3_signature.json` (và `week3_skipped_configs.json` nếu muốn tìm kiếm lại từ đầu).
   Lưu ý: xóa rồi chạy lại tuần 3 sẽ thay đổi kết quả đã đóng băng, nên chỉ làm khi bạn hiểu và muốn làm lại cả tuần 3.

## 6. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Kernel not found` / không thấy `.venv` | Chạy lệnh `ipykernel install` ở mục 2, bước 3 |
| `ModuleNotFoundError: src` hoặc không thấy `main.py` | Notebook phải chạy trong thư mục dự án; ô 1 đã tự chuyển thư mục, nếu vẫn lỗi hãy mở VS Code đúng thư mục `F:\Car_price_analysis` |
| `FileNotFoundError ... train.parquet` | Chưa chạy bước 1 (`main.py`) |
| Kernel chết/treo ở bước nặng | Chạy bước đó trong terminal (mục 5) thay vì trong notebook |
| `Existing file is NOT valid` ở bước 1 | File CSV trong `data/raw/` đã bị sửa; đổi tên file đó rồi chạy lại bước 1 để tải bản chuẩn |
| Máy quá chậm | Chỉ chạy bước 1-4; mọi kết quả nặng đã có sẵn trong `reports/` |
| Hiện cảnh báo vàng (Warning) | Bình thường, không ảnh hưởng kết quả |
