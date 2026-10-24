"""Builds notebooks/01_exploratory_data_analysis.ipynb (then execute with nbconvert). Dev utility."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# EDA – Dự đoán giá ô tô cũ (AutoScout24, Zenodo 17643343)

**Mục tiêu:** hiểu dữ liệu trước khi nghiên cứu Encoding × Scaling (tuần 2–3).
**Phạm vi:** ô tô cũ (`is_used=True`, `is_new=False`), loại `Car`, giá EUR hợp lệ. **Target:** `price` – giá chào bán (không phải giá giao dịch).

> Quy tắc tách bạch leakage: phân phối giá / số đếm / cardinality tính trên **toàn bộ dữ liệu sạch** (không dùng target để chọn biến);
> mọi quan hệ feature–giá tính trên **tập train** (`data/processed/train.parquet`). Tập test không được dùng trong notebook này."""))
C.append(code("""import json, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display, Markdown
pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.float_format', '{:,.2f}'.format)
sns.set_theme(style='whitegrid')
T = ROOT / 'reports' / 'tables'
stats = json.loads((ROOT / 'reports' / 'stats.json').read_text(encoding='utf-8'))
df = pd.read_parquet(ROOT / 'data' / 'interim' / 'autoscout24_used_common.parquet')
train = pd.read_parquet(ROOT / 'data' / 'processed' / 'train.parquet')
print('clean:', df.shape, '| train:', train.shape)"""))
C.append(md("## 2. Tổng quan dữ liệu"))
C.append(code("""raw = stats['raw']
print(f"Raw: {raw['rows']:,} dòng x {raw['cols']} cột | RAM ~{raw['ram_mb_deep']:.0f} MB | trùng toàn bộ: {raw['full_row_duplicates']} | id trùng: {raw['id_duplicates']} | dòng có VIN lặp: {raw['vin_duplicate_rows']:,}")
print('Cột rỗng 100%:', raw['all_null_columns'])
display(pd.read_csv(T / 'scope_funnel.csv'))"""))
C.append(code("""miss = pd.DataFrame({'dtype': df.dtypes.astype(str), 'pct_missing': df.isna().mean() * 100}).sort_values('pct_missing', ascending=False)
display(miss.head(15))"""))
C.append(md("## 3. Phân tích giá (toàn bộ dữ liệu sạch)"))
C.append(code("""display(pd.read_csv(T / 'price_statistics.csv'))
display(Image(str(ROOT / 'reports/figures/price_histograms.png')))
display(Image(str(ROOT / 'reports/figures/price_boxplot.png')))"""))
C.append(md("""**Nhận xét:** phân phối lệch phải rất mạnh (skew trên thang EUR rất lớn, gần đối xứng sau log). Các điểm xa trong boxplot là xe sang hợp lệ (Bugatti, Rolls-Royce…) nên không bị loại; chúng sẽ chi phối RMSE. Log-transform của target là một thí nghiệm riêng ở tuần 2 (cần so sánh RMSE trên thang EUR gốc)."""))
C.append(md("## 4. Thuộc tính số"))
C.append(code("""display(Image(str(ROOT / 'reports/figures/year_mileage_distribution.png')))
display(df[['production_year','vehicle_age','mileage_km','power_kw','engine_cc','nr_seats']].describe(percentiles=[.01,.5,.99]).T)"""))
C.append(code("""display(Image(str(ROOT / 'reports/figures/price_vs_year_mileage.png')))
display(Image(str(ROOT / 'reports/figures/numeric_correlation.png')))"""))
C.append(md("Tương quan Spearman chỉ mô tả quan hệ đơn biến trên train, không dùng để chọn hay loại biến."))
C.append(md("## 5. Biến phân loại"))
C.append(code("""display(Image(str(ROOT / 'reports/figures/categorical_counts.png')))
card = pd.read_csv(T / 'categorical_cardinality.csv')
display(card[['column','n_unique','level','pct_missing','n_cat_count_eq1','n_cat_count_lt5','n_cat_count_lt10','pct_rows_in_cats_lt10']])
display(Image(str(ROOT / 'reports/figures/cardinality.png')))"""))
C.append(code("""mvs = stats['make_model']['model_version_summary']
print({k: v for k, v in mvs.items()})
display(pd.read_csv(T / 'unseen_categories_test_vs_train.csv'))"""))
C.append(md("""`model_version` có hàng chục nghìn danh mục, phần lớn chỉ xuất hiện một lần, và đa số dòng test mang giá trị chưa thấy ở train → One-hot bùng nổ chiều; Target/Frequency encoding rất bất ổn nếu không gom nhóm hiếm (học trên train)."""))
C.append(md("## 6. Quan hệ với giá (chỉ train; luôn kèm n)"))
C.append(code("""display(Image(str(ROOT / 'reports/figures/price_relationships.png')))
for name in ['median_price_by_brand','median_price_by_fuel_type','median_price_by_transmission','median_price_by_country']:
    print(name); display(pd.read_csv(T / f'{name}.csv').head(12))"""))
C.append(code("""display(pd.read_csv(T / 'country_summary.csv'))"""))
C.append(md("Giá theo quốc gia bị nhiễu bởi cơ cấu hãng/xe khác nhau; các nhóm n nhỏ (Luxembourg, LPG, CNG…) không đủ cơ sở kết luận."))
C.append(md("## 7. Kết luận EDA"))
C.append(code("""fl = stats['flags']; sp = stats['split']
print('Số xe cũ sạch:', f"{stats['n_clean']:,}", '| train/test:', sp['n_train'], '/', sp['n_test'])
print('Cờ giá cần xem xét:', fl['price_review'], '| năm suy từ:', fl['year_source'])
print('Random split sẽ rò rỉ:', stats['random_split_leak'])"""))
C.append(md("""**Đặc trưng đáng giữ:** brand, model (make_model), fuel_type, transmission, body_type, country_code, seller_type, drive_train, body_color, production_year/vehicle_age, mileage_km, power_kw, engine_cc, nr_seats. `model_version` là biến thí nghiệm cardinality riêng.

**Vấn đề dữ liệu:** `production_year` gốc thiếu ~81% (suy từ ngày đăng ký); drive_train/body_color thiếu nhiều; giá lệch rất mạnh, có vài giá nghi lỗi; trạng thái xe không nhất quán ở ~14,7 nghìn tin (đã loại); dữ liệu thiên về xe cao cấp.

**Giả thuyết cho tuần 2–3 (cần kiểm chứng):** (1) log-target cải thiện RMSE/MAE tương đối cho mô hình tuyến tính; (2) One-hot + Ridge tốt nhưng chiều lớn khi có model_version; (3) KNN/SVR nhạy cảm với scaling và với encoding có chiều lớn; (4) Random Forest ít phụ thuộc scaling; (5) Target encoding cải thiện nếu và chỉ nếu fit trong fold, có smoothing.

**Quyết định cần tránh (leakage):** dùng `price_net`/`price_vat_rate`; fit imputer/scaler/encoder trước khi chia; dùng test để chọn đặc trưng; random split bỏ qua nhóm xe trùng; đưa VIN/id/tên người bán vào mô hình."""))
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_exploratory_data_analysis.ipynb")
