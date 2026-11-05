"""Builds notebooks/03_week2_results.ipynb (display of the week-2 results; run src.experiments first)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Tuần 2: Encoding × Scaling × Mô hình hồi quy

Notebook chỉ **hiển thị kết quả** đã tính bởi `python -m src.experiments` (đọc từ `reports/tables/week2_*.csv`).
Đánh giá bằng GroupKFold 5 fold trên **tập train**; tập test không được dùng.
Báo cáo đầy đủ: `reports/week2_report.md`."""))
C.append(code("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import json, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.float_format', '{:,.2f}'.format)
T, F = ROOT / 'reports' / 'tables', ROOT / 'reports' / 'figures'
meta = json.loads((ROOT / 'reports' / 'week2_run_meta.json').read_text(encoding='utf-8'))
print(meta)
res = pd.read_csv(T / 'week2_fold_results.csv')
print(res.shape, '| thí nghiệm:', res.experiment.value_counts().to_dict())"""))
C.append(md("## 1. Cấu hình tốt nhất của từng mô hình (64 cấu hình, giá gốc)"))
C.append(code("""best = pd.read_csv(T / 'week2_best_per_model.csv')
best[['model','encoder','scaler','mae_mean','mae_std','rmse_mean','r2_mean','medape_pct_mean','n_features_out','fit_seconds_mean']]"""))
C.append(code("""display(Image(str(F / 'week2_best_models.png')))
Image(str(F / 'week2_heatmap_mae.png'))"""))
C.append(md("MAE trung bình 5 fold (EUR); màu sáng = tốt hơn. Random Forest gần như không đổi theo scaler; KNN và SVR cải thiện rất mạnh khi có scale."))
C.append(md("## 2. Encoding: so sánh cặp theo fold với One-hot (scaler Standard)"))
C.append(code("""pe = pd.read_csv(T / 'week2_paired_encoder_vs_onehot_standard.csv')
pe.sort_values(['model','level'])"""))
C.append(md("`folds_better` = số fold (trên 5) mà encoder đó tốt hơn One-hot; `rel_change_pct` > 0 = tệ hơn One-hot."))
C.append(code("""display(Image(str(F / 'week2_dims_time.png')))"""))
C.append(md("## 3. Scaling: so sánh với No scaling"))
C.append(code("""ps = pd.read_csv(T / 'week2_paired_scaler_vs_none.csv')
ps.groupby(['model','level']).agg(delta_mae_pct=('rel_change_pct','mean'), folds_better=('folds_better','sum'), n=('n','sum'))"""))
C.append(code("""display(Image(str(F / 'week2_scaler_effect.png')))"""))
C.append(md("## 4. Log transformation của giá (thí nghiệm riêng)"))
C.append(code("""display(pd.read_csv(T / 'week2_log_target_vs_raw_mae.csv'))
display(Image(str(F / 'week2_log_target_effect.png')))"""))
C.append(md("## 4b. Log transformation của biến đầu vào lệch (mileage, power, engine)"))
C.append(code("""display(pd.read_csv(T / 'week2_log_features_vs_raw_mae.csv'))
display(Image(str(F / 'week2_log_features_effect.png')))"""))
C.append(md("Ridge hưởng lợi, KNN/SVR tệ hơn, Random Forest gần như không đổi (cây bất biến với biến đổi đơn điệu)."))
C.append(md("## 5. Thêm `model_version` (71 nghìn danh mục)"))
C.append(code("""display(pd.read_csv(T / 'week2_model_version_vs_main_mae.csv'))
display(Image(str(F / 'week2_model_version_effect.png')))"""))
C.append(md("## 6. Xác nhận trên toàn bộ train (75.457 dòng; không gồm SVR)"))
C.append(code("""p = T / 'week2_confirm_full_train.csv'
pd.read_csv(p) if p.exists() else print('Chưa chạy: python -m src.experiments --confirm')"""))
C.append(md("""## 7. Kết luận ngắn
- Random Forest cho MAE thấp nhất; SVR và KNN ở giữa (gần nhau); Ridge kém nhất với giá gốc.
- **Scaling** quyết định kết quả của KNN và SVR, không ảnh hưởng Random Forest, ảnh hưởng nhỏ lên Ridge.
- **One-hot** tốt hơn Ordinal/Frequency ở KNN, SVR, Random Forest nhưng tốn nhiều cột và thời gian hơn. Với Ridge, tập con 20k không phân biệt rõ các encoder, nhưng trên toàn train One-hot vượt rõ rệt (mục 6).
- **Log giá** giúp Ridge/SVR rất nhiều nhưng có thể làm RMSE của KNN/Random Forest tệ hơn; **log biến đầu vào** chỉ giúp Ridge, làm KNN/SVR tệ hơn.
- Siêu tham số chưa tinh chỉnh; tập test chưa dùng (việc của tuần 3)."""))
nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/03_week2_results.ipynb")
