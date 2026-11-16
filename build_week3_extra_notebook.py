"""Builds notebooks/05_week3_extra_results.ipynb (display only; train-split results, no test access)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Tuần 3 bổ sung: boosting, chẩn đoán overfit, độ quan trọng đặc trưng, đường cong học

Notebook chỉ **hiển thị** kết quả đã tính (chỉ dùng tập **train**, không đọc tập test).
**Boosting chưa được đánh giá trên test** (tập test đã dùng một lần cho các mô hình đã đóng băng).
Báo cáo đầy đủ: `reports/week3_extra_report.md`."""))
C.append(code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.float_format', '{:,.2f}'.format, 'display.max_colwidth', 90)
T, F = ROOT / 'reports' / 'tables', ROOT / 'reports' / 'figures'
best = json.loads((ROOT / 'reports' / 'week3_hgb_best.json').read_text(encoding='utf-8'))
print('Boosting được chọn:', best['final']['encoder'], '| log giá:', best['final']['log_target'])
print(best['final']['params'])"""))
C.append(md("## 1. Boosting tinh chỉnh trên toàn train (CV 5 fold)"))
C.append(code("""display(pd.read_csv(T / 'week3_hgb_vs_models_fullcv.csv'))
display(Image(str(F / 'week3_hgb.png')))"""))
C.append(code("""pd.read_csv(T / 'week3_hgb_summary.csv').head(10)"""))
C.append(md("## 2. Chẩn đoán overfit: train so với validation"))
C.append(code("""display(pd.read_csv(T / 'week3_diag_summary.csv'))
display(Image(str(F / 'week3_diag_train_valid.png')))"""))
C.append(md("## 3. Độ quan trọng đặc trưng (permutation importance trên validation)"))
C.append(code("""display(Image(str(F / 'week3_importance.png')))
imp = pd.read_csv(T / 'week3_importance_summary.csv')
imp[(imp.data == 'dev20k') & (imp.name == 'rf_tuned')].head(12)"""))
C.append(md("## 4. Đường cong học: vì sao Random Forest tinh chỉnh thua mặc định trên test?"))
C.append(code("""display(pd.read_csv(T / 'week3_rf_crossover.csv'))
display(Image(str(F / 'week3_learning_curves.png')))"""))
C.append(md("""## 5. Kết luận ngắn
- Boosting tinh chỉnh trên toàn train cho MAE CV thấp nhất trong các mô hình so sánh, nhưng **chưa có số liệu test** (chủ ý, tránh xem test lần hai).
- Các cột quyết định giá: nhóm định danh xe, công suất, tuổi xe, số km.
- Xem `reports/week3_extra_report.md` (Phụ lục A tự kiểm chứng phát biểu)."""))
nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/05_week3_extra_results.ipynb")
