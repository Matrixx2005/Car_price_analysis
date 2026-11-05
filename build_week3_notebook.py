"""Builds notebooks/04_week3_results.ipynb (display only; run the week-3 pipeline first)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Tuần 3: tinh chỉnh, đóng băng cấu hình, đánh giá cuối trên test

Notebook chỉ **hiển thị kết quả** đã tính (không chạy lại thí nghiệm, không đọc tập test).
Các lựa chọn dựa trên CV trên **train**; tập test được đánh giá **một lần** bởi `src/final_eval.py`.
Báo cáo đầy đủ: `reports/week3_report.md`."""))
C.append(code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.float_format', '{:,.2f}'.format, 'display.max_colwidth', 80)
T, F = ROOT / 'reports' / 'tables', ROOT / 'reports' / 'figures'
frozen = json.loads((ROOT / 'reports' / 'week3_frozen_config.json').read_text(encoding='utf-8'))
print('Mô hình được chọn bằng CV:', frozen['selected_model'], '| đóng băng lúc', frozen['created_at'])
print('Đánh giá test đã thực hiện:', json.loads((T / 'week3_test_evaluation_done.json').read_text())['evaluated_at'])"""))
C.append(md("## 1. Tinh chỉnh siêu tham số (CV trên train)"))
C.append(code("""tv = pd.read_csv(T / 'week3_tune_vs_default.csv')
tv[['model','n_configs_tried','n_infeasible','default_mae','tuned_mae','gain_pct','folds_tuned_better','tuned_log_target','tuned_params','any_best_mae','any_best_est_full_fit_s']]"""))
C.append(code("""display(Image(str(F / 'week3_tuning.png')))"""))
C.append(md("Vòng tròn đen = cấu hình quá đắt để huấn luyện lại trên toàn train (ước tính > 30 phút), không đủ điều kiện được chọn."))
C.append(md("## 2. Cắt xe giá cực cao khỏi tập huấn luyện"))
C.append(code("""display(pd.read_csv(T / 'week3_extremes_effect.csv'))
display(Image(str(F / 'week3_extremes.png')))"""))
C.append(md("## 3. `model_version` và gom danh mục hiếm"))
C.append(code("""display(pd.read_csv(T / 'week3_grouping_effect.csv'))
display(Image(str(F / 'week3_grouping.png')))"""))
C.append(md("## 4. Cấu hình đóng băng"))
C.append(code("""rows = []
for m, v in frozen['models'].items():
    e = v['final']
    rows.append(dict(model=m, encoder=e['encoder'], scaler=e['scaler'], log_target=e['log_target'], extreme_q=e['extreme_q'], feature_set=e['feature_set'],
                     rare_min_count=e['rare_min_count'], params=e['params'], cv_mae=e['cv']['mae_mean'], default_cv_mae=v['default']['cv']['mae_mean']))
pd.DataFrame(rows)"""))
C.append(md("## 5. Đánh giá cuối trên tập test (một lần)"))
C.append(code("""m = pd.read_csv(T / 'week3_test_metrics.csv').sort_values('mae')
ci = pd.read_csv(T / 'week3_test_bootstrap_ci.csv')
mae_ci = ci[(ci.kind == 'metric') & (ci.metric == 'mae')].set_index('name')[['ci_low', 'ci_high']]
m.merge(mae_ci, left_on='name', right_index=True).rename(columns={'ci_low': 'mae_ci_low', 'ci_high': 'mae_ci_high'})"""))
C.append(code("""display(Image(str(F / 'week3_test_models.png')))
display(Image(str(F / 'week3_cv_vs_test.png')))"""))
C.append(code("""display(ci[ci.kind == 'paired_mae_diff'])"""))
C.append(md("## 6. Phân tích phần dư của mô hình được chọn"))
C.append(code("""display(Image(str(F / 'week3_test_pred_vs_actual.png')))
display(Image(str(F / 'week3_test_errors_by_group.png')))
for k in ['price_band', 'age_band', 'country', 'brand', 'worst']:
    print(k); display(pd.read_csv(T / f'week3_test_errors_by_{k}.csv').head(12))"""))
C.append(md("""## 7. Kết luận ngắn
- Cấu hình được chốt chỉ từ CV trên train; tập test chỉ dùng một lần (có file đánh dấu và kiểm tra mã băm).
- Số liệu cụ thể và diễn giải: `reports/week3_report.md` (Phụ lục A tự kiểm chứng các phát biểu).
- Hạn chế chính: dữ liệu thiên về xe cao cấp châu Âu; kết luận chưa suy ra được cho thị trường Việt Nam."""))
nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/04_week3_results.ipynb")
