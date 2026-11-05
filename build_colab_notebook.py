"""Builds notebooks/02_colab_baseline_train.ipynb: a small baseline that runs on TRAIN only (Colab or local)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Baseline trên tập TRAIN (Colab)

Chạy thử Ridge và Random Forest trên `train.parquet`, đánh giá bằng **GroupKFold (5 fold) trên train**.
- **Không dùng tập test** (test chỉ dùng một lần ở cuối tuần 3).
- Imputer / scaler / encoder nằm trong `Pipeline` nên chỉ `fit` trên phần train của từng fold (không rò rỉ).
- `model_version` (71 nghìn danh mục) và `make_model` chưa đưa vào: đây là thí nghiệm riêng ở tuần 2.
- Đây là baseline để xem kết quả, chưa phải bộ thí nghiệm Encoding × Scaling đầy đủ."""))
C.append(code("""# Bước 1: nạp train.parquet. Trên Colab sẽ hiện nút chọn file: hãy chọn data/processed/train.parquet
import os
if not os.path.exists('train.parquet'):
    try:
        from google.colab import files
        up = files.upload()
        assert 'train.parquet' in up, 'Hãy chọn đúng file train.parquet'
    except ImportError:
        os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
import pandas as pd
path = 'train.parquet' if os.path.exists('train.parquet') else 'data/processed/train.parquet'
train = pd.read_parquet(path)
print(train.shape)
train.head()"""))
C.append(code("""import time, numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CAT = ['brand', 'model', 'fuel_type', 'transmission', 'body_type', 'country_code', 'seller_type', 'drive_train', 'body_color']
NUM = ['production_year', 'vehicle_age', 'mileage_km', 'power_kw', 'engine_cc', 'nr_seats']
X, y, groups = train[CAT + NUM], train['price'], train['group_id']

def ridge_onehot():
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('oh', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10))]), CAT)])
    return Pipeline([('pre', pre), ('m', Ridge(alpha=1.0))])

def rf_ordinal():
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('oe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), CAT)])
    return Pipeline([('pre', pre), ('m', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42))])

def log_target(p):
    return TransformedTargetRegressor(regressor=p, func=np.log1p, inverse_func=np.expm1)

experiments = {
    'Ridge + One-hot (giá gốc)': ridge_onehot(),
    'Ridge + One-hot (log giá)': log_target(ridge_onehot()),
    'RandomForest + Ordinal (giá gốc)': rf_ordinal(),
    'RandomForest + Ordinal (log giá)': log_target(rf_ordinal()),
}
cv = GroupKFold(n_splits=5)   # cùng một xe không bao giờ nằm ở cả hai phía của một fold
rows = []
for name, est in experiments.items():
    t0 = time.time()
    r = cross_validate(est, X, y, groups=groups, cv=cv, n_jobs=1,
                       scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'})
    rows.append({'thí nghiệm': name, 'RMSE (EUR)': -r['test_rmse'].mean(), 'MAE (EUR)': -r['test_mae'].mean(),
                 'R2': r['test_r2'].mean(), 'R2 std': r['test_r2'].std(), 'thời gian fit/fold (s)': r['fit_time'].mean(),
                 'tổng (s)': time.time() - t0})
    print(rows[-1]['thí nghiệm'], '-> xong')"""))
C.append(code("""res = pd.DataFrame(rows).set_index('thí nghiệm')
pd.set_option('display.float_format', '{:,.2f}'.format)
res"""))
C.append(code("""# Số chiều sau mã hóa (fit trên toàn bộ train chỉ để ĐẾM cột, không dùng để đánh giá)
for name, est in [('One-hot', ridge_onehot()), ('Ordinal', rf_ordinal())]:
    n_features = est.fit(X.head(20000), y.head(20000)).named_steps['pre'].transform(X.head(5)).shape[1]
    print(f'{name}: {n_features} cột đầu vào cho mô hình')"""))
C.append(md("""**Cách đọc kết quả**
- **RMSE/MAE** càng thấp càng tốt (đơn vị EUR); **R²** càng gần 1 càng tốt.
- RMSE bị chi phối bởi xe rất đắt, nên MAE thường dễ hiểu hơn. So sánh giá gốc và log giá để xem log-transform có giúp không.
- Số liệu là trung bình 5 fold trên train. Kết quả này **chưa** dùng tập test."""))
nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/02_colab_baseline_train.ipynb")
