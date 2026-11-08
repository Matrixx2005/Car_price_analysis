import numpy as np
import pandas as pd

from src import week2_extra
from src.week2_extra import FEATURES, run_task, subsample_groups
from tests.test_week2_pipelines import make_frame


def test_subsample_keeps_whole_groups_and_size():
    groups = pd.Series(np.repeat(np.arange(50), 4))          # 50 groups x 4 rows
    tr_idx = np.arange(200)
    kept = subsample_groups(tr_idx, groups, 0.5, seed=0)
    g = groups.iloc[kept]
    # every kept group is complete and about half of the rows are kept
    assert (g.value_counts() == 4).all()
    assert 90 <= len(kept) <= 110
    assert subsample_groups(tr_idx, groups, 1.0, seed=0) is tr_idx


def test_run_task_reports_train_and_valid_and_importance():
    df = make_frame(200)
    X, y, groups = df[FEATURES], df["price"], df["group_id"]
    tr, va = np.arange(150), np.arange(150, 200)
    rows, oof, imp = run_task("rf", "onehot", "standard", 0, 1.0, X, y, groups, tr, va, n_repeats=2)
    assert {r["split"] for r in rows} == {"train", "valid"}
    assert len(oof) == len(va) and oof["row"].tolist() == va.tolist()
    # importance is reported for the ORIGINAL columns, n_repeats values each
    assert {r["feature"] for r in imp} == set(FEATURES) and len(imp) == 2 * len(FEATURES)


def test_baseline_predicts_train_median_without_importance():
    df = make_frame(120)
    X, y, groups = df[FEATURES], df["price"], df["group_id"]
    tr, va = np.arange(100), np.arange(100, 120)
    rows, oof, imp = run_task("baseline", "-", "-", 0, 1.0, X, y, groups, tr, va, n_repeats=2)
    assert np.allclose(oof["y_pred"], y.iloc[tr].median())
    assert imp == []


def test_best_configs_reads_lowest_mean_mae(tmp_path):
    res = pd.DataFrame({"experiment": "main", "model": ["rf"] * 4, "encoder": ["onehot", "onehot", "target", "target"],
                        "scaler": "standard", "fold": [0, 1, 0, 1], "mae": [10.0, 12.0, 9.0, 20.0]})
    p = tmp_path / "r.csv"
    res.to_csv(p, index=False)
    assert week2_extra.best_configs(p) == {"rf": ("onehot", "standard")}
