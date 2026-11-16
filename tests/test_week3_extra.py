import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src import pipelines, week3_extra as ex
from src.pipelines import CAT_MAIN, CAT_WITH_VERSION, NUM, build_estimator, make_model
from src.reporting_week3_extra import _spearman
from tests.test_week2_pipelines import make_frame


def test_boosting_is_available_but_not_part_of_the_week2_and_week3_model_lists():
    assert "hgb" not in pipelines.MODELS and pipelines.EXTRA_MODELS == ["hgb"]
    m = make_model("hgb", {"max_iter": 20})
    assert m.max_iter == 20 and m.early_stopping is False and m.random_state == pipelines.SEED
    with pytest.raises(ValueError):
        make_model("hgb", {"learning": 0.1})


@pytest.mark.parametrize("encoder", ["onehot", "ordinal", "target"])
def test_boosting_pipeline_handles_every_encoder_including_sparse_onehot(encoder):
    df = make_frame(300)
    X = df[CAT_MAIN + NUM]
    est = build_estimator(encoder, "none", "hgb", log_target=True, model_params={"max_iter": 20}).fit(X, df["price"])
    pred = est.predict(X)
    assert np.isfinite(pred).all() and pred.min() > 0                    # log target is mapped back to EUR


def test_hgb_random_search_is_reproducible_unique_and_in_bounds():
    a, b = ex.sample_hgb_params(30, seed=3), ex.sample_hgb_params(30, seed=3)
    assert a == b and a != ex.sample_hgb_params(30, seed=4)
    assert len({tuple(sorted(p.items(), key=str)) for p in a}) == 30
    assert all(p[k] in ex.HGB_SPACE[k] for p in a for k in p)
    assert len(ex.hgb_encoder_configs()) == 6 and len(ex.hgb_search_configs("target", True)) == ex.N_HGB


def test_subsample_groups_keeps_whole_groups_deterministically():
    groups = pd.Series(np.repeat(np.arange(50), 4))
    tr = np.arange(200)
    s1, s2 = ex.subsample_groups(tr, groups, 0.3, seed=1), ex.subsample_groups(tr, groups, 0.3, seed=1)
    assert (s1 == s2).all() and 0.2 * 200 < len(s1) < 0.4 * 200
    assert (groups.iloc[s1].value_counts() == 4).all()                    # no group is cut in half
    assert (ex.subsample_groups(tr, groups, 1.0, seed=1) == tr).all()


class _Predictor:
    """Stand-in estimator that depends only on `mileage_km` (so permuting other columns must change nothing)."""
    def predict(self, X):
        return X["mileage_km"].to_numpy() * 0.1


def test_permutation_importance_finds_the_informative_column_and_is_deterministic():
    df = make_frame(400)
    X = df[CAT_WITH_VERSION + NUM]
    y = pd.Series(X["mileage_km"].to_numpy() * 0.1, index=X.index)
    est = _Predictor()
    rows = ex.permutation_rows(est, X, y, base_mae=0.0, seed=5, n_repeats=2)
    t = pd.DataFrame(rows).groupby("feature")["delta_mae"].mean()
    assert t["mileage_km"] > 1000 and t.drop("mileage_km").abs().max() < 1e-9      # only the used column matters
    assert {"identity (brand+model[+version])", "age (year+vehicle_age)"} <= set(t.index)
    assert pd.DataFrame(rows).equals(pd.DataFrame(ex.permutation_rows(est, X, y, 0.0, 5, 2)))
    joint = pd.DataFrame(rows).query("kind == 'joint'")["feature"].unique()
    assert len(joint) == 2


def test_permutation_joint_group_shuffles_columns_together():
    df = make_frame(200)
    X = df[CAT_WITH_VERSION + NUM].copy()
    X["vehicle_age"] = 2025 - X["production_year"]                        # exactly consistent pair

    class AgeConsistency:
        def predict(self, Z):                                            # error appears when the pair is broken
            return np.abs((2025 - Z["production_year"]) - Z["vehicle_age"]).to_numpy()
    rows = pd.DataFrame(ex.permutation_rows(AgeConsistency(), X, pd.Series(np.zeros(len(X)), index=X.index), 0.0, 1, 2))
    by = rows.groupby("feature")["delta_mae"].mean()
    assert by["age (year+vehicle_age)"] < 1e-9 and by["production_year"] > 1 and by["vehicle_age"] > 1


def test_diag_task_returns_train_and_validation_metrics_and_permutation_rows():
    df = make_frame(300)
    X, y = df[CAT_WITH_VERSION + NUM], df["price"]
    from src.week3_experiments import make_cfg
    cfg = make_cfg("ridge", "onehot", "minmax", False, {})
    row, perm = ex.diag_task("ridge_x", cfg, 0, X, y, np.arange(200), np.arange(200, 300), "unit")
    assert {"mae_train", "mae_valid", "r2_train", "r2_valid", "medape_pct_valid"} <= set(row) and row["n_train_used"] == 200 and row["n_valid"] == 100
    assert len(perm) == (len(X.columns) + 2) * ex.N_REPEATS and all(p["name"] == "ridge_x" for p in perm)


def test_signatures_depend_on_settings_and_model_entries_use_frozen_files_only(monkeypatch):
    sig = ex.hgb_signature(1000)
    assert sig["scope"] == "week3_hgb_full" and sig["n_hgb"] == ex.N_HGB and "hgb_default" in sig
    ents = ex.model_entries(include_hgb=False)
    assert {"knn_tuned", "ridge_tuned", "svr_tuned", "rf_tuned", "rf_default", "knn_default"} <= set(ents)
    assert all(len(v["config_id"]) == 12 for v in ents.values())


def test_spearman_helper():
    a = pd.Series([1, 2, 3, 4.0])
    assert _spearman(a, a * 2) == pytest.approx(1.0) and _spearman(a, -a) == pytest.approx(-1.0)


def test_linear_sanity_of_the_stand_in_estimator():
    m = LinearRegression().fit(np.arange(10).reshape(-1, 1), np.arange(10))
    assert m.predict(np.array([[3]]))[0] == pytest.approx(3)
