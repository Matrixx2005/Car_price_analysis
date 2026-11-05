import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import final_eval, pipelines, week3_experiments as w3, week3_select
from src.pipelines import DEFAULT_PARAMS, MODELS, make_model
from src.transformers import RareCategoryGrouper, drop_extreme_prices
from tests.test_week2_pipelines import make_frame

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------- transformers
def test_rare_grouper_pools_rare_and_unseen_categories_using_fit_counts_only():
    tr = pd.DataFrame({"a": ["x"] * 5 + ["y"] * 3 + ["z"]})
    g = RareCategoryGrouper(min_count=3).fit(tr)
    out = g.transform(pd.DataFrame({"a": ["x", "y", "z", "NEW", None]}))
    assert out.ravel().tolist() == ["x", "y", "__other__", "__other__", "__other__"]
    g2 = RareCategoryGrouper(min_count=4).fit(tr)
    assert g2.transform(pd.DataFrame({"a": ["y"]})).ravel().tolist() == ["__other__"]       # threshold applies to fit counts


def test_extreme_price_threshold_comes_from_the_training_fold_only():
    y_tr = pd.Series(np.arange(1, 101, dtype=float))
    keep = drop_extreme_prices(y_tr, 0.9)
    assert keep.sum() == 90 and not keep[-1] and keep[0]
    assert drop_extreme_prices(y_tr, None).all()
    # the threshold is a function of the training prices alone: adding rows elsewhere cannot change it
    assert drop_extreme_prices(y_tr, 0.9).sum() == drop_extreme_prices(y_tr.copy(), 0.9).sum()


# ---------------------------------------------------------------- model parameters
def test_default_parameters_are_the_week2_values_and_unknown_keys_are_rejected():
    assert DEFAULT_PARAMS["knn"] == {"n_neighbors": 10, "weights": "distance", "p": 2}
    assert make_model("ridge").alpha == 1.0 and make_model("rf").n_estimators == 100 and make_model("rf").min_samples_leaf == 2
    assert make_model("svr").regressor.C == 10.0 and make_model("svr").regressor.epsilon == 0.1
    assert make_model("knn", {"n_neighbors": 3}).n_neighbors == 3
    with pytest.raises(ValueError):
        make_model("ridge", {"alphaa": 1.0})


def test_random_search_is_reproducible_in_bounds_and_excludes_defaults():
    a, b = w3.sample_params("svr", 10, seed=1), w3.sample_params("svr", 10, seed=1)
    assert a == b and len(a) == 10 and a != w3.sample_params("svr", 10, seed=2)
    for params, _ in a:
        assert all(params[k] in w3.SPACES["svr"][k] for k in params)
        assert params != DEFAULT_PARAMS["svr"]
    ridge = w3.sample_params("ridge", 0)
    assert len(ridge) == 13 * 2 - 2                                          # full alpha grid x log, minus the two defaults


def test_config_id_depends_on_settings_not_on_stage():
    a = w3.make_cfg("ridge", "onehot", "minmax", False, {"alpha": 3.0}, stage="tune")
    b = w3.make_cfg("ridge", "onehot", "minmax", False, {"alpha": 3.0}, stage="combine")
    c = w3.make_cfg("ridge", "onehot", "minmax", True, {"alpha": 3.0}, stage="tune")
    d = w3.make_cfg("ridge", "onehot", "minmax", False, {"alpha": 3.0}, extreme_q=0.99)
    assert a["config_id"] == b["config_id"] and len({a["config_id"], c["config_id"], d["config_id"]}) == 3


# ---------------------------------------------------------------- run_fold3 / resume
def _fold(df):
    tr, va = np.arange(200), np.arange(200, 300)
    X, y = df[pipelines.CAT_WITH_VERSION + pipelines.NUM], df["price"]
    return X, y, tr, va


def test_run_fold3_extreme_filter_rare_grouping_and_log_target():
    df = make_frame(300)
    X, y, tr, va = _fold(df)
    base = w3.make_cfg("ridge", "onehot", "minmax", False, {})
    r0 = w3.run_fold3(dict(base, fold=0), X, y, tr, va)
    r1 = w3.run_fold3(dict(w3.make_cfg("ridge", "onehot", "minmax", False, {}, extreme_q=0.9), fold=0), X, y, tr, va)
    assert r0["n_train_used"] == 200 and r1["n_train_used"] < 200 and r1["n_valid_ge_100k"] == r0["n_valid_ge_100k"]   # eval rows untouched
    rg = w3.run_fold3(dict(w3.make_cfg("ridge", "target", "minmax", False, {}, feature_set="with_model_version", rare_min_count=50),
                           fold=0), X, y, tr, va)
    assert np.isfinite(rg["mae"])
    rl = w3.run_fold3(dict(w3.make_cfg("rf", "ordinal", "standard", True, {"n_estimators": 20}), fold=0), X, y, tr, va)
    assert np.isfinite(rl["mae"]) and rl["mae"] < 1e6                                        # back-transformed to EUR


def test_week3_runner_resumes_and_never_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(w3, "RESULTS3", tmp_path / "r3.csv")
    monkeypatch.setattr(w3, "SIG3", tmp_path / "s3.json")
    monkeypatch.setattr(w3, "week3_signature", lambda n: {"n": n})
    df = make_frame(300)
    cfgs = [w3.make_cfg("ridge", "onehot", "minmax", False, {"alpha": a}) for a in (1.0, 10.0)]
    first = w3.run_configs(cfgs, df, n_jobs=1)
    assert len(first) == 10                                              # 2 configs x 5 folds
    second = w3.run_configs(cfgs + [w3.make_cfg("ridge", "onehot", "minmax", False, {"alpha": 100.0})], df, n_jobs=1)
    assert len(second) == 15 and not second.duplicated(["config_id", "fold"]).any()


# ---------------------------------------------------------------- decision rule and freezing
def _res(rows):
    return pd.DataFrame([dict(config_id=c, fold=f, mae=m) for c, maes in rows.items() for f, m in enumerate(maes)])


def test_adopt_if_consistent_requires_lower_mean_and_enough_fold_wins():
    res = _res({"base": [10, 10, 10, 10, 10], "good": [9, 9, 9, 9, 11], "lucky": [1, 1, 11, 11, 11], "bad": [11, 11, 11, 11, 11]})
    chosen, info = w3.adopt_if_consistent(res, "base", ["good", "lucky", "bad"])
    assert chosen == "good" and info["candidates"]["good"]["wins"] == 4
    assert w3.adopt_if_consistent(res, "base", ["lucky", "bad"])[0] == "base"       # lower mean but only 2/5 wins -> keep baseline
    assert w3.adopt_if_consistent(res, "base", [])[0] == "base"


def test_selected_model_is_the_lowest_cv_mae():
    frozen = {"models": {m: {"final": {"cv": {"mae_mean": v}}} for m, v in zip(MODELS, (5.0, 3.0, 4.0, 6.0))}}
    assert min(MODELS, key=lambda m: frozen["models"][m]["final"]["cv"]["mae_mean"]) == "ridge"
    assert "selected_model" in inspect.getsource(week3_select.freeze)


# ---------------------------------------------------------------- computational feasibility
def test_full_fit_time_estimate_scales_by_model_and_budget_excludes_slow_configs():
    est = w3.estimated_full_fit_seconds
    assert est("ridge", 10, 1000, 4000) == pytest.approx(40)                    # linear
    assert est("svr", 10, 1000, 4000) == pytest.approx(160)                     # quadratic
    assert 10 * 4 < est("rf", 10, 1000, 4000) < 10 * 16                         # between linear and quadratic

    def row(cid, mae, fit):
        return dict(config_id=cid, model="svr", stage="tune", n_folds=5, mae_mean=mae, est_full_fit_s=fit)
    agg = pd.DataFrame([row("slow_best", 1.0, 40_000.0), row("ok", 2.0, 100.0), row("ok2", 3.0, 10.0)])
    assert w3.best_tune_config(agg, "svr")["config_id"] == "ok"                 # the best MAE is infeasible: next best feasible
    allslow = pd.DataFrame([row("a", 1.0, 40_000.0), row("b", 2.0, 5_000.0)])
    assert w3.best_tune_config(allslow, "svr")["config_id"] == "b"              # none feasible: the fastest one


def test_skipped_configurations_are_recorded_and_excluded(tmp_path, monkeypatch):
    first = w3.tune_configs()[0]
    sk = tmp_path / "skipped.json"
    sk.write_text(json.dumps({first["config_id"]: {"reason": "too slow"}}))
    monkeypatch.setattr(w3, "SKIPPED", sk)
    ids = {c["config_id"] for c in w3.tune_configs()}
    assert first["config_id"] not in ids and len(ids) == len(w3.tune_configs())


def test_reference_entries_are_added_for_frozen_references_only():
    frozen = {"models": {m: {"default": {"model": m}, "final": {"model": m}} for m in MODELS},
              "references": {"svr": {"model": "svr", "fit_rows": "dev_subsample"}}}
    e = final_eval.entries(frozen)
    assert "svr_cvbest_20k" in e and "rf_cvbest_20k" not in e and len(e) == 1 + 2 * len(MODELS) + 1


# ---------------------------------------------------------------- test-set protection
def test_only_final_eval_reads_the_test_split():
    for p in SRC.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        if p.name in ("final_eval.py", "reporting_week3.py", "reporting.py", "reporting_week2.py", "split_data.py", "week3_select.py"):
            continue
        assert "test.parquet" not in text, p.name
    assert "test.parquet" in inspect.getsource(final_eval.read_test)
    assert inspect.getsource(final_eval).count("read_parquet") == 1
    assert "test.parquet" not in inspect.getsource(week3_select)           # freezing never touches the test split


def test_final_eval_refuses_second_run_and_missing_or_stale_freeze(tmp_path):
    marker, frozen = tmp_path / "done.json", tmp_path / "frozen.json"
    with pytest.raises(FileNotFoundError):
        final_eval.check_ready(frozen, marker, check_hashes=False)
    frozen.write_text(json.dumps({"train_md5": "x", "week3_results_md5": "y"}))
    assert final_eval.check_ready(frozen, marker, check_hashes=False)["train_md5"] == "x"
    with pytest.raises(RuntimeError, match="changed"):
        final_eval.check_ready(frozen, marker, check_hashes=True)                   # hashes do not match the real files
    marker.write_text("{}")
    with pytest.raises(final_eval.TestAlreadyUsed):
        final_eval.check_ready(frozen, marker, check_hashes=False)


def test_cluster_bootstrap_resamples_whole_groups_and_is_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    y = rng.uniform(5000, 90000, len(groups))
    preds = {"a": y + rng.normal(0, 2000, len(y)), "b": y + rng.normal(0, 4000, len(y))}
    ci1 = final_eval.cluster_bootstrap(y, preds, groups, [("a", "b")], n_boot=200, seed=7)
    ci2 = final_eval.cluster_bootstrap(y, preds, groups, [("a", "b")], n_boot=200, seed=7)
    pd.testing.assert_frame_equal(ci1, ci2)
    m = ci1[(ci1.kind == "metric") & (ci1.name == "a") & (ci1.metric == "mae")].iloc[0]
    point = np.abs(y - preds["a"]).mean()
    assert m["ci_low"] < point < m["ci_high"]
    d = ci1[ci1.kind == "paired_mae_diff"].iloc[0]
    assert d["ci_low"] > 0 and d["share_b_better"] == 0.0                   # model b is clearly worse than a


def test_entries_cover_naive_default_and_tuned_for_every_model():
    frozen = {"models": {m: {"default": {"model": m}, "final": {"model": m}} for m in MODELS}}
    e = final_eval.entries(frozen)
    assert e["naive_median"] is None and len(e) == 1 + 2 * len(MODELS)
    assert all(f"{m}_tuned" in e and f"{m}_default" in e for m in MODELS)
