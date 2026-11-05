import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import TargetEncoder

from src import experiments, pipelines, week2_analysis
from src.encoders import FrequencyEncoder
from src.experiments import define_experiments, make_dev_sample, regression_metrics, run_fold
from src.pipelines import CAT_MAIN, CAT_WITH_VERSION, ENCODERS, MODELS, NUM, SCALERS, build_estimator, fitted_preprocessor


def make_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    brand = rng.choice(["BMW", "Audi", "Opel", "Fiat"], n)
    price = np.where(brand == "BMW", 40000, np.where(brand == "Audi", 35000, 12000)) * rng.lognormal(0, 0.2, n)
    df = pd.DataFrame({
        "brand": brand, "model": rng.choice([f"m{i}" for i in range(12)], n),
        "model_version": [f"v{i}" for i in range(n)],
        "fuel_type": rng.choice(["Diesel", "Gasoline", None], n), "transmission": rng.choice(["Manual", "Automatic"], n),
        "body_type": rng.choice(["Sedan", "Coupe"], n), "country_code": rng.choice(["DE", "IT", "NL"], n),
        "seller_type": rng.choice(["Dealer", "Private"], n), "drive_train": rng.choice(["4WD", "FWD", None], n),
        "body_color": rng.choice(["Black", "Blue", "Grey"], n),
        "production_year": rng.integers(2005, 2025, n).astype(float), "vehicle_age": rng.integers(0, 20, n).astype(float),
        "mileage_km": rng.integers(1000, 250000, n).astype(float), "power_kw": rng.integers(50, 400, n).astype(float),
        "engine_cc": np.where(rng.random(n) < 0.1, np.nan, rng.integers(900, 4000, n)), "nr_seats": rng.choice([2.0, 4.0, 5.0], n),
        "price": price, "group_id": np.repeat(np.arange(n // 2), 2)[:n],
    })
    return df


# ---------------------------------------------------------------- design of the grid
def test_grid_size_and_factors():
    cfgs = pd.DataFrame(define_experiments())
    main = cfgs[cfgs.experiment == "main"]
    assert len(main) == len(ENCODERS) * len(SCALERS) * len(MODELS) == 64
    assert set(main.encoder) == {"onehot", "ordinal", "target", "frequency"}
    assert set(main.scaler) == {"none", "standard", "minmax", "robust"}
    assert set(main.model) == {"knn", "ridge", "svr", "rf"}
    for exp in ("log_target", "log_features", "model_version"):
        assert (cfgs.experiment == exp).sum() == 16, exp
    assert len(cfgs) == 112
    assert not cfgs.duplicated(["experiment", "encoder", "scaler", "model"]).any()


def test_week2_code_never_touches_test_file():
    for mod in (experiments, pipelines, week2_analysis):
        src = inspect.getsource(mod)
        assert "test.parquet" not in src and "DATA_PROCESSED / \"test" not in src, mod.__name__
    assert "train.parquet" in inspect.getsource(experiments.load_train)


# ---------------------------------------------------------------- no leakage
def test_all_learned_steps_live_inside_the_pipeline():
    est = build_estimator("target", "standard", "ridge")
    names = [t[0] for t in est.named_steps["pre"].transformers]
    assert names == ["num", "cat"] and list(est.named_steps) == ["pre", "m"]


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
def test_scaler_statistics_come_from_train_only(scaler):
    df = make_frame()
    tr, va = df.iloc[:200], df.iloc[200:].copy()
    va["mileage_km"] = va["mileage_km"] * 1000          # would change any statistic that peeked at validation data
    est = build_estimator("ordinal", scaler, "ridge")
    X = CAT_MAIN + NUM
    est.fit(tr[X], tr["price"])
    num_sc = fitted_preprocessor(est).named_transformers_["num"].named_steps["sc"]
    j = NUM.index("mileage_km")
    ref = {"standard": ("mean_", tr["mileage_km"].mean()), "minmax": ("data_min_", tr["mileage_km"].min()),
           "robust": ("center_", tr["mileage_km"].median())}[scaler]
    assert np.isclose(getattr(num_sc, ref[0])[j], ref[1])
    est.predict(va[X])                                  # transform of validation data must not refit anything
    assert np.isclose(getattr(num_sc, ref[0])[j], ref[1])


def test_frequency_encoder_learns_on_fit_only_and_unseen_is_zero():
    tr = pd.DataFrame({"a": ["x", "x", "y", "z"]})
    enc = FrequencyEncoder().fit(tr)
    out = enc.transform(pd.DataFrame({"a": ["x", "y", "NEW"]}))
    assert np.allclose(out.ravel(), [0.5, 0.25, 0.0])
    enc2 = FrequencyEncoder().fit(pd.DataFrame({"a": ["x"] * 10}))
    assert np.allclose(enc2.transform(pd.DataFrame({"a": ["x"]})).ravel(), [1.0])   # statistics differ -> they were learned in fit


def test_target_encoder_cross_fits_training_rows():
    """Training rows must NOT be encoded with their own target: fit_transform != transform on the same data."""
    df = make_frame()
    enc = pipelines.make_encoder("target")
    Xc = df[["brand", "model"]].astype(str)
    a = enc.fit_transform(Xc, df["price"])
    b = enc.transform(Xc)
    assert not np.allclose(a, b)
    assert isinstance(enc, TargetEncoder)


def test_onehot_dummies_are_not_scaled_other_encoders_are():
    df = make_frame()
    X = CAT_MAIN + NUM
    est = build_estimator("onehot", "standard", "ridge").fit(df[X], df["price"])
    cat_out = fitted_preprocessor(est).named_transformers_["cat"].transform(df[CAT_MAIN])
    cat_out = cat_out.toarray() if hasattr(cat_out, "toarray") else cat_out
    assert set(np.unique(cat_out)) <= {0.0, 1.0}
    est2 = build_estimator("frequency", "minmax", "ridge").fit(df[X], df["price"])
    f = fitted_preprocessor(est2).named_transformers_["cat"].transform(df[CAT_MAIN])
    assert f.min() >= 0 and np.isclose(f.max(), 1.0)


def test_unseen_categories_do_not_crash_any_encoder():
    df = make_frame()
    tr, va = df.iloc[:200].copy(), df.iloc[200:].copy()
    va["brand"] = "BRAND_NEVER_SEEN"
    X = CAT_MAIN + NUM
    for enc in ENCODERS:
        est = build_estimator(enc, "standard", "ridge").fit(tr[X], tr["price"])
        assert np.isfinite(est.predict(va[X])).all(), enc


# ---------------------------------------------------------------- target transformation & metrics
def test_log_target_predictions_are_in_euro_scale():
    df = make_frame()
    X = CAT_MAIN + NUM
    est = build_estimator("ordinal", "none", "rf", log_target=True).fit(df[X], df["price"])
    pred = est.predict(df[X])
    assert pred.min() > 0 and 1000 < pred.mean() < 200000          # back-transformed with expm1, not log units


def test_regression_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y, p = rng.uniform(5000, 90000, 200), rng.uniform(5000, 90000, 200)
    m = regression_metrics(y, p)
    assert np.isclose(m["rmse"], np.sqrt(mean_squared_error(y, p)))
    assert np.isclose(m["mae"], mean_absolute_error(y, p)) and np.isclose(m["r2"], r2_score(y, p))
    assert np.isclose(m["medape_pct"], np.median(np.abs(y - p) / y) * 100)


# ---------------------------------------------------------------- analysis helpers
def test_paired_vs_baseline_counts_folds_and_sign():
    rows = []
    for fold, (a, b) in enumerate([(10, 12), (10, 13), (10, 11), (10, 14), (10, 9)]):
        rows.append(dict(experiment="main", encoder="onehot", scaler="standard", model="knn", fold=fold, mae=a))
        rows.append(dict(experiment="main", encoder="ordinal", scaler="standard", model="knn", fold=fold, mae=b))
    out = week2_analysis.paired_vs_baseline(pd.DataFrame(rows), "encoder", "onehot", {"scaler": "standard"})
    r = out.iloc[0]
    assert r["level"] == "ordinal" and r["folds_better"] == 1 and r["n"] == 5       # ordinal beats one-hot only in fold 4
    assert np.isclose(r["mean_diff"], 1.8) and np.isclose(r["rel_change_pct"], 18.0)   # diffs 2,3,1,4,-1 on a baseline of 10


def test_aggregate_means_and_std_over_folds():
    res = pd.DataFrame({"experiment": "main", "feature_set": "main", "encoder": "onehot", "scaler": "none", "model": "ridge",
                        "fold": range(5), "rmse": 2.0, "mae": [1.0, 2.0, 3.0, 4.0, 5.0], "r2": 0.5, "medape_pct": 1.0,
                        "fit_seconds": 1.0, "predict_seconds": 0.1, "n_features_out": 10})
    agg = week2_analysis.aggregate(res).iloc[0]
    assert agg["mae_mean"] == 3.0 and np.isclose(agg["mae_std"], np.std([1, 2, 3, 4, 5], ddof=1)) and agg["n_folds"] == 5


def test_incomplete_confirmation_results_are_rejected(tmp_path, capsys):
    """A partial --confirm run (e.g. interrupted) must not produce a confirmation table."""
    rows = [dict(experiment="full_train_confirm", feature_set="main", encoder=e, scaler="standard", model="knn", fold=f, rmse=1.0,
                 mae=1.0, r2=0.5, medape_pct=1.0, fit_seconds=1.0, predict_seconds=0.1, n_features_out=5)
            for e in ("onehot", "target") for f in range(3)]
    p = tmp_path / "partial.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert week2_analysis.confirm_comparison(p) is None
    assert "incomplete" in capsys.readouterr().out


# ---------------------------------------------------------------- resume, signature, confirm, log features
def test_resume_skips_finished_tasks_and_does_not_duplicate(tmp_path):
    df = make_frame(240)
    cfgs = [dict(experiment="main", feature_set="main", encoder=e, scaler="standard", model="ridge") for e in ("onehot", "ordinal")]
    path = tmp_path / "res.csv"
    first = experiments.run_experiments(cfgs, df, n_jobs=1, results_path=path, n_splits=2)
    assert len(first) == 4
    second = experiments.run_experiments(cfgs, df, n_jobs=1, results_path=path, n_splits=2)
    assert len(second) == 4 and not second.duplicated(experiments.KEY).any()
    more = cfgs + [dict(experiment="main", feature_set="main", encoder="target", scaler="standard", model="ridge")]
    third = experiments.run_experiments(more, df, n_jobs=1, results_path=path, n_splits=2)
    assert len(third) == 6                                         # only the new config was run


def test_signature_blocks_mixing_results_from_different_settings(tmp_path):
    sig, res = tmp_path / "sig.json", tmp_path / "res.csv"
    a = {"scope": "dev", "n_rows": 100, "n_splits": 5, "seed": 42}
    experiments.check_signature(sig, a, res)                       # no results yet: just records the signature
    res.write_text("experiment\nmain\n")
    experiments.check_signature(sig, dict(a), res)                 # same settings: fine
    with pytest.raises(RuntimeError, match="different settings"):
        experiments.check_signature(sig, dict(a, n_splits=3), res)
    sig.unlink()
    experiments.check_signature(sig, a, res)                       # legacy results without a signature are adopted
    assert sig.exists()


def test_best_scaler_is_the_scaler_of_the_single_best_configuration(tmp_path):
    """Regression test: the scaler must come from the best (encoder, scaler) CONFIG of each model, not from the
    scaler with the best average over encoders (which would silently change which runs --confirm needs)."""
    rows = []
    # knn: config (onehot, standard) is the best single config (3.0), but 'robust' has the best average over encoders
    for enc, sc, m in (("onehot", "standard", 3.0), ("onehot", "robust", 4.0), ("target", "standard", 20.0), ("target", "robust", 5.0)):
        rows += [dict(experiment="main", model="knn", encoder=enc, scaler=sc, mae=m, fold=f) for f in range(2)]
    rows += [dict(experiment="main", model="rf", encoder="onehot", scaler=sc, mae=m, fold=f)
             for sc, m in (("none", 5.0), ("standard", 5.1)) for f in range(2)]
    rows += [dict(experiment="log_target", model="knn", encoder="onehot", scaler="none", mae=0.1, fold=0)]   # other experiments ignored
    p = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert experiments.best_scaler_per_model(p) == {"knn": "standard", "rf": "none"}
    cfgs = experiments.define_confirm_experiments({"knn": "standard", "ridge": "minmax", "rf": "none"})
    assert len(cfgs) == 12 and {c["model"] for c in cfgs} == {"knn", "ridge", "rf"}


def test_log_features_experiment_applies_log1p_to_skewed_inputs_only():
    df = make_frame()
    X = CAT_MAIN + NUM
    est = build_estimator("ordinal", "standard", "ridge", log_features=True).fit(df[X], df["price"])
    pre = fitted_preprocessor(est)
    assert [t[0] for t in pre.transformers] == ["num_log", "num", "cat"]
    sc = pre.named_transformers_["num_log"].named_steps["sc"]
    mileage_pos = pipelines.LOG_NUM.index("mileage_km")
    assert np.isclose(sc.mean_[mileage_pos], np.log1p(df["mileage_km"]).mean())       # the scaler saw log-values
    base = build_estimator("ordinal", "standard", "ridge").fit(df[X], df["price"])
    assert "num_log" not in [t[0] for t in fitted_preprocessor(base).transformers]
    assert np.isfinite(est.predict(df[X])).all()


def test_frequency_encoder_uses_real_column_names_inside_the_pipeline():
    df = make_frame()
    est = build_estimator("frequency", "none", "ridge").fit(df[CAT_MAIN + NUM], df["price"])
    names = list(fitted_preprocessor(est).get_feature_names_out())
    assert "brand_freq" in names and not any(n.startswith("x0") for n in names)


def test_report_claims_hold_on_the_real_tables():
    """Data-dependent: re-checks every prose claim of the week-2 report against reports/tables (skipped if absent)."""
    from src import reporting_week2
    if not (reporting_week2.T / "week2_summary_all.csv").exists():
        pytest.skip("week-2 tables not generated")
    claims = reporting_week2.verify_claims(reporting_week2.build_context())
    failed = [(i, t) for i, t, ok in claims if not ok]
    assert not failed, failed


# ---------------------------------------------------------------- sampling, folds, reproducibility
def test_dev_sample_keeps_whole_groups_and_is_reproducible():
    df = make_frame(300)
    full = df.groupby("group_id").size()
    s1, s2 = make_dev_sample(df, 100), make_dev_sample(df, 100)
    assert s1.equals(s2)
    assert (s1.groupby("group_id").size() == full.loc[s1["group_id"].unique()].values).all()
    assert 90 <= len(s1) <= 110


def test_folds_have_no_group_overlap():
    df = make_frame(300)
    for tr, va in GroupKFold(n_splits=5).split(df, df["price"], df["group_id"]):
        assert set(df.iloc[tr]["group_id"]).isdisjoint(set(df.iloc[va]["group_id"]))


def test_same_seed_same_metrics():
    df = make_frame(300)
    feats = CAT_WITH_VERSION + NUM
    X, y = df[feats], df["price"]
    tr, va = np.arange(200), np.arange(200, 300)
    cfg = dict(experiment="main", feature_set="main", encoder="target", scaler="standard", model="rf", fold=0)
    a, b = run_fold(cfg, X, y, tr, va), run_fold(cfg, X, y, tr, va)
    for k in ("rmse", "mae", "r2", "n_features_out"):
        assert a[k] == b[k]


def test_model_version_adds_columns_only_in_its_feature_set():
    df = make_frame(300)
    feats = CAT_WITH_VERSION + NUM
    X, y = df[feats], df["price"]
    tr, va = np.arange(200), np.arange(200, 300)
    base = dict(experiment="main", feature_set="main", encoder="onehot", scaler="standard", model="ridge", fold=0)
    big = dict(base, experiment="model_version", feature_set="with_model_version")
    assert run_fold(big, X, y, tr, va)["n_features_out"] >= run_fold(base, X, y, tr, va)["n_features_out"]
