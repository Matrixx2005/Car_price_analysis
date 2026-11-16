"""Week-3 supplement (TRAIN split only; the test split is never read here).

Three parts, all resumable:

  hgb          gradient boosting, tuned by GroupKFold(5) CV on the FULL train split (it is fast, so no 20k subsample).
               Steps: encoder x log-target on defaults -> random search of 60 hyper-parameter sets -> optional
               model_version (target encoding, rare-category pooling), accepted only by the 4-of-5-folds rule.
  diagnostics  for the frozen final models (+ RF default, + the boosting winner), on the 20k development folds used in
               week 3: train-vs-validation metrics per fold (over/under-fitting) and permutation importance of the
               ORIGINAL input columns on the validation folds (single columns and joint groups).
  curves       learning curves on the full-train folds: RF tuned vs RF default (tests the explanation of why the tuned
               forest lost on the test split), boosting, KNN and Ridge.

Important: the boosting model is NOT evaluated on the test split. That split was evaluated exactly once (src/final_eval.py)
and must not be looked at again to compare a model that was added afterwards.

Usage:  python -m src.week3_extra --part all        (hgb, diagnostics, curves)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import config
from src.experiments import check_signature, load_train, make_dev_sample, regression_metrics, run_signature
from src.pipelines import CAT_WITH_VERSION, MODELS, NUM, make_model
from src.transformers import drop_extreme_prices
from src.week3_experiments import (N_SPLITS, adopt_if_consistent, aggregate3, cfg_from_entry, estimator_from_cfg, make_cfg,
                                   run_configs)
from src.week3_select import _entry

T = config.TABLES
HGB_RESULTS, HGB_SIG = T / "week3_hgb_fold_results.csv", T / "week3_hgb_signature.json"
HGB_BEST = config.REPORTS / "week3_hgb_best.json"
DIAG_METRICS, DIAG_PERM, DIAG_SIG = T / "week3_diag_metrics.csv", T / "week3_diag_permutation.csv", T / "week3_diag_signature.json"
CURVES, CURVES_SIG = T / "week3_learning_curves.csv", T / "week3_curves_signature.json"
SEED = config.RANDOM_SEED
N_HGB = 60
N_REPEATS = 3
FRACTIONS = [0.15, 0.3, 0.5, 1.0]
RF_CURVE_FOLDS = (0, 1, 2)      # Random Forest fits are slow: its curve uses 3 of the 5 folds at fractions below 100%
RF_DEFAULT_FULL_FROM_WEEK2 = True   # rf_default at 100% = the week-2 full-train confirmation (same config and folds)
HGB_SPACE = dict(learning_rate=[0.02, 0.05, 0.1, 0.2], max_iter=[100, 200, 400, 800, 1200], max_leaf_nodes=[15, 31, 63, 127],
                 min_samples_leaf=[5, 10, 20, 50, 100], l2_regularization=[0.0, 0.1, 1.0, 10.0], max_depth=[None, 8, 12])
HGB_ENCODERS = ["ordinal", "target", "onehot"]
HGB_RARE = [None, 10, 50]

# joint permutation groups (original columns that carry the same information are shuffled together)
JOINT_GROUPS = {"identity (brand+model[+version])": ["brand", "model", "model_version"], "age (year+vehicle_age)": ["production_year", "vehicle_age"]}


# ------------------------------------------------------------------ part 1: boosting on the full train split
def hgb_signature(n_rows: int) -> dict:
    sig = run_signature("week3_hgb_full", n_rows, N_SPLITS)
    sig.update(hgb_space=json.loads(json.dumps(HGB_SPACE, default=str)), n_hgb=N_HGB, hgb_default=repr(make_model("hgb")),
               hgb_encoders=HGB_ENCODERS, hgb_rare=HGB_RARE)
    return sig


def sample_hgb_params(n: int = N_HGB, seed: int = SEED) -> list[dict]:
    grid = [dict(zip(HGB_SPACE, v)) for v in itertools.product(*HGB_SPACE.values())]
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), size=n, replace=False))]


def hgb_encoder_configs() -> list[dict]:
    return [make_cfg("hgb", enc, "none", lt, {}, stage="hgb_encoder") for enc in HGB_ENCODERS for lt in (False, True)]


def hgb_search_configs(enc: str, log_target: bool) -> list[dict]:
    return [make_cfg("hgb", enc, "none", log_target, p, stage="hgb_tune") for p in sample_hgb_params()]


def _best_row(agg: pd.DataFrame, stages: list[str], enc: str | None = None, lt: bool | None = None) -> pd.Series:
    t = agg[agg["stage"].isin(stages) & (agg["n_folds"] == N_SPLITS)]
    if enc is not None:
        t = t[(t["encoder"] == enc) & (t["log_target"].astype(bool) == lt)]
    return t.loc[t["mae_mean"].idxmin()]


def run_hgb(n_jobs: int = -1) -> dict:
    train = load_train()
    kw = dict(results_path=HGB_RESULTS, sig_path=HGB_SIG, signature=hgb_signature(len(train)), n_jobs=n_jobs)
    res = run_configs(hgb_encoder_configs(), train, **kw)
    agg = aggregate3(res)
    base = _best_row(agg, ["hgb_encoder"])
    enc, lt = base["encoder"], bool(base["log_target"])
    print(f"best encoder/target on defaults: {enc}, log_target={lt}, MAE {base['mae_mean']:,.0f}", flush=True)
    res = run_configs(hgb_search_configs(enc, lt), train, **kw)
    agg = aggregate3(res)
    best = _best_row(agg, ["hgb_encoder", "hgb_tune"], enc, lt)
    best_cfg = cfg_from_entry(_entry(best))
    ver = [cfg_from_row_over(best, feature_set="with_model_version", rare_min_count=rc, stage="hgb_version") for rc in HGB_RARE]
    res = run_configs(ver, train, **kw)
    agg = aggregate3(res)
    chosen_id, info = adopt_if_consistent(res, best["config_id"], [c["config_id"] for c in ver])
    final = agg[agg["config_id"] == chosen_id].iloc[0]
    default = agg[(agg["stage"] == "hgb_encoder") & (agg["params"] == "{}") & (agg["encoder"] == enc) & (agg["log_target"].astype(bool) == lt)].iloc[0]
    out = {"created_at": datetime.now().isoformat(timespec="seconds"), "final": _entry(final), "default_same_encoder": _entry(default),
           "encoder_stage": agg[agg["stage"] == "hgb_encoder"][["encoder", "log_target", "mae_mean", "r2_mean", "medape_pct_mean"]].to_dict("records"),
           "n_tuned_configs": int((agg["stage"] == "hgb_tune").sum()), "version_decision": info, "tune_best_id": best_cfg["config_id"],
           "adopted_model_version": bool(chosen_id != best["config_id"])}
    HGB_BEST.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def cfg_from_row_over(row: pd.Series, **over) -> dict:
    e = _entry(row)
    e.update(over)
    stage = e.pop("stage", "")
    return cfg_from_entry(e, stage=stage)


# ------------------------------------------------------------------ shared helpers
def model_entries(include_hgb: bool = True) -> dict[str, dict]:
    """Name -> make_cfg configuration of the models analysed in parts 2 and 3 (read from frozen files; no test access)."""
    frozen = json.loads((config.REPORTS / "week3_frozen_config.json").read_text(encoding="utf-8"))
    out = {f"{m}_tuned": cfg_from_entry(frozen["models"][m]["final"]) for m in MODELS}
    out["rf_default"] = cfg_from_entry(frozen["models"]["rf"]["default"])
    out["knn_default"] = cfg_from_entry(frozen["models"]["knn"]["default"])
    if include_hgb and HGB_BEST.exists():
        out["hgb_best"] = cfg_from_entry(json.loads(HGB_BEST.read_text(encoding="utf-8"))["final"])
    return out


def _append(rows: list[dict] | pd.DataFrame, path) -> None:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if len(df):
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def subsample_groups(tr_idx: np.ndarray, groups: pd.Series, frac: float, seed: int) -> np.ndarray:
    """Random subset of the training rows made of WHOLE groups (cars), so no car is split by the subsampling."""
    if frac >= 1:
        return tr_idx
    g = groups.iloc[tr_idx]
    ug = g.unique()
    keep = set(np.random.default_rng(seed).choice(ug, size=max(1, int(round(len(ug) * frac))), replace=False))
    return tr_idx[g.isin(keep).to_numpy()]


def fit_cfg(cfg: dict, X: pd.DataFrame, y: pd.Series, tr_idx):
    """Fit a configuration on the given training rows (extreme-price threshold taken from those rows only)."""
    keep = drop_extreme_prices(y.iloc[tr_idx], cfg["extreme_q"] if not pd.isna(cfg["extreme_q"]) else None)
    used = tr_idx[keep]
    est = estimator_from_cfg(cfg)
    t0 = time.perf_counter()
    est.fit(X.iloc[used], y.iloc[used])
    return est, used, time.perf_counter() - t0


# ------------------------------------------------------------------ part 2: train-vs-validation and permutation importance
def permutation_rows(est, Xva: pd.DataFrame, yva: pd.Series, base_mae: float, seed: int, n_repeats: int = N_REPEATS) -> list[dict]:
    """Increase of validation MAE when a column (or a joint group of columns) is shuffled across validation rows."""
    rng = np.random.default_rng(seed)
    groups = {c: [c] for c in Xva.columns}
    groups.update({k: [c for c in v if c in Xva.columns] for k, v in JOINT_GROUPS.items()})
    rows = []
    for name, cols in groups.items():
        for rep in range(n_repeats):
            Xp = Xva.copy()
            Xp[cols] = Xva[cols].iloc[rng.permutation(len(Xva))].to_numpy()
            mae = float(np.abs(yva.to_numpy() - est.predict(Xp)).mean())
            rows.append({"feature": name, "kind": "joint" if name in JOINT_GROUPS else "single", "repeat": rep, "delta_mae": mae - base_mae})
    return rows


def diag_task(name, cfg, fold, X, y, tr, va, data):
    est, used, fit_s = fit_cfg(cfg, X, y, tr)
    pred_tr, pred_va = est.predict(X.iloc[used]), est.predict(X.iloc[va])
    mt, mv = regression_metrics(y.iloc[used], pred_tr), regression_metrics(y.iloc[va], pred_va)
    row = {"name": name, "data": data, "fold": fold, "n_train_used": len(used), "n_valid": len(va), "fit_seconds": fit_s,
           **{f"{k}_train": v for k, v in mt.items()}, **{f"{k}_valid": v for k, v in mv.items()}}
    perm = [dict(name=name, data=data, fold=fold, **r) for r in permutation_rows(est, X.iloc[va], y.iloc[va], mv["mae"], SEED + fold)]
    return row, perm


def diag_signature(n_rows: int) -> dict:
    ents = model_entries()
    return {**run_signature("week3_diag", n_rows, N_SPLITS), "configs": {k: v["config_id"] for k, v in ents.items()}, "n_repeats": N_REPEATS}


def run_diagnostics(n_jobs: int = -1) -> pd.DataFrame:
    train = load_train()
    dev = make_dev_sample(train)
    ents = model_entries()
    check_signature(DIAG_SIG, diag_signature(len(dev)), DIAG_METRICS)
    done = set()
    if DIAG_METRICS.exists():
        prev = pd.read_csv(DIAG_METRICS)
        done = set(zip(prev["name"], prev["data"], prev["fold"]))
    jobs = []
    for data, df in (("dev20k", dev), ("full", train)):
        X, y, g = df[CAT_WITH_VERSION + NUM], df["price"], df["group_id"]
        folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, g))
        names = [n for n in ents if (data == "dev20k" or n == "hgb_best")]       # on the full folds only the fast boosting model
        for n in names:
            for k, (tr, va) in enumerate(folds):
                if (n, data, k) not in done:
                    jobs.append((n, ents[n], k, X, y, tr, va, data))
    print(f"{len(jobs)} diagnostic tasks ({len(done)} already done)", flush=True)
    if jobs:
        t0 = time.time()
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(diag_task)(*j) for j in jobs)
        for i, (row, perm) in enumerate(gen, 1):
            _append([row], DIAG_METRICS)
            _append(perm, DIAG_PERM)
            print(f"  {i}/{len(jobs)} | {time.time() - t0:,.0f}s", flush=True)
    return pd.read_csv(DIAG_METRICS)


# ------------------------------------------------------------------ part 3: learning curves on the full-train folds
CURVE_MODELS = ["rf_default", "rf_tuned", "hgb_best", "knn_default", "knn_tuned", "ridge_tuned"]


def curve_task(name, cfg, fold, frac, X, y, groups, tr, va):
    sub = subsample_groups(tr, groups, frac, SEED + fold)
    est, used, fit_s = fit_cfg(cfg, X, y, sub)
    mt, mv = regression_metrics(y.iloc[used], est.predict(X.iloc[used])), regression_metrics(y.iloc[va], est.predict(X.iloc[va]))
    return {"name": name, "fold": fold, "fraction": frac, "n_train_used": len(used), "fit_seconds": fit_s,
            "mae_train": mt["mae"], "mae_valid": mv["mae"], "medape_train": mt["medape_pct"], "medape_valid": mv["medape_pct"]}


def curves_signature(n_rows: int) -> dict:
    ents = model_entries()
    return {**run_signature("week3_curves", n_rows, N_SPLITS), "configs": {k: ents[k]["config_id"] for k in CURVE_MODELS}, "fractions": FRACTIONS,
            "rf_folds": list(RF_CURVE_FOLDS), "rf_default_full_from_week2": RF_DEFAULT_FULL_FROM_WEEK2}


def week2_rf_default_full_rows() -> pd.DataFrame:
    """rf_default at 100% of the data: the week-2 full-train CV of Random Forest + One-hot (same model, same folds).
    Only validation metrics exist there (no train metrics)."""
    conf = pd.read_csv(T / "week2_confirm_fold_results.csv")
    c = conf[(conf["model"] == "rf") & (conf["encoder"] == "onehot")].sort_values("fold")
    return pd.DataFrame({"name": "rf_default", "fold": c["fold"].to_numpy(), "fraction": 1.0, "n_train_used": c["n_train"].to_numpy(),
                         "fit_seconds": c["fit_seconds"].to_numpy(), "mae_train": np.nan, "mae_valid": c["mae"].to_numpy(),
                         "medape_train": np.nan, "medape_valid": c["medape_pct"].to_numpy()})


def run_curves(n_jobs: int = -1) -> pd.DataFrame:
    train = load_train()
    ents = model_entries()
    check_signature(CURVES_SIG, curves_signature(len(train)), CURVES)
    X, y, g = train[CAT_WITH_VERSION + NUM], train["price"], train["group_id"]
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, g))
    done = set()
    if CURVES.exists():
        prev = pd.read_csv(CURVES)
        done = set(zip(prev["name"], prev["fold"], prev["fraction"].round(4)))
    def wanted(n, k, f):
        if n == "rf_default" and f >= 1.0 and RF_DEFAULT_FULL_FROM_WEEK2:
            return False                                      # taken from the week-2 confirmation (see below)
        return not (n.startswith("rf") and f < 1.0 and k not in RF_CURVE_FOLDS)

    jobs = [(n, ents[n], k, f, X, y, g, tr, va) for n in CURVE_MODELS for k, (tr, va) in enumerate(folds) for f in FRACTIONS
            if wanted(n, k, f) and (n, k, round(f, 4)) not in done]
    # slow jobs first so that the longest ones start early
    jobs.sort(key=lambda j: -(j[3] * (3 if j[0].startswith("rf") else 1)))
    if RF_DEFAULT_FULL_FROM_WEEK2 and not (CURVES.exists() and ((pd.read_csv(CURVES)["name"] == "rf_default") & (pd.read_csv(CURVES)["fraction"] >= 1.0)).any()):
        _append(week2_rf_default_full_rows(), CURVES)
    print(f"{len(jobs)} learning-curve tasks ({len(done)} already done)", flush=True)
    if jobs:
        t0 = time.time()
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(curve_task)(*j) for j in jobs)
        for i, row in enumerate(gen, 1):
            _append([row], CURVES)
            print(f"  {i}/{len(jobs)} | {time.time() - t0:,.0f}s", flush=True)
    return pd.read_csv(CURVES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["hgb", "diagnostics", "curves", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=-1, help="number of parallel workers (default: all logical cores)")
    args = ap.parse_args()
    # One thread per worker: boosting (OpenMP) x many parallel workers otherwise oversubscribes the CPU and the
    # whole run slows down by an order of magnitude (observed: ~10% CPU use). Workers inherit this environment.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if args.part in ("hgb", "all"):
        best = run_hgb(args.jobs)
        print("boosting winner:", best["final"]["encoder"], best["final"]["params"], f"CV MAE {best['final']['cv']['mae_mean']:,.0f}")
    if args.part in ("diagnostics", "all"):
        run_diagnostics(args.jobs)
    if args.part in ("curves", "all"):
        run_curves(args.jobs)


if __name__ == "__main__":
    main()
