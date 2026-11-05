"""Week-3 experiments on the TRAIN split only (same 20k development subsample and the same 5 GroupKFold folds as week 2).

Stages (each reads the results of the previous ones, all resumable):
  tune      hyper-parameters (+ log target yes/no) of each model, with the week-2 best encoder/scaler
  extremes  drop the most expensive TRAINING rows (threshold learned inside each training fold)
  grouping  add model_version, pooling rare categories (counts learned inside each training fold)
  combine   the winning options together, checked against the best 'tune' configuration

The hold-out test file is NOT read here: it is read once, in src/final_eval.py, after the configuration is frozen.
Usage:  python -m src.week3_experiments --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import config
from src.experiments import (CAT_MAIN, CAT_WITH_VERSION, N_DEV, N_SPLITS, NUM, check_signature, load_train, make_dev_sample,
                             regression_metrics, run_signature)
from src.pipelines import DEFAULT_PARAMS, MODELS, build_estimator, fitted_preprocessor
from src.transformers import drop_extreme_prices

RESULTS3 = config.TABLES / "week3_fold_results.csv"
SIG3 = config.TABLES / "week3_signature.json"
SEED = config.RANDOM_SEED
N_RANDOM = {"knn": 20, "svr": 20, "rf": 12}     # random draws per model (ridge uses its full alpha grid)
EXTREME_QS = [0.999, 0.99]
RARE_COUNTS = [None, 10, 50, 200]
EXPENSIVE_EUR = 100_000                         # used only to report the error on expensive cars separately
CFG_FIELDS = ["model", "encoder", "scaler", "log_target", "params", "extreme_q", "feature_set", "rare_min_count", "target_smooth"]

SPACES = {
    "knn": dict(n_neighbors=[3, 5, 7, 10, 15, 20, 30, 50], weights=["uniform", "distance"], p=[1, 2]),
    "svr": dict(C=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], epsilon=[0.02, 0.05, 0.1, 0.2], gamma=[0.012, 0.024, "scale", 0.1, 0.2]),
    "rf": dict(n_estimators=[100, 200], max_depth=[None, 15, 30], min_samples_leaf=[1, 2, 3, 5], max_features=[0.3, 0.5, 1.0]),
    "ridge": dict(alpha=[float(a) for a in np.logspace(-3, 3, 13)]),
}


# ------------------------------------------------------------------ configurations
def make_cfg(model, encoder, scaler, log_target=False, params=None, extreme_q=None, feature_set="main",
             rare_min_count=None, target_smooth="auto", stage="") -> dict:
    cfg = dict(stage=stage, model=model, encoder=encoder, scaler=scaler, log_target=bool(log_target),
               params=json.dumps(params or {}, sort_keys=True), extreme_q=extreme_q, feature_set=feature_set,
               rare_min_count=rare_min_count, target_smooth=target_smooth)
    cfg["config_id"] = config_id(cfg)
    return cfg


def config_id(cfg: dict) -> str:
    """Stable id of the settings that change the numbers (not of the stage label)."""
    core = {k: cfg[k] for k in CFG_FIELDS}
    for k in ("extreme_q", "rare_min_count"):
        core[k] = None if core[k] is None or (isinstance(core[k], float) and np.isnan(core[k])) else core[k]
    if core["rare_min_count"] is not None:
        core["rare_min_count"] = int(core["rare_min_count"])
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:12]


def week2_best_preprocessing() -> dict:
    """(encoder, scaler) of the best week-2 main-grid configuration of each model."""
    b = pd.read_csv(config.TABLES / "week2_best_per_model.csv").set_index("model")
    return {m: (b.loc[m, "encoder"], b.loc[m, "scaler"]) for m in MODELS}


def sample_params(model: str, n: int, seed: int = SEED) -> list[tuple[dict, bool]]:
    """Seeded sample (without replacement) of (params, log_target); the full grid when it is small."""
    space = SPACES[model]
    grid = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    combos = [(p, lt) for p in grid for lt in (False, True)]
    defaults = {(json.dumps({**DEFAULT_PARAMS[model]}, sort_keys=True, default=str), lt) for lt in (False, True)}
    combos = [c for c in combos if (json.dumps(c[0], sort_keys=True, default=str), c[1]) not in defaults]
    if model == "ridge" or len(combos) <= n:
        return combos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=n, replace=False)
    return [combos[i] for i in sorted(idx)]


SKIPPED = config.TABLES / "week3_skipped_configs.json"
# A configuration whose fits are so slow that it can never be refit on the full train split is not a usable candidate.
FULL_FIT_BUDGET_S = 1800.0
# time ~ n^exponent: SVR (libsvm) is about quadratic in the number of rows, Random Forest about n log n, KNN/Ridge about linear
SCALING_EXPONENT = {"svr": 2.0, "rf": 1.2, "knn": 1.0, "ridge": 1.0}


def skipped_ids() -> dict:
    """Configurations cancelled because they were infeasibly slow (id -> reason). Recorded, never silently dropped."""
    return json.loads(SKIPPED.read_text(encoding="utf-8")) if SKIPPED.exists() else {}


def n_full_train() -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(config.DATA_PROCESSED / "train.parquet").metadata.num_rows


def estimated_full_fit_seconds(model: str, fit_seconds: float, n_train_used: float, n_full: int | None = None) -> float:
    """Fit time measured inside CV (on ~16k rows) extrapolated to the full train split. A rough planning estimate."""
    n_full = n_full or n_full_train()
    return float(fit_seconds * (n_full / n_train_used) ** SCALING_EXPONENT[model])


def tune_configs() -> list[dict]:
    pre = week2_best_preprocessing()
    out = []
    for m in MODELS:
        enc, sc = pre[m]
        for lt in (False, True):                                    # the week-2 defaults are always part of the comparison
            out.append(make_cfg(m, enc, sc, lt, {}, stage="tune"))
        for params, lt in sample_params(m, N_RANDOM.get(m, 0)):
            out.append(make_cfg(m, enc, sc, lt, params, stage="tune"))
    skip = skipped_ids()
    return [c for c in out if c["config_id"] not in skip]


# ------------------------------------------------------------------ one fold
def run_fold3(cfg: dict, X: pd.DataFrame, y: pd.Series, tr_idx, va_idx) -> dict:
    cat_cols = CAT_WITH_VERSION if cfg["feature_set"] == "with_model_version" else CAT_MAIN
    est = build_estimator(cfg["encoder"], cfg["scaler"], cfg["model"], log_target=cfg["log_target"], cat_cols=cat_cols,
                          model_params=json.loads(cfg["params"]), rare_min_count=cfg["rare_min_count"],
                          target_smooth=cfg["target_smooth"])
    ytr_all = y.iloc[tr_idx]
    keep = drop_extreme_prices(ytr_all, cfg["extreme_q"])           # threshold from the training fold only
    tr_used = tr_idx[keep]
    Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
    t0 = time.perf_counter()
    est.fit(X.iloc[tr_used], y.iloc[tr_used])
    fit_s = time.perf_counter() - t0
    pred = est.predict(Xva)
    out = dict(cfg)
    out.update(regression_metrics(yva, pred))
    exp = yva.to_numpy() >= EXPENSIVE_EUR
    out.update(mae_ge_100k=float(np.abs(yva.to_numpy()[exp] - pred[exp]).mean()) if exp.any() else np.nan,
               mae_lt_100k=float(np.abs(yva.to_numpy()[~exp] - pred[~exp]).mean()),
               bias_ge_100k=float((pred[exp] - yva.to_numpy()[exp]).mean()) if exp.any() else np.nan,
               n_valid_ge_100k=int(exp.sum()), fit_seconds=fit_s, n_train_used=int(len(tr_used)), n_train=int(len(tr_idx)),
               n_features_out=int(fitted_preprocessor(est).transform(Xva.head(1)).shape[1]), fold=None)
    return out


# ------------------------------------------------------------------ runner
def week3_signature(n_rows: int) -> dict:
    sig = run_signature("week3_dev", n_rows, N_SPLITS)
    sig.update(week3_spaces=json.loads(json.dumps(SPACES, default=str)), week3_n_random=N_RANDOM, extreme_qs=EXTREME_QS,
               rare_counts=RARE_COUNTS, expensive_eur=EXPENSIVE_EUR)
    return sig


def run_configs(cfgs: list[dict], dev: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    check_signature(SIG3, week3_signature(len(dev)), RESULTS3)
    X, y, g = dev[CAT_WITH_VERSION + NUM], dev["price"], dev["group_id"]
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, g))
    done = set()
    if RESULTS3.exists():
        prev = pd.read_csv(RESULTS3)
        done = set(zip(prev["config_id"], prev["fold"]))
    seen, tasks = set(), []
    for cfg in cfgs:
        if cfg["config_id"] in seen:
            continue
        seen.add(cfg["config_id"])
        for k, (tr, va) in enumerate(folds):
            if (cfg["config_id"], k) not in done:
                tasks.append((dict(cfg, fold=k), k, tr, va))
    print(f"{len(tasks)} fold-tasks to run ({len(done)} already done)", flush=True)
    if tasks:
        t0 = time.time()
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_task)(c, X, y, tr, va) for c, _, tr, va in tasks)
        buf: list[dict] = []

        def flush():
            if buf:
                pd.DataFrame(buf).to_csv(RESULTS3, mode="a", header=not RESULTS3.exists(), index=False)
                buf.clear()
        try:
            for i, r in enumerate(gen, 1):
                buf.append(r)
                if len(buf) >= 8 or i == len(tasks):
                    flush()
                    print(f"  {i}/{len(tasks)} tasks | {time.time() - t0:,.0f}s", flush=True)
        finally:
            flush()
    return pd.read_csv(RESULTS3) if RESULTS3.exists() else pd.DataFrame()


def _task(cfg, X, y, tr, va):
    fold = cfg["fold"]
    r = run_fold3(cfg, X, y, tr, va)
    r["fold"] = fold
    return r


# ------------------------------------------------------------------ aggregation and decisions
def aggregate3(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("config_id")
    first = g[["stage", *CFG_FIELDS]].first()
    out = first.join(g[["mae", "rmse", "r2", "medape_pct", "mae_ge_100k", "mae_lt_100k", "bias_ge_100k"]].mean().add_suffix("_mean"))
    out["mae_std"] = g["mae"].std()
    out["fit_seconds_mean"] = g["fit_seconds"].mean()
    out["n_features_out"] = g["n_features_out"].mean().round().astype(int)
    out["n_folds"] = g["fold"].nunique()
    n_full = n_full_train()
    used = g["n_train_used"].mean()
    out["est_full_fit_s"] = [estimated_full_fit_seconds(m, s, u, n_full) for m, s, u in zip(out["model"], out["fit_seconds_mean"], used)]
    return out.reset_index()


def fold_mae(res: pd.DataFrame, config_id_: str) -> pd.Series:
    return res[res["config_id"] == config_id_].set_index("fold")["mae"].sort_index()


def adopt_if_consistent(res: pd.DataFrame, baseline_id: str, candidate_ids: list[str], min_wins: int = 4) -> tuple[str, dict]:
    """Decision rule used for every optional ingredient: a candidate replaces the (simpler) baseline only if its mean MAE
    is lower AND it beats the baseline in at least `min_wins` of the 5 folds. Among qualifying candidates the lowest mean
    MAE wins. Returns (chosen_id, info)."""
    base = fold_mae(res, baseline_id)
    best, info = baseline_id, {"baseline_mae": float(base.mean()), "candidates": {}}
    best_mae = base.mean()
    for cid in candidate_ids:
        c = fold_mae(res, cid)
        if len(c) < len(base) or len(base) == 0:
            continue
        wins = int((c.values < base.values).sum())
        info["candidates"][cid] = {"mae": float(c.mean()), "wins": wins}
        if c.mean() < best_mae and wins >= min_wins:
            best, best_mae = cid, c.mean()
    info["chosen"], info["chosen_mae"] = best, float(best_mae)
    return best, info


def best_tune_config(agg: pd.DataFrame, model: str) -> pd.Series:
    """Lowest mean CV MAE among the 'tune' configurations that are complete (5 folds) and computationally feasible
    (estimated full-train fit time within FULL_FIT_BUDGET_S). If none is feasible, the fastest one."""
    t = agg[(agg["model"] == model) & (agg["stage"] == "tune") & (agg["n_folds"] == N_SPLITS)]
    ok = t[t["est_full_fit_s"] <= FULL_FIT_BUDGET_S]
    return ok.loc[ok["mae_mean"].idxmin()] if len(ok) else t.loc[t["est_full_fit_s"].idxmin()]


def cfg_from_row(r: pd.Series, **over) -> dict:
    d = dict(model=r["model"], encoder=r["encoder"], scaler=r["scaler"], log_target=bool(r["log_target"]),
             params=json.loads(r["params"]), extreme_q=None if pd.isna(r["extreme_q"]) else float(r["extreme_q"]),
             feature_set=r["feature_set"], rare_min_count=None if pd.isna(r["rare_min_count"]) else int(r["rare_min_count"]),
             target_smooth=r["target_smooth"])
    d.update(over)
    return make_cfg(**d)


def extremes_configs(agg: pd.DataFrame) -> list[dict]:
    return [cfg_from_row(best_tune_config(agg, m), extreme_q=q, stage="extremes") for m in MODELS for q in EXTREME_QS]


def grouping_configs(agg: pd.DataFrame) -> list[dict]:
    out = []
    pre = week2_best_preprocessing()
    for m in MODELS:
        base = best_tune_config(agg, m)
        for enc in ("onehot", "target"):
            sc = pre[m][1]
            for rc in RARE_COUNTS:
                out.append(cfg_from_row(base, encoder=enc, scaler=sc, feature_set="with_model_version", rare_min_count=rc,
                                        stage="grouping"))
    return out


def decide(res: pd.DataFrame) -> dict:
    """Per model: best tune config -> adopt extremes option -> adopt grouping option (each with adopt_if_consistent)."""
    agg = aggregate3(res)
    out = {}
    for m in MODELS:
        base = best_tune_config(agg, m)
        ex_ids = [cfg_from_row(base, extreme_q=q)["config_id"] for q in EXTREME_QS]
        ex_ids = [i for i in ex_ids if i in set(agg["config_id"])]
        ex_choice, ex_info = adopt_if_consistent(res, base["config_id"], ex_ids)
        gr_ids = []
        for enc in ("onehot", "target"):
            for rc in RARE_COUNTS:
                cid = cfg_from_row(base, encoder=enc, scaler=week2_best_preprocessing()[m][1], feature_set="with_model_version",
                                   rare_min_count=rc)["config_id"]
                if cid in set(agg["config_id"]):
                    gr_ids.append(cid)
        gr_choice, gr_info = adopt_if_consistent(res, base["config_id"], gr_ids)
        out[m] = dict(tune_best=base["config_id"], extremes=ex_info, extremes_choice=ex_choice, grouping=gr_info, grouping_choice=gr_choice)
    return out


def combine_configs(res: pd.DataFrame) -> list[dict]:
    agg = aggregate3(res).set_index("config_id")
    d = decide(res)
    cfgs = []
    for m in MODELS:
        base = agg.loc[d[m]["tune_best"]].copy()
        base["config_id"] = d[m]["tune_best"]
        over = {}
        if d[m]["extremes_choice"] != d[m]["tune_best"]:
            over["extreme_q"] = agg.loc[d[m]["extremes_choice"], "extreme_q"]
        if d[m]["grouping_choice"] != d[m]["tune_best"]:
            g = agg.loc[d[m]["grouping_choice"]]
            over.update(encoder=g["encoder"], scaler=g["scaler"], feature_set=g["feature_set"], rare_min_count=g["rare_min_count"])
        if over:
            over = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in over.items()}
            if over.get("extreme_q") is not None:
                over["extreme_q"] = float(over["extreme_q"])
            if over.get("rare_min_count") is not None:
                over["rare_min_count"] = int(over["rare_min_count"])
            cfgs.append(cfg_from_row(base, stage="combine", **over))
    return cfgs


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "extremes", "grouping", "combine", "all"], default="all")
    args = ap.parse_args()
    dev = make_dev_sample(load_train(), N_DEV)
    print("development subsample:", dev.shape, "groups:", dev["group_id"].nunique(), flush=True)
    stages = ["tune", "extremes", "grouping", "combine"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"== stage {st}", flush=True)
        if st == "tune":
            cfgs = tune_configs()
        else:
            res = pd.read_csv(RESULTS3)
            agg = aggregate3(res)
            cfgs = {"extremes": lambda: extremes_configs(agg), "grouping": lambda: grouping_configs(agg),
                    "combine": lambda: combine_configs(res)}[st]()
        print(f"{len(cfgs)} configurations", flush=True)
        run_configs(cfgs, dev)
    meta = {"n_dev_rows": int(len(dev)), "n_splits": N_SPLITS, "finished_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (config.REPORTS / "week3_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
