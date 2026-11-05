"""Week-2 experiment runner: Encoding x Scaling x Model, cross-validated on the TRAIN split only.

The hold-out test file is never read here (it is reserved for the final evaluation in week 3).
Usage:  python -m src.experiments            (runs everything, resumable)
        python -m src.experiments --confirm  (full-train confirmation, resumable)
        python -m src.experiments --quick    (tiny smoke run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import config
from src.pipelines import (CAT_MAIN, CAT_WITH_VERSION, ENCODERS, LOG_NUM, MODELS, NUM, ONEHOT_MIN_FREQUENCY, SCALERS,
                           build_estimator, fitted_preprocessor, make_model)

RESULTS = config.TABLES / "week2_fold_results.csv"
CONFIRM_RESULTS = config.TABLES / "week2_confirm_fold_results.csv"
N_DEV = 20_000          # rows of the development subsample used for the full grid
N_SPLITS = 5
KEY = ["experiment", "feature_set", "encoder", "scaler", "model", "fold"]
N_CONFIRM_CONFIGS = 12  # 3 models x 4 encoders


# ------------------------------------------------------------------ data
def load_train() -> pd.DataFrame:
    """Reads ONLY train.parquet."""
    return pd.read_parquet(config.DATA_PROCESSED / "train.parquet")


def make_dev_sample(train: pd.DataFrame, n_rows: int = N_DEV, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Random subsample made of WHOLE groups (cars), so no car is split by the subsampling."""
    rng = np.random.default_rng(seed)
    sizes = train.groupby("group_id").size()
    order = rng.permutation(sizes.index.to_numpy())
    chosen = sizes.loc[order].cumsum()
    keep = chosen.index[: int((chosen <= n_rows).sum()) + 1]
    return train[train["group_id"].isin(keep)].reset_index(drop=True)


# ------------------------------------------------------------------ metrics
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res, ss_tot = float((err ** 2).sum()), float(((y_true - y_true.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "medape_pct": float(np.median(np.abs(err) / y_true) * 100)}


# ------------------------------------------------------------------ one fold
def run_fold(cfg: dict, X: pd.DataFrame, y: pd.Series, tr_idx, va_idx) -> dict:
    cat_cols = CAT_WITH_VERSION if cfg["feature_set"] == "with_model_version" else CAT_MAIN
    est = build_estimator(cfg["encoder"], cfg["scaler"], cfg["model"], log_target=cfg["experiment"] == "log_target",
                          cat_cols=cat_cols, num_cols=NUM, log_features=cfg["experiment"] == "log_features")
    Xtr, ytr, Xva, yva = X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx]
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = est.predict(Xva)
    pred_s = time.perf_counter() - t0
    out = dict(cfg)
    out.update(regression_metrics(yva, pred))
    out.update(fit_seconds=fit_s, predict_seconds=pred_s,
               n_features_out=int(fitted_preprocessor(est).transform(Xva.head(1)).shape[1]),
               n_train=len(tr_idx), n_valid=len(va_idx))
    return out


# ------------------------------------------------------------------ experiment definitions
def define_experiments() -> list[dict]:
    cfgs = []
    for enc in ENCODERS:                              # main grid: 4 x 4 x 4 = 64 configurations
        for sc in SCALERS:
            for m in MODELS:
                cfgs.append(dict(experiment="main", feature_set="main", encoder=enc, scaler=sc, model=m))
    for enc in ENCODERS:                              # log transformation of the TARGET (separate experiment)
        for m in MODELS:
            cfgs.append(dict(experiment="log_target", feature_set="main", encoder=enc, scaler="standard", model=m))
    for enc in ENCODERS:                              # log transformation of skewed INPUT features (separate experiment)
        for m in MODELS:
            cfgs.append(dict(experiment="log_features", feature_set="main", encoder=enc, scaler="standard", model=m))
    for enc in ENCODERS:                              # cardinality stress test: add model_version
        for m in MODELS:
            cfgs.append(dict(experiment="model_version", feature_set="with_model_version", encoder=enc,
                             scaler="standard", model=m))
    return cfgs


def best_scaler_per_model(results_path: Path = RESULTS) -> dict:
    """Scaler of the single best (encoder, scaler) configuration of each model in the main grid (lowest mean MAE
    over the folds), computed straight from the fold results (no dependency on the analysis step).
    NOT the scaler with the best average over encoders: the report and the confirmation runs use this definition."""
    res = pd.read_csv(results_path)
    cfg = res[res["experiment"] == "main"].groupby(["model", "encoder", "scaler"])["mae"].mean().reset_index()
    return cfg.loc[cfg.groupby("model")["mae"].idxmin()].set_index("model")["scaler"].to_dict()


def define_confirm_experiments(best_scaler: dict) -> list[dict]:
    """Re-run the 4 encoders on the FULL train split for the models that scale to 75k rows (not SVR),
    each with the scaler that was best for that model on the development subsample."""
    return [dict(experiment="full_train_confirm", feature_set="main", encoder=enc, scaler=best_scaler[m], model=m)
            for m in ("knn", "ridge", "rf") for enc in ENCODERS]


# ------------------------------------------------------------------ run signature (guards resuming)
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_signature(scope: str, n_rows: int, n_splits: int) -> dict:
    """Everything that changes the numbers: data, folds, seed, model settings, library version."""
    return {"scope": scope, "n_rows": int(n_rows), "n_splits": int(n_splits), "seed": config.RANDOM_SEED,
            "train_md5": _md5(config.DATA_PROCESSED / "train.parquet"), "sklearn": sklearn.__version__,
            "onehot_min_frequency": ONEHOT_MIN_FREQUENCY, "num": NUM, "cat_main": CAT_MAIN, "cat_with_version": CAT_WITH_VERSION,
            "log_numeric": LOG_NUM, "models": {m: repr(make_model(m)) for m in MODELS}}


def check_signature(sig_path: Path, signature: dict, results_path: Path) -> None:
    """Refuse to append to results produced under different settings (they would be silently mixed).
    Old results without a signature file are adopted (with a warning) because nothing else can be checked."""
    if sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))
        diff = [k for k in signature if old.get(k) != signature[k]]
        if diff and results_path.exists():
            raise RuntimeError(
                f"Existing results {results_path.name} were produced with different settings ({', '.join(diff)}). "
                f"Delete {results_path.name} and {sig_path.name} to start from scratch.")
    elif results_path.exists():
        print(f"[WARN] {results_path.name} has no signature file; adopting the current settings as its signature.", flush=True)
    sig_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ runner
def run_experiments(cfgs: list[dict], df: pd.DataFrame, n_jobs: int = -1, results_path=RESULTS,
                    n_splits: int = N_SPLITS, sig_path: Path | None = None, scope: str = "dev") -> pd.DataFrame:
    """Run all (config, fold) tasks in parallel; append finished tasks to a CSV so a crash/stop can resume.
    Results already on disk are reused only if the run signature matches (see check_signature)."""
    if sig_path is not None:
        check_signature(sig_path, run_signature(scope, len(df), n_splits), results_path)
    feats = CAT_WITH_VERSION + NUM
    X, y, groups = df[feats], df["price"], df["group_id"]
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = {tuple(r) for r in prev[KEY].itertuples(index=False, name=None)}
    tasks = []
    for cfg in cfgs:
        for k, (tr, va) in enumerate(folds):
            task_cfg = dict(cfg, fold=k)
            if tuple(task_cfg[c] for c in KEY) not in done:
                tasks.append((task_cfg, tr, va))
    print(f"{len(tasks)} fold-tasks to run ({len(done)} already done)", flush=True)
    if not tasks:
        return pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    t0 = time.time()
    gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(run_fold)(c, X, y, tr, va) for c, tr, va in tasks)
    buf: list[dict] = []

    def flush():
        if buf:
            pd.DataFrame(buf).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
            buf.clear()

    try:
        for i, res in enumerate(gen, 1):
            buf.append(res)
            if len(buf) >= 8 or i == len(tasks):
                flush()
                print(f"  {i}/{len(tasks)} tasks | {time.time() - t0:,.0f}s", flush=True)
    finally:
        flush()     # an error in one worker must not lose results that already finished
    return pd.read_csv(results_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run on 2,000 rows, 2 folds, one config per model")
    ap.add_argument("--confirm", action="store_true", help="re-run encoders on the full train split (needs main results)")
    args = ap.parse_args()
    train = load_train()
    if args.confirm:
        if not RESULTS.exists():
            raise SystemExit(f"{RESULTS.name} not found: run `python -m src.experiments` first.")
        best = best_scaler_per_model()
        print("best scaler per model on the development subsample:", best, flush=True)
        res = run_experiments(define_confirm_experiments(best), train, results_path=CONFIRM_RESULTS,
                              sig_path=config.TABLES / "week2_confirm_signature.json", scope="full_train")
        complete = bool(len(res) and res.groupby(["model", "encoder"])["fold"].nunique().eq(N_SPLITS).all()
                        and res[["model", "encoder"]].drop_duplicates().shape[0] == N_CONFIRM_CONFIGS)
        (config.REPORTS / "week2_confirm_meta.json").write_text(json.dumps(
            {"n_train_rows": int(len(train)), "n_splits": N_SPLITS, "best_scaler_per_model": best, "complete": complete,
             "finished_at": datetime.now().isoformat(timespec="seconds"), "n_result_rows": int(len(res))}, indent=2), encoding="utf-8")
        print("confirmation complete:", complete)
        return
    if args.quick:
        dev = make_dev_sample(train, 2000)
        cfgs = [dict(experiment="main", feature_set="main", encoder="target", scaler="standard", model=m) for m in MODELS]
        out = run_experiments(cfgs, dev, results_path=config.TABLES / "week2_quick_check.csv", n_splits=2)
        print(out.groupby("model")[["rmse", "mae", "r2"]].mean())
        return
    dev = make_dev_sample(train)
    print("development subsample:", dev.shape, "groups:", dev["group_id"].nunique(), flush=True)
    res = run_experiments(define_experiments(), dev, sig_path=config.TABLES / "week2_signature.json", scope="dev")
    meta = {"n_dev_rows": int(len(dev)), "n_dev_groups": int(dev["group_id"].nunique()), "n_splits": N_SPLITS,
            "finished_at": datetime.now().isoformat(timespec="seconds"), "n_result_rows": int(len(res))}
    (config.REPORTS / "week2_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
