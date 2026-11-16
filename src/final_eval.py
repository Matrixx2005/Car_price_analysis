"""FINAL evaluation on the hold-out test split. This is the ONLY module allowed to read the test split.

Rules enforced here:
  * the configuration must already be frozen (reports/week3_frozen_config.json) and unchanged since freezing
    (train file and week-3 result hashes are re-checked);
  * the test split is evaluated ONCE: a marker file is written afterwards and a second run is refused, so the test
    set cannot be used to tune anything (use the saved predictions file for further analysis);
  * every model is refit on the FULL train split with its frozen configuration, then predicts the test split;
  * (robustness) each model's predictions are saved to disk the moment that model finishes. If the run is
    interrupted, running the same command again continues with the models that are still missing. This is one and the
    same evaluation resumed, not a second look: nothing is computed from the test labels until all models exist.

Usage:  python -m src.final_eval            (run or resume)
        python -m src.final_eval --status   (show which models are already done)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.experiments import load_train, make_dev_sample, regression_metrics
from src.pipelines import CAT_WITH_VERSION, MODELS, NUM, build_estimator
from src.transformers import drop_extreme_prices
from src.week3_experiments import RESULTS3
from src.week3_select import FROZEN, file_md5

MARKER = config.TABLES / "week3_test_evaluation_done.json"
PRED_PATH = config.DATA_PROCESSED / "week3_test_predictions.parquet"
PARTIAL_DIR = config.DATA_PROCESSED / "week3_final_partial"
TEST_PATH = config.DATA_PROCESSED / "test.parquet"      # the only mention of the test file in the whole code base (besides split_data)
N_BOOT = 1000
FEATS = CAT_WITH_VERSION + NUM


class TestAlreadyUsed(RuntimeError):
    pass


def check_ready(frozen_path=FROZEN, marker=MARKER, check_hashes: bool = True) -> dict:
    """Raise unless the configuration is frozen/unchanged and the test split has not been evaluated yet."""
    if marker.exists():
        raise TestAlreadyUsed(f"The test split was already evaluated ({marker.name}). It must not be evaluated again; "
                              "use the saved predictions for any further analysis.")
    if not frozen_path.exists():
        raise FileNotFoundError("No frozen configuration: run `python -m src.week3_select` first.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if check_hashes:
        if file_md5(config.DATA_PROCESSED / "train.parquet") != frozen["train_md5"]:
            raise RuntimeError("train.parquet changed after the configuration was frozen.")
        if file_md5(RESULTS3) != frozen["week3_results_md5"]:
            raise RuntimeError("week-3 results changed after the configuration was frozen; re-freeze before evaluating.")
    return frozen


def entries(frozen: dict) -> dict[str, dict | None]:
    """Name -> configuration. None = naive baseline (train median)."""
    out: dict[str, dict | None] = {"naive_median": None}
    for m in MODELS:
        out[f"{m}_default"] = frozen["models"][m]["default"]
        out[f"{m}_tuned"] = frozen["models"][m]["final"]
    for m, cfg in frozen.get("references", {}).items():        # best-by-CV but too expensive to refit: reference only
        out[f"{m}_cvbest_20k"] = cfg
    return out


def fit_predict(cfg: dict | None, train: pd.DataFrame, test: pd.DataFrame, name: str) -> dict:
    t0 = time.perf_counter()
    if cfg is None:
        return {"name": name, "pred": np.full(len(test), float(train["price"].median())), "fit_seconds": 0.0, "n_train_used": len(train)}
    cat_cols = CAT_WITH_VERSION if cfg["feature_set"] == "with_model_version" else [c for c in CAT_WITH_VERSION if c != "model_version"]
    est = build_estimator(cfg["encoder"], cfg["scaler"], cfg["model"], log_target=cfg["log_target"], cat_cols=cat_cols, num_cols=NUM,
                          model_params=cfg["params"], rare_min_count=cfg["rare_min_count"], target_smooth=cfg["target_smooth"])
    if cfg.get("fit_rows") == "dev_subsample":                           # reference models: same 20k rows as in CV
        train = make_dev_sample(train)
    keep = drop_extreme_prices(train["price"], cfg["extreme_q"])        # threshold from the training rows used, never the test
    est.fit(train.loc[keep, FEATS], train.loc[keep, "price"])
    return {"name": name, "pred": est.predict(test[FEATS]), "fit_seconds": time.perf_counter() - t0, "n_train_used": int(keep.sum())}


# ------------------------------------------------------------------ incremental saving (resume after an interruption)
def _frozen_sha(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()


def save_partial(res: dict, listing_ids, partial_dir: Path, frozen_sha: str) -> None:
    """Write one finished model atomically (temp file, then rename) so a crash never leaves a half-written file."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    final = partial_dir / f"{res['name']}.parquet"
    tmp = final.with_suffix(".tmp")
    pd.DataFrame({"listing_id": np.asarray(listing_ids), "pred": np.asarray(res["pred"], float),
                  "fit_seconds": res["fit_seconds"], "n_train_used": res["n_train_used"], "frozen_sha": frozen_sha}).to_parquet(tmp, index=False)
    os.replace(tmp, final)


def load_partial(name: str, listing_ids, partial_dir: Path, frozen_sha: str) -> dict | None:
    """Return a finished model if its file exists, belongs to THIS frozen configuration and covers the same test rows."""
    p = partial_dir / f"{name}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if len(d) != len(listing_ids) or d["frozen_sha"].iloc[0] != frozen_sha or not (d["listing_id"].to_numpy() == np.asarray(listing_ids)).all():
        raise RuntimeError(f"{p.name} was produced by a different frozen configuration or test split; delete {partial_dir} to start over.")
    return {"name": name, "pred": d["pred"].to_numpy(), "fit_seconds": float(d["fit_seconds"].iloc[0]), "n_train_used": int(d["n_train_used"].iloc[0])}


def status(frozen: dict, listing_ids, partial_dir: Path = PARTIAL_DIR) -> dict[str, bool]:
    sha = _frozen_sha(frozen)
    return {n: load_partial(n, listing_ids, partial_dir, sha) is not None for n in entries(frozen)}


def _fit_save(name, cfg, train, test, partial_dir, sha):
    """Worker: fit, predict, and save immediately (before returning to the parent process)."""
    res = fit_predict(cfg, train, test, name)
    save_partial(res, test["listing_id"].to_numpy(), partial_dir, sha)
    return res


def compute_predictions(frozen: dict, train: pd.DataFrame, test: pd.DataFrame, partial_dir: Path = PARTIAL_DIR,
                        n_jobs: int = -1, runner=_fit_save) -> dict[str, dict]:
    """All models' predictions, reusing the ones already saved. Pending models run in parallel and are saved as they finish."""
    sha, ids = _frozen_sha(frozen), test["listing_id"].to_numpy()
    ents = entries(frozen)
    done = {n: load_partial(n, ids, partial_dir, sha) for n in ents}
    todo = [n for n, r in done.items() if r is None]
    print(f"{len(ents) - len(todo)}/{len(ents)} models already saved; fitting {len(todo)}: {todo}", flush=True)
    if todo:
        out = Parallel(n_jobs=n_jobs, verbose=5 if n_jobs != 1 else 0)(delayed(runner)(n, ents[n], train, test, partial_dir, sha) for n in todo)
        for r in out:
            done[r["name"]] = r
    return done


# ------------------------------------------------------------------ bootstrap, metrics, orchestration
def cluster_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, pairs: list[tuple[str, str]],
                      n_boot: int = N_BOOT, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """95% intervals by resampling WHOLE groups (cars) with replacement: listings of one car are not independent.
    Also returns paired differences (second minus first) of MAE for the requested model pairs."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    bounds = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1], True])
    rows = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    n_groups = len(rows)
    acc = {n: {k: [] for k in ("rmse", "mae", "r2", "medape_pct")} for n in preds}
    diffs = {p: [] for p in pairs}
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, n_groups, n_groups)])
        yb = y[idx]
        maes = {}
        for n, p in preds.items():
            m = regression_metrics(yb, p[idx])
            maes[n] = m["mae"]
            for k in acc[n]:
                acc[n][k].append(m[k])
        for a, b in pairs:
            diffs[(a, b)].append(maes[b] - maes[a])
    out = []
    for n in preds:
        for k, v in acc[n].items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            out.append({"kind": "metric", "name": n, "metric": k, "ci_low": lo, "ci_high": hi})
    for (a, b), v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out.append({"kind": "paired_mae_diff", "name": f"{b} - {a}", "metric": "mae", "ci_low": lo, "ci_high": hi,
                    "share_b_better": float(np.mean(np.asarray(v) < 0))})
    return pd.DataFrame(out)


def read_test() -> pd.DataFrame:
    """The single place where the test split is read."""
    return pd.read_parquet(TEST_PATH)


def summarise(frozen: dict, res: dict[str, dict], test: pd.DataFrame, n_boot: int = N_BOOT):
    """Metrics table, bootstrap intervals and the prediction frame from the finished models."""
    y = test["price"].to_numpy()
    preds = {n: r["pred"] for n, r in res.items()}
    sel = frozen["selected_model"]
    pairs = [(f"{m}_default", f"{m}_tuned") for m in MODELS] + [(f"{m}_tuned", f"{sel}_tuned") for m in MODELS if m != sel]
    ci = cluster_bootstrap(y, preds, test["group_id"].to_numpy(), pairs, n_boot=n_boot)
    rows = []
    for n, r in res.items():
        rows.append({"name": n, **regression_metrics(y, r["pred"]), "fit_seconds": r["fit_seconds"], "n_train_used": r["n_train_used"],
                     "is_selected": n == f"{sel}_tuned", "is_reference": n.endswith("_cvbest_20k")})
    pred_frame = test.copy()
    for n, p in preds.items():
        pred_frame[f"pred_{n}"] = p
    return pd.DataFrame(rows), ci, pred_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="show which models are already finished and exit")
    args = ap.parse_args()
    frozen = check_ready()
    train = load_train()
    test = read_test()
    assert set(train["group_id"]).isdisjoint(set(test["group_id"])), "train/test share a group"
    if args.status:
        print(json.dumps(status(frozen, test["listing_id"].to_numpy()), indent=2))
        return
    print(f"evaluating on {len(test):,} test rows; models are fitted on the full train split ({len(train):,} rows)", flush=True)
    res = compute_predictions(frozen, train, test)
    metrics, ci, pred_frame = summarise(frozen, res, test)
    metrics.to_csv(config.TABLES / "week3_test_metrics.csv", index=False)
    ci.to_csv(config.TABLES / "week3_test_bootstrap_ci.csv", index=False)
    pred_frame.to_parquet(PRED_PATH, index=False)
    MARKER.write_text(json.dumps({"evaluated_at": datetime.now().isoformat(timespec="seconds"), "selected_model": frozen["selected_model"],
                                  "n_test": int(len(test)), "test_md5": file_md5(TEST_PATH),
                                  "frozen_created_at": frozen["created_at"], "n_boot": N_BOOT}, indent=2), encoding="utf-8")
    shutil.rmtree(PARTIAL_DIR, ignore_errors=True)       # everything is now in the final files
    print(metrics.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
