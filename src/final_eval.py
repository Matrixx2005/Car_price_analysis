"""FINAL evaluation on the hold-out test split. This is the ONLY module allowed to read test.parquet.

Rules enforced here:
  * the configuration must already be frozen (reports/week3_frozen_config.json) and unchanged since freezing
    (train file and week-3 result hashes are re-checked);
  * the test split is evaluated ONCE: a marker file is written afterwards and a second run is refused, so the test
    set cannot be used to tune anything (use the saved predictions file for further analysis);
  * every model is refit on the FULL train split with its frozen configuration, then predicts the test split.

Usage:  python -m src.final_eval
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.experiments import load_train, make_dev_sample, regression_metrics
from src.pipelines import CAT_WITH_VERSION, MODELS, NUM, build_estimator
from src.transformers import drop_extreme_prices
from src.week3_select import FROZEN, file_md5
from src.week3_experiments import RESULTS3

MARKER = config.TABLES / "week3_test_evaluation_done.json"
PRED_PATH = config.DATA_PROCESSED / "week3_test_predictions.parquet"
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
    return pd.read_parquet(config.DATA_PROCESSED / "test.parquet")


def main() -> None:
    frozen = check_ready()
    train = load_train()
    test = read_test()
    assert set(train["group_id"]).isdisjoint(set(test["group_id"])), "train/test share a group"
    ents = entries(frozen)
    print(f"fitting {len(ents)} models on the full train split ({len(train):,} rows), predicting {len(test):,} test rows", flush=True)
    res = Parallel(n_jobs=-1, verbose=5)(delayed(fit_predict)(c, train, test, n) for n, c in ents.items())
    res = {r["name"]: r for r in res}
    y = test["price"].to_numpy()
    preds = {n: r["pred"] for n, r in res.items()}
    pairs = [(f"{m}_default", f"{m}_tuned") for m in MODELS]
    sel = frozen["selected_model"]
    others = [f"{m}_tuned" for m in MODELS if m != sel]
    pairs += [(o, f"{sel}_tuned") for o in others]
    ci = cluster_bootstrap(y, preds, test["group_id"].to_numpy(), pairs)
    rows = []
    for n, r in res.items():
        m = regression_metrics(y, r["pred"])
        rows.append({"name": n, **m, "fit_seconds": r["fit_seconds"], "n_train_used": r["n_train_used"],
                     "is_selected": n == f"{sel}_tuned", "is_reference": n.endswith("_cvbest_20k")})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.TABLES / "week3_test_metrics.csv", index=False)
    ci.to_csv(config.TABLES / "week3_test_bootstrap_ci.csv", index=False)
    pred_frame = test.copy()
    for n, p in preds.items():
        pred_frame[f"pred_{n}"] = p
    pred_frame.to_parquet(PRED_PATH, index=False)
    MARKER.write_text(json.dumps({"evaluated_at": datetime.now().isoformat(timespec="seconds"), "selected_model": sel,
                                  "n_test": int(len(test)), "test_md5": file_md5(config.DATA_PROCESSED / "test.parquet"),
                                  "frozen_created_at": frozen["created_at"], "n_boot": N_BOOT}, indent=2), encoding="utf-8")
    print(metrics.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
