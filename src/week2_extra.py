"""Week-2 supplementary analysis (Bài kiểm tra 2): evidence the original week-2 run did not store.

For the best (encoder, scaler) configuration of each model in the week-2 main grid, re-run the SAME
development subsample and the SAME 5 GroupKFold folds as src/experiments.py and record:

  1. train-fold vs validation-fold metrics (over/under-fitting check, criterion 7);
  2. a median baseline per fold (DummyRegressor, criterion 2 and 6);
  3. a learning curve: train/validation MAE when the training fold is subsampled to 20..100 % (whole groups);
  4. out-of-fold predictions on the validation folds (residual plot, criterion 6);
  5. permutation importance of the ORIGINAL input columns on the validation folds (criterion 9).

Only train.parquet is read; the hold-out test split is never touched.

Usage:  python -m src.week2_extra            (full run, about 10-20 minutes on 12 cores)
        python -m src.week2_extra --quick    (smoke run: 2,000 rows, 2 folds, separate *_quick files)
        python -m src.week2_extra --plots    (redraw the figures from the saved tables only)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src import config  # noqa: E402
from src.experiments import (N_DEV, N_SPLITS, RESULTS, data_fingerprint, load_train,  # noqa: E402
                             make_dev_sample, regression_metrics)
from src.pipelines import CAT_MAIN, NUM, build_estimator  # noqa: E402

T, F = config.TABLES, config.FIGURES
FEATURES = CAT_MAIN + NUM                      # the main-grid feature set (model_version is not used)
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]          # share of the training fold (whole groups) for the learning curve
CURVE_MODELS = ["knn", "ridge", "rf"]          # SVR is too slow to refit at every size; it gets the 100 % point only
N_REPEATS = 5                                  # permutation importance repeats per fold
MODEL_LABEL = {"knn": "KNN", "ridge": "Ridge", "svr": "SVR", "rf": "Random Forest", "baseline": "Baseline trung vị"}
MODEL_ORDER = ["baseline", "ridge", "knn", "svr", "rf"]


# ------------------------------------------------------------------ configuration
def best_configs(results_path=RESULTS) -> dict:
    """(encoder, scaler) of the configuration with the lowest mean validation MAE per model in the main grid,
    read from the stored week-2 fold results (same rule as week2_best_per_model.csv)."""
    res = pd.read_csv(results_path)
    cfg = res[res["experiment"] == "main"].groupby(["model", "encoder", "scaler"])["mae"].mean().reset_index()
    best = cfg.loc[cfg.groupby("model")["mae"].idxmin()]
    return {r.model: (r.encoder, r.scaler) for r in best.itertuples()}


def subsample_groups(tr_idx: np.ndarray, groups: pd.Series, frac: float, seed: int) -> np.ndarray:
    """Keep whole groups (cars) of the training fold until about `frac` of its rows are kept."""
    if frac >= 1.0:
        return tr_idx
    g = groups.iloc[tr_idx]
    sizes = g.value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cum = sizes.loc[order].cumsum()
    keep = set(cum.index[: int((cum <= frac * len(tr_idx)).sum()) + 1])
    return tr_idx[g.isin(keep).to_numpy()]


def make_estimator(model: str, encoder: str, scaler: str):
    if model == "baseline":
        return DummyRegressor(strategy="median")
    return build_estimator(encoder, scaler, model, cat_cols=CAT_MAIN, num_cols=NUM)


# ------------------------------------------------------------------ one task = (model, fold, fraction)
def run_task(model, encoder, scaler, fold, frac, X, y, groups, tr_idx, va_idx, n_repeats):
    tr_used = subsample_groups(tr_idx, groups, frac, seed=config.RANDOM_SEED + fold)
    est = make_estimator(model, encoder, scaler)
    Xtr, ytr, Xva, yva = X.iloc[tr_used], y.iloc[tr_used], X.iloc[va_idx], y.iloc[va_idx]
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pred_tr, pred_va = est.predict(Xtr), est.predict(Xva)
    base = dict(model=model, encoder=encoder, scaler=scaler, fold=fold, frac=frac,
                n_train=len(tr_used), n_valid=len(va_idx), fit_seconds=fit_s)
    rows = [dict(base, split="train", **regression_metrics(ytr, pred_tr)),
            dict(base, split="valid", **regression_metrics(yva, pred_va))]
    oof, imp = None, []
    if frac >= 1.0:
        oof = pd.DataFrame({"row": va_idx, "fold": fold, "model": model, "y_true": yva.to_numpy(), "y_pred": pred_va})
        if model != "baseline":
            r = permutation_importance(est, Xva, yva, scoring="neg_mean_absolute_error", n_repeats=n_repeats,
                                       random_state=config.RANDOM_SEED, n_jobs=1)
            for j, col in enumerate(X.columns):
                for k, v in enumerate(r.importances[j]):
                    imp.append(dict(model=model, fold=fold, repeat=k, feature=col, mae_increase=float(v)))
    return rows, oof, imp


# ------------------------------------------------------------------ runner
def run(quick: bool = False) -> None:
    suffix = "_quick" if quick else ""
    n_rows, n_splits = (2_000, 2) if quick else (N_DEV, N_SPLITS)
    fractions = [0.5, 1.0] if quick else FRACTIONS
    n_repeats = 2 if quick else N_REPEATS

    cfgs = best_configs()
    dev = make_dev_sample(load_train(), n_rows)
    X, y, groups = dev[FEATURES], dev["price"], dev["group_id"]
    # GroupKFold is deterministic given the groups: these are exactly the folds of src/experiments.py
    # (it splits the same rows; the extra model_version column it carries does not change the split).
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    print("development subsample:", dev.shape, "| best configurations:", cfgs, flush=True)

    tasks = []
    for model in ["baseline"] + list(cfgs):
        enc, sc = cfgs.get(model, ("-", "-"))
        fracs = fractions if model in CURVE_MODELS + ["baseline"] else [1.0]
        for k, (tr, va) in enumerate(folds):
            for frac in fracs:
                tasks.append((model, enc, sc, k, frac, tr, va))
    print(f"{len(tasks)} tasks", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_task)(m, e, s, k, f, X, y, groups, tr, va, n_repeats) for m, e, s, k, f, tr, va in tasks)

    folds_df = pd.DataFrame([r for rows, _, _ in out for r in rows])
    oof = pd.concat([o for _, o, _ in out if o is not None], ignore_index=True)
    oof["listing_id"] = dev["listing_id"].to_numpy()[oof["row"]]
    oof["brand"] = dev["brand"].to_numpy()[oof["row"]]
    imp = pd.DataFrame([r for _, _, im in out for r in im])

    folds_df.to_csv(T / f"week2_extra_fold_results{suffix}.csv", index=False)
    oof.drop(columns="row").to_csv(T / f"week2_extra_oof_predictions{suffix}.csv", index=False)
    imp.to_csv(T / f"week2_extra_permutation_raw{suffix}.csv", index=False)
    summarise(folds_df, imp, suffix)

    if not quick:   # consistency check against the stored week-2 results (same data, folds and settings)
        ref = pd.read_csv(RESULTS)
        ref = ref[ref["experiment"] == "main"]
        chk = []
        for model, (enc, sc) in cfgs.items():
            a = ref[(ref.model == model) & (ref.encoder == enc) & (ref.scaler == sc)].set_index("fold")["mae"]
            b = folds_df[(folds_df.model == model) & (folds_df.frac == 1.0) & (folds_df.split == "valid")].set_index("fold")["mae"]
            chk.append(dict(model=model, max_abs_diff_mae=float((a - b).abs().max())))
        print("check vs week2_fold_results.csv:", chk, flush=True)
    else:
        chk = []
    meta = {"scope": "quick" if quick else "dev", "n_dev_rows": int(len(dev)), "n_splits": n_splits,
            "fractions": fractions, "n_repeats": n_repeats, "best_configs": cfgs, "seed": config.RANDOM_SEED,
            "train_fingerprint": data_fingerprint(), "sklearn": sklearn.__version__,
            "check_vs_week2": chk, "elapsed_seconds": round(time.time() - t0, 1),
            "finished_at": datetime.now().isoformat(timespec="seconds")}
    (config.REPORTS / f"week2_extra_meta{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plots(suffix)
    print(f"done in {time.time() - t0:,.0f}s", flush=True)


# ------------------------------------------------------------------ summary tables
def summarise(folds_df: pd.DataFrame, imp: pd.DataFrame, suffix: str = "") -> None:
    full = folds_df[folds_df["frac"] == 1.0]
    metrics = ["mae", "rmse", "r2", "medape_pct"]
    s = full.groupby(["model", "encoder", "scaler", "split"])[metrics].agg(["mean", "std"])
    s.columns = [f"{m}_{a}" for m, a in s.columns]
    s = s.reset_index()
    tr = s[s.split == "train"].set_index("model")
    va = s[s.split == "valid"].set_index("model")
    tv = pd.DataFrame({"encoder": va["encoder"], "scaler": va["scaler"],
                       "mae_train_mean": tr["mae_mean"], "mae_train_std": tr["mae_std"],
                       "mae_valid_mean": va["mae_mean"], "mae_valid_std": va["mae_std"],
                       "rmse_train_mean": tr["rmse_mean"], "rmse_valid_mean": va["rmse_mean"],
                       "r2_train_mean": tr["r2_mean"], "r2_valid_mean": va["r2_mean"], "r2_valid_std": va["r2_std"],
                       "medape_train_mean": tr["medape_pct_mean"], "medape_valid_mean": va["medape_pct_mean"]})
    tv["mae_gap_eur"] = tv["mae_valid_mean"] - tv["mae_train_mean"]
    tv["mae_valid_over_train"] = tv["mae_valid_mean"] / tv["mae_train_mean"]
    tv = tv.reindex([m for m in MODEL_ORDER if m in tv.index])
    tv.to_csv(T / f"week2_extra_train_vs_valid{suffix}.csv")
    s.to_csv(T / f"week2_extra_summary{suffix}.csv", index=False)

    curve = folds_df.groupby(["model", "frac", "split"]).agg(
        n_train=("n_train", "mean"), mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    curve.to_csv(T / f"week2_extra_learning_curve{suffix}.csv", index=False)

    if len(imp):
        per_fold = imp.groupby(["model", "feature", "fold"])["mae_increase"].mean().reset_index()
        pi = per_fold.groupby(["model", "feature"])["mae_increase"].agg(["mean", "std"]).reset_index()
        pi = pi.rename(columns={"mean": "mae_increase_mean", "std": "mae_increase_std_over_folds"})
        pi["rank"] = pi.groupby("model")["mae_increase_mean"].rank(ascending=False).astype(int)
        pi.sort_values(["model", "rank"]).to_csv(T / f"week2_extra_permutation_importance{suffix}.csv", index=False)


# ------------------------------------------------------------------ figures
def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(F / name, dpi=150)
    plt.close(fig)


def plots(suffix: str = "") -> None:
    tv = pd.read_csv(T / f"week2_extra_train_vs_valid{suffix}.csv", index_col=0)
    curve = pd.read_csv(T / f"week2_extra_learning_curve{suffix}.csv")
    oof = pd.read_csv(T / f"week2_extra_oof_predictions{suffix}.csv")
    pi_path = T / f"week2_extra_permutation_importance{suffix}.csv"

    # 1) train vs validation MAE per model (+ baseline)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    idx = np.arange(len(tv))
    ax.bar(idx - 0.2, tv["mae_train_mean"], 0.4, yerr=tv["mae_train_std"], capsize=3, label="Phần huấn luyện (train fold)")
    ax.bar(idx + 0.2, tv["mae_valid_mean"], 0.4, yerr=tv["mae_valid_std"], capsize=3, label="Phần kiểm định (validation fold)")
    ax.set_xticks(idx, [MODEL_LABEL[m] for m in tv.index])
    ax.set_ylabel("MAE (EUR)")
    ax.set_title("MAE trên phần huấn luyện và phần kiểm định (trung bình ± SD qua 5 fold)")
    ax.legend()
    _save(fig, f"week2_extra_train_vs_valid{suffix}.png")

    # 2) learning curves
    models = [m for m in CURVE_MODELS if m in curve["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False)
    for ax, m in zip(axes[0], models):
        for split, lab in [("train", "Huấn luyện"), ("valid", "Kiểm định")]:
            c = curve[(curve.model == m) & (curve.split == split)].sort_values("n_train")
            ax.plot(c["n_train"], c["mae_mean"], marker="o", label=lab)
            ax.fill_between(c["n_train"], c["mae_mean"] - c["mae_std"], c["mae_mean"] + c["mae_std"], alpha=.2)
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("Số dòng huấn luyện trong fold")
        ax.set_ylabel("MAE (EUR)")
        ax.legend()
    _save(fig, f"week2_extra_learning_curve{suffix}.png")

    # 3) residuals of the out-of-fold predictions
    models = [m for m in ["ridge", "knn", "svr", "rf"] if m in oof["model"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        d = oof[oof.model == m]
        res = d["y_pred"] - d["y_true"]
        ax.scatter(d["y_pred"].clip(lower=100), res, s=3, alpha=.25)
        ax.axhline(0, color="black", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=10_000)
        ax.set_title(f"{MODEL_LABEL[m]} (MAE {res.abs().mean():,.0f} EUR)".replace(",", "."))
        n_low = int((d["y_pred"] < 100).sum())
        if n_low:   # predictions <= 100 EUR (incl. negative prices) are drawn at 100 on the log axis
            ax.text(0.02, 0.04, f"{n_low} dự đoán < 100 EUR (gồm giá âm) vẽ tại 100",
                    transform=ax.transAxes, fontsize=8, color="darkred")
        ax.set_xlabel("Giá dự đoán (EUR, thang log)")
        ax.set_ylabel("Phần dư = dự đoán − thực tế (EUR)")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    _save(fig, f"week2_extra_residuals{suffix}.png")

    # 4) permutation importance
    if pi_path.exists():
        pi = pd.read_csv(pi_path)
        models = [m for m in ["ridge", "knn", "svr", "rf"] if m in pi["model"].unique()]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for ax, m in zip(axes.ravel(), models):
            d = pi[pi.model == m].sort_values("mae_increase_mean")
            ax.barh(d["feature"], d["mae_increase_mean"], xerr=d["mae_increase_std_over_folds"], capsize=2)
            ax.set_title(MODEL_LABEL[m])
            ax.set_xlabel("MAE tăng khi hoán vị cột (EUR)")
        for ax in axes.ravel()[len(models):]:
            ax.axis("off")
        _save(fig, f"week2_extra_permutation_importance{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run: 2,000 rows, 2 folds, files with suffix _quick")
    ap.add_argument("--plots", action="store_true", help="only redraw the figures from the saved tables")
    args = ap.parse_args()
    if args.plots:
        plots()
        return
    run(quick=args.quick)


if __name__ == "__main__":
    main()
