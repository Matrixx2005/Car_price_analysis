"""Week-3 tables and figures.

cv_*  functions use only the TRAIN-split results (week3_fold_results.csv).
test_* functions use only the saved predictions of the single final evaluation (never the raw test split).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.pipelines import MODELS  # noqa: E402
from src.week2_analysis import ENC_LABEL, MODEL_LABEL  # noqa: E402
from src.week3_experiments import FULL_FIT_BUDGET_S, RESULTS3, aggregate3, fold_mae  # noqa: E402

T, F = config.TABLES, config.FIGURES
PRED_PATH = config.DATA_PROCESSED / "week3_test_predictions.parquet"
PRICE_BINS = [0, 10_000, 20_000, 40_000, 80_000, 160_000, 1e9]
PRICE_LABELS = ["<10k", "10-20k", "20-40k", "40-80k", "80-160k", ">=160k"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=130)
    plt.close(fig)
    return name


# ----------------------------------------------------------------------------------- CV stages (train only)
def tune_vs_default(res: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """'tuned' = best FEASIBLE configuration (estimated full-train fit time within the budget); 'any' = best regardless of cost."""
    from src.week3_experiments import best_tune_config
    rows = []
    for m in MODELS:
        t = agg[(agg.model == m) & (agg.stage == "tune") & (agg.n_folds == 5)]
        best = best_tune_config(agg, m)
        anyb = t.loc[t.mae_mean.idxmin()]
        d = t[(t.params == "{}") & (~t.log_target.astype(bool))].iloc[0]
        b, a = fold_mae(res, best.config_id), fold_mae(res, d.config_id)
        rows.append(dict(model=m, n_configs_tried=len(t), n_infeasible=int((t.est_full_fit_s > FULL_FIT_BUDGET_S).sum()), default_mae=d.mae_mean,
                         tuned_mae=best.mae_mean, gain_pct=(best.mae_mean / d.mae_mean - 1) * 100,
                         folds_tuned_better=int((b.values < a.values).sum()), tuned_params=best.params, tuned_log_target=bool(best.log_target),
                         tuned_est_full_fit_s=best.est_full_fit_s, any_best_mae=anyb.mae_mean, any_best_params=anyb.params,
                         any_best_log_target=bool(anyb.log_target), any_best_est_full_fit_s=anyb.est_full_fit_s,
                         default_rmse=d.rmse_mean, tuned_rmse=best.rmse_mean, default_r2=d.r2_mean, tuned_r2=best.r2_mean,
                         default_medape=d.medape_pct_mean, tuned_medape=best.medape_pct_mean))
    return pd.DataFrame(rows)


def extremes_effect(res: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    from src.week3_experiments import best_tune_config, cfg_from_row, EXTREME_QS
    rows = []
    by = agg.set_index("config_id")
    for m in MODELS:
        base = best_tune_config(agg, m)
        opts = [(None, base.config_id)] + [(q, cfg_from_row(base, extreme_q=q)["config_id"]) for q in EXTREME_QS]
        for q, cid in opts:
            if cid not in by.index or by.loc[cid, "n_folds"] != 5:
                continue
            r = by.loc[cid]
            rows.append(dict(model=m, extreme_q=q if q is not None else "none", mae=r.mae_mean, mae_lt_100k=r.mae_lt_100k_mean, mae_ge_100k=r.mae_ge_100k_mean,
                             bias_ge_100k=r.bias_ge_100k_mean, rmse=r.rmse_mean,
                             folds_better_than_none=int((fold_mae(res, cid).values < fold_mae(res, base.config_id).values).sum()) if q is not None else np.nan,
                             rel_change_pct=(r.mae_mean / base.mae_mean - 1) * 100))
    return pd.DataFrame(rows)


def grouping_effect(res: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    from src.week3_experiments import RARE_COUNTS, best_tune_config, cfg_from_row, week2_best_preprocessing
    by = agg.set_index("config_id")
    rows = []
    for m in MODELS:
        base = best_tune_config(agg, m)
        sc = week2_best_preprocessing()[m][1]
        for enc in ("onehot", "target"):
            for rc in RARE_COUNTS:
                cid = cfg_from_row(base, encoder=enc, scaler=sc, feature_set="with_model_version", rare_min_count=rc)["config_id"]
                if cid not in by.index or by.loc[cid, "n_folds"] != 5:
                    continue
                r = by.loc[cid]
                rows.append(dict(model=m, encoder=enc, rare_min_count=rc if rc is not None else "none", mae=r.mae_mean, baseline_mae=base.mae_mean,
                                 rel_change_pct=(r.mae_mean / base.mae_mean - 1) * 100, n_features_out=r.n_features_out,
                                 folds_better=int((fold_mae(res, cid).values < fold_mae(res, base.config_id).values).sum())))
    return pd.DataFrame(rows)


def fig_tuning(agg: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, m in zip(axes, MODELS):
        t = agg[(agg.model == m) & (agg.stage == "tune") & (agg.n_folds == 5)].sort_values("mae_mean").reset_index(drop=True)
        d = t[(t.params == "{}") & (~t.log_target.astype(bool))]
        ax.scatter(t.index, t.mae_mean, c=np.where(t.log_target.astype(bool), "#c0504d", "#3b6ea5"), s=25,
                   marker="o")
        slow = t[t.est_full_fit_s > FULL_FIT_BUDGET_S]
        ax.scatter(slow.index, slow.mae_mean, facecolors="none", edgecolors="k", s=90, label="quá đắt cho toàn train")
        ax.scatter(d.index, d.mae_mean, marker="*", s=220, c="gold", edgecolor="k", zorder=5, label="mặc định tuần 2")
        ax.set(title=f"{MODEL_LABEL[m]} ({len(t)} cấu hình)", xlabel="cấu hình (xếp theo MAE)", ylabel="MAE CV (EUR)")
        ax.scatter([], [], c="#3b6ea5", label="giá gốc")
        ax.scatter([], [], c="#c0504d", label="log giá")
        if m == "knn":
            ax.legend(fontsize=8)
    return _save(fig, "week3_tuning.png")


def fig_extremes(ex: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, col, title in ((axes[0], "mae_lt_100k", "MAE trên xe < 100.000 EUR"), (axes[1], "mae_ge_100k", "MAE trên xe >= 100.000 EUR")):
        p = ex.pivot(index="model", columns="extreme_q", values=col).loc[MODELS]
        p.rename(index=MODEL_LABEL).plot.bar(ax=ax, rot=0)
        ax.set(title=title, ylabel="MAE (EUR)", xlabel="")
        ax.legend(title="cắt xe giá trên phân vị", fontsize=8)
    return _save(fig, "week3_extremes.png")


def fig_grouping(gr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, enc in zip(axes, ("onehot", "target")):
        p = gr[gr.encoder == enc].pivot(index="model", columns="rare_min_count", values="rel_change_pct").loc[MODELS]
        p = p[[c for c in ["none", 10, 50, 200] if c in p.columns]]
        sns.heatmap(p.rename(index=MODEL_LABEL), annot=True, fmt="+.1f", cmap="RdYlGn_r", center=0, ax=ax)
        ax.set(title=f"+ model_version, {ENC_LABEL[enc]}: thay đổi MAE (%) so với không có\n(âm = tốt hơn; cột = ngưỡng gom danh mục hiếm)",
               xlabel="gom danh mục < N lần", ylabel="")
        ax.title.set_fontsize(10)
    return _save(fig, "week3_grouping.png")


def cv_tables_and_figures() -> dict:
    res = pd.read_csv(RESULTS3)
    agg = aggregate3(res)
    agg.to_csv(T / "week3_summary_all.csv", index=False)
    tv = tune_vs_default(res, agg)
    tv.to_csv(T / "week3_tune_vs_default.csv", index=False)
    figs = [fig_tuning(agg)]
    ex = gr = None
    if (agg.stage == "extremes").any():
        ex = extremes_effect(res, agg)
        ex.to_csv(T / "week3_extremes_effect.csv", index=False)
        figs.append(fig_extremes(ex))
    if (agg.stage == "grouping").any():
        gr = grouping_effect(res, agg)
        gr.to_csv(T / "week3_grouping_effect.csv", index=False)
        figs.append(fig_grouping(gr))
    return {"agg": agg, "tv": tv, "ex": ex, "gr": gr, "figures": figs}


# ----------------------------------------------------------------------------------- final test (saved predictions only)
def band(series: pd.Series, bins, labels) -> pd.Series:
    return pd.cut(series, bins, labels=labels, right=False)


def group_errors(df: pd.DataFrame, pred: str, key: str, min_n: int) -> pd.DataFrame:
    e = df[pred] - df["price"]
    g = pd.DataFrame({"key": df[key], "abs": e.abs(), "signed": e, "ape": e.abs() / df["price"] * 100, "price": df["price"]}).groupby("key", observed=True)
    out = g.agg(n=("abs", "size"), mae=("abs", "mean"), bias=("signed", "mean"), medape_pct=("ape", "median"), median_price=("price", "median")).reset_index()
    return out[out.n >= min_n].sort_values("mae", ascending=False).reset_index(drop=True)


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive check AFTER the single evaluation (never used to choose a model): how much do a handful of label errors
    and the worst predictions drive RMSE/R²? Excludes the rows flagged as suspicious prices in the week-1 data screen."""
    from src.experiments import regression_metrics
    flags = pd.read_parquet(config.DATA_INTERIM / "autoscout24_used_common.parquet", columns=["listing_id", "flag_price_review"])
    flagged = set(flags.loc[flags["flag_price_review"], "listing_id"])
    mask = ~df["listing_id"].isin(flagged)
    rows = []
    for c in [c for c in df.columns if c.startswith("pred_")]:
        err = (df[c] - df["price"]).to_numpy()
        sse = np.sort(err ** 2)[::-1]
        m_all, m_clean = regression_metrics(df["price"], df[c]), regression_metrics(df.loc[mask, "price"], df.loc[mask, c])
        rows.append({"name": c.removeprefix("pred_"), "n_flagged_excluded": int((~mask).sum()), "mae": m_all["mae"], "mae_without_flagged": m_clean["mae"],
                     "rmse": m_all["rmse"], "rmse_without_flagged": m_clean["rmse"], "r2": m_all["r2"], "r2_without_flagged": m_clean["r2"],
                     "share_sse_top5_errors_pct": sse[:5].sum() / sse.sum() * 100})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def test_analysis(selected: str) -> dict:
    """Residual analysis of the selected model on the saved test predictions."""
    df = pd.read_parquet(PRED_PATH)
    pred = f"pred_{selected}_tuned"
    df["price_band"] = band(df["price"], PRICE_BINS, PRICE_LABELS)
    df["age_band"] = band(df["vehicle_age"], [0, 2, 4, 7, 10, 15, 100], ["0-1", "2-3", "4-6", "7-9", "10-14", "15+"])
    tabs = {"price_band": group_errors(df, pred, "price_band", 1), "age_band": group_errors(df, pred, "age_band", 1),
            "brand": group_errors(df, pred, "brand", 100), "country": group_errors(df, pred, "country_code", 100),
            "fuel_type": group_errors(df, pred, "fuel_type", 100)}
    tabs["price_band"] = tabs["price_band"].set_index("key").loc[PRICE_LABELS].reset_index()
    tabs["age_band"] = tabs["age_band"].set_index("key").reindex(["0-1", "2-3", "4-6", "7-9", "10-14", "15+"]).reset_index()
    worst = df.assign(pred=df[pred], err=df[pred] - df["price"], abs_err=(df[pred] - df["price"]).abs()).nlargest(10, "abs_err")
    tabs["worst"] = worst[["brand", "model", "production_year", "mileage_km", "country_code", "price", "pred", "err"]].reset_index(drop=True)
    for k, v in tabs.items():
        v.to_csv(T / f"week3_test_errors_by_{k}.csv", index=False)
    sens = sensitivity(df)
    sens.to_csv(T / "week3_test_sensitivity.csv", index=False)
    tabs["sensitivity"] = sens

    figs = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(df["price"], df[pred], s=4, alpha=.25)
    lim = [df["price"].min() * 0.9, df["price"].max() * 1.1]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=(max(lim[0], 100), lim[1]), xlabel="Giá thực (EUR)", ylabel="Giá dự đoán (EUR)",
              title=f"{MODEL_LABEL[selected]}: dự đoán so với thực tế (test, trục log)")
    pb = tabs["price_band"]
    ax[1].bar(pb["key"].astype(str), pb["mae"], color="#3b6ea5")
    for i, (m, n) in enumerate(zip(pb["mae"], pb["n"])):
        ax[1].text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax[1].set(title="MAE theo khoảng giá thực (test)", xlabel="giá thực", ylabel="MAE (EUR)")
    figs.append(_save(fig, "week3_test_pred_vs_actual.png"))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    b = tabs["brand"].head(12)
    ax[0].barh([f"{k} (n={n})" for k, n in zip(b["key"], b["n"])][::-1], b["mae"][::-1], color="#3b6ea5")
    ax[0].set(title="MAE theo hãng (n >= 100, test)", xlabel="MAE (EUR)")
    c = tabs["country"]
    ax[1].bar(c["key"], c["mae"], color="#3b6ea5")
    ax[1].set(title="MAE theo quốc gia (n >= 100)", ylabel="MAE (EUR)")
    ag = tabs["age_band"]
    ax[2].bar(ag["key"].astype(str), ag["medape_pct"], color="#3b6ea5")
    ax[2].set(title="MedAPE (%) theo tuổi xe", xlabel="tuổi xe (năm)", ylabel="%")
    figs.append(_save(fig, "week3_test_errors_by_group.png"))
    return {"tabs": tabs, "figures": figs, "df": df, "pred_col": pred}


def fig_models_on_test(metrics: pd.DataFrame, ci: pd.DataFrame) -> str:
    m = metrics.sort_values("mae").reset_index(drop=True)
    c = ci[(ci.kind == "metric") & (ci.metric == "mae")].set_index("name")
    lo = m["mae"].to_numpy() - c.loc[m["name"], "ci_low"].to_numpy()
    hi = c.loc[m["name"], "ci_high"].to_numpy() - m["mae"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c0504d" if s else ("#9bb7d4" if n.endswith("default") else ("#9a9a9a" if n.endswith("cvbest_20k") else "#3b6ea5"))
              for n, s in zip(m["name"], m["is_selected"])]
    ax.barh(m["name"][::-1], m["mae"][::-1], xerr=[lo[::-1], hi[::-1]], color=colors[::-1], capsize=4)
    ax.set(title="MAE trên tập test (khoảng tin cậy 95% theo bootstrap nhóm xe)\nđỏ = mô hình được chọn bằng CV; xanh nhạt = mặc định tuần 2; xám = tham khảo (quá đắt)",
           xlabel="MAE (EUR)")
    ax.title.set_fontsize(10)
    return _save(fig, "week3_test_models.png")


def fig_cv_vs_test(frozen: dict, metrics: pd.DataFrame) -> str:
    t = metrics.set_index("name")
    cv = [frozen["models"][m]["final"]["cv"]["mae_mean"] for m in MODELS]
    te = [t.loc[f"{m}_tuned", "mae"] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, cv, 0.4, label="CV trên train (tập con 20k)", color="#3b6ea5")
    ax.bar(x + 0.2, te, 0.4, label="Test (huấn luyện trên toàn train)", color="#c0504d")
    ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS])
    ax.set(ylabel="MAE (EUR)", title="MAE ước lượng bằng CV so với MAE trên test (cấu hình đã tinh chỉnh)")
    ax.legend()
    ax.title.set_fontsize(10)
    return _save(fig, "week3_cv_vs_test.png")


def main() -> None:
    """CV tables/figures always; test residual analysis only if the single final evaluation has been done."""
    out = cv_tables_and_figures()
    print("CV figures:", out["figures"])
    if (T / "week3_test_evaluation_done.json").exists():
        print("test figures:", test_tables_and_figures()["figures"])
    else:
        print("final evaluation not done yet: run `python -m src.final_eval` first")


def test_tables_and_figures() -> dict:
    frozen = json.loads((config.REPORTS / "week3_frozen_config.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(T / "week3_test_metrics.csv")
    ci = pd.read_csv(T / "week3_test_bootstrap_ci.csv")
    out = test_analysis(frozen["selected_model"])
    out["figures"] += [fig_models_on_test(metrics, ci), fig_cv_vs_test(frozen, metrics)]
    out.update(frozen=frozen, metrics=metrics, ci=ci)
    return out


if __name__ == "__main__":
    main()
