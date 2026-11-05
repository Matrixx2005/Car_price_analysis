"""Aggregate week-2 fold results into tables and figures (all numbers come from week2_fold_results.csv)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402

T, F = config.TABLES, config.FIGURES
GROUP = ["experiment", "feature_set", "encoder", "scaler", "model"]
METRICS = ["rmse", "mae", "r2", "medape_pct"]
ENC_LABEL = {"onehot": "One-hot", "ordinal": "Ordinal", "target": "Target", "frequency": "Frequency"}
MODEL_LABEL = {"knn": "KNN", "ridge": "Ridge", "svr": "SVR", "rf": "Random Forest"}
SCALER_LABEL = {"none": "None", "standard": "Standard", "minmax": "MinMax", "robust": "Robust"}


def load_results(path=T / "week2_fold_results.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def aggregate(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(GROUP)
    out = g[METRICS].mean().add_suffix("_mean")
    out = out.join(g[METRICS].std().add_suffix("_std"))
    out["fit_seconds_mean"] = g["fit_seconds"].mean()
    out["predict_seconds_mean"] = g["predict_seconds"].mean()
    out["n_features_out"] = g["n_features_out"].mean().round().astype(int)
    out["n_folds"] = g["fold"].nunique()
    return out.reset_index()


def main_grid(agg: pd.DataFrame) -> pd.DataFrame:
    return agg[agg["experiment"] == "main"].copy()


def best_per_model(agg: pd.DataFrame, metric: str = "mae_mean") -> pd.DataFrame:
    m = main_grid(agg)
    return m.loc[m.groupby("model")[metric].idxmin()].sort_values(metric).reset_index(drop=True)


def effect_table(agg: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Mean MAE per model for each level of `factor` (encoder or scaler), averaged over the other factor."""
    m = main_grid(agg)
    t = m.pivot_table(index="model", columns=factor, values="mae_mean", aggfunc="mean")
    t["spread_max_minus_min"] = t.max(axis=1) - t.min(axis=1)
    t["spread_pct_of_min"] = t["spread_max_minus_min"] / t.drop(columns="spread_max_minus_min").min(axis=1) * 100
    return t


def paired_comparison(res: pd.DataFrame, exp_a: str, exp_b: str, metric: str = "mae") -> pd.DataFrame:
    """Fold-level paired difference (b - a) for configs present in both experiments (same encoder/model;
    scaler is 'standard' in the non-main experiments, so compare against main/standard)."""
    a = res[(res["experiment"] == exp_a) & (res["scaler"] == "standard")]
    b = res[res["experiment"] == exp_b]
    m = a.merge(b, on=["encoder", "model", "fold"], suffixes=("_a", "_b"))
    m["diff"] = m[f"{metric}_b"] - m[f"{metric}_a"]
    g = m.groupby(["model", "encoder"])
    out = g.agg(a_mean=(f"{metric}_a", "mean"), b_mean=(f"{metric}_b", "mean"), mean_diff=("diff", "mean"),
                folds_b_better=("diff", lambda d: int((d < 0).sum())), folds=("diff", "size")).reset_index()
    out["rel_change_pct"] = out["mean_diff"] / out["a_mean"] * 100
    return out


def paired_vs_baseline(res: pd.DataFrame, factor: str, baseline: str, fixed: dict, metric: str = "mae") -> pd.DataFrame:
    """Fold-level paired difference of every level of `factor` against `baseline`, inside the main grid, with the
    other factor fixed (e.g. factor='encoder', baseline='onehot', fixed={'scaler': 'standard'}).
    folds_better counts the folds (out of 5) in which the level beats the baseline: 5/5 or 0/5 is a consistent effect."""
    m = res[res["experiment"] == "main"]
    for k, v in fixed.items():
        m = m[m[k] == v]
    other = "scaler" if factor == "encoder" else "encoder"
    keys = ["model", "fold"] + ([other] if other not in fixed else [])
    base = m[m[factor] == baseline].set_index(keys)[metric]
    rows = []
    for lvl in m[factor].unique():
        if lvl == baseline:
            continue
        cur = m[m[factor] == lvl].set_index(keys)[metric]
        d = (cur - base).dropna().reset_index(name="diff")
        for model, g in d.groupby("model"):
            rows.append(dict(model=model, level=lvl, baseline=baseline, mean_diff=g["diff"].mean(),
                             rel_change_pct=g["diff"].mean() / base.xs(model, level="model").mean() * 100,
                             folds_better=int((g["diff"] < 0).sum()), n=len(g)))
    return pd.DataFrame(rows)


def confirm_comparison(path=T / "week2_confirm_fold_results.csv") -> pd.DataFrame | None:
    """Does the encoder ranking found on the 20k development subsample hold on the full train split?"""
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    if raw.groupby(["model", "encoder"])["fold"].nunique().ne(5).any() or raw[["model", "encoder"]].drop_duplicates().shape[0] != 12:
        print(f"[WARN] {path.name} is incomplete (need 12 configs x 5 folds); run `python -m src.experiments --confirm` to resume. "
              "Confirmation table not written.")
        return None
    full = aggregate(raw)
    dev = aggregate(load_results())
    dev = dev[dev["experiment"] == "main"]
    rows = []
    for _, r in full.iterrows():
        d = dev[(dev["model"] == r["model"]) & (dev["encoder"] == r["encoder"]) & (dev["scaler"] == r["scaler"])].iloc[0]
        rows.append(dict(model=r["model"], encoder=r["encoder"], scaler=r["scaler"], mae_dev=d["mae_mean"], mae_full=r["mae_mean"],
                         rmse_full=r["rmse_mean"], r2_full=r["r2_mean"], medape_full=r["medape_pct_mean"],
                         fit_seconds_full=r["fit_seconds_mean"], n_features=r["n_features_out"], folds=int(r["n_folds"])))
    out = pd.DataFrame(rows)
    out["rank_dev"] = out.groupby("model")["mae_dev"].rank().astype(int)
    out["rank_full"] = out.groupby("model")["mae_full"].rank().astype(int)
    return out


# ----------------------------------------------------------------------------- figures
def fig_heatmaps(agg: pd.DataFrame, metric: str = "mae_mean", name: str = "week2_heatmap_mae.png", title="MAE (EUR)") -> str:
    m = main_grid(agg)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, model in zip(axes, ["knn", "ridge", "svr", "rf"]):
        p = m[m["model"] == model].pivot(index="encoder", columns="scaler", values=metric)
        p = p.loc[["onehot", "ordinal", "target", "frequency"], ["none", "standard", "minmax", "robust"]]
        sns.heatmap(p, annot=True, fmt=",.0f", cmap="viridis_r", ax=ax, cbar=False)
        ax.set_title(f"{MODEL_LABEL[model]} – {title}")
        ax.set_xlabel("Scaler")
        ax.set_ylabel("Encoder" if model == "knn" else "")
        ax.set_xticklabels([SCALER_LABEL[t.get_text()] for t in ax.get_xticklabels()])
        ax.set_yticklabels([ENC_LABEL[t.get_text()] for t in ax.get_yticklabels()], rotation=0)
    fig.tight_layout()
    fig.savefig(F / name, dpi=130)
    plt.close(fig)
    return name


def fig_dims_time(agg: pd.DataFrame) -> str:
    m = main_grid(agg)
    m = m[m["scaler"] == "standard"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    d = m.pivot(index="encoder", columns="model", values="n_features_out").loc[list(ENC_LABEL)]
    d.rename(index=ENC_LABEL, columns=MODEL_LABEL).iloc[:, :1].plot.bar(ax=ax[0], legend=False, color="#3b6ea5", rot=0)
    ax[0].set(title="Số chiều đầu vào mô hình sau mã hóa (main, 15 biến)", ylabel="Số cột")
    for c in ax[0].containers:
        ax[0].bar_label(c, fmt="%d")
    t = m.pivot(index="encoder", columns="model", values="fit_seconds_mean").loc[list(ENC_LABEL)]
    t.rename(index=ENC_LABEL, columns=MODEL_LABEL).plot.bar(ax=ax[1], rot=0)
    ax[1].set_yscale("log")
    ax[1].set(title="Thời gian huấn luyện trung bình mỗi fold (s, trục log)", ylabel="giây")
    fig.tight_layout()
    fig.savefig(F / "week2_dims_time.png", dpi=130)
    plt.close(fig)
    return "week2_dims_time.png"


def fig_log_vs_raw(res: pd.DataFrame, exp: str = "log_target", fname: str = "week2_log_target_effect.png",
                   label: str = "Log của giá") -> str:
    """Heatmaps of the paired % change in MAE and RMSE of experiment `exp` vs the main grid (scaler Standard)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, metric, title in ((ax[0], "mae", "MAE"), (ax[1], "rmse", "RMSE")):
        c = paired_comparison(res, "main", exp, metric)
        p = c.pivot(index="encoder", columns="model", values="rel_change_pct").loc[list(ENC_LABEL)]
        sns.heatmap(p.rename(index=ENC_LABEL, columns=MODEL_LABEL), annot=True, fmt="+.1f", cmap="RdYlGn_r", center=0, ax=a)
        a.set_title(f"{label}: thay đổi {title} (%)\n(âm = tốt hơn dữ liệu gốc)", fontsize=11)
    fig.tight_layout()
    fig.savefig(F / fname, dpi=130)
    plt.close(fig)
    return fname


def fig_best_models(agg: pd.DataFrame) -> str:
    """Best main-grid configuration per model: mean MAE with the fold-to-fold standard deviation as error bar."""
    b = best_per_model(agg)
    labels = [f"{MODEL_LABEL[m]}\n{ENC_LABEL[e]} + {SCALER_LABEL[s]}" for m, e, s in zip(b["model"], b["encoder"], b["scaler"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, b["mae_mean"], yerr=b["mae_std"], capsize=6, color="#3b6ea5")
    for i, (v, s) in enumerate(zip(b["mae_mean"], b["mae_std"])):
        ax.text(i, v + s + 250, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)      # above the error bar, not on it
    ax.set(title="Cấu hình tốt nhất của từng mô hình\n(MAE trung bình 5 fold; thanh sai số = độ lệch chuẩn giữa các fold)",
           ylabel="MAE (EUR)", ylim=(0, float((b["mae_mean"] + b["mae_std"]).max()) * 1.12))
    ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(F / "week2_best_models.png", dpi=130)
    plt.close(fig)
    return "week2_best_models.png"


def fig_model_version(res: pd.DataFrame) -> str:
    cmp = paired_comparison(res, "main", "model_version", "mae")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    p = cmp.pivot(index="encoder", columns="model", values="rel_change_pct").loc[list(ENC_LABEL)]
    sns.heatmap(p.rename(index=ENC_LABEL, columns=MODEL_LABEL), annot=True, fmt="+.1f", cmap="RdYlGn_r", center=0, ax=ax[0])
    ax[0].set_title("Thêm model_version: thay đổi MAE (%) – âm = tốt hơn")
    dims = res[res["experiment"].isin(["main", "model_version"]) & (res["scaler"] == "standard") & (res["model"] == "ridge")] \
        .groupby(["experiment", "encoder"])["n_features_out"].mean().unstack(0).loc[list(ENC_LABEL)]
    dims.rename(index=ENC_LABEL, columns={"main": "9 biến phân loại", "model_version": "+ model_version"}).plot.bar(ax=ax[1], rot=0)
    ax[1].set_yscale("log")
    ax[1].set(title="Số cột sau mã hóa (trục log)", ylabel="số cột")
    fig.tight_layout()
    fig.savefig(F / "week2_model_version_effect.png", dpi=130)
    plt.close(fig)
    return "week2_model_version_effect.png"


def fig_scaler_effect(agg: pd.DataFrame) -> str:
    m = main_grid(agg)
    base = m[m["scaler"] == "none"].set_index(["encoder", "model"])["mae_mean"]
    rows = []
    for sc in ["standard", "minmax", "robust"]:
        s = m[m["scaler"] == sc].set_index(["encoder", "model"])["mae_mean"]
        rows.append(((s - base) / base * 100).rename(sc))
    d = pd.concat(rows, axis=1).reset_index().melt(id_vars=["encoder", "model"], var_name="scaler", value_name="change_pct")
    g = sns.catplot(data=d, x="encoder", y="change_pct", hue="scaler", col="model", kind="bar", height=3.6, aspect=1.0,
                    col_order=["knn", "ridge", "svr", "rf"], order=list(ENC_LABEL))
    g.set_axis_labels("Encoder", "Thay đổi MAE so với No scaling (%)")
    g.set_titles("{col_name}")
    g.figure.suptitle("Ảnh hưởng của scaler (âm = tốt hơn không scale)", y=1.03)
    g.savefig(F / "week2_scaler_effect.png", dpi=130)
    plt.close("all")
    return "week2_scaler_effect.png"


def build_tables_and_figures() -> dict:
    res = load_results()
    agg = aggregate(res)
    agg.to_csv(T / "week2_summary_all.csv", index=False)
    main_grid(agg).sort_values("mae_mean").to_csv(T / "week2_main_grid_ranked.csv", index=False)
    best_per_model(agg).to_csv(T / "week2_best_per_model.csv", index=False)
    enc = effect_table(agg, "encoder")
    sca = effect_table(agg, "scaler")
    enc.to_csv(T / "week2_effect_encoder.csv")
    sca.to_csv(T / "week2_effect_scaler.csv")
    paired_comparison(res, "main", "log_target", "mae").to_csv(T / "week2_log_target_vs_raw_mae.csv", index=False)
    paired_comparison(res, "main", "log_target", "rmse").to_csv(T / "week2_log_target_vs_raw_rmse.csv", index=False)
    if (res["experiment"] == "log_features").any():
        paired_comparison(res, "main", "log_features", "mae").to_csv(T / "week2_log_features_vs_raw_mae.csv", index=False)
        paired_comparison(res, "main", "log_features", "rmse").to_csv(T / "week2_log_features_vs_raw_rmse.csv", index=False)
    paired_comparison(res, "main", "model_version", "mae").to_csv(T / "week2_model_version_vs_main_mae.csv", index=False)
    paired_vs_baseline(res, "encoder", "onehot", {"scaler": "standard"}).to_csv(T / "week2_paired_encoder_vs_onehot_standard.csv", index=False)
    paired_vs_baseline(res, "scaler", "none", {}).to_csv(T / "week2_paired_scaler_vs_none.csv", index=False)
    conf = confirm_comparison()
    if conf is not None:
        conf.to_csv(T / "week2_confirm_full_train.csv", index=False)
    else:   # never leave a stale/partial confirmation table behind
        (T / "week2_confirm_full_train.csv").unlink(missing_ok=True)
    figs = [fig_heatmaps(agg), fig_heatmaps(agg, "medape_pct_mean", "week2_heatmap_medape.png", "MedAPE (%)"),
            fig_heatmaps(agg, "rmse_mean", "week2_heatmap_rmse.png", "RMSE (EUR)"), fig_heatmaps(agg, "r2_mean", "week2_heatmap_r2.png", "R²"),
            fig_dims_time(agg), fig_log_vs_raw(res), fig_model_version(res), fig_scaler_effect(agg), fig_best_models(agg)]
    if (res["experiment"] == "log_features").any():
        figs.append(fig_log_vs_raw(res, "log_features", "week2_log_features_effect.png", "Log của biến đầu vào lệch"))
    return {"agg": agg, "res": res, "enc_effect": enc, "scaler_effect": sca, "figures": figs}


if __name__ == "__main__":
    out = build_tables_and_figures()
    print(out["figures"])
