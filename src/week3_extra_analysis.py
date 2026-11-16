"""Tables and figures of the week-3 supplement (boosting on the full train split, diagnostics, learning curves).
Reads only the CSV/JSON results written by src/week3_extra.py (train split only)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.week3_experiments import aggregate3  # noqa: E402
from src.week3_extra import CURVES, DIAG_METRICS, DIAG_PERM, HGB_BEST, HGB_RESULTS  # noqa: E402

T, F = config.TABLES, config.FIGURES
NAME_LABEL = {"knn_default": "KNN (mặc định)", "knn_tuned": "KNN (tinh chỉnh)", "ridge_default": "Ridge (mặc định, giá gốc)", "ridge_tuned": "Ridge (tinh chỉnh)",
              "svr_tuned": "SVR (tinh chỉnh)", "hgb_default": "Boosting (mặc định)",
              "rf_default": "Random Forest (mặc định)", "rf_tuned": "Random Forest (tinh chỉnh)", "hgb_best": "Boosting (tinh chỉnh)"}
FEATURE_LABEL = {"identity (brand+model[+version])": "ĐỊNH DANH xe (hãng+mẫu)", "age (year+vehicle_age)": "TUỔI xe (năm+tuổi)"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=130)
    plt.close(fig)
    return name


def label(n: str) -> str:
    return NAME_LABEL.get(n, n)


# ------------------------------------------------------------------ boosting
def hgb_tables() -> dict:
    res = pd.read_csv(HGB_RESULTS)
    agg = aggregate3(res)
    cols = ["config_id", "stage", "encoder", "log_target", "feature_set", "rare_min_count", "params", "mae_mean", "mae_std", "rmse_mean", "r2_mean",
            "medape_pct_mean", "fit_seconds_mean", "n_features_out"]
    agg[cols].sort_values("mae_mean").to_csv(T / "week3_hgb_summary.csv", index=False)
    best = json.loads(HGB_BEST.read_text(encoding="utf-8"))
    final_id, enc_default_id = best["final"]["config_id"], best["default_same_encoder"]["config_id"]

    # full-train CV of the other models at their default settings (week-2 confirmation, identical folds)
    conf = pd.read_csv(T / "week2_confirm_fold_results.csv")
    rows = []
    for m in ("knn", "ridge", "rf"):
        c = conf[conf["model"] == m]
        c = c[c["encoder"] == "onehot"]
        rows.append({"model": f"{m}_default", "source": "tuần 2, mặc định, toàn train", "mae_mean": c["mae"].mean(), "mae_std": c["mae"].std(),
                     "rmse_mean": c["rmse"].mean(), "r2_mean": c["r2"].mean(), "medape_mean": c["medape_pct"].mean(), "fit_seconds": c["fit_seconds"].mean(),
                     "folds": c.sort_values("fold")["mae"].to_numpy()})
    for nm, cid, src in (("hgb_default", enc_default_id, "tuần 3, mặc định, toàn train"), ("hgb_best", final_id, "tuần 3, đã tinh chỉnh, toàn train")):
        s = res[res["config_id"] == cid].sort_values("fold")
        rows.append({"model": nm, "source": src, "mae_mean": s["mae"].mean(), "mae_std": s["mae"].std(), "rmse_mean": s["rmse"].mean(), "r2_mean": s["r2"].mean(),
                     "medape_mean": s["medape_pct"].mean(), "fit_seconds": s["fit_seconds"].mean(), "folds": s["mae"].to_numpy()})
    cmp_ = pd.DataFrame(rows)
    hb = cmp_.set_index("model").loc["hgb_best", "folds"]
    cmp_["hgb_best_wins_folds"] = [int((hb < f).sum()) if m != "hgb_best" else np.nan for m, f in zip(cmp_["model"], cmp_["folds"])]
    cmp_.drop(columns="folds").to_csv(T / "week3_hgb_vs_models_fullcv.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    t = agg[agg["stage"].isin(["hgb_encoder", "hgb_tune", "hgb_version"]) & (agg["n_folds"] == 5)].sort_values("mae_mean").reset_index(drop=True)
    ax[0].scatter(t.index, t["mae_mean"], c=np.where(t["stage"] == "hgb_encoder", "gold", "#3b6ea5"), s=25, edgecolor="k", linewidth=0.3)
    ax[0].set(title=f"Boosting: {len(t)} cấu hình (CV 5 fold trên toàn train)", xlabel="cấu hình (xếp theo MAE)", ylabel="MAE CV (EUR)")
    c = cmp_.copy()
    ax[1].bar([label(m) for m in c["model"]], c["mae_mean"], yerr=c["mae_std"], capsize=5,
              color=["#9bb7d4" if "default" in m else "#c0504d" for m in c["model"]])
    ax[1].set(title="MAE CV trên toàn train (cùng các fold)", ylabel="MAE (EUR)")
    ax[1].tick_params(axis="x", rotation=25, labelsize=8)
    _save(fig, "week3_hgb.png")
    return {"agg": agg, "cmp": cmp_.drop(columns="folds"), "best": best}


# ------------------------------------------------------------------ diagnostics
def diag_summary() -> pd.DataFrame:
    d = pd.read_csv(DIAG_METRICS)
    g = d.groupby(["name", "data"])
    out = g[["mae_train", "mae_valid", "r2_train", "r2_valid", "medape_pct_train", "medape_pct_valid", "fit_seconds"]].mean().reset_index()
    out = out.rename(columns={"medape_pct_train": "medape_train", "medape_pct_valid": "medape_valid"})
    out["mae_std_valid"] = g["mae_valid"].std().to_numpy()
    out["valid_over_train_mae"] = out["mae_valid"] / out["mae_train"]
    out["r2_gap"] = out["r2_train"] - out["r2_valid"]
    out["n_folds"] = g["fold"].nunique().to_numpy()
    out = out.sort_values(["data", "mae_valid"]).reset_index(drop=True)
    out.to_csv(T / "week3_diag_summary.csv", index=False)
    return out


def importance_summary() -> pd.DataFrame:
    p = pd.read_csv(DIAG_PERM)
    per_fold = p.groupby(["name", "data", "fold", "feature", "kind"])["delta_mae"].mean().reset_index()
    g = per_fold.groupby(["name", "data", "feature", "kind"])["delta_mae"]
    out = g.mean().rename("delta_mae").reset_index()
    out["delta_mae_std_folds"] = g.std().to_numpy()
    out["folds_positive"] = g.apply(lambda s: int((s > 0).sum())).to_numpy()
    single = out[out["kind"] == "single"].copy()
    tot = single.groupby(["name", "data"])["delta_mae"].transform(lambda s: s.clip(lower=0).sum())
    out.loc[single.index, "share_pct"] = single["delta_mae"].clip(lower=0) / tot * 100
    out["rank_in_model"] = out.groupby(["name", "data"])["delta_mae"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["data", "name", "rank_in_model"]).reset_index(drop=True)
    out.to_csv(T / "week3_importance_summary.csv", index=False)
    return out


def fig_diag(ds: pd.DataFrame) -> str:
    d = ds[ds["data"] == "dev20k"].sort_values("mae_valid")
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    ax[0].bar(x - 0.2, d["mae_train"], 0.4, label="train (trong fold)", color="#9bb7d4")
    ax[0].bar(x + 0.2, d["mae_valid"], 0.4, yerr=d["mae_std_valid"], capsize=4, label="validation", color="#c0504d")
    ax[0].set_xticks(x, [label(n) for n in d["name"]], rotation=25, fontsize=8)
    ax[0].set(title="MAE trên train và validation (5 fold, tập con 20k)", ylabel="MAE (EUR)")
    ax[0].legend()
    r = d[d["name"] != "knn_default"]          # KNN with distance weights has train MAE ~0 (each point is its own neighbour): ratio meaningless
    ax[1].bar([label(n) for n in r["name"]], r["valid_over_train_mae"], color="#3b6ea5")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set(title="Tỷ lệ MAE validation / MAE train (1 = không có khoảng cách)\n(không vẽ KNN mặc định: MAE train ≈ 0 nên tỷ lệ vô nghĩa)", ylabel="tỷ lệ")
    ax[1].title.set_fontsize(10)
    ax[1].tick_params(axis="x", rotation=25, labelsize=8)
    return _save(fig, "week3_diag_train_valid.png")


def fig_importance(imp: pd.DataFrame) -> str:
    d = imp[(imp["data"] == "dev20k") & (imp["kind"] == "single")]
    piv = d.pivot(index="feature", columns="name", values="share_pct").fillna(0)
    piv = piv[[c for c in NAME_LABEL if c in piv.columns]]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    j = imp[(imp["data"] == "dev20k") & (imp["kind"] == "joint")].pivot(index="feature", columns="name", values="delta_mae")
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    sns.heatmap(piv.rename(columns=label), annot=True, fmt=".0f", cmap="viridis_r", ax=ax[0], cbar_kws={"label": "% tổng mức tăng MAE"})
    ax[0].set(title="Độ quan trọng hoán vị của từng cột gốc (% của tổng; tập con 20k, validation)", ylabel="")
    ax[0].tick_params(axis="x", rotation=30, labelsize=8)
    jj = j[[c for c in NAME_LABEL if c in j.columns]].rename(index=FEATURE_LABEL, columns=label)
    sns.heatmap(jj, annot=True, fmt=",.0f", cmap="viridis_r", ax=ax[1], cbar_kws={"label": "tăng MAE (EUR)"})
    ax[1].set(title="Hoán vị cả nhóm cột cùng thông tin (tăng MAE, EUR)", ylabel="")
    ax[1].tick_params(axis="x", rotation=30, labelsize=8)
    return _save(fig, "week3_importance.png")


# ------------------------------------------------------------------ learning curves
def curves_summary() -> pd.DataFrame:
    c = pd.read_csv(CURVES)
    g = c.groupby(["name", "fraction"])
    out = g[["n_train_used", "mae_train", "mae_valid", "medape_valid", "fit_seconds"]].mean().reset_index()
    out["mae_valid_std"] = g["mae_valid"].std().to_numpy()
    out["n_folds"] = g["fold"].nunique().to_numpy()
    out.to_csv(T / "week3_learning_curves_summary.csv", index=False)
    # paired fold comparison of the two forests at each fraction
    rows = []
    for fr, d in c.groupby("fraction"):
        a = d[d["name"] == "rf_default"].set_index("fold")["mae_valid"]
        b = d[d["name"] == "rf_tuned"].set_index("fold")["mae_valid"]
        common = a.index.intersection(b.index)
        if len(common):
            rows.append({"fraction": fr, "n_train_used": d[d["name"] == "rf_default"]["n_train_used"].mean(), "rf_default_mae": a.loc[common].mean(),
                         "rf_tuned_mae": b.loc[common].mean(), "tuned_minus_default": (b.loc[common] - a.loc[common]).mean(),
                         "folds_tuned_better": int((b.loc[common] < a.loc[common]).sum()), "n_folds": len(common)})
    pd.DataFrame(rows).sort_values("fraction").to_csv(T / "week3_rf_crossover.csv", index=False)
    return out


def fig_curves(cs: pd.DataFrame) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for n, d in cs.groupby("name"):
        d = d.sort_values("n_train_used")
        ls = "-" if "tuned" in n or n == "hgb_best" else "--"
        ax[0].errorbar(d["n_train_used"], d["mae_valid"], yerr=d["mae_valid_std"], ls=ls, marker="o", ms=4, capsize=2, label=label(n))
    ax[0].set(xscale="log", title="Đường cong học: MAE validation theo số dòng huấn luyện (toàn train, 5 fold)", xlabel="số dòng huấn luyện (trục log)",
              ylabel="MAE validation (EUR)")
    ax[0].legend(fontsize=8)
    cr = pd.read_csv(T / "week3_rf_crossover.csv")
    ax[1].plot(cr["n_train_used"], cr["rf_default_mae"], "o--", label=label("rf_default"))
    ax[1].plot(cr["n_train_used"], cr["rf_tuned_mae"], "o-", label=label("rf_tuned"), color="#c0504d")
    ax[1].set(xscale="log", title="Hai cấu hình Random Forest: ai tốt hơn ở từng cỡ dữ liệu?", xlabel="số dòng huấn luyện (trục log)", ylabel="MAE validation (EUR)")
    ax[1].legend()
    for _, r in cr.iterrows():
        ax[1].annotate(f"{int(r['folds_tuned_better'])}/{int(r['n_folds'])}", (r["n_train_used"], r["rf_tuned_mae"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    return _save(fig, "week3_learning_curves.png")


def all_tables_and_figures() -> dict:
    out = {"hgb": hgb_tables() if HGB_RESULTS.exists() and HGB_BEST.exists() else None, "figures": ["week3_hgb.png"]}
    if DIAG_METRICS.exists() and DIAG_PERM.exists():
        out["diag"], out["imp"] = diag_summary(), importance_summary()
        out["figures"] += [fig_diag(out["diag"]), fig_importance(out["imp"])]
    if CURVES.exists():
        out["curves"] = curves_summary()
        out["figures"].append(fig_curves(out["curves"]))
    return out


if __name__ == "__main__":
    print(all_tables_and_figures()["figures"])
