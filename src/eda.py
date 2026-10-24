"""EDA helpers: tables -> reports/tables, figures -> reports/figures. All numbers come from the data.

Label-free analyses (counts, cardinality, distributions) use the whole cleaned set.
Target-relationship analyses (price by category, correlations) use the TRAIN split only, so no
test information can influence feature decisions.
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PRICE_LABEL = "Giá chào bán (EUR)"


def _save(fig, name: str) -> str:
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def cardinality_level(n: int) -> str:
    return "low (<=10)" if n <= 10 else "medium (11-50)" if n <= 50 else "high (>50)"


# ---------------------------------------------------------------- cardinality
def categorical_cardinality(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df:
            continue
        s = df[c]
        nn = s.dropna()
        vc = nn.astype(str).value_counts()
        n = len(df)
        as_str = nn.astype(str)
        collide = as_str.groupby(as_str.str.casefold()).nunique()
        fmt_bad = int((as_str != as_str.str.strip()).sum() + (as_str.str.contains(r"[^\x20-\x7EÀ-ÿ]", regex=True)).sum())
        rows.append({
            "column": c,
            "n_unique": int(len(vc)),
            "n_non_missing": int(len(nn)),
            "pct_missing": round(s.isna().mean() * 100, 3),
            "level": cardinality_level(len(vc)),
            "top_category": vc.index[0] if len(vc) else None,
            "top_category_share_pct": round(vc.iloc[0] / max(len(nn), 1) * 100, 2) if len(vc) else np.nan,
            "top10_share_pct": round(vc.iloc[:10].sum() / max(len(nn), 1) * 100, 2),
            "n_cat_count_eq1": int((vc == 1).sum()),
            "n_cat_count_lt5": int((vc < 5).sum()),
            "n_cat_count_lt10": int((vc < 10).sum()),
            "pct_cats_lt5": round((vc < 5).mean() * 100, 2) if len(vc) else np.nan,
            "pct_cats_lt10": round((vc < 10).mean() * 100, 2) if len(vc) else np.nan,
            "pct_rows_in_cats_lt5": round(vc[vc < 5].sum() / n * 100, 3),
            "pct_rows_in_cats_lt10": round(vc[vc < 10].sum() / n * 100, 3),
            "n_case_variant_groups": int((collide > 1).sum()),
            "n_values_format_issue": fmt_bad,
        })
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False).reset_index(drop=True)


def unseen_category_rate(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Share of TEST rows whose category never occurs in TRAIN (matters for target/frequency encoders)."""
    rows = []
    for c in cols:
        seen = set(train[c].dropna().astype(str))
        t = test[c].dropna().astype(str)
        rows.append({"column": c, "n_test_non_missing": len(t),
                     "n_unseen_categories": int(t[~t.isin(seen)].nunique()),
                     "pct_test_rows_unseen": round((~t.isin(seen)).mean() * 100, 3)})
    return pd.DataFrame(rows)


def make_model_analysis(df: pd.DataFrame) -> dict:
    out = {}
    b = df["brand"].dropna().astype(str)
    out["brand_top20"] = b.value_counts().head(20).rename_axis("brand").reset_index(name="n")
    fold = b.groupby(b.str.casefold()).nunique()
    out["brand_spelling_variants"] = fold[fold > 1]
    out["brand_n"] = int(b.nunique())
    out["brand_missing"] = int(df["brand"].isna().sum())
    m = df.dropna(subset=["brand", "model"])
    out["model_n"] = int(m["model"].nunique())
    out["model_top20"] = m.groupby(["brand", "model"]).size().sort_values(ascending=False).head(20).reset_index(name="n")
    out["models_per_brand"] = m.groupby("brand")["model"].nunique().sort_values(ascending=False).rename("n_models").reset_index()
    brands_per_model = m.groupby("model")["brand"].nunique()
    out["model_names_shared_across_brands"] = brands_per_model[brands_per_model > 1].sort_values(ascending=False).rename("n_brands").reset_index()
    out["model_missing"] = int(df["model"].isna().sum())
    out["make_model_n"] = int(df["make_model"].nunique()) if "make_model" in df else None
    mv = df["model_version"].dropna().astype(str)
    vc = mv.value_counts()
    tokens = mv.str.split().str.len()
    out["model_version_summary"] = {
        "n_unique": int(vc.size), "n_non_missing": int(len(mv)), "n_missing": int(df["model_version"].isna().sum()),
        "n_singletons": int((vc == 1).sum()), "pct_cats_singleton": round((vc == 1).mean() * 100, 2),
        "n_lt5": int((vc < 5).sum()), "n_lt10": int((vc < 10).sum()),
        "pct_rows_in_cats_lt10": round(vc[vc < 10].sum() / len(mv) * 100, 2),
        "median_tokens": float(tokens.median()), "mean_tokens": round(float(tokens.mean()), 2),
        "pct_with_ge4_tokens": round((tokens >= 4).mean() * 100, 2),
        "pct_with_power_or_displacement": round(mv.str.contains(r"\b\d{2,3}\s?(?:kw|ps|hp|cv)\b|\b\d\.\d\b|\b\d{4}\s?cc\b",
                                                                 case=False, regex=True).mean() * 100, 2),
        "pct_with_drivetrain_or_gearbox_words": round(mv.str.contains(r"\b(?:4x4|awd|xdrive|quattro|4matic|dsg|tiptronic|pdk|automatic|aut\.?)\b",
                                                                       case=False, regex=True).mean() * 100, 2),
    }
    out["model_version_top20"] = vc.head(20).rename_axis("model_version").reset_index(name="n")
    return out


def country_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Scope decision table (descriptive; uses price only to describe markets, not to select features)."""
    g = df.groupby("country_code", dropna=False)["price"]
    t = g.agg(n="size", median_price="median", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9)).reset_index()
    t["share_pct"] = (t["n"] / t["n"].sum() * 100).round(2)
    t["flag_small_sample"] = t["n"] < 1000
    return t.sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- price / numeric (full cleaned set)
def price_tables(df: pd.DataFrame) -> pd.DataFrame:
    p = df["price"]
    q = p.quantile([.001, .01, .05, .25, .5, .75, .95, .99, .999])
    t = pd.DataFrame({"statistic": ["count", "mean", "std", "min"] + [f"p{x*100:g}" for x in q.index] + ["max", "skew", "skew_log1p"],
                      "value": [len(p), p.mean(), p.std(), p.min()] + list(q.values) + [p.max(), p.skew(), np.log1p(p).skew()]})
    return t


def price_figures(df: pd.DataFrame) -> list[str]:
    paths = []
    p = df["price"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(p[p <= p.quantile(.99)], bins=80, color="#3b6ea5")
    ax[0].set(title="Phân phối giá (đến phân vị 99%, trục tuyến tính)", xlabel=PRICE_LABEL, ylabel="Số tin đăng")
    ax[1].hist(np.log10(p), bins=80, color="#3b6ea5")
    ax[1].set(title="Phân phối log10(giá) – toàn bộ dữ liệu", xlabel="log10(giá EUR)", ylabel="Số tin đăng")
    paths.append(_save(fig, "price_histograms.png"))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.boxplot(p, vert=False, showfliers=True, flierprops=dict(markersize=2, alpha=.3))
    ax.set_xscale("log")
    ax.set(title="Boxplot giá (trục log) – các điểm xa là xe sang hợp lệ, không bị loại", xlabel=PRICE_LABEL)
    paths.append(_save(fig, "price_boxplot.png"))
    return paths


def numeric_figures(df: pd.DataFrame) -> list[str]:
    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    yr = df["production_year"].dropna()
    ax[0].hist(yr, bins=np.arange(yr.min(), yr.max() + 2) - .5, color="#3b6ea5")
    ax[0].set(title="Phân phối năm sản xuất", xlabel="Năm", ylabel="Số tin đăng")
    km = df["mileage_km"].dropna()
    ax[1].hist(km[km <= km.quantile(.99)], bins=80, color="#3b6ea5")
    ax[1].set(title="Phân phối số km (đến phân vị 99%)", xlabel="Số km", ylabel="Số tin đăng")
    paths.append(_save(fig, "year_mileage_distribution.png"))
    return paths


def categorical_count_figures(df: pd.DataFrame) -> list[str]:
    paths = []
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for a, (col, title, k) in zip(ax.ravel(), [("brand", "Top hãng xe", 15), ("make_model", "Top mẫu xe (hãng + mẫu)", 15),
                                               ("fuel_type", "Nhiên liệu", 10), ("transmission", "Hộp số", 5),
                                               ("country_code", "Quốc gia", 10), ("body_type", "Kiểu thân xe", 10)]):
        vc = df[col].value_counts().head(k)[::-1]
        a.barh(vc.index.astype(str), vc.values, color="#3b6ea5")
        a.set(title=title, xlabel="Số tin đăng")
    paths.append(_save(fig, "categorical_counts.png"))
    return paths


def cardinality_figure(card: pd.DataFrame) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    c = card.sort_values("n_unique")
    ax[0].barh(c["column"], c["n_unique"], color="#3b6ea5")
    ax[0].set_xscale("log")
    ax[0].set(title="Số danh mục duy nhất (trục log)", xlabel="Số danh mục")
    ax[1].barh(c["column"], c["pct_rows_in_cats_lt10"], color="#c0504d")
    ax[1].set(title="% bản ghi thuộc danh mục xuất hiện < 10 lần", xlabel="% bản ghi")
    return _save(fig, "cardinality.png")


# ---------------------------------------------------------------- target relationships (TRAIN only)
def _grouped_median(train: pd.DataFrame, col: str, min_n: int, top: int | None = None) -> pd.DataFrame:
    g = train.groupby(col)["price"].agg(n="size", median="median").reset_index()
    g = g[g["n"] >= min_n].sort_values("median", ascending=False)
    return g.head(top) if top else g


def _bar_with_n(ax, g, col, title, horizontal=True):
    labels = [f"{a} (n={n:,})" for a, n in zip(g[col].astype(str), g["n"])]
    ax.barh(labels[::-1], g["median"].values[::-1], color="#3b6ea5")
    ax.set(title=title, xlabel="Giá trung vị (EUR)")


def relationship_tables_and_figures(train: pd.DataFrame) -> dict:
    t = {}
    t["median_price_by_brand"] = _grouped_median(train, "brand", min_n=100)
    t["median_price_by_fuel_type"] = _grouped_median(train, "fuel_type", min_n=30)
    t["median_price_by_transmission"] = _grouped_median(train, "transmission", min_n=30)
    t["median_price_by_country"] = _grouped_median(train, "country_code", min_n=30)
    tr = train.copy()
    tr["year_group"] = pd.cut(tr["production_year"], [1949, 2004, 2009, 2014, 2017, 2019, 2021, 2023, 2026],
                              labels=["<=2004", "2005-09", "2010-14", "2015-17", "2018-19", "2020-21", "2022-23", "2024-26"])
    tr["mileage_group"] = pd.cut(tr["mileage_km"], [-1, 1000, 10000, 30000, 60000, 100000, 150000, 200000, 1e7],
                                 labels=["<=1k", "1-10k", "10-30k", "30-60k", "60-100k", "100-150k", "150-200k", ">200k"])
    for col, name in (("year_group", "median_price_by_year_group"), ("mileage_group", "median_price_by_mileage_group")):
        g = tr.groupby(col, observed=True)["price"].agg(n="size", median="median").reset_index()
        t[name] = g
    for name, g in t.items():
        g.to_csv(config.TABLES / f"{name}.csv", index=False)

    paths = []
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    _bar_with_n(ax[0, 0], t["median_price_by_brand"].head(25), "brand", "Giá trung vị theo hãng (n >= 100, train)")
    _bar_with_n(ax[0, 1], t["median_price_by_fuel_type"], "fuel_type", "Giá trung vị theo nhiên liệu (n >= 30)")
    _bar_with_n(ax[0, 2], t["median_price_by_transmission"], "transmission", "Giá trung vị theo hộp số (n >= 30)")
    for a, key, col, title in ((ax[1, 0], "median_price_by_year_group", "year_group", "Giá trung vị theo nhóm năm"),
                               (ax[1, 1], "median_price_by_mileage_group", "mileage_group", "Giá trung vị theo nhóm km")):
        g = t[key]
        a.bar([f"{x}\n(n={n:,})" for x, n in zip(g[col].astype(str), g["n"])], g["median"], color="#3b6ea5")
        a.set(title=title, ylabel="Giá trung vị (EUR)")
        a.tick_params(axis="x", labelsize=8)
    order = t["median_price_by_country"]["country_code"].tolist()
    sub = train[train["country_code"].isin(order)]
    sns.boxplot(data=sub, x="country_code", y="price", order=order, ax=ax[1, 2], fliersize=1)
    ax[1, 2].set_yscale("log")
    ax[1, 2].set(title="Phân bố giá theo quốc gia (trục log)", xlabel="Quốc gia", ylabel=PRICE_LABEL)
    paths.append(_save(fig, "price_relationships.png"))

    s = train.sample(min(len(train), 20000), random_state=config.RANDOM_SEED)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].scatter(s["production_year"], s["price"], s=3, alpha=.25)
    ax[0].set_yscale("log")
    ax[0].set(title="Giá theo năm sản xuất (mẫu 20.000 dòng train)", xlabel="Năm sản xuất", ylabel=PRICE_LABEL)
    ax[1].scatter(s["mileage_km"], s["price"], s=3, alpha=.25)
    ax[1].set_yscale("log")
    ax[1].set_xlim(0, 400000)
    ax[1].set(title="Giá theo số km (mẫu 20.000 dòng train)", xlabel="Số km", ylabel=PRICE_LABEL)
    paths.append(_save(fig, "price_vs_year_mileage.png"))

    num = train[["price", "production_year", "vehicle_age", "mileage_km", "power_kw", "engine_cc", "nr_seats"]].copy()
    num["log10_price"] = np.log10(num["price"])
    corr = num.corr(method="spearman")
    corr.to_csv(config.TABLES / "numeric_spearman_corr_train.csv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Tương quan Spearman (train) – chỉ mô tả, không dùng để loại biến")
    paths.append(_save(fig, "numeric_correlation.png"))
    t["figures"] = paths
    t["corr"] = corr
    return t
