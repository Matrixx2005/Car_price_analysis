"""Week-1 orchestration: download -> profile -> clean -> validate -> schema -> features -> split -> EDA -> reports.

Run:  python main.py            (re-runs everything from the raw CSV)
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import config, download_dataset, eda, reporting
from src.adapters.autoscout24_adapter import to_common_schema
from src.adapters.schema import CATEGORICAL_FEATURES
from src.cleaning import clean_pipeline
from src.data_validation import has_errors, issues_to_frame, run_validation, write_validation_report
from src.feature_engineering import add_features, feature_audit
from src.load_data import read_raw
from src.profiling import profile_dataset, write_profile_report
from src.split_data import build_group_ids, group_split, save_split


def main() -> int:
    T = config.TABLES
    stats: dict = {"started_at": datetime.now().isoformat(timespec="seconds")}

    print("[1/9] Verify / download raw dataset")
    if download_dataset.main() != 0:
        print("Dataset unavailable -> stopping (no fake data is generated).")
        return 1

    print("[2/9] Profile raw dataset")
    prof = profile_dataset()
    write_profile_report(prof)
    stats["raw"] = prof["stats"]
    raw = prof.pop("df")
    raw_issues = run_validation(raw.rename(columns={}), "raw")

    print("[3/9] Clean (scope: used passenger cars, EUR, valid price)")
    clean, log, value_log, scope_ct = clean_pipeline(raw)
    flog = log.to_frame()
    flog.to_csv(T / "scope_funnel.csv", index=False)
    value_log.to_csv(T / "implausible_values_log.csv", index=False)
    scope_ct.to_csv(T / "used_scope_flag_combinations.csv", index=False)
    clean.to_parquet(config.DATA_INTERIM / "autoscout24_used_clean.parquet", index=False)
    clean_issues = run_validation(clean, "clean")
    write_validation_report({"Raw (trước cleaning)": raw_issues, "Sau cleaning": clean_issues})
    issues_to_frame(clean_issues).to_csv(T / "validation_after_cleaning.csv", index=False)
    if has_errors(clean_issues):
        print("[ERROR] Validation errors remain after cleaning:")
        print(issues_to_frame(clean_issues).query("severity == 'ERROR' and count > 0"))
        return 1
    stats["funnel"] = flog.to_dict("records")
    stats["n_clean"] = len(clean)
    stats["flags"] = {"price_review": int(clean["flag_price_review"].sum()),
                      "future_registration": int(clean["flag_future_registration"].sum()),
                      "year_source": clean["year_source"].value_counts().to_dict(),
                      "mileage_zero": int((clean["mileage_km"] == 0).sum()),
                      "rental_kept": int(clean["is_rental"].fillna(False).sum()),
                      "invalid_vin_format": int(clean["vin"].notna().sum() - (clean["vin"].astype(str).str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")).sum())}
    stats["implausible_values"] = value_log.to_dict("records")
    clean[clean["flag_price_review"]][["make", "model", "price", "mileage_km", "production_year", "country_code"]] \
        .sort_values("price", ascending=False).to_csv(T / "price_review_list.csv", index=False)

    print("[4/9] Common schema + feature engineering")
    df = add_features(to_common_schema(clean))
    df.to_parquet(config.DATA_INTERIM / "autoscout24_used_common.parquet", index=False)
    audit = feature_audit(df)
    audit.to_csv(T / "feature_audit.csv", index=False)

    print("[5/9] Cardinality, make/model, country analysis")
    card = eda.categorical_cardinality(df, CATEGORICAL_FEATURES)
    card.to_csv(T / "categorical_cardinality.csv", index=False)
    mm = eda.make_model_analysis(df)
    for k in ("brand_top20", "model_top20", "models_per_brand", "model_names_shared_across_brands", "model_version_top20"):
        mm[k].to_csv(T / f"{k}.csv", index=False)
    country = eda.country_analysis(df)
    country.to_csv(T / "country_summary.csv", index=False)
    price_tab = eda.price_tables(df)
    price_tab.to_csv(T / "price_statistics.csv", index=False)
    stats["cardinality"] = card.to_dict("records")
    stats["make_model"] = {k: (v if not isinstance(v, (pd.DataFrame, pd.Series)) else None) for k, v in mm.items()}
    stats["make_model"]["brand_spelling_variants"] = mm["brand_spelling_variants"].to_dict()
    stats["country"] = country.to_dict("records")
    stats["price_stats"] = dict(zip(price_tab["statistic"], price_tab["value"]))
    stats["missing_pct_main_features"] = {c: round(df[c].isna().mean() * 100, 2) for c in df.columns if c in audit["column"].values}

    print("[6/9] Leakage-safe split (group-based, seed 42)")
    groups = build_group_ids(df)
    tr_idx, te_idx = group_split(df, groups)
    meta = save_split(df, groups, tr_idx, te_idx)
    stats["split"] = meta
    # What a naive random split would have done (for the report only; NOT used downstream)
    rs_tr, rs_te = train_test_split(df.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    stats["random_split_leak"] = {
        "test_rows_with_same_group_in_train": int(groups.loc[rs_te].isin(set(groups.loc[rs_tr])).sum()),
        "n_test": len(rs_te),
        "group_split_same_metric": int(groups.loc[te_idx].isin(set(groups.loc[tr_idx])).sum()),
    }
    train, test = df.loc[tr_idx], df.loc[te_idx]
    assert meta["group_overlap_train_test"] == 0 and meta["index_overlap_train_test"] == 0
    unseen = eda.unseen_category_rate(train, test, CATEGORICAL_FEATURES)
    unseen.to_csv(T / "unseen_categories_test_vs_train.csv", index=False)
    stats["unseen"] = unseen.to_dict("records")
    pd.read_parquet(config.DATA_PROCESSED / "train.parquet").drop(columns=["listing_id", "group_id"]) \
        .sample(1000, random_state=config.RANDOM_SEED).to_csv(config.DATA_SAMPLES / "train_sample_1000.csv", index=False)

    print("[7/9] Figures")
    figs = eda.price_figures(df) + eda.numeric_figures(df) + eda.categorical_count_figures(df) + [eda.cardinality_figure(card)]
    rel = eda.relationship_tables_and_figures(train)
    figs += rel["figures"]
    stats["figures"] = [f.replace(str(config.PROJECT_ROOT) + "\\", "") for f in figs]
    stats["train_median_price_year_group"] = rel["median_price_by_year_group"].to_dict("records")
    stats["train_corr_with_price"] = rel["corr"]["price"].round(3).to_dict()

    print("[8/9] Lineage")
    stats["lineage"] = {
        "raw_md5": json.loads((config.DATA_RAW / "SOURCE.json").read_text(encoding="utf-8"))["md5_verified"],
        "seed": config.RANDOM_SEED, "test_size": config.TEST_SIZE, "snapshot_date": config.SNAPSHOT_DATE,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (config.REPORTS / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[9/9] Reports")
    reporting.write_all()
    print("Done. n_clean =", len(clean), "| train/test =", meta["n_train"], "/", meta["n_test"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
