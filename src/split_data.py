"""Leakage-safe train/test split.

Evidence (see reports): the data has no listing-time column, and the same physical car appears in
several listings (shared VIN, or identical technical fingerprint when VIN is missing). A plain
random split would put copies of one car in both train and test, so we use GroupShuffleSplit.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.adapters.schema import FEATURES, TARGET
from src.cleaning import is_valid_vin

FINGERPRINT_COLS = ["brand", "model", "model_version", "registration_date", "mileage_km", "power_kw", "country_code"]


def build_group_ids(df: pd.DataFrame) -> pd.Series:
    """Connected components over two linkage keys: (a) valid 17-char VIN, (b) technical fingerprint
    (brand, model, version, registration month, mileage, power, country) when mileage and registration
    date are known. Listings sharing any key land in the same group. Others form singleton groups."""
    n = len(df)
    vin_key = df["vin"].where(is_valid_vin(df["vin"])) if "vin" in df else pd.Series(np.nan, index=df.index)
    fp_ok = df["mileage_km"].notna() & df["registration_date"].notna()
    parts = df[FINGERPRINT_COLS].astype("object").fillna("NA").astype(str)
    fp_key = parts.iloc[:, 0].str.cat([parts[c] for c in FINGERPRINT_COLS[1:]], sep="|").where(fp_ok)
    rows, cols, off = [], [], n
    for key in (vin_key, fp_key):
        codes, uniques = pd.factorize(key)
        m = codes >= 0
        rows.append(np.arange(n)[m])
        cols.append(codes[m] + off)
        off += len(uniques)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(off, off))
    _, labels = connected_components(g, directed=False)
    return pd.Series(labels[:n], index=df.index, name="group_id")


def group_split(df: pd.DataFrame, groups: pd.Series, test_size: float = config.TEST_SIZE,
                seed: int = config.RANDOM_SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=groups))
    return df.index[tr], df.index[te]


def split_summary(df, groups, tr_idx, te_idx) -> dict:
    sizes = groups.value_counts()
    return {
        "n_total": int(len(df)), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "test_fraction": round(len(te_idx) / len(df), 4),
        "n_groups": int(groups.nunique()),
        "n_groups_with_multiple_listings": int((sizes > 1).sum()),
        "listings_in_multi_listing_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "group_overlap_train_test": int(len(set(groups.loc[tr_idx]) & set(groups.loc[te_idx]))),
        "index_overlap_train_test": int(len(set(tr_idx) & set(te_idx))),
        "seed": config.RANDOM_SEED,
        "method": "GroupShuffleSplit (groups = VIN ∪ technical-fingerprint connected components)",
    }


def save_split(df: pd.DataFrame, groups: pd.Series, tr_idx, te_idx, out_dir=config.DATA_PROCESSED) -> dict:
    """Write train/test parquet (features + target + listing_id + group_id) and manifest + metadata.
    Identifying columns (VIN, seller name, description, address) are NOT written."""
    cols = ["listing_id"] + [c for c in FEATURES if c in df.columns] + [TARGET]
    data = df[cols].assign(group_id=groups)
    train, test = data.loc[tr_idx].reset_index(drop=True), data.loc[te_idx].reset_index(drop=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    manifest = pd.concat([train[["listing_id", "group_id"]].assign(split="train"),
                          test[["listing_id", "group_id"]].assign(split="test")])
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    meta = split_summary(df, groups, tr_idx, te_idx)
    meta.update(features=[c for c in FEATURES if c in df.columns], target=TARGET,
                dtypes={c: str(t) for c, t in train.dtypes.items()})
    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
