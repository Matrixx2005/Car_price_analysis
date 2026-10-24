import numpy as np
import pandas as pd

from src.split_data import build_group_ids, group_split, save_split, split_summary


def _frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(n)],
        "brand": rng.choice(["BMW", "Audi"], n), "model": rng.choice(["A", "B", "C"], n),
        "model_version": [f"v{i}" for i in range(n)],
        "registration_date": ["2020-01-01"] * n, "mileage_km": rng.integers(1000, 200000, n).astype(float),
        "power_kw": 100.0, "country_code": "DE", "price": rng.integers(5000, 90000, n).astype(float),
        "vin": pd.Series([None] * n, dtype="object"),
    })
    # 40 cars each listed 3 times with the same VIN (different ids); VINs valid (17 chars)
    for k in range(40):
        idx = [k * 3, k * 3 + 1, k * 3 + 2]
        df.loc[idx, "vin"] = f"WBA{k:014d}"
    # fingerprint twins without VIN
    df.loc[300:304, ["brand", "model", "model_version", "mileage_km"]] = ["BMW", "A", "same", 12345.0]
    return df


def test_train_test_disjoint_indices():                                    # 12
    df = _frame()
    g = build_group_ids(df)
    tr, te = group_split(df, g)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(df)


def test_no_group_leakage(  ):                                             # 13
    df = _frame()
    g = build_group_ids(df)
    tr, te = group_split(df, g)
    assert set(g.loc[tr]).isdisjoint(set(g.loc[te]))
    assert g.iloc[0] == g.iloc[1] == g.iloc[2]                      # same VIN -> same group
    assert g.loc[300:304].nunique() == 1                            # same fingerprint -> same group
    assert split_summary(df, g, tr, te)["group_overlap_train_test"] == 0


def test_same_seed_same_split():                                           # 14
    df = _frame()
    g = build_group_ids(df)
    a = group_split(df, g, seed=42)
    b = group_split(df, g, seed=42)
    c = group_split(df, g, seed=7)
    assert list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])
    assert list(a[1]) != list(c[1])


def test_split_sizes_and_saved_files(tmp_path):                            # 16
    from src.adapters.schema import FEATURES
    df = _frame()
    for c in FEATURES:
        if c not in df:
            df[c] = 1.0
    g = build_group_ids(df)
    tr, te = group_split(df, g)
    meta = save_split(df, g, tr, te, out_dir=tmp_path)
    train, test = pd.read_parquet(tmp_path / "train.parquet"), pd.read_parquet(tmp_path / "test.parquet")
    assert len(train) + len(test) == len(df) == meta["n_total"]
    assert abs(len(test) / len(df) - 0.2) < 0.06
    assert "vin" not in train.columns and "seller_company_name" not in train.columns
    assert "price" in train.columns and set(train["listing_id"]).isdisjoint(test["listing_id"])
