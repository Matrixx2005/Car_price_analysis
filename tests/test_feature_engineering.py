import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from src import cleaning, feature_engineering, split_data
from src.adapters import autoscout24_adapter
from src.adapters.autoscout24_adapter import to_common_schema
from src.adapters.schema import COMMON_COLUMNS, FEATURES, TARGET
from src.cleaning import clean_pipeline
from src.feature_engineering import add_features, make_make_model, select_xy


def test_adapter_maps_columns(raw):                                        # 8
    clean, *_ = clean_pipeline(raw)
    out = to_common_schema(clean)
    for new in ("brand", "fuel_type", "engine_cc", "currency", "listing_id"):
        assert new in out.columns
    for old in ("make", "fuel_category", "cylinders_volume_cc", "price_currency", "id"):
        assert old not in out.columns
    assert out["brand"].tolist() == clean["make"].tolist()
    assert (out["vehicle_condition"] == "used").all()
    assert out.columns[0] == "listing_id"


def test_adapter_does_not_invent_columns():
    out = to_common_schema(pd.DataFrame({"make": ["BMW"], "price": [1.0]}))
    assert set(out.columns) == {"brand", "price"}


def test_make_model_exact_and_missing_safe():                              # 9
    b = pd.Series(["BMW", "Audi", None, "Volvo"])
    m = pd.Series(["X5", "X5", "A3", None])
    out = make_make_model(b, m)
    assert out.iloc[0] == "BMW X5" and out.iloc[1] == "Audi X5"      # same model, different brands stay distinct
    assert out.iloc[2] is np.nan or pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


def test_vehicle_age_uses_snapshot_year_not_today():
    df = pd.DataFrame({"production_year": [2020.0, 2026.0, np.nan]})
    out = feature_engineering.add_vehicle_age(df, reference_year=2025)
    assert out["vehicle_age"].iloc[0] == 5 and out["vehicle_age"].iloc[1] == 0 and np.isnan(out["vehicle_age"].iloc[2])


def test_target_not_in_features(raw):                                      # 11
    assert TARGET not in FEATURES
    clean, *_ = clean_pipeline(raw)
    df = add_features(to_common_schema(clean))
    X, y = select_xy(df)
    assert TARGET not in X.columns and len(X) == len(y)
    with pytest.raises(ValueError):
        select_xy(df, features=FEATURES + [TARGET])


def test_leakage_columns_not_features():
    for c in ("price_net", "price_vat_rate", "price_tax_deductible", "price_negotiable", "vin", "listing_id", "description"):
        assert c not in FEATURES
    assert {"listing_id", "currency", "vehicle_condition"}.isdisjoint(FEATURES)
    assert set(FEATURES) - set(COMMON_COLUMNS) >= {"vehicle_age", "make_model"}


def test_no_fit_on_full_data_in_week1():                                   # 15
    """Week-1 modules must not import/call any fitted transformer (imputer/scaler/encoder)."""
    banned = {"StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer", "KNNImputer", "OneHotEncoder",
              "OrdinalEncoder", "TargetEncoder", "LabelEncoder", "fit_transform", "fit"}
    for mod in (cleaning, feature_engineering, split_data, autoscout24_adapter):
        tree = ast.parse(inspect.getsource(mod))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
            | {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not (banned & used), f"{mod.__name__} uses {banned & used}"
