import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_raw(n: int = 12) -> pd.DataFrame:
    """Small synthetic raw frame with AutoScout24 column names (tests only; never used in reports)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": [f"id{i}" for i in range(n)],
        "price": [10000 + 1000 * i for i in range(n)],
        "price_currency": ["EUR"] * n,
        "make": ["BMW", "Audi", "BMW", "Porsche"] * (n // 4),
        "model": ["X5", "A4", "X1", "911"] * (n // 4),
        "model_version": [f"v{i}" for i in range(n)],
        "mileage_km_raw": [50000.0 + 1000 * i for i in range(n)],
        "mileage_km": [f"{50000 + 1000 * i:,} km" for i in range(n)],
        "registration_date": ["2018-05-01"] * n,
        "production_year": [np.nan] * n,
        "vehicle_type": ["Car"] * n,
        "body_type": ["Sedan"] * n,
        "is_used": [True] * n, "is_new": [False] * n, "is_preregistered": [False] * n, "is_rental": [False] * n,
        "offer_type": ["U"] * n,
        "country_code": ["DE", "IT", "NL", "DE"] * (n // 4),
        "seller_type": ["Dealer"] * n,
        "fuel_category": ["Gasoline"] * n,
        "transmission": ["Manual"] * n,
        "nr_seats": [5.0] * n, "power_kw": [100.0] * n, "cylinders_volume_cc": [2000.0] * n,
        "vin": pd.Series([None] * n, dtype="object"), "seller_company_name": ["x"] * n,
    })
    for c in ("is_used", "is_new", "is_preregistered", "is_rental"):
        df[c] = df[c].astype("boolean")   # same nullable dtype the real pipeline produces
    return df


@pytest.fixture
def raw():
    return make_raw()
