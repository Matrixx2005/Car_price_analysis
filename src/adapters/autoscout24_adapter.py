"""AutoScout24 -> common schema. Only maps columns that really exist; invents nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.adapters.schema import COMMON_COLUMNS

RENAME_MAP = {
    "make": "brand",
    "fuel_category": "fuel_type",
    "cylinders_volume_cc": "engine_cc",
    "price_currency": "currency",
    "id": "listing_id",
}


def derive_vehicle_condition(df: pd.DataFrame) -> pd.Series | None:
    """'used' / 'new' / 'preregistered' / 'unknown' from the three real flags (no guessing from mileage)."""
    if not {"is_used", "is_new"}.issubset(df.columns):
        return None
    used = df["is_used"].fillna(False).astype(bool)
    new = df["is_new"].fillna(False).astype(bool)
    pre = df["is_preregistered"].fillna(False).astype(bool) if "is_preregistered" in df else pd.Series(False, index=df.index)
    cond = np.select([used & ~new & ~pre, new & ~used, pre & ~new], ["used", "new", "preregistered"], "unknown")
    return pd.Series(cond, index=df.index)


def to_common_schema(df: pd.DataFrame, keep_extra: bool = True) -> pd.DataFrame:
    """Rename mapped columns present in df and add vehicle_condition; common columns come first."""
    out = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    cond = derive_vehicle_condition(df)
    if cond is not None:
        out["vehicle_condition"] = cond
    first = [c for c in COMMON_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in first] if keep_extra else []
    return out[first + rest]
