"""Cleaning pipeline (raw AutoScout24 column names in, raw names out).

Principles: never touch the raw file; every row-removing rule is logged with count + reason;
valid expensive cars are NOT dropped; implausible technical values become NaN (row is kept);
no imputation here (imputation statistics are learned on train in the week-2 pipeline).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "undefined"}
LONG_TEXT_COLUMNS = {"description", "equipment_comfort", "equipment_entertainment",
                     "equipment_extra", "equipment_safety"}
BOOL_COLUMNS = ["is_used", "is_new", "is_preregistered", "is_rental", "had_accident",
                "has_full_service_history", "non_smoking", "price_negotiable",
                "price_tax_deductible", "seller_is_dealer", "has_particle_filter"]
COMMERCIAL_BODY_TYPES = {"transporter", "panel van", "van-high roof", "box", "flatbed van",
                         "flatbed+tarpaulin"}

# Rules (documented in reports/cleaning_log.md)
MIN_VALID_PRICE = 100          # EUR; price=1 is a placeholder ("price on request")
MAX_VALID_MILEAGE = 1_000_000  # km
VALID_SEATS = (1, 9)
VALID_POWER_KW = (10, 1500)
MAX_VALID_ENGINE_CC = 10_000
VALID_YEAR = (1950, 2026)      # 2026 allowed: model year can exceed snapshot year
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

KEEP_COLUMNS = [
    "id", "vin", "price", "price_currency", "price_net", "price_vat_rate", "price_tax_deductible",
    "price_negotiable", "make", "model", "model_version", "mileage_km", "registration_date",
    "production_year", "registration_year", "year_source", "vehicle_type", "body_type", "nr_seats",
    "nr_doors", "body_color", "power_kw", "transmission", "drive_train", "cylinders_volume_cc",
    "fuel_category", "is_used", "is_new", "is_preregistered", "is_rental", "offer_type",
    "country_code", "seller_type", "seller_company_name", "nr_prev_owners", "had_accident",
    "flag_price_review", "flag_future_registration",
]


@dataclass
class CleaningLog:
    steps: list = field(default_factory=list)

    def add(self, step: str, rule: str, before: int, after: int, reason: str = "") -> None:
        self.steps.append({"step": step, "rule": rule, "rows_before": before, "rows_after": after,
                           "rows_removed": before - after, "reason": reason})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------- low-level normalisers ----------
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in out.columns]
    return out


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/collapse whitespace and turn missing-value tokens into NaN (short text columns only)."""
    out = df.copy()
    for c in out.columns:
        if c in LONG_TEXT_COLUMNS or not (pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object):
            continue
        s = out[c].astype("object")
        s = s.where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        s = s.mask(s.notna() & s.astype(str).str.lower().isin(MISSING_TOKENS))
        out[c] = s
    return out


def to_bool(s: pd.Series) -> pd.Series:
    """True/False, 1/0, 'true'/'false', 'yes'/'no' -> nullable boolean; anything else -> <NA>."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    mapping = {"true": True, "1": True, "1.0": True, "yes": True, "y": True, "t": True,
               "false": False, "0": False, "0.0": False, "no": False, "n": False, "f": False}
    return s.astype("object").map(lambda v: mapping.get(str(v).strip().lower(), pd.NA) if pd.notna(v) else pd.NA).astype("boolean")


def parse_numeric(s: pd.Series) -> pd.Series:
    """'12,500' / '9,900 km' / '1.945 kg'-style strings -> float (thousand commas removed)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = s.astype("object").map(lambda v: re.sub(r"[^0-9.\-]", "", str(v).replace(",", "")) if pd.notna(v) else np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def is_valid_vin(s: pd.Series) -> pd.Series:
    return s.astype("object").map(lambda v: bool(VIN_RE.match(str(v).upper())) if pd.notna(v) else False).astype(bool)


# ---------- main steps ----------
def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_strings(normalize_column_names(df))
    for c in BOOL_COLUMNS:
        if c in out:
            out[c] = to_bool(out[c])
    out["price"] = parse_numeric(out["price"])
    # mileage: use the numeric raw column when present, fall back to parsing the text column
    if "mileage_km_raw" in out:
        m = pd.to_numeric(out["mileage_km_raw"], errors="coerce")
        if "mileage_km" in out:
            m = m.fillna(parse_numeric(out["mileage_km"]))
        out["mileage_km"] = m
    elif "mileage_km" in out:
        out["mileage_km"] = parse_numeric(out["mileage_km"])
    for c in ("power_kw", "cylinders_volume_cc", "nr_seats", "nr_doors", "production_year", "nr_prev_owners", "price_net"):
        if c in out:
            out[c] = parse_numeric(out[c])
    if "vin" in out:
        out["vin"] = out["vin"].astype("object").map(lambda v: str(v).upper() if pd.notna(v) else np.nan)
    for c in ("price_currency", "country_code"):
        if c in out:
            out[c] = out[c].astype("object").map(lambda v: str(v).upper() if pd.notna(v) else np.nan)
    return out


def derive_year(df: pd.DataFrame, snapshot: str = config.SNAPSHOT_DATE) -> pd.DataFrame:
    """production_year = raw production_year if valid else registration year (source recorded).

    A registration date after the snapshot date is impossible for a listed used car: flagged and
    not used to derive the year.
    """
    out = df.copy()
    reg = pd.to_datetime(out["registration_date"], errors="coerce")
    future = reg > pd.Timestamp(snapshot)
    out["flag_future_registration"] = future.fillna(False)
    reg_year = reg.dt.year.where(~future)
    out["registration_year"] = reg_year
    prod = out["production_year"].where(out["production_year"].between(*VALID_YEAR))
    out["year_source"] = np.where(prod.notna(), "production_year",
                                  np.where(reg_year.notna(), "registration_date", "missing"))
    out["production_year"] = prod.fillna(reg_year)
    out.loc[~out["production_year"].between(*VALID_YEAR), "production_year"] = np.nan
    return out


def null_implausible_values(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Measurements that are physically implausible -> NaN (row kept). Counted per rule."""
    out = df.copy()

    def _null(col, mask, rule):
        mask = mask.fillna(False) & out[col].notna()
        log.append({"column": col, "rule": rule, "n_set_to_nan": int(mask.sum())})
        out.loc[mask, col] = np.nan

    _null("nr_seats", (out["nr_seats"] < VALID_SEATS[0]) | (out["nr_seats"] > VALID_SEATS[1]), f"seats outside {VALID_SEATS}")
    _null("power_kw", (out["power_kw"] < VALID_POWER_KW[0]) | (out["power_kw"] > VALID_POWER_KW[1]), f"power_kw outside {VALID_POWER_KW}")
    electric = out["fuel_category"].astype(str).str.lower().eq("electric")
    _null("cylinders_volume_cc", (out["cylinders_volume_cc"] > MAX_VALID_ENGINE_CC) | ((out["cylinders_volume_cc"] <= 0) & ~electric),
          f"engine_cc > {MAX_VALID_ENGINE_CC} or <= 0 for non-electric (electric keeps 0)")
    _null("mileage_km", (out["mileage_km"] < 0) | (out["mileage_km"] > MAX_VALID_MILEAGE), f"mileage < 0 or > {MAX_VALID_MILEAGE:,} km")
    return out


def flag_price_review(df: pd.DataFrame, ratio: float = 20.0, min_group: int = 5) -> pd.Series:
    """Flag (do NOT drop) prices far above the make+model median or of '999999' placeholder form.

    This is a data-error screen only: it is not used to select features or tune models.
    """
    key = df["make"].astype(str) + "|" + df["model"].astype(str)
    grp = df.groupby(key)["price"]
    med = grp.transform("median")
    n = grp.transform("size")
    far = (n >= min_group) & (df["price"] > ratio * med)
    placeholder = df["price"].astype("Int64").astype(str).str.fullmatch(r"9{6,}")
    return (far | placeholder).fillna(False)


def filter_used_cars(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    n0 = len(df)
    keep = (df["is_used"] == True) & (df["is_new"] == False)  # noqa: E712  (nullable booleans)
    out = df[keep.fillna(False)]
    log.add("used_only", "is_used == True and is_new == False (NaN counts as unconfirmed)", n0, len(out),
            "new cars, pre-registered/demo cars and records without a confirmed used flag")
    if "offer_type" in out:
        n1 = len(out)
        out = out[out["offer_type"].isna() | out["offer_type"].eq("U")]
        log.add("offer_type", "offer_type == 'U' (or missing)", n1, len(out), "offer types other than 'U' (used) are unconfirmed")
    return out


def filter_passenger_cars(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    n0 = len(df)
    out = df[df["vehicle_type"].astype(str).str.lower().eq("car")]
    log.add("vehicle_type", "vehicle_type == 'Car'", n0, len(out), "non-passenger vehicles (e.g. Transporter)")
    n1 = len(out)
    out = out[~out["body_type"].astype(str).str.lower().isin(COMMERCIAL_BODY_TYPES)]
    log.add("commercial_body_type", f"body_type not in {sorted(COMMERCIAL_BODY_TYPES)}", n1, len(out),
            "commercial body types listed under vehicle_type=Car")
    return out


def filter_currency_and_price(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    n0 = len(df)
    out = df[df["price_currency"].eq("EUR")]
    log.add("currency", "price_currency == 'EUR'", n0, len(out), "other/missing currency (no conversion performed)")
    n1 = len(out)
    out = out[out["price"].notna()]
    log.add("price_not_null", "price is numeric and not null", n1, len(out), "unparseable/missing target")
    n2 = len(out)
    out = out[out["price"] > 0]
    log.add("price_positive", "price > 0", n2, len(out), "zero/negative price")
    n3 = len(out)
    out = out[out["price"] >= MIN_VALID_PRICE]
    log.add("price_min", f"price >= {MIN_VALID_PRICE} EUR", n3, len(out), "placeholder prices (e.g. price = 1)")
    return out


def drop_content_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Same VALID 17-char VIN + same price, mileage, registration date and country = same car, same ad."""
    n0 = len(df)
    valid = is_valid_vin(df["vin"])
    key_cols = ["vin", "price", "mileage_km", "registration_date", "country_code"]
    dup = valid & df.duplicated(subset=key_cols, keep="first")
    out = df[~dup]
    log.add("content_duplicates", "drop repeated (valid VIN, price, mileage, registration_date, country)", n0, len(out),
            "identical listings re-posted under a different id")
    return out


def clean_pipeline(raw: pd.DataFrame, snapshot: str = config.SNAPSHOT_DATE):
    """Return (clean_df, CleaningLog, implausible_value_log, scope_crosstab)."""
    log = CleaningLog()
    df = standardize_types(raw)
    log.add("raw", "start (after type/whitespace/missing-token normalisation only)", len(df), len(df))
    scope_ct = (df.groupby(["is_used", "is_new", "is_preregistered", "is_rental", "offer_type"], dropna=False)
                .size().rename("n").reset_index())
    df = filter_used_cars(df, log)
    df = filter_passenger_cars(df, log)
    df = filter_currency_and_price(df, log)
    df = drop_content_duplicates(df, log)
    df = derive_year(df, snapshot)
    value_log: list = []
    df = null_implausible_values(df, value_log)
    df["flag_price_review"] = flag_price_review(df)
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    return df[cols].reset_index(drop=True), log, pd.DataFrame(value_log), scope_ct
