"""Data validation: returns structured issues instead of silently passing bad data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config

REQUIRED_COLUMNS = ["id", "price", "price_currency", "make", "model", "mileage_km", "registration_date",
                    "is_used", "is_new", "vehicle_type", "country_code"]
BOOL_COLS = ["is_used", "is_new", "is_preregistered", "is_rental"]


@dataclass
class Issue:
    check: str
    severity: str  # ERROR | WARNING | INFO
    count: int
    message: str


def check_required_columns(df, required=REQUIRED_COLUMNS) -> list[Issue]:
    missing = [c for c in required if c not in df.columns]
    return [Issue("required_columns", "ERROR", len(missing), f"missing required columns: {missing}")] if missing else \
           [Issue("required_columns", "INFO", 0, "all required columns present")]


def check_target(df, col="price", min_price=100) -> list[Issue]:
    if col not in df:
        return [Issue("target", "ERROR", 1, f"target column '{col}' not found")]
    s = df[col]
    out = []
    if not pd.api.types.is_numeric_dtype(s):
        out.append(Issue("target_dtype", "ERROR", len(s), f"target dtype is {s.dtype}, expected numeric"))
        return out
    for name, mask, sev in [("target_null", s.isna(), "ERROR"), ("target_nonpositive", s <= 0, "ERROR"),
                            ("target_below_min", (s > 0) & (s < min_price), "WARNING")]:
        n = int(mask.sum())
        out.append(Issue(name, sev if n else "INFO", n, f"{n} rows ({name})"))
    return out


def check_currency(df, col="price_currency", expected="EUR") -> list[Issue]:
    if col not in df:
        return [Issue("currency", "ERROR", 1, f"'{col}' not found")]
    bad = int((df[col] != expected).sum())
    return [Issue("currency", "ERROR" if bad else "INFO", bad, f"{bad} rows with currency != {expected}")]


def check_year(df, col="production_year", valid=(1950, 2026)) -> list[Issue]:
    if col not in df:
        return [Issue("year", "WARNING", 1, f"'{col}' not found")]
    s = df[col]
    bad = int((s.notna() & ~s.between(*valid)).sum())
    return [Issue("year_range", "ERROR" if bad else "INFO", bad, f"{bad} rows with {col} outside {valid}"),
            Issue("year_missing", "INFO", int(s.isna().sum()), f"{int(s.isna().sum())} rows with missing {col}")]


def check_mileage(df, col="mileage_km", max_km=1_000_000) -> list[Issue]:
    if col not in df:
        return [Issue("mileage", "WARNING", 1, f"'{col}' not found")]
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return [Issue("mileage_dtype", "ERROR", len(s), f"{col} dtype {s.dtype}, expected numeric")]
    neg, big = int((s < 0).sum()), int((s > max_km).sum())
    return [Issue("mileage_negative", "ERROR" if neg else "INFO", neg, f"{neg} negative mileage values"),
            Issue("mileage_extreme", "WARNING" if big else "INFO", big, f"{big} rows above {max_km:,} km"),
            Issue("mileage_zero", "INFO", int((s == 0).sum()), f"{int((s == 0).sum())} used cars with 0 km (kept, flagged)")]


def check_booleans(df, cols=BOOL_COLS) -> list[Issue]:
    out = []
    for c in cols:
        if c in df:
            ok = pd.api.types.is_bool_dtype(df[c]) or str(df[c].dtype) == "boolean"
            n_na = int(df[c].isna().sum())
            out.append(Issue(f"bool_{c}", "ERROR" if not ok else ("WARNING" if n_na else "INFO"), n_na,
                             f"{c}: dtype={df[c].dtype}, {n_na} missing"))
    return out


def check_duplicates(df, id_col="id") -> list[Issue]:
    out = []
    n_full = int(df.duplicated().sum())
    out.append(Issue("duplicate_rows", "WARNING" if n_full else "INFO", n_full, f"{n_full} fully duplicated rows"))
    if id_col in df:
        n_id = int(df[id_col].duplicated().sum())
        out.append(Issue("duplicate_id", "ERROR" if n_id else "INFO", n_id, f"{n_id} duplicated {id_col}"))
    return out


def check_invalid_values(df) -> list[Issue]:
    out = []
    for col, lo, hi in [("nr_seats", 1, 9), ("power_kw", 10, 1500), ("cylinders_volume_cc", 0, 10000)]:
        if col in df:
            n = int((df[col].notna() & ~df[col].between(lo, hi)).sum())
            out.append(Issue(f"range_{col}", "WARNING" if n else "INFO", n, f"{n} rows with {col} outside [{lo}, {hi}]"))
    return out


def run_validation(df: pd.DataFrame, stage: str = "clean") -> list[Issue]:
    issues = check_required_columns(df)
    if issues[0].severity == "ERROR":
        return issues
    for fn in (check_target, check_currency, check_year, check_mileage, check_booleans, check_duplicates, check_invalid_values):
        issues += fn(df)
    return issues


def has_errors(issues) -> bool:
    return any(i.severity == "ERROR" and i.count > 0 for i in issues)


def issues_to_frame(issues) -> pd.DataFrame:
    return pd.DataFrame([i.__dict__ for i in issues])


def write_validation_report(stages: dict, path: Path = config.REPORTS / "validation_report.md") -> None:
    L = ["# Validation report", "", "_Sinh tự động bởi `src/data_validation.py`._", ""]
    for stage, issues in stages.items():
        L += [f"## {stage}", "", "| Kiểm tra | Mức | Số dòng | Chi tiết |", "|---|---|---:|---|"]
        L += [f"| {i.check} | {i.severity} | {i.count:,} | {i.message} |" for i in issues]
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
