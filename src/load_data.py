"""Reading helpers and a first inspection of the raw AutoScout24 CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src import config


def peek_csv(path: Path = config.RAW_PATH, nrows: int = 10) -> dict:
    """Inspect the first rows: delimiter, encoding, columns. Long text is truncated."""
    with open(path, "r", encoding="utf-8", newline="") as f:  # raises on non-UTF-8 header block
        sample = f.read(200_000)
    try:
        dialect = csv.Sniffer().sniff(sample[:50_000], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    head = pd.read_csv(path, nrows=nrows, sep=delimiter, encoding="utf-8", low_memory=False)
    return {"delimiter": delimiter, "columns": list(head.columns), "head": head}


def truncate_long_strings(df: pd.DataFrame, width: int = 60) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include="object"):
        out[c] = out[c].astype("string").str.slice(0, width)
    return out


def read_raw(usecols=None, dtype=None, chunksize: int | None = None, path: Path = config.RAW_PATH, **kw):
    """Read the raw CSV. All strings stay strings; pass chunksize to stream."""
    return pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize,
                       encoding="utf-8", low_memory=False, **kw)


def count_rows_cols(path: Path = config.RAW_PATH) -> tuple[int, int]:
    """Exact data-row count via the CSV parser (handles newlines inside quoted fields)."""
    n = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000, encoding="utf-8"):
        n += len(chunk)
    ncols = len(pd.read_csv(path, nrows=0).columns)
    return n, ncols
