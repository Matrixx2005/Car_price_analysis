import hashlib

import numpy as np
import pandas as pd

from src import config
from src.cleaning import clean_pipeline, standardize_types, to_bool
from src.data_validation import (check_required_columns, check_target, has_errors, run_validation)


def test_required_columns_detected(raw):                                   # 1
    assert not has_errors(check_required_columns(raw))
    issues = check_required_columns(raw.drop(columns=["price"]))
    assert has_errors(issues) and "price" in issues[0].message


def test_target_numeric_conversion():                                      # 2
    df = pd.DataFrame({"price": ["12,500", "9000", "n/a", "abc", None]})
    out = standardize_types(df.assign(price_currency="EUR"))
    assert out["price"].iloc[0] == 12500 and out["price"].iloc[1] == 9000
    assert out["price"].iloc[2:].isna().all()


def test_validation_flags_bad_target():
    df = pd.DataFrame({"price": [100.0, 0.0, -5.0, np.nan, 50.0]})
    names = {i.check: i.count for i in check_target(df)}
    assert names["target_null"] == 1 and names["target_nonpositive"] == 2 and names["target_below_min"] == 1


def test_boolean_labels_parsed():                                          # 5
    s = pd.Series(["True", "false", "1", "0", "YES", "no", None, "maybe"])
    out = to_bool(s)
    assert out.tolist()[:6] == [True, False, True, False, True, False]
    assert out.isna().tolist()[6:] == [True, True]
    assert str(to_bool(pd.Series([True, False])).dtype) == "boolean"


def test_missing_values_do_not_crash(raw):                                 # 10
    raw.loc[0:3, ["make", "model", "mileage_km_raw", "mileage_km", "registration_date", "country_code"]] = np.nan
    clean, *_ = clean_pipeline(raw)
    assert len(clean) > 0
    assert not has_errors(run_validation(clean))


def test_raw_file_not_modified():                                          # 7
    """Cleaning works on copies; also verify the real raw file (if present) still matches its MD5."""
    raw_df = pd.DataFrame({"price": ["1"], "price_currency": ["EUR"]})
    before = raw_df.copy(deep=True)
    standardize_types(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
    src = config.DATA_RAW / "SOURCE.json"
    if config.RAW_PATH.exists() and src.exists():
        import json
        h = hashlib.md5()
        with open(config.RAW_PATH, "rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        assert h.hexdigest() == json.loads(src.read_text(encoding="utf-8"))["md5_published"]
