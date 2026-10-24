import numpy as np

from src.cleaning import clean_pipeline, is_valid_vin
from tests.conftest import make_raw


def test_new_and_preregistered_cars_excluded(raw):                         # 4
    raw.loc[0, ["is_used", "is_new"]] = [False, True]
    raw.loc[1, ["is_used", "is_preregistered"]] = [False, True]
    raw.loc[2, "is_used"] = np.nan          # unconfirmed
    raw.loc[3, ["is_used", "is_new"]] = [True, True]  # contradictory
    clean, log, *_ = clean_pipeline(raw)
    assert set(clean["id"]).isdisjoint({"id0", "id1", "id2", "id3"})
    assert len(clean) == len(raw) - 4


def test_bad_prices_removed_by_rule(raw):                                  # 3
    raw.loc[0, "price"] = 0
    raw.loc[1, "price"] = -10
    raw.loc[2, "price"] = 1          # placeholder < 100
    raw.loc[3, "price"] = np.nan
    clean, *_ = clean_pipeline(raw)
    assert set(clean["id"]).isdisjoint({"id0", "id1", "id2", "id3"})
    assert (clean["price"] >= 100).all()


def test_expensive_valid_cars_kept(raw):                                   # 6
    raw.loc[0, ["make", "model", "price", "mileage_km_raw", "mileage_km"]] = ["Bugatti", "Chiron", 5_000_000.0, 1200.0, "1,200 km"]
    clean, *_ = clean_pipeline(raw)
    assert "id0" in set(clean["id"])
    assert clean.loc[clean["id"] == "id0", "price"].iloc[0] == 5_000_000


def test_non_eur_and_commercial_removed(raw):
    raw.loc[0, "price_currency"] = "CHF"
    raw.loc[1, "vehicle_type"] = "Transporter"
    raw.loc[2, "body_type"] = "Panel van"
    clean, *_ = clean_pipeline(raw)
    assert set(clean["id"]).isdisjoint({"id0", "id1", "id2"})


def test_implausible_values_become_nan_but_rows_kept(raw):
    raw.loc[0, "nr_seats"] = 255
    raw.loc[1, "power_kw"] = 8115
    raw.loc[2, "cylinders_volume_cc"] = 67500
    clean, *_ = clean_pipeline(raw)
    assert len(clean) == len(raw)
    assert clean.loc[clean["id"] == "id0", "nr_seats"].isna().all()
    assert clean.loc[clean["id"] == "id1", "power_kw"].isna().all()
    assert clean.loc[clean["id"] == "id2", "cylinders_volume_cc"].isna().all()


def test_content_duplicates_removed_only_with_valid_vin(raw):
    vin = "WBA12345678901234"
    assert is_valid_vin(raw["vin"].fillna("x")).sum() == 0
    raw.loc[[0, 1], "vin"] = vin
    raw.loc[[0, 1], ["price", "mileage_km_raw", "mileage_km", "country_code", "registration_date"]] = [20000, 1000.0, "1,000 km", "DE", "2020-01-01"]
    clean, log, *_ = clean_pipeline(raw)
    assert len(clean) == len(raw) - 1
    # same duplicated content WITHOUT a VIN must not be deleted
    raw2 = make_raw()
    raw2.loc[[0, 1], ["price", "mileage_km_raw", "mileage_km", "country_code", "registration_date"]] = [20000, 1000.0, "1,000 km", "DE", "2020-01-01"]
    clean2, *_ = clean_pipeline(raw2)
    assert len(clean2) == len(raw2)


def test_row_counts_logged_and_consistent(raw):                            # 16
    raw.loc[0, "is_new"] = True
    clean, log, *_ = clean_pipeline(raw)
    f = log.to_frame()
    assert (f["rows_before"] - f["rows_removed"] == f["rows_after"]).all()
    assert f.iloc[-1]["rows_after"] == len(clean)
    assert f["rows_before"].iloc[1:].tolist() == f["rows_after"].iloc[:-1].tolist()


def test_year_falls_back_to_registration_and_future_flagged(raw):
    raw.loc[0, "production_year"] = 2015
    raw.loc[1, "registration_date"] = "2026-06-01"   # after 2025-11-08 snapshot
    clean, *_ = clean_pipeline(raw)
    c = clean.set_index("id")
    assert c.loc["id0", "production_year"] == 2015 and c.loc["id0", "year_source"] == "production_year"
    assert c.loc["id2", "production_year"] == 2018 and c.loc["id2", "year_source"] == "registration_date"
    assert bool(c.loc["id1", "flag_future_registration"]) and np.isnan(c.loc["id1", "production_year"])
