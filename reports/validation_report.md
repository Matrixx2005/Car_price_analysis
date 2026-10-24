# Validation report

_Sinh tự động bởi `src/data_validation.py`._

## Raw (trước cleaning)

| Kiểm tra | Mức | Số dòng | Chi tiết |
|---|---|---:|---|
| required_columns | INFO | 0 | all required columns present |
| target_null | INFO | 0 | 0 rows (target_null) |
| target_nonpositive | INFO | 0 | 0 rows (target_nonpositive) |
| target_below_min | WARNING | 8 | 8 rows (target_below_min) |
| currency | INFO | 0 | 0 rows with currency != EUR |
| year_range | INFO | 0 | 0 rows with production_year outside (1950, 2026) |
| year_missing | INFO | 96,110 | 96110 rows with missing production_year |
| mileage_dtype | ERROR | 118,382 | mileage_km dtype str, expected numeric |
| bool_is_used | INFO | 0 | is_used: dtype=bool, 0 missing |
| bool_is_new | INFO | 0 | is_new: dtype=bool, 0 missing |
| bool_is_preregistered | INFO | 0 | is_preregistered: dtype=bool, 0 missing |
| bool_is_rental | INFO | 0 | is_rental: dtype=bool, 0 missing |
| duplicate_rows | INFO | 0 | 0 fully duplicated rows |
| duplicate_id | INFO | 0 | 0 duplicated id |
| range_nr_seats | WARNING | 10 | 10 rows with nr_seats outside [1, 9] |
| range_power_kw | WARNING | 35 | 35 rows with power_kw outside [10, 1500] |
| range_cylinders_volume_cc | WARNING | 6 | 6 rows with cylinders_volume_cc outside [0, 10000] |

## Sau cleaning

| Kiểm tra | Mức | Số dòng | Chi tiết |
|---|---|---:|---|
| required_columns | INFO | 0 | all required columns present |
| target_null | INFO | 0 | 0 rows (target_null) |
| target_nonpositive | INFO | 0 | 0 rows (target_nonpositive) |
| target_below_min | INFO | 0 | 0 rows (target_below_min) |
| currency | INFO | 0 | 0 rows with currency != EUR |
| year_range | INFO | 0 | 0 rows with production_year outside (1950, 2026) |
| year_missing | INFO | 23 | 23 rows with missing production_year |
| mileage_negative | INFO | 0 | 0 negative mileage values |
| mileage_extreme | INFO | 0 | 0 rows above 1,000,000 km |
| mileage_zero | INFO | 14 | 14 used cars with 0 km (kept, flagged) |
| bool_is_used | INFO | 0 | is_used: dtype=boolean, 0 missing |
| bool_is_new | INFO | 0 | is_new: dtype=boolean, 0 missing |
| bool_is_preregistered | INFO | 0 | is_preregistered: dtype=boolean, 0 missing |
| bool_is_rental | INFO | 0 | is_rental: dtype=boolean, 0 missing |
| duplicate_rows | INFO | 0 | 0 fully duplicated rows |
| duplicate_id | INFO | 0 | 0 duplicated id |
| range_nr_seats | INFO | 0 | 0 rows with nr_seats outside [1, 9] |
| range_power_kw | INFO | 0 | 0 rows with power_kw outside [10, 1500] |
| range_cylinders_volume_cc | INFO | 0 | 0 rows with cylinders_volume_cc outside [0, 10000] |
