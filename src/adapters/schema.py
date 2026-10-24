"""Common data schema shared by every data source (AutoScout24 now, Vietnamese data later).

Any new source only needs an adapter that renames/derives columns into these names.
"""
TARGET = "price"

# Canonical columns. Not every column must be populated by every source.
COMMON_COLUMNS = [
    "listing_id", "brand", "model", "model_version", "production_year", "mileage_km",
    "fuel_type", "transmission", "body_type", "country_code", "seller_type",
    "engine_cc", "power_kw", "nr_seats", "price", "currency", "vehicle_condition",
]

# Columns that are metadata / filter conditions, never training features.
METADATA_COLUMNS = ["listing_id", "currency", "vehicle_condition"]

# Secondary categorical attributes kept in the common schema when the source has them.
OPTIONAL_COLUMNS = ["drive_train", "body_color"]

NUMERIC_FEATURES = ["production_year", "vehicle_age", "mileage_km", "power_kw", "engine_cc", "nr_seats"]
CATEGORICAL_FEATURES = [
    "brand", "model", "model_version", "make_model",
    "fuel_type", "transmission", "body_type", "country_code", "seller_type",
    "drive_train", "body_color",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
