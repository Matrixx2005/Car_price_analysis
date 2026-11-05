"""Week-2 pipelines: Encoding x Scaling x Model. Everything that learns from data (imputer, encoder,
scaler, target encoder) is a step INSIDE the Pipeline, so cross-validation fits it on the training fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler, TargetEncoder)
from sklearn.svm import SVR

from src import config
from src.encoders import FrequencyEncoder
from src.transformers import RareCategoryGrouper

# Main feature set (week-1 decision): model_version is a separate cardinality experiment.
CAT_MAIN = ["brand", "model", "fuel_type", "transmission", "body_type", "country_code", "seller_type",
            "drive_train", "body_color"]
CAT_WITH_VERSION = CAT_MAIN + ["model_version"]
NUM = ["production_year", "vehicle_age", "mileage_km", "power_kw", "engine_cc", "nr_seats"]
LOG_NUM = ["mileage_km", "power_kw", "engine_cc"]   # right-skewed measurements (week-1 EDA), used by the log_features experiment

ENCODERS = ["onehot", "ordinal", "target", "frequency"]
SCALERS = ["none", "standard", "minmax", "robust"]
MODELS = ["knn", "ridge", "svr", "rf"]

ONEHOT_MIN_FREQUENCY = 10   # categories seen < 10 times in the training fold are pooled into one "infrequent" column
SEED = config.RANDOM_SEED


def make_scaler(name: str):
    return {"none": "passthrough", "standard": StandardScaler(), "minmax": MinMaxScaler(),
            "robust": RobustScaler()}[name]


def make_encoder(name: str, target_smooth="auto"):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=ONEHOT_MIN_FREQUENCY)
    if name == "ordinal":
        # unseen category -> -1. Order is arbitrary (alphabetical): the "false order" risk studied in the report.
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "target":
        # sklearn's TargetEncoder cross-fits internally (cv=5) and smooths, so training rows are not
        # encoded with their own target.
        return TargetEncoder(target_type="continuous", smooth=target_smooth,
                             cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
    if name == "frequency":
        return FrequencyEncoder()
    raise ValueError(name)


# Week-2 defaults (untuned). Week 3 overrides them through `params`; any key outside this set is rejected.
DEFAULT_PARAMS = {
    "ridge": {"alpha": 1.0},
    "knn": {"n_neighbors": 10, "weights": "distance", "p": 2},
    "svr": {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    "rf": {"n_estimators": 100, "min_samples_leaf": 2, "max_depth": None, "max_features": 1.0},
}


def make_model(name: str, params: dict | None = None, n_jobs: int = 1):
    """Model with the week-2 defaults, overridden by `params` (week-3 tuning). Unknown keys raise."""
    if name not in DEFAULT_PARAMS:
        raise ValueError(name)
    unknown = set(params or {}) - set(DEFAULT_PARAMS[name])
    if unknown:
        raise ValueError(f"unknown {name} parameters: {sorted(unknown)}")
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "ridge":
        return Ridge(**p)
    if name == "knn":
        return KNeighborsRegressor(**p, n_jobs=n_jobs)
    if name == "svr":
        # SVR's epsilon/C are defined in target units, so the target is standardised inside the model
        # (this is part of the SVR definition, not the "log transformation" experiment).
        return TransformedTargetRegressor(regressor=SVR(kernel="rbf", **p), transformer=StandardScaler())
    return RandomForestRegressor(**p, n_jobs=n_jobs, random_state=SEED)


def build_preprocessor(encoder: str, scaler: str, cat_cols=CAT_MAIN, num_cols=NUM,
                       log_features: bool = False, rare_min_count: int | None = None,
                       target_smooth="auto") -> ColumnTransformer:
    """Numeric: median impute (+ missing indicators) -> scaler.
    Categorical: impute 'missing' (a category of its own) -> encoder -> scaler.
    One-hot dummies are NOT scaled (they are already 0/1); the other encoders produce numeric columns
    that are scaled like numeric features.

    log_features=True (separate experiment): the right-skewed measurements in LOG_NUM get log1p BEFORE
    imputation and scaling. log1p is a stateless row-wise function, so it cannot leak anything.

    rare_min_count (week 3): categories seen fewer than this many times in the TRAINING fold are pooled into one
    shared label before encoding (counts are learned inside the pipeline)."""
    def num_branch(log: bool) -> Pipeline:
        steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if log else []
        return Pipeline(steps + [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", make_scaler(scaler))])

    cat_steps = [("imp", SimpleImputer(strategy="constant", fill_value="missing"))]
    if rare_min_count is not None:
        cat_steps.append(("rare", RareCategoryGrouper(min_count=rare_min_count)))
    cat_steps.append(("enc", make_encoder(encoder, target_smooth)))
    if encoder != "onehot":
        cat_steps.append(("sc", make_scaler(scaler)))
    cat = Pipeline(cat_steps)
    if log_features:
        skewed = [c for c in LOG_NUM if c in num_cols]
        rest = [c for c in num_cols if c not in skewed]
        parts = [("num_log", num_branch(True), skewed), ("num", num_branch(False), rest), ("cat", cat, list(cat_cols))]
    else:
        parts = [("num", num_branch(False), list(num_cols)), ("cat", cat, list(cat_cols))]
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_estimator(encoder: str, scaler: str, model: str, log_target: bool = False,
                    cat_cols=CAT_MAIN, num_cols=NUM, log_features: bool = False, model_params: dict | None = None,
                    rare_min_count: int | None = None, target_smooth="auto", n_jobs: int = 1):
    pipe = Pipeline([("pre", build_preprocessor(encoder, scaler, cat_cols, num_cols, log_features, rare_min_count, target_smooth)),
                     ("m", make_model(model, model_params, n_jobs))])
    if log_target:
        # fit on log1p(price); predictions are mapped back with expm1 so metrics are in EUR
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def fitted_preprocessor(est):
    """The fitted ColumnTransformer inside a (possibly target-wrapped) estimator."""
    pipe = est.regressor_ if isinstance(est, TransformedTargetRegressor) else est
    return pipe.named_steps["pre"]
