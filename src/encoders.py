"""Custom encoders that sklearn does not provide. Statistics are learned in fit() only, so inside a
Pipeline they are learned from the training fold and never from validation/test data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency in the data seen in fit().

    One output column per input column (no dimension blow-up). A category never seen in fit gets 0.0
    (it has frequency zero in the training data); this is the documented behaviour for unseen categories.
    """

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        self.freq_ = {c: X[c].astype(str).value_counts(normalize=True).to_dict() for c in self.columns_}
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = np.zeros((len(X), len(self.columns_)), dtype=float)
        for j, c in enumerate(self.columns_):
            out[:, j] = X[c].astype(str).map(self.freq_[c]).fillna(0.0).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        """Use the real column names when the pipeline passes them (e.g. 'brand_freq'), else positional names."""
        names = list(input_features) if input_features is not None else self.columns_
        return np.asarray([f"{c}_freq" for c in names], dtype=object)

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        cols = getattr(self, "columns_", [f"x{i}" for i in range(np.asarray(X).shape[1])])
        return pd.DataFrame(np.asarray(X, dtype=object), columns=cols)
