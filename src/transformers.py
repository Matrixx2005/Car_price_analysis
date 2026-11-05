"""Week-3 transformers. Statistics are learned in fit() only (from the training fold inside a Pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

OTHER_LABEL = "__other__"


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Pool categories seen fewer than `min_count` times in fit() into one shared label.

    Categories never seen in fit() are also mapped to that label, so the encoders that follow always meet a
    category they know. The threshold is applied to counts from the training fold only.
    """

    def __init__(self, min_count: int = 10, other_label: str = OTHER_LABEL):
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):
        X = self._frame(X, fit=True)
        self.columns_ = list(X.columns)
        self.keep_ = {c: set(X[c].dropna().astype(str).value_counts().loc[lambda s: s >= self.min_count].index) for c in self.columns_}
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        X = self._frame(X, fit=False)
        out = np.empty((len(X), len(self.columns_)), dtype=object)
        for j, c in enumerate(self.columns_):
            s = X[c].astype(str)
            out[:, j] = np.where(s.isin(self.keep_[c]), s, self.other_label)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) if input_features is not None else self.columns_, dtype=object)

    def _frame(self, X, fit: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        cols = [f"x{i}" for i in range(np.asarray(X).shape[1])] if fit else self.columns_
        return pd.DataFrame(np.asarray(X, dtype=object), columns=cols)


def drop_extreme_prices(y_train: pd.Series, q: float | None) -> np.ndarray:
    """Boolean mask of TRAINING rows to keep: price <= the q-quantile of the training fold's own prices.

    The threshold comes only from the training fold (never from validation/test rows) and the mask is applied to the
    training rows only; evaluation rows are never removed.
    """
    if q is None:
        return np.ones(len(y_train), dtype=bool)
    return (y_train.to_numpy() <= np.quantile(y_train.to_numpy(), q))
