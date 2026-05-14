"""Train-only feature scaler for the 10 price-derived state features.

The 5 positional features are bounded by design and are NOT scaled here.

Usage
-----
    scaler = PriceFeatureScaler()
    scaler.fit(train_price_features)
    train_scaled = scaler.transform(train_price_features)
    test_scaled  = scaler.transform(test_price_features)   # reuses train stats
    scaler.save("models/<run-id>/scaler.pkl")
    ...
    scaler = PriceFeatureScaler.load("models/<run-id>/scaler.pkl")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .state_builder import N_PRICE_FEATURES, PRICE_FEATURE_NAMES


class PriceFeatureScaler:
    """Thin wrapper around ``sklearn.preprocessing.StandardScaler`` that:

    - enforces the 10-column ``PRICE_FEATURE_NAMES`` schema,
    - drops NaN rows during ``fit`` (the warmup region),
    - preserves NaN rows during ``transform`` (the env's warmup loop ignores them),
    - persists via ``joblib``.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._fit_n_rows: int = 0

    # ----------------------------------------------------------------- fit/transform

    def fit(self, price_features: pd.DataFrame) -> "PriceFeatureScaler":
        """Fit on the train split's price features. NaN rows are dropped."""
        self._check_schema(price_features)
        clean = price_features.dropna(axis=0, how="any")
        if len(clean) == 0:
            raise ValueError("fit() received zero non-NaN rows.")
        self._scaler = StandardScaler()
        self._scaler.fit(clean.to_numpy(dtype="float64"))
        self._fit_n_rows = len(clean)
        return self

    def transform(self, price_features: pd.DataFrame) -> np.ndarray:
        """Transform any split using the train-fit stats. NaN in, NaN out (per row)."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fit yet. Call fit() first.")
        self._check_schema(price_features)
        arr = price_features.to_numpy(dtype="float64")
        out = np.full_like(arr, fill_value=np.nan, dtype="float64")
        mask = ~np.isnan(arr).any(axis=1)
        if mask.any():
            out[mask] = self._scaler.transform(arr[mask])
        return out.astype(np.float32)

    def fit_transform(self, price_features: pd.DataFrame) -> np.ndarray:
        return self.fit(price_features).transform(price_features)

    # ----------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Persist the fitted scaler with joblib."""
        if self._scaler is None:
            raise RuntimeError("Cannot save an unfit scaler.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "feature_names": PRICE_FEATURE_NAMES,
            "scaler": self._scaler,
            "fit_n_rows": self._fit_n_rows,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PriceFeatureScaler":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != PRICE_FEATURE_NAMES:
            raise ValueError(
                f"Scaler at {path} was fit on a different feature schema:\n"
                f"  saved: {payload['feature_names']}\n"
                f"  current: {PRICE_FEATURE_NAMES}"
            )
        obj = cls()
        obj._scaler = payload["scaler"]
        obj._fit_n_rows = int(payload["fit_n_rows"])
        return obj

    # ----------------------------------------------------------------- introspection

    @property
    def mean_(self) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Scaler not fit yet.")
        return self._scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Scaler not fit yet.")
        return self._scaler.scale_

    @property
    def n_features_(self) -> int:
        return N_PRICE_FEATURES

    @property
    def fit_n_rows(self) -> int:
        return self._fit_n_rows

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _check_schema(df: pd.DataFrame) -> None:
        if list(df.columns) != list(PRICE_FEATURE_NAMES):
            raise ValueError(
                f"price_features columns must match PRICE_FEATURE_NAMES.\n"
                f"  expected: {list(PRICE_FEATURE_NAMES)}\n"
                f"  got:      {list(df.columns)}"
            )
