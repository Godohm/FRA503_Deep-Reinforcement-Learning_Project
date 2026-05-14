"""M3 feature-engineering tests.

Covers:
- indicators.py: log_returns, rsi, macd, stochastic_k, atr
- state_builder.py: compute_price_features (per-day, no leakage), assemble_state
- normalization.py: train-only fit, NaN passthrough, save/load round-trip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import atr, log_returns, macd, rsi, stochastic_k
from src.features.normalization import PriceFeatureScaler
from src.features.state_builder import (
    N_PRICE_FEATURES,
    POSITIONAL_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    STATE_DIM,
    assemble_state,
    compute_price_features,
)


# --------------------------------------------------------------------------- helpers


def _make_session_frame(day: str, n_bars: int = 200, seed: int = 0,
                       point_size: float = 1e-5) -> pd.DataFrame:
    """Build a synthetic single-session DataFrame in the shape that preprocess.py emits."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{day} 02:00:00", tz="UTC")  # 09:00 UTC+7
    idx = pd.date_range(start=start, periods=n_bars, freq="1min")
    # Random walk close, smooth-ish.
    close = 1.10000 + np.cumsum(rng.normal(0, 1e-5, size=n_bars))
    high = close + np.abs(rng.normal(0, 5e-6, size=n_bars))
    low = close - np.abs(rng.normal(0, 5e-6, size=n_bars))
    open_ = close + rng.normal(0, 1e-6, size=n_bars)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(1, 50, size=n_bars),
            "spread": rng.integers(1, 20, size=n_bars),
            "real_volume": np.zeros(n_bars, dtype="int64"),
            "session_day": pd.Series([pd.Timestamp(day).date()] * n_bars).values,
            "bar_idx_in_day": np.arange(n_bars, dtype="int64"),
        },
        index=idx,
    )
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------- indicators


def test_log_returns_hand_check():
    close = pd.Series([1.0, 1.01, 1.0201], dtype="float64")
    out = log_returns(close, period=1)
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1], np.log(1.01 / 1.0))
    np.testing.assert_allclose(out.iloc[2], np.log(1.0201 / 1.01))


def test_log_returns_warmup_nan():
    close = pd.Series(np.arange(1, 11, dtype="float64"))
    out = log_returns(close, period=5)
    assert out.iloc[:5].isna().all()
    assert not out.iloc[5:].isna().any()


def test_rsi_extremes():
    # All-up series → RSI should saturate near 100 once warmed up.
    close = pd.Series(np.arange(1, 50, dtype="float64"))
    r = rsi(close, period=14)
    assert r.iloc[14:].min() > 99.0, "RSI must saturate near 100 on a monotone-up series."
    # All-down series → RSI near 0.
    close = pd.Series(np.arange(50, 1, -1, dtype="float64"))
    r = rsi(close, period=14)
    assert r.iloc[14:].max() < 1.0, "RSI must saturate near 0 on a monotone-down series."


def test_macd_shape_and_warmup():
    close = pd.Series(np.linspace(1.0, 2.0, 100), dtype="float64")
    line, sig, hist = macd(close)
    assert len(line) == len(sig) == len(hist) == 100
    # Histogram should be defined once signal EMA is warm (slow + signal - 1 = 26 + 9 - 1 = 34).
    assert hist.iloc[40:].notna().all()


def test_stochastic_k_range():
    close = pd.Series(np.linspace(1.0, 2.0, 50))
    high = close + 0.01
    low = close - 0.01
    k = stochastic_k(high, low, close, period=14)
    valid = k.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr_positive():
    close = pd.Series(np.linspace(1.0, 1.5, 50))
    high = close + 0.001
    low = close - 0.001
    a = atr(high, low, close, period=14)
    valid = a.dropna()
    assert (valid > 0).all()


# --------------------------------------------------------------------------- no-look-ahead


def test_indicators_no_lookahead():
    """Shift-input ⇒ shift-output: append a future bar and assert past values are unchanged."""
    close = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1e-5, 200)) + 1.1)
    high = close + 0.0001
    low = close - 0.0001

    base_rsi = rsi(close, 14)
    base_macd_hist = macd(close)[2]
    base_stoch = stochastic_k(high, low, close, 14)
    base_atr = atr(high, low, close, 14)
    base_ret5 = log_returns(close, 5)

    # Mutate FUTURE bars (last 50) and verify earlier bars don't change.
    perturbed_close = close.copy()
    perturbed_close.iloc[150:] = close.iloc[150:] + 0.01
    perturbed_high = high.copy()
    perturbed_high.iloc[150:] = high.iloc[150:] + 0.01
    perturbed_low = low.copy()
    perturbed_low.iloc[150:] = low.iloc[150:] + 0.01

    np.testing.assert_array_equal(rsi(perturbed_close, 14).iloc[:150].values,
                                  base_rsi.iloc[:150].values)
    np.testing.assert_array_equal(macd(perturbed_close)[2].iloc[:150].values,
                                  base_macd_hist.iloc[:150].values)
    np.testing.assert_array_equal(stochastic_k(perturbed_high, perturbed_low, perturbed_close, 14)
                                  .iloc[:150].values,
                                  base_stoch.iloc[:150].values)
    np.testing.assert_array_equal(atr(perturbed_high, perturbed_low, perturbed_close, 14)
                                  .iloc[:150].values,
                                  base_atr.iloc[:150].values)
    np.testing.assert_array_equal(log_returns(perturbed_close, 5).iloc[:150].values,
                                  base_ret5.iloc[:150].values)


# --------------------------------------------------------------------------- state_builder


def test_price_features_columns_and_order():
    df = _make_session_frame("2024-03-04", n_bars=120)
    feats = compute_price_features(df, point_size=1e-5)
    assert list(feats.columns) == list(PRICE_FEATURE_NAMES)
    assert len(feats) == len(df)


def test_price_features_spread_in_price_units():
    df = _make_session_frame("2024-03-04", n_bars=80)
    feats = compute_price_features(df, point_size=1e-5)
    expected = df["spread"].astype("float64") * 1e-5
    np.testing.assert_allclose(feats["spread"].values, expected.values)


def test_price_features_warmup_nan_then_valid():
    df = _make_session_frame("2024-03-04", n_bars=200)
    feats = compute_price_features(df, point_size=1e-5)
    # 60m return needs 60 bars; MACD with signal needs slow+signal-1 = 34 bars.
    # ATR/RSI need 14. So everything is valid from bar 60 onwards.
    valid = feats.iloc[60:]
    assert valid.notna().all().all(), "All features must be valid from bar 60 onwards."


def test_price_features_per_day_no_cross_session_leak():
    """Bar 0 of day 2 should NOT incorporate day 1's prices."""
    day1 = _make_session_frame("2024-03-04", n_bars=120, seed=1)
    day2 = _make_session_frame("2024-03-05", n_bars=120, seed=2)
    full = pd.concat([day1, day2], axis=0)
    feats_full = compute_price_features(full, point_size=1e-5)

    # ret_1m at bar 0 of day 2 must be NaN (no prior bar within that session).
    day2_bar0_idx = day2.index[0]
    assert pd.isna(feats_full.loc[day2_bar0_idx, "ret_1m"])

    # If we run day 2 in isolation, bar-0 ret_1m is also NaN — same result.
    feats_day2 = compute_price_features(day2, point_size=1e-5)
    assert pd.isna(feats_day2.iloc[0]["ret_1m"])
    # And bars 60+ in day 2 should match exactly whether or not day 1 is in the input.
    np.testing.assert_array_equal(
        feats_full.loc[day2.index[60:], "ret_60m"].values,
        feats_day2.iloc[60:]["ret_60m"].values,
    )


def test_assemble_state_shape_and_order():
    price = np.arange(N_PRICE_FEATURES, dtype="float32")  # [0..9]
    positional = [0.5, 1.0, 0.01, -0.002, 0.3]
    state = assemble_state(price, positional)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    # Indices 0..9 == price features
    np.testing.assert_array_equal(state[:N_PRICE_FEATURES], price)
    # Indices 10..14 == positional in declared order (TL, POS, PR, DR, HT)
    np.testing.assert_allclose(state[N_PRICE_FEATURES:], positional, rtol=0, atol=1e-7)


def test_assemble_state_wrong_lengths_raise():
    with pytest.raises(ValueError):
        assemble_state(np.zeros(9), [0.0] * 5)
    with pytest.raises(ValueError):
        assemble_state(np.zeros(10), [0.0] * 4)


def test_positional_feature_names_count():
    assert len(POSITIONAL_FEATURE_NAMES) == 5
    assert POSITIONAL_FEATURE_NAMES == ("TL", "POS", "PR", "DR", "HT")


# --------------------------------------------------------------------------- normalization


def test_scaler_fit_train_only_and_transform_test(tmp_path: Path):
    train_df = _make_session_frame("2024-03-04", n_bars=200, seed=10)
    test_df = _make_session_frame("2024-03-05", n_bars=200, seed=20)
    train_feats = compute_price_features(train_df, point_size=1e-5)
    test_feats = compute_price_features(test_df, point_size=1e-5)

    scaler = PriceFeatureScaler().fit(train_feats)
    train_scaled = scaler.transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    # Scaler stats should be derived ONLY from train (not test).
    train_clean = train_feats.dropna(how="any").to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_clean.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(scaler.scale_, train_clean.std(axis=0, ddof=0), rtol=1e-6)

    # Transformed train should be ~ N(0, 1) on non-NaN rows.
    train_scaled_valid = train_scaled[~np.isnan(train_scaled).any(axis=1)]
    np.testing.assert_allclose(train_scaled_valid.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(train_scaled_valid.std(axis=0, ddof=0), 1.0, atol=1e-5)

    # Test transform reuses train stats — manually verify on a single row.
    test_clean_row = test_feats.dropna(how="any").iloc[0].to_numpy()
    expected = (test_clean_row - scaler.mean_) / scaler.scale_
    test_scaled_valid = test_scaled[~np.isnan(test_scaled).any(axis=1)]
    np.testing.assert_allclose(test_scaled_valid[0], expected, rtol=1e-6)


def test_scaler_nan_passthrough():
    train_df = _make_session_frame("2024-03-04", n_bars=120, seed=5)
    train_feats = compute_price_features(train_df, point_size=1e-5)
    scaler = PriceFeatureScaler().fit(train_feats)
    scaled = scaler.transform(train_feats)
    # First few rows have NaN features (warmup); the scaled output should preserve NaN there.
    assert np.isnan(scaled[:5]).all()


def test_scaler_save_load_roundtrip(tmp_path: Path):
    train_df = _make_session_frame("2024-03-04", n_bars=200, seed=7)
    train_feats = compute_price_features(train_df, point_size=1e-5)
    scaler = PriceFeatureScaler().fit(train_feats)
    p = scaler.save(tmp_path / "scaler.pkl")
    assert p.exists()

    reloaded = PriceFeatureScaler.load(p)
    np.testing.assert_allclose(reloaded.mean_, scaler.mean_)
    np.testing.assert_allclose(reloaded.scale_, scaler.scale_)

    a = scaler.transform(train_feats)
    b = reloaded.transform(train_feats)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_scaler_rejects_bad_schema():
    df = pd.DataFrame({"foo": [1.0, 2.0], "bar": [3.0, 4.0]})
    with pytest.raises(ValueError):
        PriceFeatureScaler().fit(df)
