import numpy as np
import pandas as pd

from features.engineer import build_features
from regimes.detect import label_regime


def test_no_lookahead_bias():
    close = pd.Series(np.linspace(100, 124, 25), name="close")
    df = pd.DataFrame({"close": close})
    result = build_features(df)
    assert pd.isna(result["ret_1d"].iloc[0])
    assert pd.isna(result["ret_5d"].iloc[4])
    assert pd.isna(result["vol_20d"].iloc[19])
    assert pd.isna(result["rsi"].iloc[:13]).all()
    assert pd.notna(result["rsi"].iloc[13])


def test_rsi_matches_expected_direction():
    close = pd.Series([100, 101, 102, 103, 104], name="close")
    df = pd.DataFrame({"close": close})
    result = build_features(df)
    assert result["rsi"].iloc[-1] > 50


def test_regime_labels_are_stable():
    close = pd.Series(np.linspace(100, 110, 100), name="close")
    df = pd.DataFrame({"close": close})
    df = build_features(df)
    regime = label_regime(df)
    assert regime.iloc[0] == regime.iloc[1]
    assert regime.iloc[0] == regime.iloc[3]
