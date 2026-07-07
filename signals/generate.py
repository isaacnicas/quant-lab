import itertools
from typing import Dict, List

import numpy as np

THRESHOLDS = {
    "ret_1d_z": [1.0, 0.5, -0.5, -1.0],
    "ret_5d_z": [1.0, 0.5, -0.5, -1.0],
    "ret_20d_z": [1.0, 0.5, -0.5, -1.0],
    "vol_20d_z": [1.0, 0.5, -0.5, -1.0],
    "vol_60d_z": [1.0, 0.5, -0.5, -1.0],
    "rsi": [30, 50, 70],
    "macd_line_z": [1.0, 0.5, -0.5, -1.0],
    "macd_signal_z": [1.0, 0.5, -0.5, -1.0],
}


def generate_signals(features: Dict[str, np.ndarray], max_combinations: int = 200) -> List[str]:
    """Generate rule-based signal strings, capped at max_combinations to avoid explosion."""
    feature_combinations = []
    for feature, values in THRESHOLDS.items():
        if feature in features:
            for threshold in values:
                feature_combinations.append((feature, ">", threshold))
                feature_combinations.append((feature, "<", threshold))

    signals = []
    for r in range(1, 4):
        for combo in itertools.combinations(feature_combinations, r):
            features_used = [f for f, _, _ in combo]
            if len(features_used) != len(set(features_used)):
                # Same feature referenced twice in one combo is always either
                # contradictory (e.g. "x > 1.0 AND x < 1.0") or redundant
                # (e.g. "x > 1.0 AND x > 0.5") — skip it.
                continue
            signals.append(" AND ".join(f"{f} {op} {v}" for f, op, v in combo))
            if len(signals) >= max_combinations:
                return signals[:max_combinations]
    return signals[:max_combinations]


if __name__ == "__main__":
    sample = {"ret_1d_z": np.random.randn(100), "rsi": np.random.randint(0, 100, 100), "macd_line_z": np.random.randn(100)}
    out = generate_signals(sample)
    print(f"Generated {len(out)} signals")
