"""
Anomaly Threshold Selection Strategies
Author: Batch 11 - Predictive Maintenance System
References:
- Review 1 Presentation Slides: Slide 4 & 7 (Phase 5: Threshold Selection)
- IJACSA 2025 Paper: Section V-C (Anomaly Detection Process) & Section VI-E
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class AnomalyThresholdSelector:
    """
    Implements and compares multiple thresholding techniques:
    1. Statistical Standard Deviation: threshold = mean + k * std
    2. Quantile/Percentile: threshold = quantile(q)
    3. Dynamic / Adaptive Thresholding: Rolling window statistics that adapt to compressor load
    """
    def __init__(self, method: str = "std", k: float = 3.0, percentile: float = 98.0):
        self.method = method
        self.k = k
        self.percentile = percentile
        self.threshold_value = None

    def fit(self, normal_errors: np.ndarray) -> float:
        """
        Fits threshold on errors computed from normal operating regime.
        """
        if self.method == "std":
            mean_val = np.mean(normal_errors)
            std_val = np.std(normal_errors)
            self.threshold_value = mean_val + self.k * std_val
        elif self.method == "percentile":
            self.threshold_value = np.percentile(normal_errors, self.percentile)
        elif self.method == "iqr":
            q25, q75 = np.percentile(normal_errors, [25, 75])
            iqr = q75 - q25
            self.threshold_value = q75 + (self.k * iqr)
        else:
            raise ValueError(f"Unknown thresholding method: {self.method}")

        return float(self.threshold_value)

    def predict(self, errors: np.ndarray) -> np.ndarray:
        if self.threshold_value is None:
            raise RuntimeError("Threshold selector has not been fitted! Call fit() first.")
        return (errors > self.threshold_value).astype(int)

    @staticmethod
    def compute_adaptive_threshold(errors: np.ndarray, window_size: int = 50, k: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic / Adaptive thresholding using rolling window statistics:
        T_t = mu_{t-w:t} + k * sigma_{t-w:t}
        Prevents false alarms during high-load operational transitions.
        """
        s = pd.Series(errors)
        rolling_mean = s.rolling(window=window_size, min_periods=1).mean()
        rolling_std = s.rolling(window=window_size, min_periods=1).std().fillna(0)
        adaptive_threshold = rolling_mean + (k * rolling_std)
        anomalies = (s > adaptive_threshold).astype(int).values
        return anomalies, adaptive_threshold.values

    @staticmethod
    def compare_threshold_strategies(normal_errors: np.ndarray, test_errors: np.ndarray, true_labels: np.ndarray) -> pd.DataFrame:
        """
        Evaluates various threshold candidate multipliers (k=2.0, 2.5, 3.0, 3.5, 95th, 99th pct)
        to identify the optimal precision-recall trade-off.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        strategies = [
            ("Mean + 2.0*Std", "std", 2.0, None),
            ("Mean + 2.5*Std", "std", 2.5, None),
            ("Mean + 3.0*Std", "std", 3.0, None),
            ("Mean + 3.5*Std", "std", 3.5, None),
            ("95th Percentile", "percentile", None, 95.0),
            ("98th Percentile", "percentile", None, 98.0),
            ("99th Percentile", "percentile", None, 99.0),
        ]

        results = []
        for name, method, k_val, pct_val in strategies:
            selector = AnomalyThresholdSelector(method=method, k=k_val or 3.0, percentile=pct_val or 95.0)
            thresh = selector.fit(normal_errors)
            preds = selector.predict(test_errors)

            p = precision_score(true_labels, preds, zero_division=0)
            r = recall_score(true_labels, preds, zero_division=0)
            f1 = f1_score(true_labels, preds, zero_division=0)

            results.append({
                "Strategy": name,
                "Threshold": thresh,
                "Precision": p,
                "Recall": r,
                "F1-Score": f1,
                "Detected Anomalies": int(np.sum(preds))
            })

        return pd.DataFrame(results)
