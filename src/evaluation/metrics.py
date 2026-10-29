"""
Evaluation Metrics and Lead-Time Assessment
Author: Batch 11 - Predictive Maintenance System
References:
- IJACSA 2025 Paper: Section V-D (Evaluation Metrics), Table I & Table II
- Review 1 Presentation Slides: Slide 7 & Slide 8
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, List

from src.data.labels import FAILURE_PERIODS


def calculate_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Computes reconstruction fidelity metrics: MAE, MSE, RMSE.
    Matches Table I from IJACSA paper:
    Model | MAE | MSE | RMSE
    """
    error = original - reconstructed
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes anomaly classification metrics:
    Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
    Matches Table II from IJACSA paper:
    Model | Accuracy | Precision | Recall | F1-score
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_score": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }


def calculate_lead_time_to_failure(timestamps: np.ndarray, predictions: np.ndarray, lookahead_hours: float = 12.0) -> List[Dict[str, Any]]:
    """
    Computes early warning lead-time before each failure event.
    Evaluates whether the system successfully alerts operators before catastrophic breakdown occurs.
    """
    ts_series = pd.to_datetime(timestamps)
    pred_series = pd.Series(predictions, index=ts_series)
    lead_time_results = []

    for failure in FAILURE_PERIODS:
        fail_start = pd.to_datetime(failure["start"])
        window_start = fail_start - pd.Timedelta(hours=lookahead_hours)

        # Slice predictions in the lookahead window leading up to failure start
        pre_failure_preds = pred_series[(pred_series.index >= window_start) & (pred_series.index <= fail_start)]

        if pre_failure_preds.empty:
            lead_time_results.append({
                "Failure_ID": failure["id"],
                "Failure_Type": failure["failure_type"],
                "Failure_Start": str(fail_start),
                "First_Alert": "N/A",
                "Lead_Time_Hours": 0.0,
                "Status": "Not evaluated in supplied data",
            })
            continue

        # Find first sustained alert (at least 2 consecutive positive alerts or positive alert)
        anomaly_indices = pre_failure_preds[pre_failure_preds == 1].index

        if len(anomaly_indices) > 0:
            first_alert_time = anomaly_indices[0]
            lead_time_seconds = (fail_start - first_alert_time).total_seconds()
            lead_time_hours = max(0.0, lead_time_seconds / 3600.0)
            status = "Early Warning Detected"
        else:
            first_alert_time = None
            lead_time_hours = 0.0
            status = "Missed Early Warning"

        lead_time_results.append({
            "Failure_ID": failure["id"],
            "Failure_Type": failure["failure_type"],
            "Failure_Start": str(fail_start),
            "First_Alert": str(first_alert_time) if first_alert_time else "N/A",
            "Lead_Time_Hours": round(lead_time_hours, 2),
            "Status": status
        })

    return lead_time_results
