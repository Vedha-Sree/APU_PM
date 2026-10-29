"""
Ground Truth Failure Labels and Maintenance Events for MetroPT-3
Author: Batch 11 - Predictive Maintenance System
References:
- Veloso et al. (Scientific Data 2022)
- Davari et al. (IEEE DSAA 2021)
- Akshitha Mary et al. (IJACSA 2025)
"""

import pandas as pd
import numpy as np

# Official failure periods documented by metro maintenance teams
FAILURE_PERIODS = [
    {
        "id": "F1",
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:00",
        "failure_type": "Air Leak",
        "severity": "High Stress",
        "report": "Pressure drop in pneumatic system, continuous compressor loading"
    },
    {
        "id": "F2",
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "failure_type": "Air Leak",
        "severity": "High Stress",
        "report": "Air dryer valve malfunction, scheduled maintenance performed"
    },
    {
        "id": "F3",
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "failure_type": "Air Leak",
        "severity": "High Stress",
        "report": "Critical leak in compressor intake, extensive continuous operation"
    },
    {
        "id": "F4",
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "failure_type": "Air Leak",
        "severity": "High Stress",
        "report": "Intake valve stuck open, abnormal motor current surge"
    }
]


def annotate_failures(df: pd.DataFrame, timestamp_col: str = "timestamp", early_warning_hours: float = 2.0) -> pd.DataFrame:
    """
    Annotates the dataframe with:
    - 'is_anomaly': 1 if timestamp falls inside any confirmed failure period, else 0.
    - 'is_early_warning': 1 if timestamp falls within early_warning_hours before any failure.
    - 'failure_id': ID of the failure event ('F1', 'F2', etc.) or 'Normal'.
    """
    df = df.copy()
    if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df["is_anomaly"] = 0
    df["is_early_warning"] = 0
    df["failure_id"] = "Normal"

    for failure in FAILURE_PERIODS:
        start_ts = pd.to_datetime(failure["start"])
        end_ts = pd.to_datetime(failure["end"])
        warning_start_ts = start_ts - pd.Timedelta(hours=early_warning_hours)

        # Flag true failure window
        failure_mask = (df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)
        df.loc[failure_mask, "is_anomaly"] = 1
        df.loc[failure_mask, "failure_id"] = failure["id"]

        # Flag early warning incubation window (pre-failure state)
        warning_mask = (df[timestamp_col] >= warning_start_ts) & (df[timestamp_col] < start_ts)
        df.loc[warning_mask, "is_early_warning"] = 1

    return df


def get_failure_summary():
    """Returns a pandas DataFrame summary of ground-truth failures."""
    return pd.DataFrame(FAILURE_PERIODS)
