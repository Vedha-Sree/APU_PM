"""
Data Preprocessing and Sequence Generation for MetroPT-3
Author: Batch 11 - Predictive Maintenance System
References:
- IJACSA 2025 Paper: Section V-A Preprocessing
- Best ML Practices: Chronological split & Strict Featurization Ordering
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.labels import annotate_failures

ANALOG_FEATURES = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"
]

DIGITAL_FEATURES = [
    "COMP", "DV_electric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"
]

ALL_SENSOR_FEATURES = ANALOG_FEATURES + DIGITAL_FEATURES


class TimeSeriesSequenceDataset(Dataset):
    """PyTorch Dataset for sliding-window time series sequences."""
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None, timestamps: Optional[np.ndarray] = None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.timestamps = timestamps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class MetroPTDataPreprocessor:
    def __init__(self, feature_cols: Optional[List[str]] = None, window_size: int = 30, step_size: int = 5):
        self.feature_cols = feature_cols or ALL_SENSOR_FEATURES
        self.window_size = window_size
        self.step_size = step_size
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.raw_row_count = 0

    def load_data(self, csv_path: str, sample_step: int = 1) -> pd.DataFrame:
        """
        Loads the CSV, standardizes column names, parses timestamp,
        handles missing values using forward-fill, and annotates ground-truth anomalies.
        sample_step: downsampling step if training on local machines (e.g., 5 or 10s for fast iteration).
        """
        print(f"Loading data from: {csv_path} (step={sample_step})...")
        df = pd.read_csv(csv_path)
        self.raw_row_count = len(df)

        # Standardize column names (strip whitespace and common variations)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        if "DV_eletric" in df.columns and "DV_electric" not in df.columns:
            df.rename(columns={"DV_eletric": "DV_electric"}, inplace=True)

        # Ensure timestamp is parsed and sorted chronologically
        timestamp_col = None
        for cand in ["timestamp", "Time", "datetime", "date"]:
            if cand in df.columns:
                timestamp_col = cand
                break

        if timestamp_col is None:
            # First column is usually timestamp if unnamed or named index
            timestamp_col = df.columns[1] if df.columns[0] in ["Unnamed:_0", "index"] else df.columns[0]

        df["timestamp"] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if sample_step > 1:
            df = df.iloc[::sample_step].reset_index(drop=True)

        # Ensure all required features are present
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing expected sensor columns in dataset: {missing_cols}")

        # Data Cleaning: Forward-fill followed by backward-fill and feature mean for residual NaNs
        df[self.feature_cols] = df[self.feature_cols].ffill().bfill()
        for col in self.feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

        # Annotate ground truth failure labels
        df = annotate_failures(df, timestamp_col="timestamp")
        print(f"Loaded {len(df):,} records spanning {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Ground-truth anomalies: {df['is_anomaly'].sum():,} points ({df['is_anomaly'].mean()*100:.2f}%)")
        return df

    def split_train_test_chronological(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/test split.
        Per Section V-A and ML best practices: 80% train (normal baseline) and 20% test (unseen evaluation).
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"Chronological Split -> Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
        return train_df, test_df

    def fit_scaler(self, train_df: pd.DataFrame) -> None:
        """
        Fit scaler STRICTLY on normal data in training set to avoid data leakage.
        """
        # Train solely on normal operating regime
        normal_train = train_df[train_df["is_anomaly"] == 0]
        if len(normal_train) == 0:
            raise ValueError("Cannot fit the scaler: the training period contains no normal observations.")

        self.scaler.fit(normal_train[self.feature_cols].values)
        self.is_fitted = True
        print(f"[OK] StandardScaler fitted on {len(normal_train):,} normal records.")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted! Call fit_scaler() first.")
        return self.scaler.transform(df[self.feature_cols].values)

    def create_sequences(self, data_scaled: np.ndarray, labels: Optional[np.ndarray] = None, timestamps: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Creates 3D sliding window sequences of shape [num_windows, window_size, num_features].
        Label for window is 1 if any point inside window is an anomaly.
        """
        num_records = len(data_scaled)
        sequences = []
        seq_labels = []
        seq_timestamps = []

        for i in range(0, num_records - self.window_size + 1, self.step_size):
            window = data_scaled[i : i + self.window_size]
            sequences.append(window)

            if labels is not None:
                # If any timestamp in window is an anomaly, label sequence as anomaly
                window_label = 1 if np.any(labels[i : i + self.window_size] == 1) else 0
                seq_labels.append(window_label)

            if timestamps is not None:
                # Window timestamp is the last timestamp in the window (current inference time)
                seq_timestamps.append(timestamps.iloc[i + self.window_size - 1])

        seq_arr = np.array(sequences, dtype=np.float32)
        label_arr = np.array(seq_labels, dtype=np.int32) if labels is not None else None
        ts_arr = np.array(seq_timestamps) if timestamps is not None else None

        return seq_arr, label_arr, ts_arr

    def build_dataloaders(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 64,
        calibration_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Build normal-only training batches and chronological calibration/test windows.
        """
        self.fit_scaler(train_df)
        train_scaled = self.transform(train_df)
        test_scaled = self.transform(test_df)

        X_train, y_train, ts_train = self.create_sequences(
            train_scaled, train_df["is_anomaly"].values, train_df["timestamp"]
        )
        X_test, y_test, ts_test = self.create_sequences(
            test_scaled, test_df["is_anomaly"].values, test_df["timestamp"]
        )

        normal_train_mask = y_train == 0
        X_train = X_train[normal_train_mask]
        y_train = y_train[normal_train_mask]
        ts_train = ts_train[normal_train_mask]
        if len(X_train) == 0:
            raise ValueError("No fully normal training windows remain after excluding failure periods.")

        X_calibration = np.empty((0, self.window_size, len(self.feature_cols)), dtype=np.float32)
        y_calibration = np.empty(0, dtype=np.int32)
        ts_calibration = np.empty(0, dtype="datetime64[ns]")
        if calibration_df is not None:
            calibration_scaled = self.transform(calibration_df)
            X_calibration, y_calibration, ts_calibration = self.create_sequences(
                calibration_scaled,
                calibration_df["is_anomaly"].values,
                calibration_df["timestamp"],
            )

        print(
            f"Generated Sequences -> X_train: {X_train.shape} (normal only) | "
            f"X_calibration: {X_calibration.shape} | X_test: {X_test.shape}"
        )
        if y_test is not None:
            print(f"Anomaly sequences in test set: {np.sum(y_test):,} / {len(y_test):,} ({np.mean(y_test)*100:.2f}%)")

        train_dataset = TimeSeriesSequenceDataset(X_train, y_train, ts_train)
        test_dataset = TimeSeriesSequenceDataset(X_test, y_test, ts_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            "num_features": len(self.feature_cols),
            "window_size": self.window_size,
            "feature_cols": self.feature_cols,
            "ts_train": ts_train,
            "ts_test": ts_test,
            "y_test": y_test,
            "X_test": X_test,
            "X_calibration": X_calibration,
            "y_calibration": y_calibration,
            "ts_calibration": ts_calibration,
            "X_train": X_train,
        }

        return train_loader, test_loader, metadata
