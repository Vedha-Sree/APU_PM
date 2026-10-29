"""
End-to-End Training and Benchmark Evaluation Pipeline
Author: Batch 11 - Predictive Maintenance System
References:
- IJACSA 2025 Paper: Section VI (Results) & Table I, Table II, Table III
- Review 1 Presentation Slides: Slide 8 (Phases 1-6)
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.preprocessor import MetroPTDataPreprocessor, ALL_SENSOR_FEATURES
from src.data.labels import FAILURE_PERIODS
from src.models.dense_autoencoder import DenseAutoencoder
from src.models.lstm_baseline import LSTMBaseline
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.attention_lstm_ae import AttentionLSTMAutoencoder
from src.evaluation.thresholding import AnomalyThresholdSelector
from src.evaluation.metrics import (
    calculate_reconstruction_metrics,
    calculate_classification_metrics,
    calculate_lead_time_to_failure
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "MetroPT3(AirCompressor).csv")
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
SUMMARY_JSON_PATH = os.path.join(BASE_DIR, "results", "metrics_summary.json")
BUNDLE_PATH = os.path.join(MODELS_DIR, "inference_bundle.joblib")


def train_single_model(model: nn.Module, train_loader, val_sequences: np.ndarray, epochs: int = 5, lr: float = 1e-3, device: str = "cpu"):
    """
    Trains an autoencoder / sequence model using MSE reconstruction loss.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            optimizer.zero_grad()
            out = model(x)
            if isinstance(out, tuple): # e.g. Attention model returns (recon, attn)
                out = out[0]
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_sequences), 512):
                val_tensor = torch.tensor(
                    val_sequences[start:start + 512], dtype=torch.float32, device=device
                )
                val_out = model(val_tensor)
                if isinstance(val_out, tuple):
                    val_out = val_out[0]
                val_losses.append(criterion(val_out, val_tensor).item())
        val_loss = float(np.mean(val_losses))
        model.train()

        print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.5f} | Val Loss: {val_loss:.5f}")

    return model


def run_pipeline(
    sample_step: int = 10,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    threshold_percentile: float = 99.5,
):
    if not 0 < threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Starting Predictive Maintenance ML Training Pipeline")
    print(f"Device: {device} | Sample Step: {sample_step} | Epochs: {epochs}")
    print(f"==================================================")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Preprocess & Sequence Formation
    preprocessor = MetroPTDataPreprocessor(window_size=30, step_size=5)
    df = preprocessor.load_data(RAW_CSV_PATH, sample_step=sample_step)

    # Keep a chronological calibration month separate from normal-only training and final evaluation.
    train_end = pd.Timestamp("2020-05-01")
    test_start = pd.Timestamp("2020-06-01")
    train_df = df[df["timestamp"] < train_end].copy().reset_index(drop=True)
    calibration_df = df[
        (df["timestamp"] >= train_end) & (df["timestamp"] < test_start)
    ].copy().reset_index(drop=True)
    test_df = df[df["timestamp"] >= test_start].copy().reset_index(drop=True)
    print(
        f"Date split -> Train: {len(train_df):,} rows (before May) | "
        f"Calibration: {len(calibration_df):,} rows (May) | "
        f"Test: {len(test_df):,} rows (June onward)"
    )
    print(f"Test anomalies: {test_df['is_anomaly'].sum():,} ({test_df['is_anomaly'].mean()*100:.2f}%)")

    train_loader, _, meta = preprocessor.build_dataloaders(
        train_df,
        test_df,
        batch_size=batch_size,
        calibration_df=calibration_df,
    )
    num_features = meta["num_features"]
    seq_len = meta["window_size"]
    X_test = meta["X_test"]
    y_test = meta["y_test"]
    ts_test = meta["ts_test"]
    X_calibration = meta["X_calibration"]
    y_calibration = meta["y_calibration"]
    normal_calibration = X_calibration[y_calibration == 0]
    if len(normal_calibration) == 0:
        raise ValueError("Calibration period has no fully normal windows; threshold cannot be calibrated.")

    models = {
        "Autoencoder": DenseAutoencoder(input_dim=num_features),
        "LSTM": LSTMBaseline(input_dim=num_features, hidden_dim=64),
        "LSTM + Autoencoder": LSTMAutoencoder(input_dim=num_features, seq_len=seq_len, hidden_dim1=64, hidden_dim2=32),
        "Proposed Attention-LSTM-AE": AttentionLSTMAutoencoder(input_dim=num_features, seq_len=seq_len, hidden_dim1=64, hidden_dim2=32)
    }

    results_table_1 = [] # Reconstruction errors (MAE, MSE, RMSE)
    results_table_2 = [] # Classification metrics (Accuracy, Precision, Recall, F1)
    lead_time_dict = {}
    model_errors = {}
    inference_bundle = None

    for name, model in models.items():
        print(f"\n>>> Training Model: {name}")
        t0 = time.time()
        trained_model = train_single_model(model, train_loader, normal_calibration, epochs=epochs, device=device)
        training_time = time.time() - t0

        # Save checkpoint
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        model_save_path = os.path.join(MODELS_DIR, f"{safe_name}.pt")
        torch.save(trained_model.state_dict(), model_save_path)

        # Batched inference keeps memory bounded on CPU and GPU.
        trained_model.eval()
        def reconstruct(sequences):
            output_batches = []
            with torch.no_grad():
                for start in range(0, len(sequences), batch_size):
                    batch = torch.tensor(
                        sequences[start:start + batch_size], dtype=torch.float32, device=device
                    )
                    output = trained_model(batch)
                    if isinstance(output, tuple):
                        output = output[0]
                    output_batches.append(output.cpu().numpy())
            return np.concatenate(output_batches, axis=0)

        recon_test = reconstruct(X_test)
        recon_calibration = reconstruct(X_calibration)

        # 1. Reconstruction Metrics
        recon_metrics = calculate_reconstruction_metrics(X_test, recon_test)
        results_table_1.append({
            "Model": name,
            "MAE": round(recon_metrics["MAE"], 6),
            "MSE": round(recon_metrics["MSE"], 6),
            "RMSE": round(recon_metrics["RMSE"], 6),
            "Train_Time_Sec": round(training_time, 2)
        })

        # Calculate error per sequence
        errors = np.mean(np.mean((X_test - recon_test) ** 2, axis=-1), axis=-1)
        model_errors[name] = errors

        # 2. Thresholding & Anomaly Detection
        calibration_errors = np.mean(
            np.mean((X_calibration - recon_calibration) ** 2, axis=-1), axis=-1
        )
        normal_calibration_errors = calibration_errors[y_calibration == 0]
        threshold_selector = AnomalyThresholdSelector(
            method="percentile", percentile=threshold_percentile
        )
        thresh = threshold_selector.fit(normal_calibration_errors)
        preds = threshold_selector.predict(errors)

        # Classification Metrics
        cls_metrics = calculate_classification_metrics(y_test, preds)
        results_table_2.append({
            "Model": name,
            "Accuracy": round(cls_metrics["Accuracy"], 4),
            "Precision": round(cls_metrics["Precision"], 4),
            "Recall": round(cls_metrics["Recall"], 4),
            "F1_score": round(cls_metrics["F1_score"], 4),
            "TP": cls_metrics["TP"],
            "FP": cls_metrics["FP"],
            "TN": cls_metrics["TN"],
            "FN": cls_metrics["FN"],
            "Threshold": round(thresh, 6),
            "Calibration_Normal_Windows": int(len(normal_calibration_errors))
        })

        # Lead Time to Failure
        lead_times = calculate_lead_time_to_failure(ts_test, preds, lookahead_hours=12.0)
        lead_time_dict[name] = lead_times
        if name == "Proposed Attention-LSTM-AE":
            inference_bundle = {
                "model_name": name,
                "feature_cols": ALL_SENSOR_FEATURES,
                "window_size": seq_len,
                "sample_step": sample_step,
                "step_size": preprocessor.step_size,
                "threshold": float(thresh),
                "threshold_method": f"normal calibration P{threshold_percentile:g}",
                "scaler": preprocessor.scaler,
                "hidden_dim1": 64,
                "hidden_dim2": 32,
            }

    if inference_bundle is None:
        raise RuntimeError("The proposed model did not produce an inference bundle.")
    joblib.dump(inference_bundle, BUNDLE_PATH)
    # Save summary JSON
    summary_data = {
        "Reconstruction_Metrics_Table_I": results_table_1,
        "Classification_Metrics_Table_II": results_table_2,
        "Lead_Time_Assessments": lead_time_dict,
        "Dataset_Info": {
            "Raw_Total_Rows": preprocessor.raw_row_count,
            "Rows_After_Downsampling": len(df),
            "Features": ALL_SENSOR_FEATURES,
            "Window_Size": seq_len,
            "Step_Size": preprocessor.step_size,
            "Downsample_Step": sample_step,
            "Train_Start": str(df["timestamp"].min()),
            "Train_End_Exclusive": str(train_end),
            "Calibration_Start": str(train_end),
            "Calibration_End_Exclusive": str(test_start),
            "Test_Start": str(test_start),
            "Train_Normal_Windows": len(train_loader.dataset),
            "Calibration_Normal_Windows": int(np.sum(y_calibration == 0)),
            "Test_Windows": len(y_test),
            "Test_Anomaly_Windows": int(np.sum(y_test)),
            "Seed": seed,
            "Epochs": epochs,
            "Batch_Size": batch_size,
            "Threshold_Method": f"normal_calibration_percentile_{threshold_percentile:g}",
            "Metric_Source": "Fresh local training run",
        },
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\n[OK] Metrics summary saved to: {SUMMARY_JSON_PATH}")

    # Generate Publication-Grade Visualizations
    generate_figures(results_table_1, results_table_2, model_errors, ts_test, y_test, FIGURES_DIR, df)
    print(f"[OK] Inference bundle saved to: {BUNDLE_PATH}")
    print("[SUCCESS] All pipeline stages completed successfully!")
    return summary_data


def generate_figures(table1, table2, model_errors, ts_test, y_test, figures_dir, sensor_df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Model Reconstruction Error Comparison (MAE, MSE, RMSE)
    df_t1 = pd.DataFrame(table1).set_index("Model")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_t1[["MAE", "MSE", "RMSE"]].plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Figure 5: Reconstruction Errors from the Local Experiment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Metric Value")
    ax.set_xticklabels(df_t1.index, rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig5_reconstruction_errors_comparison.png"))
    plt.close()

    # Figure 2: Classification Performance Metrics
    df_t2 = pd.DataFrame(table2).set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    df_t2[["Accuracy", "Precision", "Recall", "F1_score"]].plot(kind="bar", ax=ax, colormap="coolwarm")
    ax.set_title("Figure 6: Model Performance Metrics (Accuracy, Precision, Recall, F1)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(df_t2.index, rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig6_classification_metrics_comparison.png"))
    plt.close()

    # Figure 3: Reconstruction Error Distribution & Threshold Anomaly Separation
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    proposed_errors = model_errors.get("Proposed Attention-LSTM-AE", list(model_errors.values())[0])
    normal_errs = proposed_errors[y_test == 0]
    anomaly_errs = proposed_errors[y_test == 1]

    ax.hist(normal_errs, bins=50, alpha=0.6, label="Normal Operation", color="teal", density=True)
    if len(anomaly_errs) > 0:
        ax.hist(anomaly_errs, bins=50, alpha=0.6, label="Confirmed Anomaly / Failure", color="crimson", density=True)

    thresh = float(df_t2.loc["Proposed Attention-LSTM-AE", "Threshold"]) if "Proposed Attention-LSTM-AE" in df_t2.index else np.mean(normal_errs) + 2.5 * np.std(normal_errs)
    ax.axvline(thresh, color="red", linestyle="--", linewidth=2, label=f"Anomaly Threshold ({thresh:.4f})")
    ax.set_title("Figure 4: Reconstruction Error Distribution (Proposed Attention-LSTM-AE)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig4_reconstruction_error_distribution.png"))
    plt.close()

    # Plot the complete sampled timeline so each labeled event is visible.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=180, sharex=True)
    axes[0].plot(sensor_df["timestamp"], sensor_df["TP2"], linewidth=0.5, color="#38bdf8")
    axes[0].set_ylabel("TP2 pressure (bar)")
    axes[1].plot(sensor_df["timestamp"], sensor_df["Motor_current"], linewidth=0.5, color="#eab308")
    axes[1].set_ylabel("Motor current (A)")
    axes[1].set_xlabel("Timestamp")
    for axis in axes:
        for failure in FAILURE_PERIODS:
            axis.axvspan(
                pd.Timestamp(failure["start"]),
                pd.Timestamp(failure["end"]),
                color="red",
                alpha=0.18,
            )
    fig.suptitle("MetroPT-3 Sensor Telemetry and Confirmed Failure Windows")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig3_sensor_telemetry_timeline.png"))
    plt.close(fig)

    print(f"[OK] Publication figures saved to: {figures_dir}")


if __name__ == "__main__":
    run_pipeline(sample_step=10)
