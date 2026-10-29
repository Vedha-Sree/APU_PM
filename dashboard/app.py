"""APU-Guardian dashboard backed by the locally trained Attention-LSTM-AE."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src

import torch

from src.data.labels import FAILURE_PERIODS, annotate_failures
from src.models.attention_lstm_ae import AttentionLSTMAutoencoder
from src.scheduling.maintenance_planner import MaintenanceScheduler

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "MetroPT3(AirCompressor).csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "metrics_summary.json")
BUNDLE_PATH = os.path.join(PROJECT_ROOT, "results", "models", "inference_bundle.joblib")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "models", "proposed_attention-lstm-ae.pt"
)

st.set_page_config(
    page_title="APU-Guardian | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def load_trained_model(bundle_modified_time):
    if not os.path.isfile(BUNDLE_PATH) or not os.path.isfile(CHECKPOINT_PATH):
        return None

    bundle = joblib.load(BUNDLE_PATH)
    model = AttentionLSTMAutoencoder(
        input_dim=len(bundle["feature_cols"]),
        seq_len=bundle["window_size"],
        hidden_dim1=bundle["hidden_dim1"],
        hidden_dim2=bundle["hidden_dim2"],
    )
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, bundle


@st.cache_data
def load_sensor_data(sample_step):
    features = {
        "timestamp",
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
        "COMP",
        "DV_electric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
    }

    def include_column(column):
        normalized = column.strip().replace(" ", "_")
        return normalized in features or normalized == "DV_eletric"

    df = pd.read_csv(DATA_PATH, usecols=include_column)
    df.columns = [column.strip().replace(" ", "_") for column in df.columns]
    if "DV_eletric" in df.columns and "DV_electric" not in df.columns:
        df = df.rename(columns={"DV_eletric": "DV_electric"})
    missing = features.difference(df.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").iloc[::sample_step].reset_index(drop=True)
    return annotate_failures(df, timestamp_col="timestamp")


def score_sequences(model, bundle, df):
    features = bundle["feature_cols"]
    values = bundle["scaler"].transform(df[features].to_numpy())
    window_size = bundle["window_size"]
    step_size = bundle["step_size"]
    if len(values) < window_size:
        raise ValueError(f"At least {window_size} sampled rows are required for inference.")

    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window_size, axis=0
    ).transpose(0, 2, 1)[::step_size]
    timestamps = df["timestamp"].iloc[
        np.arange(window_size - 1, len(df), step_size)
    ].reset_index(drop=True)

    scores = []
    attention = []
    with torch.inference_mode():
        for start in range(0, len(windows), 256):
            batch = torch.as_tensor(
                windows[start:start + 256].copy(), dtype=torch.float32
            )
            reconstruction, weights = model(batch)
            errors = torch.mean((batch - reconstruction) ** 2, dim=(1, 2))
            scores.extend(errors.cpu().numpy().tolist())
            attention.extend(weights.cpu().numpy())

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "error": scores,
            "attention": attention,
        }
    )


@st.cache_data(show_spinner="Scoring the sampled historical telemetry...")
def load_scored_telemetry(bundle_modified_time):
    model, bundle = load_trained_model(bundle_modified_time)
    df = load_sensor_data(bundle["sample_step"])
    return df, score_sequences(model, bundle, df)


def evaluate_modified_window(model, bundle, values):
    scaled = bundle["scaler"].transform(values)
    window = torch.as_tensor(scaled[None, :, :], dtype=torch.float32)
    with torch.inference_mode():
        reconstruction, _ = model(window)
        return float(torch.mean((window - reconstruction) ** 2).item())


st.title("APU-Guardian | Predictive Maintenance")
st.caption(
    "Offline MetroPT-3 demonstration using trained local model artifacts. "
    "This dashboard does not connect to live industrial equipment."
)

if not os.path.isfile(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}. Add the MetroPT-3 CSV and restart.")
    st.stop()

bundle_modified_time = os.path.getmtime(BUNDLE_PATH) if os.path.isfile(BUNDLE_PATH) else 0
loaded = load_trained_model(bundle_modified_time)
if loaded is None:
    st.warning(
        "No trained inference bundle is available. Run "
        "`python src/models/train_pipeline.py` from the project folder before using model features."
    )
    st.stop()

model, bundle = loaded
df, scored = load_scored_telemetry(bundle_modified_time)
threshold = float(bundle["threshold"])
if not np.isfinite(threshold) or threshold <= 0:
    st.error("The trained model bundle contains an invalid anomaly threshold.")
    st.stop()

scored["is_alert"] = scored["error"] > threshold
scored["health_index"] = np.clip(
    100 * (1 - 0.6 * scored["error"] / threshold), 0, 100
)

nav = st.sidebar.radio(
    "Dashboard",
    [
        "Sensor & Health Monitor",
        "Model Evaluation",
        "Temporal Attention",
        "Maintenance Dispatcher",
        "What-if Simulator",
    ],
)
st.sidebar.caption(f"Model: {bundle['model_name']}")
st.sidebar.caption(f"Threshold method: {bundle['threshold_method']}")
st.sidebar.caption(f"Threshold: {threshold:.6f} reconstruction MSE")

if nav == "Sensor & Health Monitor":
    st.subheader("Sampled sensor telemetry and model predictions")
    max_start = max(0, len(df) - 500)
    start = st.slider("Telemetry window start", 0, max_start, 0, step=max(1, bundle["step_size"]))
    end = min(start + 500, len(df))
    view = df.iloc[start:end]
    end_time = view["timestamp"].iloc[-1]
    window_scores = scored[scored["timestamp"] <= end_time]
    current = window_scores.iloc[-1]
    scheduler = MaintenanceScheduler(
        threshold_warning=threshold * 0.75,
        threshold_critical=threshold,
    )
    health = float(current["health_index"])
    recent = window_scores["health_index"].to_numpy()
    intervals = window_scores["timestamp"].diff().dt.total_seconds().dropna()
    hours_per_score = (
        float(intervals.median() / 3600) if not intervals.empty else 0.0
    )
    rul = scheduler.estimate_remaining_useful_life(recent, hours_per_score)

    cols = st.columns(4)
    cols[0].metric("Equipment health index", f"{health:.1f}%")
    cols[1].metric(
        "Detected state",
        "ANOMALY" if current["is_alert"] else "NORMAL",
        f"MSE {current['error']:.5f}",
    )
    cols[2].metric("Estimated RUL", f"{rul:.1f} h" if rul is not None else "Not estimable")
    cols[3].metric("Alerts in selected history", int(window_scores["is_alert"].sum()))

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=view["timestamp"], y=view["TP2"], name="TP2 pressure"))
    figure.add_trace(
        go.Scatter(x=view["timestamp"], y=view["Motor_current"], name="Motor current")
    )
    figure.update_layout(
        template="plotly_dark",
        title="Sensor history (units differ; use hover for values)",
        height=420,
        xaxis_title="Timestamp",
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        window_scores[["timestamp", "error", "health_index", "is_alert"]].tail(20),
        width="stretch",
    )

elif nav == "Model Evaluation":
    st.subheader("Held-out chronological evaluation")
    if not os.path.isfile(SUMMARY_PATH):
        st.info("No evaluation summary exists. Run the training pipeline first.")
        st.stop()
    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        summary = json.load(file)
    info = summary.get("Dataset_Info", {})
    st.write(
        f"Metric source: **{info.get('Metric_Source', 'unspecified')}**. "
        f"Test begins **{info.get('Test_Start', 'unknown')}**; "
        f"{info.get('Test_Windows', 'unknown')} windows evaluated."
    )
    reconstruction = pd.DataFrame(summary.get("Reconstruction_Metrics_Table_I", []))
    classification = pd.DataFrame(summary.get("Classification_Metrics_Table_II", []))
    if reconstruction.empty or classification.empty:
        st.warning("The saved summary does not contain model evaluation tables.")
    else:
        st.markdown("#### Reconstruction errors")
        st.dataframe(reconstruction, width="stretch")
        st.markdown("#### Anomaly classification")
        st.dataframe(classification, width="stretch")
        st.plotly_chart(
            px.bar(
                classification,
                x="Model",
                y=["Precision", "Recall", "F1_score"],
                barmode="group",
                template="plotly_dark",
                title="Held-out anomaly metrics",
            ),
            width="stretch",
        )
        st.markdown("#### Proposed-model lead-time assessment")
        lead_times = summary.get("Lead_Time_Assessments", {}).get(bundle["model_name"], [])
        if lead_times:
            st.dataframe(pd.DataFrame(lead_times), width="stretch")
        else:
            st.info("The summary contains no lead-time evaluation for the proposed model.")
        st.caption(
            "Window-level metrics overlap in time because the sliding-window step is shorter "
            "than the window length. These are offline benchmark results, not live guarantees."
        )

elif nav == "Temporal Attention":
    st.subheader("Attention weights for an actual scored window")
    selected = st.slider(
        "Scored window", 0, len(scored) - 1, len(scored) - 1
    )
    row = scored.iloc[selected]
    weights = np.asarray(row["attention"])
    chart = px.bar(
        x=np.arange(len(weights)),
        y=weights,
        labels={"x": "Timestep in the window", "y": "Attention weight"},
        title=f"Model attention for window ending {row['timestamp']}",
        template="plotly_dark",
    )
    st.plotly_chart(chart, width="stretch")
    st.metric("Reconstruction MSE", f"{row['error']:.6f}", "ANOMALY" if row["is_alert"] else "NORMAL")
    st.info(
        "These are model attention weights over timesteps, not proof of causal attribution "
        "and not sensor-specific explanations."
    )

elif nav == "Maintenance Dispatcher":
    st.subheader("Maintenance recommendation from the selected telemetry history")
    st.dataframe(pd.DataFrame(FAILURE_PERIODS), width="stretch")
    selected = st.slider(
        "Latest scored window to assess", 0, len(scored) - 1, len(scored) - 1
    )
    history = scored.iloc[:selected + 1]
    current = history.iloc[-1]
    intervals = history["timestamp"].diff().dt.total_seconds().dropna()
    hours_per_score = float(intervals.median() / 3600) if not intervals.empty else 0.0
    scheduler = MaintenanceScheduler(threshold * 0.75, threshold)
    rul = scheduler.estimate_remaining_useful_life(
        history["health_index"].to_numpy(), hours_per_score
    )
    ticket = scheduler.generate_maintenance_ticket(
        current_error=float(current["error"]),
        health_index=float(current["health_index"]),
        rul_hours=rul,
        current_timestamp=str(current["timestamp"]),
    )
    st.json(ticket)
    st.caption(
        "RUL is a short-window linear extrapolation when degradation is observable; "
        "otherwise it is reported as not estimable."
    )

else:
    st.subheader("Model-backed what-if sensor perturbation")
    if len(df) < bundle["window_size"]:
        st.error("The dataset does not contain enough rows for the configured model window.")
        st.stop()
    base = df[bundle["feature_cols"]].iloc[-bundle["window_size"]:].to_numpy(copy=True)
    pressure_drop = st.slider("TP2 pressure reduction (bar)", 0.0, 5.0, 1.0, 0.1)
    current_surge = st.slider("Motor current increase (A)", 0.0, 8.0, 2.0, 0.2)
    stuck_valve = st.checkbox("Set COMP inactive in the final 10 timesteps")
    altered = base.copy()
    pressure_index = bundle["feature_cols"].index("TP2")
    current_index = bundle["feature_cols"].index("Motor_current")
    altered[-10:, pressure_index] -= pressure_drop
    altered[-10:, current_index] += current_surge
    if stuck_valve:
        comp_index = bundle["feature_cols"].index("COMP")
        altered[-10:, comp_index] = 0
    base_error = evaluate_modified_window(model, bundle, base)
    changed_error = evaluate_modified_window(model, bundle, altered)
    cols = st.columns(3)
    cols[0].metric("Original model MSE", f"{base_error:.6f}")
    cols[1].metric("Perturbed model MSE", f"{changed_error:.6f}")
    cols[2].metric(
        "Perturbed classification",
        "ANOMALY" if changed_error > threshold else "NORMAL",
        f"Threshold {threshold:.6f}",
    )
    st.caption(
        "The what-if panel changes recent measured sensor values and scores the altered "
        "window with the trained model; it is a simulation, not a physical-failure prediction."
    )
