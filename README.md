# APU-Guardian: AI-Driven Predictive Maintenance System 🛡️

<p align="center">
  <img src="https://img.shields.io/badge/SDG%209-Industry%20%26%20Innovation-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SDG%2012-Responsible%20Consumption-gold?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-blue?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Domain-Industrial%20AI-blueviolet?style=for-the-badge" />
</p>

---

## 📌 Project Overview

**APU-Guardian** is an offline predictive-maintenance prototype for the MetroPT-3 air production unit (APU) dataset. It trains reconstruction-based time-series anomaly detectors, evaluates them against documented failure windows, and demonstrates maintenance recommendations in a local dashboard. It is not connected to live equipment and does not guarantee advance warning of failures.

> **Course-Based Project (CBP) | Batch 11**
> B. Rishika Reddy | E. Tejaswi | J. Vedha Sree

---

## 🎯 Objectives

| Objective | Description |
|-----------|-------------|
| **Equipment Health Monitoring** | Inspect sampled historical sensor channels (pressure, temperature, current, valve states) |
| **Anomaly Detection** | Unsupervised temporal anomaly detection using reconstruction error analysis |
| **Failure Evaluation** | Evaluate anomaly windows against documented failure periods on a held-out chronological period |
| **Maintenance Scheduling** | Generate demonstration maintenance tickets; estimate RUL only when a degradation trend is observed |

---

## 🗂️ Dataset: MetroPT-3

> **Source**: UCI Machine Learning Repository (ID: 791). The downloaded CSV and ZIP are intentionally excluded from this public repository because each exceeds GitHub's 100 MB file limit; use the downloader below to obtain them locally.
> **Paper**: *Davari, N., Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. (2021). Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. IEEE DSAA 2021.*

| Property | Value |
|----------|-------|
| **Instances** | 1,516,948 observations |
| **Sampling Rate** | Approximately one observation every 10 seconds in the source dataset; the default experiment retains every 10th row |
| **Model window** | 30 sampled observations (about 50 minutes) with a 5-observation step (about 8 minutes) |
| **Duration** | February – September 2020 (7 months) |
| **Features** | 15 signals (7 analogue + 8 digital) |
| **Ground-Truth Failures** | 4 confirmed Air Leak events (unlabeled) |

### 🔬 Features Used

#### Analogue Sensors (Continuous, in bar / °C / A)
| Feature | Description |
|---------|-------------|
| `TP2` | Pressure on the compressor output |
| `TP3` | Pressure at the pneumatic panel |
| `H1` | Pressure drop at cyclonic separator filter |
| `DV_pressure` | Pressure drop at air dryer discharge valves |
| `Reservoirs` | Downstream reservoir pressure |
| `Motor_current` | Current draw (one phase of 3-phase motor) |
| `Oil_temperature` | Compressor lubrication oil temperature |

#### Digital Sensors (Binary Control Signals)
| Feature | Description |
|---------|-------------|
| `COMP` | Intake valve active state |
| `DV_electric` | Outlet valve control |
| `Towers` | Active drying tower (1 or 2) |
| `MPG` | Load startup activation |
| `LPS` | Low-pressure alarm (<7 bar) |
| `Pressure_switch` | Dryer tower discharge trigger |
| `Oil_level` | Oil level alarm |
| `Caudal_impulses` | Air flow pulse counter |

---

## 🧠 Model Architectures

### 1. Dense Autoencoder (Baseline)
Standard feedforward encoder-decoder: Input → 64 → 32 → 16 (bottleneck) → 32 → 64 → Output.

### 2. Stacked LSTM (Baseline)
2-layer LSTM capturing temporal sensor dependencies for sequence reconstruction.

### 3. LSTM-Autoencoder baseline
Encoder: 2-layer LSTM → compressed latent vector (RepeatVector) → Decoder: 2-layer LSTM → Time-Distributed Dense output.

### 4. ⭐ Proposed: Attention-Enhanced LSTM-Autoencoder
Same architecture as #3 **plus** a Bahdanau-style **Temporal Attention Layer** over encoder outputs:
- Dynamically assigns importance weights $α_t$ to each timestep in the window.
- Returns attention weights over timesteps for inspection. These weights are not causal explanations and do not identify which sensor caused an alert.

---

## 📊 Experimental Results

The committed metrics below come from a fresh local training run; they are not copied from a paper or a dashboard fallback. The experiment trains on normal windows before May 2020, calibrates on normal windows in May, and evaluates on June 2020 onward. Results and split details are in `results/metrics_summary.json`.

### Held-out classification results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Autoencoder | 0.9673 | 0.2941 | 0.0769 | 0.1220 |
| LSTM | 0.9645 | 0.1938 | 0.0641 | 0.0963 |
| LSTM + Autoencoder | 0.9673 | 0.3084 | 0.0846 | 0.1328 |
| Proposed Attention-LSTM-AE | 0.9673 | 0.2981 | 0.0795 | 0.1255 |

The test set contains 390 anomalous windows out of 13,197. The proposed model detected 31 and missed 359; it produced 73 false-positive windows. The percentile threshold reduced false positives from 391 to 73 compared with the earlier mean-plus-2.5-standard-deviations threshold on the same held-out split, and improved F1 from 0.0858 to 0.1255, at the cost of slightly lower recall. This is a threshold trade-off, not proof of reliable failure prediction: it did not issue an alert in the 12-hour pre-failure window for either held-out event (F3 or F4). F1 and F2 are outside this test period and are reported as not evaluated. **These results do not support operational deployment or a claim that the system predicts failures reliably.** The pipeline and dashboard are end-to-end implemented, but model efficacy remains a limitation requiring further research and independent validation.

---

## 📁 Project Structure

```
AI CBP/
├── data/
│   ├── raw/                         # Downloaded locally; not committed
│   └── processed/                   # Cached sequences (auto-generated)
├── src/
│   ├── data/
│   │   ├── downloader.py            # Automated UCI dataset downloader
│   │   ├── labels.py                # Ground-truth failure period annotations
│   │   └── preprocessor.py          # Normalization, sliding windows, DataLoaders
│   ├── runtime.py                   # Windows runtime DLL setup for PyTorch
│   ├── models/
│   │   ├── dense_autoencoder.py     # Feedforward Autoencoder baseline
│   │   ├── lstm_baseline.py         # Stacked LSTM baseline
│   │   ├── lstm_autoencoder.py      # Hybrid LSTM-Autoencoder (IJACSA)
│   │   ├── attention_lstm_ae.py     # ⭐ Proposed Attention-LSTM-AE
│   │   └── train_pipeline.py        # End-to-end training & evaluation pipeline
│   ├── evaluation/
│   │   ├── thresholding.py          # Statistical & adaptive threshold strategies
│   │   └── metrics.py               # Reconstruction + Classification metrics
│   └── scheduling/
│       └── maintenance_planner.py   # Health Index, RUL estimation, ticket dispatch
├── dashboard/
│   └── app.py                       # 🖥️ Streamlit Interactive Web Dashboard
├── PM_Final_Review.pptx             # Updated review deck with local experiment results
├── results/
│   ├── models/                      # Saved PyTorch checkpoints (.pt)
│   │   └── inference_bundle.joblib  # Fitted scaler + model inference settings
│   ├── figures/                     # Generated result plots
│   └── metrics_summary.json         # Local benchmark results
├── tests/
│   └── test_core.py                 # Metrics, model, event, and scheduler tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies (Python 3.10+)
```bash
pip install -r requirements.txt
```

### 2. Download Dataset (Automatic)
```bash
python src/data/downloader.py
```

### 3. Run Training Pipeline
```bash
python src/models/train_pipeline.py
```

### 4. Run the core tests
```bash
python -m unittest discover -s tests
```

### 5. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at **http://localhost:8501**. The dashboard requires the trained checkpoint and inference bundle produced by step 3. It scores historical telemetry offline; it does not provide a live IIoT connection.

### Reproducibility notes

- The training pipeline uses a chronological train/calibration/test split. Training windows overlapping labeled failures are excluded.
- The default threshold is the 99.5th percentile of normal calibration scores. Its selection was made on calibration-only score sweeps and it is never fitted on the final test period.
- Test metrics are window-level; adjacent windows overlap and are not independent observations.
- RUL is a short-window linear extrapolation and is reported as unavailable if the history is too short or stable.

---

## 🌍 Sustainable Development Goals (SDG) Alignment

| SDG | Target | APU-Guardian Contribution |
|-----|--------|--------------------------|
| **SDG 9** – Industry, Innovation & Infrastructure | 9.1, 9.4 | Resilient, sustainable transportation infrastructure via predictive uptime management |
| **SDG 12** – Responsible Consumption & Production | 12.2, 12.5 | Prevents premature component disposal; cuts energy waste from parasitic compressor loading |

---

## 📚 References

1. Davari, N., Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. (2021). *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry*. IEEE DSAA 2021.
2. Veloso, B., Gama, J., Ribeiro, R.P., Pereira, P.M. (2022). *MetroPT: A benchmark dataset for predictive maintenance*. Scientific Data.
3. Akshitha Mary, A., Chandra Sekar, P. (2025). *AI-Driven Predictive Maintenance for Metro Operations Using LSTM Autoencoder*. IJACSA, Vol. 16 No. 4.

---

*APU-Guardian v1.0 | Batch 11 Course-Based Project*
