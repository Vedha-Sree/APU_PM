"""
Stacked LSTM Baseline for Time-Series Anomaly Detection
Author: Batch 11 - Predictive Maintenance System
Reference: IJACSA 2025 Paper - Section V-B (1) LSTM Architecture
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMBaseline, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, num_features]
        Outputs reconstructed / predicted sequence of identical shape.
        """
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        out = self.fc(lstm_out)    # [batch_size, seq_len, input_dim]
        return out

    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error_per_timestep = torch.mean((x - reconstructed) ** 2, dim=-1)
            if reduction == "mean":
                return torch.mean(error_per_timestep, dim=-1)
            return error_per_timestep
