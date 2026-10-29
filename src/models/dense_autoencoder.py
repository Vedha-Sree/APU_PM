"""
Dense Autoencoder Baseline for Sensor Time-Series Anomaly Detection
Author: Batch 11 - Predictive Maintenance System
Reference: IJACSA 2025 Paper - Section V-B (2) Autoencoder Architecture
"""

import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, bottleneck_dim: int = 16):
        super(DenseAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, bottleneck_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, num_features]
        Processes each timestep through the dense autoencoder.
        """
        batch_size, seq_len, num_features = x.shape
        x_flat = x.view(-1, num_features)
        latent = self.encoder(x_flat)
        reconstructed_flat = self.decoder(latent)
        reconstructed = reconstructed_flat.view(batch_size, seq_len, num_features)
        return reconstructed

    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Computes reconstruction error (MSE or MAE per sequence).
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error_per_timestep = torch.mean((x - reconstructed) ** 2, dim=-1) # [batch, seq_len]
            if reduction == "mean":
                return torch.mean(error_per_timestep, dim=-1) # [batch]
            return error_per_timestep
