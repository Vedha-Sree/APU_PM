"""
LSTM-Autoencoder (Encoder-Decoder) for Multivariate Time-Series Anomaly Detection
Author: Batch 11 - Predictive Maintenance System
Reference: IJACSA 2025 Paper - Section V-B (3) and Figure 3
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout: float = 0.2):
        super(LSTMEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, (h_n, _) = self.lstm2(out1)
        # Latent representation is the final hidden state of the second LSTM
        bottleneck = h_n[-1] # [batch_size, hidden_dim2]
        return bottleneck, out2


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim: int, seq_len: int, hidden_dim2: int = 64, hidden_dim1: int = 128, dropout: float = 0.2):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(hidden_dim2, hidden_dim2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.time_distributed_dense = nn.Linear(hidden_dim1, output_dim)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        # bottleneck: [batch_size, hidden_dim2]
        # RepeatVector: expand latent vector to sequence length
        repeated = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch_size, seq_len, hidden_dim2]
        out1, _ = self.lstm1(repeated)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)
        reconstruction = self.time_distributed_dense(out2) # [batch_size, seq_len, output_dim]
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Combined LSTM-Autoencoder Architecture (Fig. 3 from IJACSA 2025 Paper)
    """
    def __init__(self, input_dim: int, seq_len: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout: float = 0.2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(input_dim, hidden_dim1, hidden_dim2, dropout)
        self.decoder = LSTMDecoder(input_dim, seq_len, hidden_dim2, hidden_dim1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed

    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error_per_timestep = torch.mean((x - reconstructed) ** 2, dim=-1)
            if reduction == "mean":
                return torch.mean(error_per_timestep, dim=-1)
            return error_per_timestep
