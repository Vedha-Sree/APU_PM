"""
Attention-Enhanced LSTM-Autoencoder for Explainable Industrial Anomaly Detection
Author: Batch 11 - Predictive Maintenance System
References:
- Review 1 Presentation Slides: Slide 4 & Slide 6 (Attention Mechanism Exploration)
- Jiang et al. (A2-LSTM for Predictive Maintenance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    Computes Bahdanau-style temporal attention weights across sequence timesteps.
    Identifies which timestamps contribute most to normal vs anomalous compressor behavior.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        energy = torch.tanh(self.attn(encoder_outputs)) # [batch_size, seq_len, hidden_dim]
        scores = self.v(energy).squeeze(-1)            # [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # Context vector as weighted sum of encoder hidden representations
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1) # [batch_size, hidden_dim]
        return context_vector, attention_weights


class AttentionLSTMAutoencoder(nn.Module):
    """
    Proposed Model: LSTM-Autoencoder with Temporal Attention Mechanism.
    Combines sequence learning, bottleneck dimensionality reduction, and dynamic temporal focusing.
    """
    def __init__(self, input_dim: int, seq_len: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout: float = 0.2):
        super(AttentionLSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim2 = hidden_dim2

        # Encoder
        self.enc_lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.enc_dropout = nn.Dropout(dropout)
        self.enc_lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)

        # Attention Layer
        self.attention = TemporalAttention(hidden_dim2)

        # Decoder
        self.dec_lstm1 = nn.LSTM(hidden_dim2, hidden_dim2, batch_first=True)
        self.dec_dropout = nn.Dropout(dropout)
        self.dec_lstm2 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim1, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, input_dim]
        out1, _ = self.enc_lstm1(x)
        out1 = self.enc_dropout(out1)
        out2, _ = self.enc_lstm2(out1) # [batch_size, seq_len, hidden_dim2]

        # Apply Temporal Attention over encoder outputs
        context_vector, attn_weights = self.attention(out2) # [batch_size, hidden_dim2], [batch_size, seq_len]

        # Decoder repeats the attention context vector across the sequence
        repeated = context_vector.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch_size, seq_len, hidden_dim2]
        dec1, _ = self.dec_lstm1(repeated)
        dec1 = self.dec_dropout(dec1)
        dec2, _ = self.dec_lstm2(dec1)
        reconstruction = self.fc_out(dec2) # [batch_size, seq_len, input_dim]

        return reconstruction, attn_weights

    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            reconstructed, attn_weights = self.forward(x)
            error_per_timestep = torch.mean((x - reconstructed) ** 2, dim=-1)
            if reduction == "mean":
                return torch.mean(error_per_timestep, dim=-1), attn_weights
            return error_per_timestep, attn_weights
