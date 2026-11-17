"""Independent neural forecasters for paired sensible/latent flux series."""
from __future__ import annotations

from ..registry import LEVEL2

import torch
from torch import nn
import torch.nn.functional as F


def default_config(name: str, hidden_size: int | None = None,
                   dropout: float = 0.1) -> dict:
    if name in LEVEL2:
        from .advanced import advanced_config
        return advanced_config(name, hidden_size, dropout)
    if name == "mlp":
        hidden = [128, 64] if hidden_size is None else [
            int(hidden_size), max(8, int(hidden_size) // 2)
        ]
        return {"hidden_sizes": hidden, "dropout": float(dropout)}
    if name == "tcn":
        return {
            "channels": 32 if hidden_size is None else int(hidden_size),
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "dropout": float(dropout),
        }
    if name == "gru":
        return {
            "hidden_size": 64 if hidden_size is None else int(hidden_size),
            "num_layers": 1,
            "dropout": float(dropout),
        }
    raise ValueError(f"Unknown neural model: {name}")


class MLPForecaster(nn.Module):
    """Flattened-history MLP with a direct multi-horizon output."""

    def __init__(self, input_length: int, horizon: int, hidden_sizes,
                 dropout: float, input_channels: int = 2):
        super().__init__()
        layers = []
        width = input_length * input_channels
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(width, int(hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            width = int(hidden)
        layers.append(nn.Linear(width, horizon * 2))
        self.network = nn.Sequential(*layers)
        self.horizon = horizon

    def forward(self, x):
        return self.network(x.reshape(x.shape[0], -1)).reshape(
            x.shape[0], self.horizon, 2
        )


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left-only temporal padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        y = self.dropout(self.activation(self.conv1(x)))
        y = self.dropout(self.activation(self.conv2(y)))
        return self.activation(y + residual)


class TCNForecaster(nn.Module):
    """Causal dilated TCN with a direct multi-horizon output head."""

    def __init__(self, horizon: int, channels: int, kernel_size: int,
                 dilations, dropout: float, input_channels: int = 2):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            TemporalResidualBlock(channels, kernel_size, int(d), dropout)
            for d in dilations
        ])
        self.head = nn.Linear(channels, horizon * 2)
        self.horizon = horizon

    def forward(self, x):
        # Input is [batch, time, channels].
        y = self.input_projection(x.transpose(1, 2))
        y = self.blocks(y)
        return self.head(y[:, :, -1]).reshape(x.shape[0], self.horizon, 2)


class GRUForecaster(nn.Module):
    """Single-layer GRU followed by a direct multi-horizon output head."""

    def __init__(self, horizon: int, hidden_size: int, num_layers: int,
                 dropout: float, input_channels: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon * 2)
        self.horizon = horizon

    def forward(self, x):
        sequence, _ = self.gru(x)
        last = self.dropout(sequence[:, -1])
        return self.head(last).reshape(x.shape[0], self.horizon, 2)


def build_model(name: str, input_length: int, horizon: int,
                config: dict | None = None, input_channels: int = 2) -> nn.Module:
    config = config or default_config(name)
    if name in LEVEL2:
        from .advanced import advanced_model
        return advanced_model(name, input_length, horizon, input_channels, config)
    if name == "mlp":
        return MLPForecaster(
            input_length, horizon, config["hidden_sizes"], config["dropout"],
            input_channels=input_channels
        )
    if name == "tcn":
        return TCNForecaster(
            horizon, config["channels"], config["kernel_size"],
            config["dilations"], config["dropout"],
            input_channels=input_channels
        )
    if name == "gru":
        return GRUForecaster(
            horizon, config["hidden_size"], config["num_layers"],
            config["dropout"], input_channels=input_channels
        )
    raise ValueError(f"Unknown neural model: {name}")
