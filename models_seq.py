import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMBackbone(nn.Module):
    def __init__(self, input_features: int = 1, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, L, 1)
        out, _ = self.lstm(x)
        mean_pool = out.mean(dim=1)
        logits = self.proj(mean_pool)
        return logits


class CNNBiLSTMBackbone(nn.Module):
    def __init__(self, seq_len: int, conv_channels: int = 64, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        # Conv on (B, 1, L)
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # After two pools, length becomes L//4
        self.out_len = seq_len // 4
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, L, 1) -> (B, 1, L)
        x_c = x.transpose(1, 2)
        x_c = F.relu(self.bn1(self.conv1(x_c)))
        x_c = self.pool(x_c)
        x_c = F.relu(self.bn2(self.conv2(x_c)))
        x_c = self.pool(x_c)
        # (B, C, L') -> (B, L', C)
        x_seq = x_c.transpose(1, 2)
        out, _ = self.lstm(x_seq)
        mean_pool = out.mean(dim=1)
        logits = self.proj(mean_pool)
        return logits 