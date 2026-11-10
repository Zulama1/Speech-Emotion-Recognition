import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCRNN(nn.Module):
    """
    CRNN architecture: 2D CNN -> Bidirectional LSTM -> Fully Connected Classifier
    Input shape: (batch_size, 1, n_mels, time_steps)
    """
    def __init__(
        self,
        num_classes: int = 8,
        n_mels: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Feature Extractor (CNN)
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        # Dimension reduction along frequency axis (128 -> 64 -> 32 -> 16)
        freq_reduced = n_mels // (2 ** 3)
        self.feature_dim = 128 * freq_reduced

        # Sequence Modeling (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN Feature maps: (batch, 128, n_mels // 8, time_steps // 8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 2. Reshape for LSTM: (batch, time, feature_dim)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, channels * freq)

        # 3. Temporal aggregation
        lstm_out, _ = self.lstm(x)
        out = torch.mean(lstm_out, dim=1)  # Temporal mean pooling

        # 4. Logits
        return self.classifier(out)