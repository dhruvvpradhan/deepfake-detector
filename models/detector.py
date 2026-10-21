import torch
import torch.nn as nn
import torchvision.models as models
from models.layers import FFTFrequencyLayer

class DualBranchDetector(nn.Module):
    def __init__(self, sequence_length=20, hidden_dim=256, fft_dim=128, dropout=0.5):
        super().__init__()
        self.seq_len = sequence_length

        # ----------------------------------------------------
        # Branch 1: Spatial-Temporal Stream (Motion & Blending)
        # ----------------------------------------------------
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        # Strip final FC layer -> Outputs 2048-dim feature representation
        self.spatial_backbone = nn.Sequential(*list(resnext.children())[:-1])
        
        self.temporal_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # ----------------------------------------------------
        # Branch 2: Frequency Domain Stream (GAN Grid Residuals)
        # ----------------------------------------------------
        self.fft_layer = FFTFrequencyLayer()
        self.fft_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, fft_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fft_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # ----------------------------------------------------
        # Multi-Modal Fusion & Classification Head
        # ----------------------------------------------------
        fused_dim = (hidden_dim * 2) + fft_dim  # 512 + 128 = 640
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)

        # 1. Spatial-Temporal Processing
        spatial_feats = self.spatial_backbone(x_flat).view(b, t, -1)
        lstm_out, _ = self.temporal_lstm(spatial_feats)
        branch1_out = lstm_out[:, -1, :]  # Terminal frame output: (B, 512)

        # 2. Frequency Domain Processing
        fft_spectra = self.fft_layer(x_flat)
        fft_feats = self.fft_cnn(fft_spectra).view(b, t, -1)
        branch2_out = torch.mean(fft_feats, dim=1)  # Temporal mean pool: (B, 128)

        # 3. Concatenation and Binary Classification
        fused = torch.cat((branch1_out, branch2_out), dim=1)  # (B, 640)
        prediction = self.classifier(fused)                   # (B, 1)

        return prediction
