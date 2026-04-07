"""
Model definitions for the Shoplifting Detection CNN from scratch.
Identical to src/model.py – re-exposed here for clean Django imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models


# ─────────────────────────────────────────────────────────────────────────────
class ThreeDCNN(nn.Module):
    """3-D Convolutional Neural Network for video classification."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1     = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
class CNNRNN(nn.Module):
    """Frame-level CNN feature extractor + LSTM temporal model."""

    def __init__(self, num_classes=2, hidden_size=128, num_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.lstm    = nn.LSTM(512, hidden_size, num_layers,
                               batch_first=True, dropout=0.3)
        self.fc      = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.fc(self.dropout(out[:, -1, :]))


# ─────────────────────────────────────────────────────────────────────────────
class VideoTransformer(nn.Module):
    """Frame-level CNN + Transformer encoder for video classification."""

    def __init__(self, num_classes=2, d_model=512, nhead=8,
                 num_layers=4, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.d_model = d_model

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).view(B, T, self.d_model)
        feats = feats + self.pos_embedding[:, :T, :]
        out   = self.transformer(feats)
        return self.fc(self.dropout(out.mean(dim=1)))


# ─────────────────────────────────────────────────────────────────────────────
class PretrainedR3D(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super(PretrainedR3D, self).__init__()
        
        self.model = video_models.r3d_18(pretrained=True)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        # x shape: (batch, num_frames, C, H, W)
        batch_size, num_frames, C, H, W = x.shape
        
        # Permute to (batch, C, num_frames, H, W) for 3D CNN
        x = x.permute(0, 2, 1, 3, 4)
        
        return self.model(x)
