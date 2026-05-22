# policies/encoders/cnn_encoder.py
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)  # (B, 128)