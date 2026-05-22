# policies/encoders/resnet_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 224

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.backbone = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.feature_dim = 512

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        x = (x - self.mean) / self.std

        with torch.inference_mode():
            features = self.backbone(x)

        features = features.flatten(1)

        return features