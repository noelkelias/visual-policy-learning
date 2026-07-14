# policies/encoders/dinov2_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 224

        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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
            features = self.model(x)

        features = features / features.norm(dim=-1, keepdim=True)

        return features