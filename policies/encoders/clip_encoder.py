# policies/encoders/clip_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 224

        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711]
            ).view(1, 3, 1, 1)
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        x = (x - self.mean) / self.std

        with torch.inference_mode():
            features = self.model.encode_image(x)

        # Must match training feature extraction (training.ipynb)
        features = features / features.norm(dim=-1, keepdim=True)

        return features