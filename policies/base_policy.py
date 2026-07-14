# policies/base_policy.py
import torch.nn as nn

class BasePolicy(nn.Module):
    def __init__(self, encoder, action_head):
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head

    def forward(self, image):
        features = self.encoder(image)
        return self.action_head(features)