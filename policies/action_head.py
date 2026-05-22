# policies/action_head.py

import torch
import torch.nn as nn


class ActionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            # IMPORTANT:
            # Predict actuator-space actions directly.
            # NO tanh.
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)