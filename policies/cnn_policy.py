# policies/cnn_policy.py
from policies.base_policy import BasePolicy
from policies.encoders.cnn_encoder import CNNEncoder
from policies.action_head import ActionHead

class CNNPolicy(BasePolicy):
    def __init__(self, action_dim):
        encoder = CNNEncoder()
        head = ActionHead(128, action_dim)
        super().__init__(encoder, head)