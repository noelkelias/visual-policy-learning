# policies/dinov2_policy.py
from policies.base_policy import BasePolicy
from policies.encoders.dinov2_encoder import DINOv2Encoder
from policies.action_head import ActionHead

class DINOv2Policy(BasePolicy):
    def __init__(self, action_dim):
        encoder = DINOv2Encoder()
        head = ActionHead(384, action_dim)
        super().__init__(encoder, head)