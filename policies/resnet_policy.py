# policies/resnet_policy.py
from policies.base_policy import BasePolicy
from policies.encoders.resnet_encoder import ResNetEncoder
from policies.action_head import ActionHead

class ResNetPolicy(BasePolicy):
    def __init__(self, action_dim):
        encoder = ResNetEncoder()
        head = ActionHead(512, action_dim)
        super().__init__(encoder, head)