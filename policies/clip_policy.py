# policies/clip_policy.py
from policies.base_policy import BasePolicy
from policies.encoders.clip_encoder import CLIPEncoder
from policies.action_head import ActionHead

class CLIPPolicy(BasePolicy):
    def __init__(self, action_dim):
        encoder = CLIPEncoder()
        head = ActionHead(512, action_dim)
        super().__init__(encoder, head)