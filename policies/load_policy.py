# policies/load_policy.py

import torch

from policies.base_policy import BasePolicy
from policies.action_head import ActionHead


def load_policy(
    encoder,
    encoder_dim,
    action_dim,
    path,
    device="cuda",
):

    # =================================
    # LOAD CHECKPOINT
    # =================================

    ckpt = torch.load(
        path,
        map_location=device,
    )

    if isinstance(ckpt, dict) and "input_dim" in ckpt:
        ckpt_dim = int(ckpt["input_dim"])
        if ckpt_dim != encoder_dim:
            print(
                f"Using checkpoint input_dim={ckpt_dim} "
                f"(encoder_dim arg was {encoder_dim})"
            )
        encoder_dim = ckpt_dim

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # =================================
    # BUILD POLICY
    # =================================

    head = ActionHead(
        encoder_dim,
        action_dim,
    )

    model = BasePolicy(
        encoder,
        head,
    ).to(device)

    # =================================
    # OLD FORMAT COMPATIBILITY
    # =================================
    #
    # OLD:
    # net.0.weight
    #
    # NEW:
    # action_head.net.0.weight
    #
    # =================================

    if "net.0.weight" in state_dict:

        state_dict = {
            f"action_head.{k}": v
            for k, v in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("\nLoaded:", path)

    encoder_prefix = "encoder."
    missing_head = [k for k in missing if not k.startswith(encoder_prefix)]
    missing_encoder = [k for k in missing if k.startswith(encoder_prefix)]

    if missing_encoder and not missing_head:
        print("(encoder weights come from pretrained backbone, not checkpoint)")

    elif missing_head:
        print("\nMissing keys (action head / policy):")
        for k in missing_head:
            print(k)

    if unexpected:

        print("\nUnexpected keys:")

        for k in unexpected:
            print(k)

    model.eval()

    return model