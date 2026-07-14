# policies/load_policy.py

import os

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
    """Load an imitation (BC) policy checkpoint. Unchanged API."""

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
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        # Training checkpoint: {"model": action-head weights, ...}
        state_dict = ckpt["model"]
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

    # Legacy flat keys -> action_head.* 
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


def load_rl_policy(
    algo,
    path,
    *,
    backend="scratch",
    make_env_fn=None,
    device="cpu",
):
    """
    Load a state-based RL policy (PPO / SAC).

    backend:
      scratch -> our PPO/SAC (.pt under policies/ppo_policy.py or sac_policy.py)
      sb3     -> SB3 PPO/SAC (.zip)
    """
    algo = algo.lower()
    backend = backend.lower()

    if backend == "scratch":
        if algo == "ppo":
            from policies.ppo_policy import load_ppo_policy

            return load_ppo_policy(path, device=device)
        if algo == "sac":
            from policies.sac_policy import load_sac_policy

            return load_sac_policy(path, device=device)
        raise ValueError(f"Unknown algo: {algo}")

    if backend == "sb3":
        if algo == "ppo":
            from policies.ppo_sb3_policy import load_ppo_sb3_for_eval

            # SB3 PPO trains under VecNormalize; eval needs those stats.
            if make_env_fn is None:
                raise ValueError(
                    "SB3 PPO requires make_env_fn so VecNormalize "
                    "(vecnormalize.pkl next to model.zip) can be loaded for eval"
                )
            return load_ppo_sb3_for_eval(
                path, make_env_fn=make_env_fn, device=device
            )
        if algo == "sac":
            from policies.sac_sb3_policy import load_sac_sb3_policy

            return load_sac_sb3_policy(path, device=device)
        raise ValueError(f"Unknown algo: {algo}")

    raise ValueError(f"Unknown backend: {backend}")


def resolve_rl_checkpoint_stem(run_name, seed, backend="scratch"):
    """models/rl/checkpoints/{run}/seed_{seed}/model(.pt|.zip)."""
    seed_dir = os.path.join(
        "models", "rl", "checkpoints", run_name, f"seed_{seed}"
    )
    return os.path.join(seed_dir, "model")
