# training/train_rl.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import csv
import argparse

from envs.panda_reach_state_env import PandaReachStateEnv
from policies.ppo_policy import PPOPolicy
from policies.sac_policy import SACPolicy


# =========================
# TRAIN (PPO / SAC)
# =========================
def train_ppo_sac(args, seed):

    device = args.device

    if device == "auto":

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    seed_dir = os.path.join(
        "models", "rl", "checkpoints", args.run_name, f"seed_{seed}"
    )
    os.makedirs(seed_dir, exist_ok=True)

    curve_path = os.path.join(seed_dir, "learning_curve.csv")
    model_path = os.path.join(seed_dir, "model")

    env = PandaReachStateEnv(
        model_path=args.model_path,
        render_mode=False,
        max_steps=args.max_steps,
        physics_steps=args.physics_steps,
        include_ee_delta=True,
        sparse_reward=False,
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    act_low = env.action_space.low.astype("float32")
    act_high = env.action_space.high.astype("float32")

    print(f"\n{args.algo.upper()}  {args.run_name}/seed_{seed}")
    print(f"Timesteps : {args.total_timesteps}")
    print(f"Save dir  : {seed_dir}\n")

    if args.algo == "ppo":

        policy = PPOPolicy(
            obs_dim,
            act_dim,
            act_low,
            act_high,
            hidden=64,
            device=device,
            gamma=args.gamma,
        )

        policy.learn(
            env,
            args.total_timesteps,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            seed=seed,
            curve_path=curve_path,
            max_steps=args.max_steps,
        )

    else:

        policy = SACPolicy(
            obs_dim,
            act_dim,
            act_low,
            act_high,
            hidden=256,
            device=device,
        )

        policy.learn(
            env,
            args.total_timesteps,
            lr=args.lr,
            gamma=args.gamma,
            tau=0.005,
            buffer_size=100_000,
            batch_size=256,
            learning_starts=100,
            seed=seed,
            curve_path=curve_path,
            max_steps=args.max_steps,
        )

    saved = policy.save(model_path)

    env.close()

    print(f"Saved {saved}")
    print(f"Curve {curve_path}")


# =========================
# TRAIN (SB3)
# =========================
def train_sb3(args, seed):

    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from policies.ppo_sb3_policy import PPOSb3Policy
    from policies.sac_sb3_policy import SACSb3Policy

    class EpisodeLogCallback(BaseCallback):
        """Append episode reward/length to CSV (Monitor infos)."""

        def __init__(self, log_path, verbose=0):
            super().__init__(verbose)
            self.log_path = log_path
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            if not os.path.exists(log_path):
                with open(log_path, "w", newline="") as f:
                    csv.writer(f).writerow(
                        ["timesteps", "episode_reward", "episode_length"]
                    )

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" not in info:
                    continue
                ep = info["episode"]
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [int(self.num_timesteps), float(ep["r"]), float(ep["l"])]
                    )
            return True

    seed_dir = os.path.join(
        "models", "rl", "checkpoints", args.run_name, f"seed_{seed}"
    )
    os.makedirs(seed_dir, exist_ok=True)

    curve_path = os.path.join(seed_dir, "learning_curve.csv")
    model_path = os.path.join(seed_dir, "model")
    vecnorm_path = os.path.join(seed_dir, "vecnormalize.pkl")

    # PPO uses obs/reward norm; SAC does not
    normalize = args.algo == "ppo"

    env = PandaReachStateEnv(
        model_path=args.model_path,
        render_mode=False,
        max_steps=args.max_steps,
        physics_steps=args.physics_steps,
        include_ee_delta=True,
        sparse_reward=False,
    )
    env = Monitor(env)  # episode reward/length for CSV log
    # SB3 APIs (and VecNormalize) expect a VecEnv, not a raw Gym env
    env = DummyVecEnv([lambda e=env: e])

    if normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=args.gamma,
        )

    print(f"\nSB3 {args.algo.upper()}  {args.run_name}/seed_{seed}")
    print(f"Normalize : {normalize}")
    print(f"Timesteps : {args.total_timesteps}")
    print(f"Save dir  : {seed_dir}\n")

    if args.algo == "ppo":

        policy = PPOSb3Policy(
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.2,
            verbose=1,
            seed=seed,
            device=args.device,
        )

    else:

        policy = SACSb3Policy(
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            learning_starts=100,
            verbose=1,
            seed=seed,
            device=args.device,
        )

    policy.learn(
        total_timesteps=args.total_timesteps,
        callback=EpisodeLogCallback(curve_path),
        progress_bar=False,
    )

    policy.save(model_path)

    if normalize:

        env.save(vecnorm_path)
        print(f"Saved {vecnorm_path}")

    env.close()

    print(f"Saved {model_path}.zip")
    print(f"Curve {curve_path}")


# =========================
# TRAIN
# =========================
def train(args):

    seeds = [int(s) for s in args.seeds.split(",")]

    for seed in seeds:

        if args.backend == "scratch":
            train_ppo_sac(args, seed)
        else:
            train_sb3(args, seed)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        required=True,
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["scratch", "sb3"],
        default="scratch",
        help="scratch = our PPO/SAC; sb3 = Stable-Baselines3",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--physics_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="envs/panda/scene.xml",
    )

    args = parser.parse_args()

    train(args)
