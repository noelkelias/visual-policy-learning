# policies/ppo_sb3_policy.py

import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =========================
# POLICY
# =========================
class PPOSb3Policy:
    """Stable-Baselines3 PPO wrapper."""

    def __init__(
        self,
        env,
        policy="MlpPolicy",
        model=None,
        vec_normalize=None,
        **ppo_kwargs,
    ):
        # If set, predict() normalizes raw obs with saved running stats.
        self.vec_normalize = vec_normalize

        if model is not None:
            self.model = model
            return

        defaults = {"verbose": 1}
        defaults.update(ppo_kwargs)
        self.model = PPO(policy, env, **defaults)

    def save(self, path):
        self.model.save(path)

    def predict(self, obs, deterministic=True):
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)

        action, _states = self.model.predict(
            obs,
            deterministic=deterministic,
        )
        return action

    def learn(self, total_timesteps, callback=None, **kwargs):
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs,
        )


# =========================
# LOAD
# =========================
def load_vecnormalize(model_path, venv, *, required=False):
    """Load vecnormalize.pkl next to model.zip."""
    if venv is None:
        if required:
            raise ValueError("venv is required to load VecNormalize for SB3 PPO")
        return None

    seed_dir = os.path.dirname(model_path)
    vec_path = os.path.join(seed_dir, "vecnormalize.pkl")

    if not os.path.exists(vec_path):
        if required:
            raise FileNotFoundError(
                f"SB3 PPO eval needs VecNormalize stats at {vec_path}"
            )
        return None

    vec_normalize = VecNormalize.load(vec_path, venv)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def load_ppo_sb3_policy(path, env=None, device="auto"):
    """Load weights; attach vecnormalize.pkl when env (VecEnv) is provided."""
    model = PPO.load(path, env=env, device=device)
    vec_normalize = load_vecnormalize(path, env, required=False)
    return PPOSb3Policy(env=None, model=model, vec_normalize=vec_normalize)


def load_ppo_sb3_for_eval(path, make_env_fn, device="auto"):
    """
    Load PPO + VecNormalize stats for closed-loop eval on a raw env.

    make_env_fn: zero-arg -> gymnasium env (used only to load VecNormalize).
    """
    model = PPO.load(path, device=device)

    def _thunk():
        return Monitor(make_env_fn())

    venv = DummyVecEnv([_thunk])
    vec_normalize = load_vecnormalize(path, venv, required=True)
    venv.close()

    return PPOSb3Policy(env=None, model=model, vec_normalize=vec_normalize)
