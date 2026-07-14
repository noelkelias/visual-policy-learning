# policies/sac_sb3_policy.py

from stable_baselines3 import SAC


# =========================
# POLICY
# =========================
class SACSb3Policy:
    """Stable-Baselines3 SAC wrapper."""

    def __init__(self, env, policy="MlpPolicy", model=None, **sac_kwargs):
        if model is not None:
            self.model = model
            return

        defaults = {"verbose": 1}
        defaults.update(sac_kwargs)
        self.model = SAC(policy, env, **defaults)

    def save(self, path):
        self.model.save(path)

    def predict(self, obs, deterministic=True):
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
def load_sac_sb3_policy(path, env=None, device="auto"):
    """Load a saved SAC checkpoint (SB3 SAC checkpoint loader)."""
    return SACSb3Policy(env=None, model=SAC.load(path, env=env, device=device))
