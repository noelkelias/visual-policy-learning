# training/rl_utils.py

import numpy as np


# =========================
# ACTION SCALING
# =========================
def scale_action(action, low, high):
    """Env [low, high] -> [-1, 1]."""

    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(scaled, low, high):
    """[-1, 1] -> env [low, high]."""

    return low + 0.5 * (scaled + 1.0) * (high - low)


# =========================
# RUNNING STATS
# =========================
class RunningMeanStd:

    def __init__(self, shape=(), epsilon=1e-4):

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):

        x = np.asarray(x, dtype=np.float64)

        if self.mean.ndim == 0:

            x = x.reshape(-1)
            batch_mean = np.array(x.mean(), dtype=np.float64)
            batch_var = np.array(x.var(), dtype=np.float64)
            batch_count = x.size

        else:

            if x.ndim == 1:
                x = x[None, :]

            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):

        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total

        self.mean = new_mean
        self.var = M2 / total
        self.count = total

    def std(self):
        return float(np.sqrt(np.asarray(self.var) + 1e-8))


# =========================
# OBS / REWARD NORM
# =========================
class ObsRewardNormalizer:
    """Running obs + return normalization (VecNormalize-style)."""

    def __init__(
        self,
        obs_shape,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    ):

        self.obs_rms = RunningMeanStd(obs_shape)
        self.ret_rms = RunningMeanStd(())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.ret = 0.0

    def normalize_obs(self, obs, update=True):

        if update:
            self.obs_rms.update(obs)

        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

        return np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)

    def normalize_reward(self, reward, update=True):

        reward = float(np.asarray(reward).reshape(-1)[0])

        if update:
            self.ret = float(self.ret * self.gamma + reward)
            self.ret_rms.update([self.ret])

        reward = reward / self.ret_rms.std()

        return float(np.clip(reward, -self.clip_reward, self.clip_reward))

    def reset_ret(self):
        self.ret = 0.0

    def state_dict(self):

        return {
            "obs_mean": self.obs_rms.mean,
            "obs_var": self.obs_rms.var,
            "obs_count": self.obs_rms.count,
            "ret_mean": self.ret_rms.mean,
            "ret_var": self.ret_rms.var,
            "ret_count": self.ret_rms.count,
            "gamma": self.gamma,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
        }

    def load_state_dict(self, state):

        self.obs_rms.mean = np.asarray(state["obs_mean"], dtype=np.float64)
        self.obs_rms.var = np.asarray(state["obs_var"], dtype=np.float64)
        self.obs_rms.count = float(state.get("obs_count", 1e4))

        self.ret_rms.mean = np.asarray(state.get("ret_mean", 0.0), dtype=np.float64)
        self.ret_rms.var = np.asarray(state.get("ret_var", 1.0), dtype=np.float64)
        self.ret_rms.count = float(state.get("ret_count", 1e4))

        self.gamma = float(state.get("gamma", self.gamma))
        self.clip_obs = float(state.get("clip_obs", self.clip_obs))
        self.clip_reward = float(state.get("clip_reward", self.clip_reward))
