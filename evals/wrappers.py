import numpy as np


class LightingWrapper:
    def __init__(self, env, brightness=0.5):
        self.env = env
        self.brightness = brightness

    def _mod(self, img):
        return np.clip(
            img.astype(np.float32) * self.brightness,
            0,
            255
        ).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs["image"] = self._mod(obs["image"])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        obs["image"] = self._mod(obs["image"])
        return obs, reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class OcclusionWrapper:
    def __init__(self, env):
        self.env = env

    def _mod(self, img):
        img = img.copy()
        h, w, _ = img.shape
        img[h//3:h//3+16, w//3:w//3+16] = 0
        return img

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs["image"] = self._mod(obs["image"])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        obs["image"] = self._mod(obs["image"])
        return obs, reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class GaussianNoiseWrapper:
    def __init__(self, env, noise_std=10):
        self.env = env
        self.noise_std = noise_std

    def _mod(self, img):
        noise = np.random.randn(*img.shape) * self.noise_std
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs["image"] = self._mod(obs["image"])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        obs["image"] = self._mod(obs["image"])
        return obs, reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()