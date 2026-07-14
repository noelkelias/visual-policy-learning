# envs/panda_reach_state_env.py

import numpy as np
from gymnasium import spaces

from envs.panda_reach_env import PandaReachEnv


class PandaReachStateEnv(PandaReachEnv):
    """
    Same reach task / actions as PandaReachEnv, but observations are a flat
    privileged state vector (no RGB).

    Layout:
      [qpos_arm (7), qvel_arm (7), target_pos (3)]
      optionally + [ee_pos - target_pos (3)]

    Reward:
      dense (default): same shaped reward as PandaReachEnv
      sparse: +10 on success, else 0
    """

    def __init__(
        self,
        model_path="envs/panda/scene.xml",
        image_width=64,
        image_height=64,
        render_mode=False,
        max_steps=100,
        verbose=False,
        physics_steps=1,
        include_ee_delta=True,
        sparse_reward=False,
    ):

        super().__init__(
            model_path=model_path,
            image_width=image_width,
            image_height=image_height,
            render_mode=render_mode,
            max_steps=max_steps,
            verbose=verbose,
            physics_steps=physics_steps,
        )

        self.include_ee_delta = include_ee_delta
        self.sparse_reward = sparse_reward

        obs_dim = self.arm_dofs + self.arm_dofs + 3

        if include_ee_delta:
            obs_dim += 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # -----------------------------
    # OBS (override image Dict)
    # -----------------------------
    def _get_obs(self):

        arm = self.arm_dofs

        qpos = self.data.qpos[:arm].astype(np.float32)
        qvel = self.data.qvel[:arm].astype(np.float32)
        target = np.asarray(self.target, dtype=np.float32)

        parts = [qpos, qvel, target]

        if self.include_ee_delta:
            ee = self.data.xpos[self.ee_body_id].astype(np.float32)
            parts.append(ee - target)

        return np.concatenate(parts).astype(np.float32)

    # -----------------------------
    # STEP (optional sparse reward)
    # -----------------------------
    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        if self.sparse_reward:
            reward = 10.0 if info.get("success", False) else 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        # RGB helper for optional rollout videos (needs render_mode=True).
        if not self.render_mode or self.renderer is None:
            return np.zeros(
                (self.image_height, self.image_width, 3),
                dtype=np.uint8,
            )

        self.renderer.update_scene(self.data)
        return self.renderer.render().astype(np.uint8)
