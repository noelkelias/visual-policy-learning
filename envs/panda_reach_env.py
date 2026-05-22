import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PandaReachEnv(gym.Env):

    def __init__(
        self,
        model_path="envs/panda/scene.xml",
        image_width=64,
        image_height=64,
        render_mode=False,
        max_steps=100,
        verbose=False,
        physics_steps=1,
    ):

        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.image_width = image_width
        self.image_height = image_height
        self.max_steps = max_steps
        self.step_count = 0
        self.verbose = verbose

        self.physics_steps = physics_steps

        self.arm_dofs = 7

        self.renderer = None
        if self.render_mode:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width
            )

        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]

        self.action_space = spaces.Box(
            low=ctrl_low.astype(np.float32),
            high=ctrl_high.astype(np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8
            )
        })

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand"
        )

        self.success_threshold = 0.05
        self.target = np.array([0.55, 0.0, 0.4])

        target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        self.target_mocap_id = int(self.model.body_mocapid[target_body_id])

    def set_target(self, target_pos):
        """Move the visible goal marker (and success criterion)."""
        self.target = np.asarray(target_pos, dtype=np.float64).copy()
        self.data.mocap_pos[self.target_mocap_id] = self.target
        mujoco.mj_forward(self.model, self.data)

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.data.qvel[:] = 0

        self.set_target(
            [
                np.random.uniform(0.45, 0.60),
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(0.30, 0.50),
            ]
        )

        return self._get_obs(), self._get_info()

    # -----------------------------
    # STEP (FAST)
    # -----------------------------
    def step(self, action):

        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        dist = info["distance"]

        reward = -dist
        reward -= 0.01 * np.linalg.norm(self.data.qvel[:self.arm_dofs])

        if info["success"]:
            reward += 10.0

        terminated = info["success"]
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    # -----------------------------
    # STATE SET
    # -----------------------------
    def set_state(self, qpos):
        qpos = np.asarray(qpos).copy()

        self.data.qpos[:len(qpos)] = qpos[:self.model.nq]
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        return self.data.qpos.copy()

    # -----------------------------
    # OBS
    # -----------------------------
    def _get_obs(self):

        # FAST PATH: no renderer at all
        if not self.render_mode or self.renderer is None:
            return {
                "image": np.zeros(
                    (self.image_height, self.image_width, 3),
                    dtype=np.uint8
                )
            }

        self.renderer.update_scene(self.data)
        img = self.renderer.render().astype(np.uint8)

        return {"image": img}

    # -----------------------------
    # INFO
    # -----------------------------
    def _get_info(self):
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        dist = np.linalg.norm(ee_pos - self.target)

        return {
            "distance": float(dist),
            "success": bool(dist < self.success_threshold),
            "ee_pos": ee_pos,
            "target_pos": self.target.copy(),
        }

    def render(self):
        return self._get_obs()["image"]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()