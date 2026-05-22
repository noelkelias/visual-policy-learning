import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco


class ScriptedPolicy:

    def __init__(
        self,
        model,
        data,
        ee_body_name="hand"
    ):

        self.model = model
        self.data = data

        self.arm_dofs = 7

        # =====================================================
        # END EFFECTOR
        # =====================================================
        self.ee_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            ee_body_name
        )

        if self.ee_id == -1:
            raise ValueError(
                f"EE body '{ee_body_name}' not found"
            )

        # =====================================================
        # IK PARAMETERS
        # =====================================================
        self.kp = 12.0

        self.damping = 0.05

        self.step_size = 0.25

    # =========================================================
    # PREDICT
    # =========================================================
    def predict(self, info):

        ee_pos = info["ee_pos"]

        target = info["target_pos"]

        # =====================================================
        # POSITION ERROR
        # =====================================================
        error = target - ee_pos

        error = np.clip(
            error,
            -0.2,
            0.2
        )

        # =====================================================
        # JACOBIAN
        # =====================================================
        jacp = np.zeros(
            (3, self.model.nv)
        )

        jacr = np.zeros(
            (3, self.model.nv)
        )

        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacp,
            jacr,
            self.ee_id
        )

        J = jacp[:, :self.arm_dofs]

        # =====================================================
        # DAMPED LEAST SQUARES IK
        # =====================================================
        JJt = J @ J.T

        J_pinv = J.T @ np.linalg.inv(
            JJt + (
                self.damping ** 2
            ) * np.eye(3)
        )

        dq = J_pinv @ (
            self.kp * error
        )

        # =====================================================
        # LIMIT JOINT MOTION
        # =====================================================
        dq = np.clip(
            dq,
            -2.0,
            2.0
        )

        # =====================================================
        # CURRENT CONFIGURATION
        # =====================================================
        q_current = self.data.qpos[
            :self.arm_dofs
        ].copy()

        # =====================================================
        # POSITION TARGET
        # =====================================================
        q_target = (
            q_current
            + self.step_size * dq
        )

        # =====================================================
        # JOINT LIMITS
        # =====================================================
        ctrl_low = self.model.actuator_ctrlrange[
            :self.arm_dofs,
            0
        ]

        ctrl_high = self.model.actuator_ctrlrange[
            :self.arm_dofs,
            1
        ]

        q_target = np.clip(
            q_target,
            ctrl_low,
            ctrl_high
        )

        # =====================================================
        # FULL ACTION VECTOR
        # =====================================================
        action = np.zeros(
            self.model.nu,
            dtype=np.float32
        )

        action[:self.arm_dofs] = q_target

        # keep gripper fixed
        if self.model.nu > self.arm_dofs:

            action[7] = 0.0

        return action