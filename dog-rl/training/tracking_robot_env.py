import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from pathlib import Path

joint_names = [
    "FL_j1", "FL_j2", "FL_j3",
    "FR_j1", "FR_j2", "FR_j3",
    "BL_j1", "BL_j2", "BL_j3",
    "BR_j1", "BR_j2", "BR_j3",
]

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "scene.xml"
)

class TrackDogEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        print(f"Actuators detected: {self.model.nu}")

        self.base_id = self.model.body("base_link").id

        self.joint_ids = [
            self.model.joint(name).id for name in joint_names
        ]
        self.qpos_ids = np.array([
            self.model.jnt_qposadr[j] for j in self.joint_ids
        ])
        self.qvel_ids = np.array([
            self.model.jnt_dofadr[j] for j in self.joint_ids
        ])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.frame_skip = 10
        self.max_steps = 1000
        self.step_count = 0

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        self.prev_action = np.zeros(12, dtype=np.float32)

        if not np.all(self.model.actuator_ctrllimited):
            raise ValueError("All actuators needs ctrlrange limits")

    def _get_obs(self):

        joint_pos = self.data.qpos[self.qpos_ids]
        joint_vel = self.data.qvel[self.qvel_ids]

        base_pos = self.data.body("base_link").xpos
        base_quat = self.data.body("base_link").xquat
        rotation = self.data.body("base_link").xmat.reshape(3,3)

        base_ang_vel = self.data.qvel[3:6]
        base_vel = self.data.qvel[0:3]
        local_lin_vel = rotation.T @ base_vel

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            base_pos,
            base_quat,
            base_ang_vel,
            local_lin_vel
        ])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.data.ctrl[:] = (
            self.ctrl_low + self.ctrl_high) /2

        mujoco.mj_forward(self.model, self.data)
        self.initial_xy = (self.data.body("base_link").xpos[:2].copy())
        self.prev_action = np.zeros(12, dtype=np.float32)

        return self._get_obs(), {}

    def step(self,  action):
        target_height = 0.2
        forward_weight = 8.0
        lateral_weight = 0.5
        yaw_weight = 4.0
        upright_weight = 1.0
        height_weight = 1.0
        energy_weight = 0.01
        joint_vel_weight = 0.001
        smooth_weight = 0.001
        max_action_change = 0.90

        
        action = np.clip(action, -1.0, 1.0)
        filtered_action = self.prev_action + np.clip(action - self.prev_action, -max_action_change, max_action_change)

        target = self.ctrl_low + ((filtered_action + 1.0) / 2.0) * (self.ctrl_high - self.ctrl_low)

        self.data.ctrl[:] = target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # target_vx = self.command[0]
        # target_vy = self.command[1]
        # target_yaw = self.command[2]

        target_vx = 0.0
        target_vy = 0.5
        target_yaw = 0.0

        #Getting raw data from robot-workspace
        rotation = self.data.body("base_link").xmat.reshape(3,3)

        body_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_id,
            body_vel,
            1
        )
        local_ang_vel = body_vel[:3]
        local_lin_vel = body_vel[3:6]
        
        #Measuring our goal values
        vx = local_lin_vel[0]
        vy = local_lin_vel[1]
        yaw = local_ang_vel[2]
        vx_err = vx - target_vx
        vy_err = vy - target_vy
        yaw_err = yaw - target_yaw

        joint_vel = self.data.qvel[6:18]
        upright = self.data.body("base_link").xmat[8]
        base_height = self.data.body("base_link").xpos[2]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2,1]**2 + rotation[2,2]**2))
        roll = np.arctan2(rotation[2,1], rotation[2,2])

        forward_reward = forward_weight * np.exp(-5.0 * vy_err**2)
        lateral_reward = lateral_weight * np.exp(-5.0 * vx_err**2)
        yaw_reward = yaw_weight * np.exp( -5.0 * yaw_err**2)
        upright_reward = upright_weight * np.exp(-5.0 * (pitch**2 + roll**2))
        height_reward = height_weight * np.exp( -5.0 * (base_height - target_height) ** 2)
        energy_reward = -1 * energy_weight * np.sum(filtered_action**2)
        joint_vel_reward = -1 * joint_vel_weight * np.sum(joint_vel**2)
        smooth_reward = -smooth_weight * np.sum((filtered_action - self.prev_action)**2)
        
        reward = (forward_reward + lateral_reward + yaw_reward + upright_reward + height_reward + energy_reward + joint_vel_reward + smooth_reward)

        terminated = bool(
            base_height < 0.05 or upright < 0.15
        )

        truncated = self.step_count >= self.max_steps

        self.prev_action = filtered_action.copy()
        return(
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {}
        )

