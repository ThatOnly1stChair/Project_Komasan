import time
import mujoco.viewer
from stable_baselines3 import PPO
from robot_env import RobotDogEnv
from tracking_robot_env import TrackDogEnv

render_interval = 2

env = TrackDogEnv()
policy = PPO.load("policies/walk_forward0")


obs, info = env.reset()
print(f"Environment space: {env.observation_space.shape}, {obs.shape}")


print("Environment observation shape:", env.observation_space.shape)
print("Actual observation shape:", obs.shape)
print("Policy observation shape:", policy.observation_space.shape)

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        start = time.time()
        for _ in range(render_interval):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:

                obs, info = env.reset()

            viewer.sync()
            simulated_dt = (env.model.opt.timestep * env.frame_skip)
            remaining = simulated_dt - (time.time() - start)

            if remaining > 0:
                time.sleep(remaining)

env.close()
