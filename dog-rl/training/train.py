from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from robot_env import RobotDogEnv
from pathlib import Path
import os

MODEL_PATH = "policies/robot_dog_idle0.zip"

env = Monitor(RobotDogEnv())

check_env(env)

if os.path.exists(MODEL_PATH):
    print("Loading existing policy")
    model = PPO.load(
        MODEL_PATH,
        env=env,
        tensorboard_log="./logs/"
    )
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./logs/"
    )

model.learn(total_timesteps=2000000, reset_num_timesteps=False, progress_bar=True)

Path("policies").mkdir(exist_ok=True)
model.save(MODEL_PATH)

env.close()