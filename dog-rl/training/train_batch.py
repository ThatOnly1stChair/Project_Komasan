from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
import os

from robot_env import RobotDogEnv
from tracking_robot_env import TrackDogEnv

MODEL_PATH = "policies/robot_dog_walk14.zip"
#robot_dog_walk13 set with goal of range of -0.5 to -0.15

n_envs = 4

def make_env():
    return Monitor(TrackDogEnv())

def main():
    test_env = TrackDogEnv()
    check_env(test_env)
    test_env.close()
    env = SubprocVecEnv([make_env for n_env in range(n_envs)])

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
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            device="cpu",
            tensorboard_log="./logs/"
        ) 

    model.learn(total_timesteps=500000, reset_num_timesteps=False, progress_bar=True)

    Path("policies").mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    env.close()

if __name__ == "__main__":
    main()