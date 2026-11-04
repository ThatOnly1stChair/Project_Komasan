from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from robot_env import RobotDogEnv

def make_env():
    return RobotDogEnv()

if __name__ == "__main__":
    env = SubprocVecEnv( [make_env for _ in range(8)])

    model = PPO.load("policies/robot_dog_ppo")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=40,
        deterministic=True
    )

    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    env.close()