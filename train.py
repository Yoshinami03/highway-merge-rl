import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.merge_env import MergeEnv, RoadSpec

def make_env():
    return MergeEnv(seed=0, road=RoadSpec(1000.0, 300.0, 600.0, 1, 0), dt=0.1, spawn_rate=0.08)

if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    obs, _ = env.reset()
    ep_r = 0.0
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_r += r
        if done or trunc:
            break
    print(ep_r)
