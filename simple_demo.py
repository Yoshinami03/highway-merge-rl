#!/usr/bin/env python3
"""
シンプルな可視化デモ
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.merge_env import MergeEnv, RoadSpec

def simple_demo():
    """シンプルな可視化デモ"""
    print("Highway Merge RL - Simple Visualization Demo")
    print("=" * 50)
    
    # 環境作成（レンダリング有効）
    env = MergeEnv(seed=42, road=RoadSpec(1000.0, 300.0, 600.0, 1, 0), 
                   dt=0.1, spawn_rate=0.08, render_mode="human")
    
    # モデル学習
    print("Training model...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    
    # 可視化実行
    print("\nRunning visualization...")
    print("Red car: Agent (trained)")
    print("Blue cars: Traffic")
    print("Yellow area: Merge zone")
    
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        
        # レンダリング
        env.render()
        
        # 重要なイベントを表示
        if 'event' in info:
            print(f"Step {step}: {info['event']} (Reward: {reward:.3f})")
        
        if done or trunc:
            print(f"Episode finished! Total reward: {total_reward:.3f}")
            break
    
    env.close()

if __name__ == "__main__":
    try:
        simple_demo()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')

