#!/usr/bin/env python3
"""
学習済みモデルの可視化スクリプト
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.merge_env import MergeEnv, RoadSpec

def load_trained_model():
    """学習済みモデルを読み込む"""
    # 新しい環境を作成（レンダリング用）
    env = MergeEnv(seed=42, road=RoadSpec(1000.0, 300.0, 600.0, 1, 0), dt=0.1, spawn_rate=0.08, render_mode="human")
    
    # モデルを学習（実際のプロジェクトでは保存されたモデルを読み込む）
    print("学習済みモデルを読み込み中...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)  # 短時間で学習
    
    return model, env

def visualize_episode(model, env, max_steps=2000):
    """1エピソードの可視化"""
    obs, _ = env.reset()
    total_reward = 0.0
    step_count = 0
    
    print("エピソード開始 - 可視化中...")
    print("赤い車: エージェント（学習済み）")
    print("青い車: 交通車両")
    print("黄色エリア: 合流ゾーン")
    
    for step in range(max_steps):
        # モデルから行動を取得
        action, _ = model.predict(obs, deterministic=True)
        
        # 環境でステップ実行
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # レンダリング
        env.render()
        
        # イベント情報を表示
        if 'event' in info:
            print(f"Step {step_count}: {info['event']} (Reward: {reward:.3f})")
        
        # エピソード終了
        if done or trunc:
            break
    
    print(f"\nエピソード終了!")
    print(f"総ステップ数: {step_count}")
    print(f"総報酬: {total_reward:.3f}")
    
    return total_reward, step_count

def compare_strategies():
    """ランダム行動と学習済みモデルの比較"""
    print("=== ランダム行動 vs 学習済みモデル 比較 ===")
    
    # 学習済みモデル
    model, env = load_trained_model()
    
    # 学習済みモデルでの実行
    print("\n1. 学習済みモデル:")
    trained_reward, trained_steps = visualize_episode(model, env)
    
    # ランダム行動での実行
    print("\n2. ランダム行動:")
    obs, _ = env.reset()
    random_reward = 0.0
    random_steps = 0
    
    for step in range(2000):
        # ランダム行動
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        random_reward += reward
        random_steps += 1
        
        env.render()
        
        if 'event' in info:
            print(f"Step {random_steps}: {info['event']} (Reward: {reward:.3f})")
        
        if done or trunc:
            break
    
    print(f"\n=== 比較結果 ===")
    print(f"学習済みモデル: {trained_reward:.3f} (ステップ: {trained_steps})")
    print(f"ランダム行動: {random_reward:.3f} (ステップ: {random_steps})")
    print(f"改善率: {((trained_reward - random_reward) / abs(random_reward) * 100):.1f}%")
    
    env.close()

def analyze_behavior(model, env):
    """エージェントの行動分析"""
    print("\n=== 行動分析 ===")
    
    obs, _ = env.reset()
    actions_taken = []
    positions = []
    speeds = []
    lane_changes = 0
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        
        actions_taken.append(action)
        positions.append(env.agent.x)
        speeds.append(env.agent.v)
        
        if action[1] == 1:  # 車線変更
            lane_changes += 1
        
        if done or trunc:
            break
    
    # 分析結果を表示
    print(f"平均速度: {np.mean(speeds):.2f} m/s")
    print(f"最大速度: {np.max(speeds):.2f} m/s")
    print(f"車線変更回数: {lane_changes}")
    print(f"最終位置: {positions[-1]:.2f} m")
    
    # グラフで可視化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # 位置の変化
    ax1.plot(positions)
    ax1.axvline(x=300, color='orange', linestyle='--', label='合流開始')
    ax1.axvline(x=600, color='orange', linestyle='--', label='合流終了')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('エージェントの位置変化')
    ax1.legend()
    ax1.grid(True)
    
    # 速度の変化
    ax2.plot(speeds)
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('エージェントの速度変化')
    ax2.grid(True)
    
    # 行動の分布
    throttle_actions = [a[0] for a in actions_taken]
    lane_actions = [a[1] for a in actions_taken]
    
    ax3.hist(throttle_actions, bins=3, alpha=0.7, label='Throttle')
    ax3_twin = ax3.twinx()
    ax3_twin.hist(lane_actions, bins=2, alpha=0.7, color='red', label='Lane Change')
    ax3.set_xlabel('Action Value')
    ax3.set_ylabel('Throttle Frequency')
    ax3_twin.set_ylabel('Lane Change Frequency')
    ax3.set_title('行動分布')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Highway Merge RL - 可視化デモ")
    print("=" * 50)
    
    try:
        # 比較実行
        compare_strategies()
        
        # 行動分析
        model, env = load_trained_model()
        analyze_behavior(model, env)
        
    except KeyboardInterrupt:
        print("\n可視化を終了します...")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        plt.close('all')

