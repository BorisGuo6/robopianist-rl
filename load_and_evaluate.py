#!/usr/bin/env python3
"""
加载训练好的模型并进行评估
"""

import numpy as np
import pickle
import jax
from pathlib import Path
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers
import dm_env_wrappers as wrappers
from robopianist_rl import sac, specs

def load_model(model_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("模型加载成功!")
    print(f"模型包含的组件: {list(model_data.keys())}")
    return model_data

def create_environment():
    """创建评估环境"""
    env = suite.load(
        environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0",
        seed=42,
        task_kwargs=dict(
            n_steps_lookahead=10,
            trim_silence=True,
            gravity_compensation=True,
            reduced_action_space=True,
            control_timestep=0.05,
            wrong_press_termination=False,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            primitive_fingertip_collisions=True,
            change_color_on_activation=True,
        ),
    )
    
    # 添加包装器
    env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    env = robopianist_wrappers.MidiEvaluationWrapper(environment=env, deque_size=1)
    
    return env

def evaluate_model(model_data, env, num_episodes=5):
    """使用训练好的模型进行评估"""
    print(f"开始评估模型，运行 {num_episodes} 个回合...")
    
    # 创建 SAC agent (不训练，只用于推理)
    spec = specs.EnvironmentSpec.make(env)
    agent = sac.SAC.initialize(spec=spec, seed=42)
    
    # 加载训练好的参数
    agent = agent.replace(
        actor=agent.actor.replace(params=model_data['actor_params']),
        critic=agent.critic.replace(params=model_data['critic_params']),
        target_critic=agent.target_critic.replace(params=model_data['target_critic_params']),
        temp=agent.temp.replace(params=model_data['temp_params'])
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        timestep = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"  回合 {episode + 1}/{num_episodes}...")
        
        while not timestep.last():
            # 使用训练好的模型生成动作
            action = agent.eval_actions(timestep.observation)
            timestep = env.step(action)
            
            episode_reward += timestep.reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"    奖励: {episode_reward:.3f}, 长度: {episode_length}")
    
    # 获取最终统计信息
    try:
        stats = env.get_statistics()
        print(f"\n环境统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except:
        print("\n无法获取环境统计信息")
    
    try:
        music_stats = env.get_musical_metrics()
        print(f"\n音乐评估指标:")
        for key, value in music_stats.items():
            print(f"  {key}: {value}")
    except:
        print("\n无法获取音乐评估指标")
    
    # 计算平均性能
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n=== 评估结果 ===")
    print(f"平均奖励: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    print(f"平均回合长度: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最佳奖励: {np.max(episode_rewards):.3f}")
    print(f"最差奖励: {np.min(episode_rewards):.3f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': np.std(episode_rewards),
        'avg_length': avg_length,
        'std_length': np.std(episode_lengths),
        'best_reward': np.max(episode_rewards),
        'worst_reward': np.min(episode_rewards)
    }

def main():
    """主函数"""
    print("=== RoboPianist 模型加载和评估 ===\n")
    
    # 查找模型文件
    model_dir = Path("/tmp/robopianist/rl")
    model_files = list(model_dir.glob("**/final_model.pkl"))
    
    if not model_files:
        print("未找到模型文件!")
        print("请确保:")
        print("1. 训练已经完成并保存了模型")
        print("2. 模型文件路径正确")
        print(f"搜索路径: {model_dir}")
        return
    
    # 选择最新的模型
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"找到模型文件: {latest_model}")
    
    # 加载模型
    try:
        model_data = load_model(latest_model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 创建环境
    try:
        env = create_environment()
        print("环境创建成功!")
    except Exception as e:
        print(f"创建环境失败: {e}")
        return
    
    # 评估模型
    try:
        results = evaluate_model(model_data, env, num_episodes=3)
        print(f"\n评估完成! 平均奖励: {results['avg_reward']:.3f}")
    except Exception as e:
        print(f"评估失败: {e}")

if __name__ == "__main__":
    main()
