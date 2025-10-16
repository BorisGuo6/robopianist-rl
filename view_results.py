#!/usr/bin/env python3
"""
查看 RoboPianist 训练结果的脚本
"""

import wandb
try:
    import pandas as pd
except ImportError:
    pd = None

def view_training_results():
    """查看训练结果"""
    print("=== RoboPianist 训练结果查看器 ===\n")
    
    # 1. 连接到 wandb
    try:
        # 获取最近的运行记录
        api = wandb.Api()
        runs = api.runs("robopianist", per_page=10)
        
        print("最近的训练运行:")
        for i, run in enumerate(runs):
            print(f"{i+1}. {run.name}")
            print(f"   状态: {run.state}")
            print(f"   开始时间: {run.created_at}")
            print(f"   运行时长: {run.summary.get('_wandb', {}).get('runtime', 'N/A')}")
            print()
        
        # 选择最新的运行
        if runs:
            latest_run = runs[0]
            print(f"正在查看最新运行: {latest_run.name}")
            
            # 获取运行历史
            history = latest_run.history()
            print(f"总训练步数: {len(history)}")
            
            # 显示关键指标
            if len(history) > 0:
                print("\n=== 训练指标摘要 ===")
                
                # 训练指标
                train_cols = [col for col in history.columns if col.startswith('train/')]
                if train_cols:
                    print("训练指标:")
                    for col in train_cols[:5]:  # 显示前5个训练指标
                        if col in history.columns:
                            final_value = history[col].iloc[-1]
                            if pd and pd.isna(final_value):
                                final_value = "N/A"
                            print(f"  {col}: {final_value}")
                
                # 评估指标
                eval_cols = [col for col in history.columns if col.startswith('eval/')]
                if eval_cols:
                    print("\n评估指标:")
                    for col in eval_cols[:5]:  # 显示前5个评估指标
                        if col in history.columns:
                            final_value = history[col].iloc[-1]
                            if pd and pd.isna(final_value):
                                final_value = "N/A"
                            print(f"  {col}: {final_value}")
                
                # 音乐指标
                music_cols = [col for col in history.columns if 'music' in col.lower() or 'note' in col.lower()]
                if music_cols:
                    print("\n音乐评估指标:")
                    for col in music_cols[:3]:
                        if col in history.columns:
                            final_value = history[col].iloc[-1]
                            if pd and pd.isna(final_value):
                                final_value = "N/A"
                            print(f"  {col}: {final_value}")
            
            # 提供 wandb 链接
            print(f"\n=== 详细结果查看 ===")
            print(f"Weights & Biases 链接: {latest_run.url}")
            print("您可以在浏览器中打开上述链接查看详细的训练曲线和指标")
            
    except Exception as e:
        print(f"无法连接到 wandb: {e}")
        print("请确保:")
        print("1. 已登录 wandb: wandb login")
        print("2. 网络连接正常")
        print("3. 训练确实使用了 wandb 记录")

def create_evaluation_script():
    """创建评估脚本"""
    script_content = '''#!/usr/bin/env python3
"""
使用训练好的模型进行评估的脚本
注意：由于原始训练脚本没有保存模型权重，这个脚本需要修改训练脚本以保存模型
"""

import numpy as np
import jax
import jax.numpy as jnp
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers
import dm_env_wrappers as wrappers

def load_and_evaluate_model():
    """加载并评估模型"""
    print("=== RoboPianist 模型评估 ===\\n")
    
    # 创建环境
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
    
    print("环境创建成功！")
    print(f"动作空间维度: {env.action_spec().shape}")
    print(f"观察空间维度: {env.observation_spec()}")
    
    # 注意：这里需要加载训练好的模型权重
    # 由于原始训练脚本没有保存模型，需要修改训练脚本添加模型保存功能
    
    print("\\n要使用此脚本，需要:")
    print("1. 修改训练脚本添加模型保存功能")
    print("2. 加载保存的模型权重")
    print("3. 在环境中运行模型并评估性能")

if __name__ == "__main__":
    load_and_evaluate_model()
'''
    
    with open('/home/boris/workspace/robopianist/robopianist-rl/evaluate_model.py', 'w') as f:
        f.write(script_content)
    
    print("已创建评估脚本: evaluate_model.py")

if __name__ == "__main__":
    view_training_results()
    print("\n" + "="*50)
    create_evaluation_script()
