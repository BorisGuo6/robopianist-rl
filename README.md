# RoboPianist RL 训练代码

这个目录包含了用于训练 RoboPianist 强化学习模型的代码。

## 核心文件

### 训练相关
- **`train.py`** - 主训练脚本，使用 SAC 算法训练钢琴演奏策略
- **`run.sh`** - 训练启动脚本，包含训练参数配置
- **`sac.py`** - SAC (Soft Actor-Critic) 算法实现
- **`replay.py`** - 经验回放缓冲区实现

### 网络和模型
- **`networks.py`** - 神经网络架构定义 (MLP, Ensemble 等)
- **`distributions.py`** - 概率分布定义 (TanhNormal 等)
- **`specs.py`** - 环境规范定义

### 评估和查看结果
- **`load_and_evaluate.py`** - 加载训练好的模型并进行评估
- **`view_results.py`** - 查看训练结果和指标

## 使用方法

### 1. 开始训练
```bash
# 激活环境
conda activate pianist

# 开始训练
bash run.sh
```

### 2. 查看训练结果
```bash
# 查看 wandb 训练结果
python view_results.py
```

### 3. 评估训练好的模型
```bash
# 加载并评估模型
python load_and_evaluate.py
```

## 训练输出

训练完成后，以下文件会被保存：
- **模型权重**: `/tmp/robopianist/rl/[run_name]/final_model.pkl`
- **检查点**: `/tmp/robopianist/rl/[run_name]/checkpoint_*.pkl`
- **训练日志**: 自动上传到 Weights & Biases (wandb)

## 环境要求

- Python 3.10
- JAX
- Flax
- RoboPianist
- MuJoCo
- Weights & Biases

## 注意事项

1. 确保已安装所有依赖包
2. 训练需要较长时间 (通常几小时到几天)
3. 模型权重会自动保存，训练中断后可从检查点恢复
4. 训练结果会同步到 wandb，可在网页界面查看详细指标