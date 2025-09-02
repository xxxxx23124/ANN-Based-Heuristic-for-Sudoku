# RL-Sudoku-Solver

本项目旨在使用强化学习训练一个能解决数独问题的智能体。

## 项目背景

传统的监督学习方法在处理具有多解或无解的数独时存在局限。本项目转向强化学习，将数独求解建模为一个决策过程，使模型能学习策略而非记忆固定答案。

## 技术方案

-   **方法**: 强化学习 (Reinforcement Learning)，可能采用 PPO 算法。
-   **模型**: 继承自 [RL_mamba2-selfAttention-ppo](https://github.com/xxxxx23124/RL_mamba2-selfAttention-ppo.git) 的神经网络架构，用于处理棋盘状态。
-   **核心机制**:
    -   **Agent (模型)** 在数独**环境 (棋盘)** 中执行**动作 (填数字)**。
    -   根据动作的后果（如是否合法、是否完成）获得**奖励**。
    -   通过不断试错优化其决策**策略**。

## 当前状态

**开发中**。当前主要任务是完成ppo算法的编写。有点累，想休息一会。ppo算法的模板可能会继承 [RL_mamba2-selfAttention-ppo/legacy
/ppo_transformer_ssm.py](https://github.com/xxxxx23124/RL_mamba2-selfAttention-ppo/blob/main/legacy/ppo_transformer_ssm.py)。会使用GAE。