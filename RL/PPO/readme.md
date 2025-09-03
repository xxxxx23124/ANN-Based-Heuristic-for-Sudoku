# PPO在这里实现
此目录包含专为显存受限环境设计的近端策略优化 (PPO) 算法实现。

### 核心设计思路

实现严格遵循以下原则以最大化显存效率：

1.  **固定 `batch_size = 1`**：
    在数据收集（Rollout）阶段，每次只处理单个环境实例，避免了并行环境带来的显存开销。

2.  **轨迹级梯度累积**：
    -   **梯度累加**：在每个轨迹的训练循环中计算出的梯度，会被累积下来，但**不立即**执行优化器步骤。
    -   **延迟更新**：在处理完设定的批次中所有轨迹后，才使用累积的梯度执行一次参数更新。

3.  **分块（Chunking）与缓存（Caching）**：
    -   在处理长序列轨迹时，将数据切分为固定大小的块。
    -   利用模型的自回归缓存机制，在块之间传递隐藏状态，实现了在处理长序列时显存占用的恒定性。


参考实现:
1. [Coding PPO From Scratch With PyTorch](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)
2. [PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners)
3. [RL_mamba2-selfAttention-ppo/legacy/ppo_transformer_ssm.py](https://github.com/xxxxx23124/RL_mamba2-selfAttention-ppo/blob/main/legacy/ppo_transformer_ssm.py)