#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
from ANN.Networks.TimeSpaceSudokuModel import TimeSpaceSudokuModel
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache


class SudokuAgent:
    """
    一个使用 TimeSpaceSudokuModel 与 SudokuEnv 交互的强化学习智能体。
    """

    def __init__(self, grid_size: int, device: torch.device):
        """
        初始化智能体。

        Args:
            grid_size (int): 数独棋盘的边长 (例如，9)。
            device (torch.device): 模型和计算所用的设备 (例如, torch.device('cuda'))。
        """
        self.grid_size = grid_size
        self.device = device

        # 初始化 Actor-Critic 模型
        self.actor_critic = TimeSpaceSudokuModel(grid_size=grid_size, device=device)
        self.actor_critic.to(device)
        self.actor_critic.eval()  # 默认设置为评估模式，在训练时再切换

        # 初始化用于自回归推理的缓存列表
        self.cache_list: list[list[Mamba2InferenceCache]] | None = None

    def _preprocess_observation(self, obs_board: np.ndarray, initial_board: np.ndarray) -> torch.Tensor:
        """
        将环境的棋盘观测值转换为模型所需的输入张量。

        Args:
            obs_board (np.ndarray): 当前的棋盘状态 (H, W)。
            initial_board (np.ndarray): 初始谜题的棋盘 (H, W)。

        Returns:
            torch.Tensor: 形状为 (1, 1, H, W, 10) 的模型输入张量。
        """
        # 创建一个 (H, W, 10) 的零张量
        # C=10 的设计:
        # - 通道 0-8: 对应数字 1-9 的 one-hot 编码
        # - 通道 9: 标记哪些是初始给定的、不可更改的数字
        model_input = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.float32)

        # 填充 one-hot 编码 (通道 0-8)
        # 遍历棋盘，对于非零数字 n，在第 n-1 通道上标记为 1
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                num = obs_board[r, c]
                if num != 0:
                    model_input[r, c, num - 1] = 1.0

        # 填充不可变位置掩码 (通道 9)
        # 初始棋盘中非零的位置是不可变的
        immutable_mask = (initial_board != 0).astype(np.float32)
        model_input[:, :, 9] = immutable_mask

        # 转换为 PyTorch 张量并移动到指定设备
        tensor_obs = torch.from_numpy(model_input).to(self.device)

        # 增加批次 (B) 和序列 (S) 维度，使其成为 (1, 1, H, W, C)
        final_tensor = tensor_obs.unsqueeze(0).unsqueeze(0)

        return final_tensor

    def step(self, obs: dict, initial_board: np.ndarray, deterministic: bool = False) -> tuple[int, float]:
        """
        根据当前观测值选择一个动作。

        Args:
            obs (dict): 来自 SudokuEnv 环境的观测字典，包含 'observation' 和 'legal_actions_mask'。
            initial_board (np.ndarray): 初始谜题的棋盘，用于生成模型输入。
            deterministic (bool): 是否使用确定性策略 (选择概率最高的动作)。False 则进行采样。

        Returns:
            tuple[int, float]: (选择的动作, 评估的价值)。
        """
        with torch.no_grad():
            # 预处理观测数据
            model_input = self._preprocess_observation(obs['observation'], initial_board)

            # 从模型获取输出，并传入/更新缓存
            # self.cache_list 在第一次调用时为 None，之后会包含上一时间步的缓存
            action_logits, value, self.cache_list = self.actor_critic(model_input, self.cache_list)
            # action_logits shape: (1, 1, grid_size**3)
            # value shape: (1, 1, 1)

            # 应用合法动作掩码
            action_logits = action_logits.squeeze(0).squeeze(0)  # 降维至 (grid_size**3)

            # SudokuEnv 中没有 action_mask，而是 legal_actions_mask
            # 注意：SudokuEnv 中的 get_legal_actions_mask() 实际上是标记了 "可填" 的位置，
            # 而不是逻辑上合法的动作。这与 AlphaGo 的 MCTS 概念不同。
            legal_mask = torch.from_numpy(obs['legal_actions_mask']).to(self.device)

            # 将掩码为 0 的位置 (非法动作) 在 logits 中对应的值设为负无穷
            masked_logits = action_logits.masked_fill(legal_mask == 0, -1e9)
            action_probs = F.softmax(masked_logits, dim=-1)

            # 从概率分布中选择一个动作
            if deterministic:
                # 确定性选择：选择概率最高的动作
                final_action = torch.argmax(action_probs, dim=-1)
            else:
                # 随机性采样：从分布中抽取一个样本
                final_action = torch.multinomial(action_probs, num_samples=1)

            # 提取标量值
            final_action_int = final_action.item()
            value_float = value.squeeze().item()

            return final_action_int, value_float

    def reset(self):
        """
        重置智能体的内部状态，主要是清除缓存。
        应在每个新游戏（episode）开始时调用。
        """
        self.cache_list = None
        print("SudokuAgent cache has been reset.")



from game_collector import GameCollector
from Env.sudoku_env import SudokuEnv

if __name__ == '__main__':
    # 初始化环境和智能体
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_size = 9

    collector = GameCollector(difficulty=0.5, concentration=20)
    env = SudokuEnv(game_collector=collector, max_episode_steps=150)  # 最多填满所有格子

    agent = SudokuAgent(grid_size=grid_size, device=device)

    # 运行一个完整的游戏（episode）
    obs, info = env.reset()
    # 必须保存初始棋盘，因为它在整个 episode 中用于生成模型输入
    initial_board_for_episode = env._initial_board.copy()

    agent.reset()  # 每个新游戏开始时重置智能体缓存

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        env.render()  # 显示当前棋盘

        # 智能体决策
        action, value = agent.step(
            {'observation': obs, 'legal_actions_mask': info['legal_actions_mask']},
            initial_board=initial_board_for_episode,
            deterministic=False  # 在训练时设为 False 以便探索
        )

        # 与环境交互
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        print(f"Step: {step_count}, Action: {action}, Value: {value:.4f}, Reward: {reward}")
        if "error" in info:
            print(f"Agent Error: {info['error']}")

    print("\n--- Episode Finished ---")
    env.render()  # 显示最终棋盘
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward}")
    if terminated:
        print("Result: Game solved or ended with a final state.")
    if truncated:
        print("Result: Episode truncated due to max steps.")