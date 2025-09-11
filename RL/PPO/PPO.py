import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

from Env.sudoku_env import SudokuEnv
from Agent.SudokuAgent import SudokuAgent
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache


class PPO:
    """
    近端策略优化 (PPO) 算法实现。
    专为显存受限环境设计，核心思路是基于轨迹的梯度累积。
    """

    def __init__(self,
                 env: SudokuEnv,
                 agent: SudokuAgent,
                 epochs: int,
                 lr: float,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 target_kl: float = 0.005,
                 chunk_size_tuple: tuple[int, ...] = (64, 32, 16, 8, 4, 1),
                 device: torch.device = torch.device('cuda')):
        """
        初始化 PPO 训练器。

        Args:
            env (SudokuEnv): 训练用的 Gym 环境。
            agent (SudokuAgent): 包含 Actor-Critic 网络的智能体。
            epochs (int): 在每批数据上训练的周期数。
            lr (float): 学习率。
            gamma (float): 折扣因子。
            gae_lambda (float): GAE (广义优势估计) 的 lambda 参数。
            clip_coef (float): PPO 裁剪范围的系数。
            ent_coef (float): 熵奖励的系数，鼓励探索。
            vf_coef (float): 价值函数损失的系数。
            target_kl (float): approx_kl的阈值，如果超过，就跳出循环。
            chunk_size_list tuple: 处理序列数据时的分块大小，降序。
            device (torch.device): 计算设备。
        """
        self.env = env
        self.agent = agent
        self.device = device

        # --- 超参数 ---
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.chunk_size_tuple = chunk_size_tuple

        self.optimizer = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=lr)

    def _collect_one_trajectory(self) -> dict:
        """从环境中收集一条完整的轨迹 (episode)。"""
        trajectory:dict = {
            "obs": [],
            "initial_boards": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "terminateds": [],
            "truncateds": [],
            "legal_masks": []
        }

        # 重置环境和智能体状态
        obs_dict, info_dict = self.env.reset()
        initial_board = self.env._initial_board.copy()
        self.agent.reset()

        terminated, truncated = False, False

        while not terminated and not truncated:
            # --- 智能体决策 ---
            # 预处理观测值
            model_input = self.agent._preprocess_observation(obs_dict, initial_board)

            # 从模型获取动作概率、价值和log-prob
            with torch.no_grad():
                action_logits, value, self.agent.cache_list = self.agent.actor_critic(
                    model_input, self.agent.cache_list
                )

            # 应用掩码并采样动作
            action_logits = action_logits.squeeze(0).squeeze(0)
            legal_mask = torch.from_numpy(info_dict['legal_actions_mask']).to(self.device)
            dist = self._get_masked_distribution(action_logits, legal_mask)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            # --- 与环境交互 ---
            action_int = action.item()
            next_obs_dict, reward, terminated, truncated, next_info_dict = self.env.step(action_int)

            # 将reward转换为tensor类型
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            # --- 存储数据 ---
            trajectory["obs"].append(obs_dict)
            trajectory["initial_boards"].append(initial_board)  # 存储每个时间步的初始棋盘
            trajectory["actions"].append(action)
            trajectory["log_probs"].append(log_prob)
            trajectory["rewards"].append(reward)
            trajectory["values"].append(value.squeeze())
            trajectory["terminateds"].append(terminated)
            trajectory["truncateds"].append(truncated)
            trajectory["legal_masks"].append(legal_mask)

            # 更新状态
            obs_dict, info_dict = next_obs_dict, next_info_dict

        # 将列表转换为张量
        for key, val in trajectory.items():
            if key in ["actions", "log_probs", "rewards", "values", "legal_masks"]:
                trajectory[key] = torch.stack(val)

        return trajectory

    def _get_masked_distribution(self, logits: torch.Tensor, mask: torch.Tensor):
        """根据合法动作掩码创建分类分布。"""
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def _compute_advantages_and_returns(self, trajectory: dict):
        """使用 GAE 计算优势和回报"""
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        terminateds = trajectory["terminateds"]
        truncateds = trajectory["truncateds"]

        T = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0

        for t in reversed(range(T)):
            # 判断下一步是否存在
            if t == T - 1:
                next_nonterminal = 0.0
                next_value = 0.0
            else:
                # 如果下一步是 truncated 而不是 terminated，仍然要 bootstrap
                # 所以 nonterminal 的判断应该基于当前步的状态
                next_nonterminal = 1.0 - (terminateds[t] or truncateds[t])
                # 如果当前步是 terminated，则下一个状态的价值为0
                next_value = values[t + 1] * (1.0 - terminateds[t])

            # 计算 delta
            # 在 `truncated` 情况下, next_value 是非零的, 但 next_nonterminal 是零.
            # delta 的计算应该用 next_value, 而 GAE 的递归应该用 next_nonterminal
            delta = rewards[t] + self.gamma * next_value - values[t]

            # 计算 GAE
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam

        returns = advantages + values
        return advantages, returns

    def learn(self, total_timesteps: int, trajectories_per_update: int):
        """
        主训练循环。

        Args:
            total_timesteps (int): 训练的总时间步数。
            trajectories_per_update (int): 每一次参数更新需要收集的轨迹数量。
        """
        global_step = 0
        update_count = 0

        while global_step < total_timesteps:
            update_start_time = time.time()
            update_count += 1
            print(f"--- Update #{update_count} ---")

            # --- 轨迹级梯度累积 ---
            self.optimizer.zero_grad()  # 在每次大更新前清零梯度

            batch_trajectories = []
            rollout_timesteps = 0

            # --- 收集一批轨迹 ---
            print(f"Collecting {trajectories_per_update} trajectories...")
            for _ in tqdm(range(trajectories_per_update), desc="Rollout"):
                traj = self._collect_one_trajectory()
                batch_trajectories.append(traj)
                rollout_timesteps += len(traj["rewards"])

            global_step += rollout_timesteps
            print(f"Collected {rollout_timesteps} timesteps. Total timesteps: {global_step}/{total_timesteps}")

            # --- 多 Epoch 训练 ---
            for epoch in range(self.epochs):

                # --- 对每条轨迹进行学习 ---

                # 切换到训练模式
                self.agent.actor_critic.train()

                # 跨 trajectory 累积梯度，但在同一个 epoch 内更新一次
                # 清空可能未被清空的梯度
                self.optimizer.zero_grad()

                # 用来累加所有轨迹的损失
                total_loss = torch.zeros(1, device=self.device)

                break_epoch_loop = False

                for traj_idx, trajectory in enumerate(batch_trajectories):
                    print(
                        f"  Training on trajectory {traj_idx + 1}/{len(batch_trajectories)} (length: {len(trajectory['rewards'])})")

                    # 计算 GAE 和回报
                    advantages, returns = self._compute_advantages_and_returns(trajectory)
                    # 标准化优势 (可选但推荐)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    obs_np = np.array(trajectory["obs"])
                    initial_boards_np = np.array(trajectory["initial_boards"])
                    old_actions = trajectory["actions"].to(self.device)
                    old_log_probs = trajectory["log_probs"].to(self.device)
                    legal_masks = trajectory["legal_masks"].to(self.device)

                    trajectory_len = len(obs_np)
                    # 用来累加轨迹中所有子块的损失
                    trajectory_loss = torch.zeros(1, device=self.device)

                    # trajectory_weight：让一条 trajectory 内所有 chunk 的权重和为 1，
                    # 从而得到“按时间步均匀平均”的 loss。
                    trajectory_weight = 0.0

                    # --- 分块处理 (Chunking) ---
                    # 遍历序列数据块
                    processed_len = 0
                    inference_cache:Mamba2InferenceCache | None = None

                    new_log_probs = []

                    while processed_len < trajectory_len:
                        remaining_len = trajectory_len - processed_len

                        # 决定当前块的大小
                        chunk_size = 1
                        for test_size in self.chunk_size_tuple:
                            if remaining_len >= test_size:
                                chunk_size = test_size
                                break

                        start = processed_len
                        end = start + chunk_size

                        processed_len += chunk_size

                        # 准备数据块
                        # np(S, H, W) -> np(chunk_size, H, W)
                        obs_chunk = obs_np[start:end]
                        initial_boards_chunk = initial_boards_np[start:end]

                        # 转换成模型输入格式 np(chunk_size, H, W) -> tensor(1, chunk_size, H, W, 10)
                        model_input = self._preprocess_batch(obs_chunk, initial_boards_chunk)

                        # 前向传播并使用缓存
                        action_logits, new_values, inference_cache = self.agent.actor_critic(model_input, cache_list=inference_cache)

                        # 计算新 log_probs, 熵, 和价值
                        dist = self._get_masked_distribution(action_logits.squeeze(0), legal_masks[start:end])
                        sub_new_log_probs = dist.log_prob(old_actions[start:end])
                        entropy = dist.entropy()

                        # 积累log_probs
                        new_log_probs.append(sub_new_log_probs)

                        # 计算损失
                        # 策略损失 (Policy Loss)
                        ratio = torch.exp(sub_new_log_probs - old_log_probs[start:end])
                        # 与 pg_loss1 = advantages[start:end] * ratio
                        # pg_loss2 = advantages[start:end] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        # pg_loss = (-torch.min(pg_loss1, pg_loss2)).mean() 等效
                        pg_loss1 = -advantages[start:end] * ratio
                        pg_loss2 = -advantages[start:end] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # 价值损失 (Value Loss)
                        v_loss = F.mse_loss(new_values.squeeze(), returns[start:end])

                        # 熵损失 (Entropy Loss)
                        entropy_loss = entropy.mean()

                        # 子损失 loss_chunk
                        loss_chunk = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                        w = chunk_size  # 每个时间步权重 = 1
                        trajectory_loss = trajectory_loss + loss_chunk * w
                        trajectory_weight = trajectory_weight + w

                    # 总损失
                    total_loss = total_loss + trajectory_loss/trajectory_weight

                    # list -> tensor
                    new_log_probs = torch.cat(new_log_probs, dim=0)

                    # 这条轨迹的 Approximating KL Divergence
                    logratios = new_log_probs - old_log_probs
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    print(f"  Training on trajectory {traj_idx + 1}/{len(batch_trajectories)} (approx kl: {approx_kl:>16})")
                    if approx_kl > self.target_kl:
                        #break_epoch_loop = True
                        #break
                        pass

                # --- 执行参数更新 ---
                if not break_epoch_loop:
                    # 反向传播，累积梯度
                    # 梯度会按比例缩放，以正确处理不同轨迹的贡献
                    avg_loss = total_loss / trajectories_per_update
                    avg_loss.backward()
                    # 使用累积的梯度更新网络参数
                    nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), 0.5)  # 梯度裁剪
                    self.optimizer.step()

                self.agent.actor_critic.eval()  # 切换回评估模式

                update_duration = time.time() - update_start_time
                print(
                    f"Update finished in {update_duration:.2f}s. Avg reward: {np.mean([t['rewards'].sum().item() for t in batch_trajectories]):.2f}")
                print("-" * 25)

                if break_epoch_loop:
                    break

    def _preprocess_batch(self, obs_chunk: np.ndarray, initial_boards_batch: np.ndarray) -> torch.Tensor:
        """为一批观测数据（一个块）进行预处理。"""
        chunk_size = len(obs_chunk)
        H, W = obs_chunk[0].shape
        C = 10  # 模型通道数

        model_input = np.zeros((chunk_size, H, W, C), dtype=np.float32)

        for i in range(chunk_size):
            obs = obs_chunk[i]
            initial_board = initial_boards_batch[i]

            # 填充 one-hot
            for r in range(H):
                for c in range(W):
                    num = obs[r, c]
                    if num != 0:
                        model_input[i, r, c, num - 1] = 1.0

            # 填充不可变掩码
            model_input[i, :, :, 9] = (initial_board != 0).astype(np.float32)

        tensor_obs = torch.from_numpy(model_input).to(self.device)
        # 增加批次维度 (因为我们的网络总是接收一个批次，即使B=1)
        final_tensor = tensor_obs.unsqueeze(0)

        return final_tensor
