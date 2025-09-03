import torch

from game_collector import GameCollector
from Env.sudoku_env import SudokuEnv
from Agent.SudokuAgent import SudokuAgent
from RL.PPO.PPO import PPO

if __name__ == "__main__":
    # --- 初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 环境参数
    GRID_SIZE = 9

    # 训练超参数
    DIFFICULTY = 0.2
    MAX_EPISODE_STEPS = 120
    TOTAL_TIMESTEPS = 1_000_000_000_000
    TRAJECTORIES_PER_UPDATE = 1  # 每次更新收集n条轨迹
    EPOCHS_PER_UPDATE = 4  # 在每批数据上训练n个周期
    LEARNING_RATE = 1e-5
    CHUNK_SIZE_TUPLE = (64, 32, 16, 8, 4, 1)  # 每次处理n个时间步的数据

    # --- 创建环境和智能体 ---
    game_collector = GameCollector(difficulty=DIFFICULTY, concentration=20)
    env = SudokuEnv(game_collector=game_collector, max_episode_steps=MAX_EPISODE_STEPS)
    agent = SudokuAgent(grid_size=GRID_SIZE, device=device)

    # --- 创建并启动 PPO 训练 ---
    ppo_trainer = PPO(
        env=env,
        agent=agent,
        epochs=EPOCHS_PER_UPDATE,
        lr=LEARNING_RATE,
        chunk_size_tuple=CHUNK_SIZE_TUPLE,
        device=device
    )

    ppo_trainer.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        trajectories_per_update=TRAJECTORIES_PER_UPDATE
    )

    print("Training finished.")
    # 保存模型
    # torch.save(agent.actor_critic.state_dict(), "ppo_sudoku_model.pth")