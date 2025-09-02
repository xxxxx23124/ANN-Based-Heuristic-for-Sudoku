#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from game_collector import GameCollector
from sudoku.sudoku import UnsolvableSudoku

class SudokuEnv(gym.Env):
    """
    一个用于强化学习的数独 Gym 环境。

    - 状态 (Observation): 一个 9x9 的 numpy 数组，0 代表空格。
    - 动作 (Action): 一个整数，范围在 0 到 728 之间。
        - 动作 `a` 被解码为 (行, 列, 数字)。
        - `row = a // 81`
        - `col = (a % 81) // 9`
        - `num = (a % 9) + 1`
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, game_collector: GameCollector,
                 render_mode: str | None = None,
                 max_episode_steps: int = 150):
        super().__init__()

        self.game_collector = game_collector
        self.grid_size = game_collector.sub_grid_size ** 2
        self.sub_grid_size = game_collector.sub_grid_size
        self.action_space_size = self.grid_size * self.grid_size * self.grid_size  # 9*9*9 = 729
        self.max_episode_steps = max_episode_steps  # 保存最大步数
        self._episode_steps = 0                     # 当前 episode 的步数计数器

        # 定义动作空间: 729个离散动作
        self.action_space = spaces.Discrete(self.action_space_size)

        # 定义状态空间: 9x9 的棋盘，每个格子可以是 0 (空) 或 1-9
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int8
        )

        self.render_mode = render_mode
        self._current_board: np.ndarray = None
        self._initial_board: np.ndarray = None
        self._empty_cells_initial: list[tuple[int, int]] = []

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)  # gymnasium 要求调用父类的reset来处理seed
        self._episode_steps = 0  # 重置步数计数器

        # 获取一个新的数独谜题，保证有解
        while True:
            puzzle = self.game_collector.collect_games(1)[0]
            try:
                puzzle.solve(assert_solvable=True)
                break
            except UnsolvableSudoku:
                continue

        # 初始化棋盘状态
        self._initial_board = np.array(
            [[val if val is not None else 0 for val in row] for row in puzzle.board],
            dtype=np.int8
        )
        self._current_board = self._initial_board.copy()


        # 记录初始的空格位置
        self._empty_cells_initial = list(zip(*np.where(self._initial_board == 0)))

        if self.render_mode == "human":
            self.render()

        return self._current_board, self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 解码动作
        row, col, num = self._decode_action(action)
        # 在执行任何有效步骤之前，增加步数计数器
        self._episode_steps += 1

        # 检查动作是否在初始空格上
        if (row, col) not in self._empty_cells_initial:
            # 对于无效动作（尝试修改固定数字），给予较大惩罚并保持状态不变
            reward = -20  # 或者一个更大的惩罚值
            terminated = False
            truncated = False
            info = self._get_info()
            info["error"] = f"Invalid move: cell ({row}, {col}) is a pre-filled number."
            return self._current_board, reward, terminated, truncated, info

        # 计算奖励并更新棋盘
        reward = 0

        # 检查是否造成明显冲突
        if not self._is_move_legal(row, col, num):
            reward += -10  # 冲突惩罚

        # 每走一步都应用一个小的负奖励
        reward += -1

        # 应用动作到棋盘
        self._current_board[row, col] = num

        # 检查游戏是否结束
        terminated = False
        truncated = False

        # 检查是否完成 (棋盘已满)
        if np.all(self._current_board != 0):
            # 棋盘已满，检查是否是正确解
            if self._is_board_complete_and_valid(self._current_board):
                reward += 150  # 成功解出
                terminated = True
            else:
                # 棋盘填满但答案错误
                reward += -50  # 可以设置一个较大的最终惩罚
                terminated = True

        # 检查是否达到最大步数
        if not terminated and self._episode_steps >= self.max_episode_steps:
            truncated = True # episode 因超时而结束，而非达到最终状态

        if self.render_mode == "human":
            self.render()

        return self._current_board, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_board(self._current_board)

    def close(self):
        # 如有需要，在这里清理资源
        pass

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        """将一维动作解码为 (行, 列, 数字)"""
        row = action // (self.grid_size * self.grid_size)
        col = (action % (self.grid_size * self.grid_size)) // self.grid_size
        num = (action % self.grid_size) + 1
        return row, col, num

    def _is_move_legal(self, row: int, col: int, num: int) -> bool:
        """检查在 (row, col) 填入 num 是否会与棋盘上其他数字冲突"""
        # 检查行冲突 (排除当前格子自身)
        if num in np.delete(self._current_board[row, :], col):
            return False

        # 检查列冲突 (排除当前格子自身)
        if num in np.delete(self._current_board[:, col], row):
            return False

        # 检查 3x3 子宫格冲突
        start_row, start_col = (self.sub_grid_size * (row // self.sub_grid_size),
                                self.sub_grid_size * (col // self.sub_grid_size))
        subgrid = self._current_board[start_row:start_row + self.sub_grid_size,
                  start_col:start_col + self.sub_grid_size].flatten()

        # 从一维化的子宫格中移除当前格子的值再检查
        # 计算当前格子在 3x3 子宫格内的相对索引
        subgrid_idx = (row % self.sub_grid_size) * self.sub_grid_size + (col % self.sub_grid_size)
        if num in np.delete(subgrid, subgrid_idx):
            return False

        return True

    def _is_board_complete_and_valid(self, board: np.ndarray) -> bool:
        """
        检查一个已填满的 9x9 棋盘是否是一个有效的数独解。
        """
        # 检查是否已填满 (作为前置条件，虽然调用时已知)
        if np.any(board == 0):
            return False

        # 检查每一行
        for i in range(self.grid_size):
            if len(np.unique(board[i, :])) != self.grid_size:
                return False

        # 检查每一列
        for j in range(self.grid_size):
            if len(np.unique(board[:, j])) != self.grid_size:
                return False

        # 检查每一个 3x3 子宫格
        for i in range(0, self.grid_size, self.sub_grid_size):
            for j in range(0, self.grid_size, self.sub_grid_size):
                subgrid = board[i:i + self.sub_grid_size, j:j + self.sub_grid_size]
                if len(np.unique(subgrid)) != self.grid_size:
                    return False

        # 如果所有检查都通过，则棋盘是一个有效的解
        return True

    def get_legal_actions_mask(self) -> np.ndarray:
        """
        返回一个布尔掩码，标记哪些动作是合法的。
        True 代表合法动作。
        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        for r, c in self._empty_cells_initial:
            # 对于每一个初始空格，9个填入动作都是“可选”的
            # 不在这里检查逻辑冲突，只检查位置是否可选
            start_idx = r * 81 + c * 9
            mask[start_idx: start_idx + 9] = True
        return mask

    def _get_info(self) -> dict:
        """返回关于当前状态的附加信息"""
        return {
            "legal_actions_mask": self.get_legal_actions_mask()
        }

    def _render_board(self, board: np.ndarray):
        """在控制台打印棋盘"""
        print("-" * 25)
        for i in range(self.grid_size):
            if i % self.sub_grid_size == 0 and i != 0:
                print("-" * 25)
            row_str = ""
            for j in range(self.grid_size):
                if j % self.sub_grid_size == 0 and j != 0:
                    row_str += "| "

                num = board[i, j]
                # 对初始给定的数字使用不同颜色或样式，以作区分
                if (i, j) not in self._empty_cells_initial and num != 0:
                    # 使用 ANSI escape code 加粗显示
                    row_str += f"\033[1m{num}\033[0m "
                else:
                    row_str += f"{num if num != 0 else '.'} "
            print(f"| {row_str}|")
        print("-" * 25)
