#!/usr/bin/env python3

import numpy as np

# https://pypi.com.cn/project/py-sudoku/1.0.3/
from sudoku import Sudoku

class GameCollector:
    def __init__(self,
                 difficulty: float,
                 concentration:float,
                 sub_grid_size: int = 3,
                 ) -> None:
        """
        Args:
            difficulty (float): 中心难度值，必须在 (0, 1) 之间。
            concentration (float): 难度分布的集中程度。值越大，采样结果越接近中心难度。
            sub_grid_size (int): 小格子的大小，默认为3
        """
        self.current_difficulty = difficulty
        self.concentration = concentration
        self.sub_grid_size=sub_grid_size
        if not 0 < difficulty < 1:
            raise ValueError("Difficulty must be between 0 and 1.")
        if concentration <= 0:
            raise ValueError("Concentration must be a positive number.")
        if self.sub_grid_size <= 0:
            raise ValueError("Sub-grid size must be a positive number.")

    def collect_games(self, game_size: int) -> list[Sudoku]:
        """
        收集游戏的底层样本。
        game_size (int): 收集的游戏数
        """
        if game_size < 0:
            raise ValueError("Game size must be a positive number.")

        games_list = []
        while len(games_list) < game_size:
            puzzle = Sudoku(self.sub_grid_size).difficulty(difficulty=self.__sample_difficulty())
            games_list.append(puzzle)
        return games_list

    def set_current_difficulty(self, target_difficulty: float) -> None:
        if not 0 < target_difficulty < 1:
            raise ValueError("Target Difficulty must be between 0 and 1.")
        self.current_difficulty = target_difficulty

    def __sample_difficulty(self) -> float:
        """
        从以 current_difficulty 为中心的 β 分布中采样一个难度值。
        Mean = α / (α + β)
        α = current_difficulty * concentration
        β = (1 - current_difficulty) * concentration
        """
        alpha = self.current_difficulty * self.concentration
        beta = (1.0 - self.current_difficulty) * self.concentration
        sampled_difficulty = np.random.beta(alpha, beta)
        return sampled_difficulty

if __name__ == "__main__":
    print(type(Sudoku(3).difficulty(difficulty=0.5)))
    x = GameCollector(difficulty=0.5,concentration=10)
    l = x.collect_games(128)
    print(l[1].board)
    print(l[1].solve().board)

    l[1].show()
    print("-----------------------------------------------")
    l[1].solve().show()
