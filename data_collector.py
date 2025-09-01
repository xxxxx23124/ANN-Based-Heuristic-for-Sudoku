#!/usr/bin/env python3

import numpy as np
import time

from game_collector import GameCollector
from sudoku import Sudoku
from sudoku.sudoku import UnsolvableSudoku


class DataCollector:
    def __init__(self, game_collector: GameCollector):
        """
        初始化数据转换器。

        Args:
            game_collector (GameCollector): 一个 GameCollector 实例，用于生成原始数独游戏。
        """
        self.game_collector = game_collector
        self.grid_size = game_collector.sub_grid_size ** 2
        self.sub_grid_size = game_collector.sub_grid_size

    def collect_training_data(self,
                              num_games: int,
                              error_injection_prob: float = 0.5,
                              cunning_error_prob: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        生成训练模型的数据集。
        可以向棋盘中注入错误，以训练模型识别冲突并“回溯”。

        Args:
            num_games (int): 要生成和处理的游戏数量。
            error_injection_prob (float): 向棋盘注入错误的概率，介于0和1之间。
            cunning_error_prob (float): 在决定注入错误时，有多大可能性注入“狡猾”的错误。(1 - cunning_error_prob) 的概率会注入简单的随机错误。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - X (np.ndarray): 输入数据，形状为 (num_samples, grid_size, grid_size, grid_size + 1)。
            - y (np.ndarray): 标签数据，形状为 (num_samples,)。标签值范围为 0-8 (填数字1-9) 或 9 (回溯)。
        """
        sudoku_puzzles = self.game_collector.collect_games(num_games)
        X_list, y_list = [], []

        for puzzle in sudoku_puzzles:
            try:
                solved_puzzle = puzzle.solve()
                # 检查返回的是否是有效的Sudoku对象
                if solved_puzzle is None:
                    # 如果谜题本身无解 (py-sudoku库有时会生成)，跳过
                    continue
                solution_board_raw = solved_puzzle.board
            except UnsolvableSudoku:  # py-sudoku 库在确定无解时会抛出这个异常
                continue

            # 统一处理 None -> 0
            solution_board = np.array([[val if val is not None else 0 for val in row] for row in solution_board_raw], dtype=int)
            # 同样处理 problem_board
            problem_board_raw = puzzle.board
            problem_board = np.array([[val if val is not None else 0 for val in row] for row in problem_board_raw], dtype=int)

            empty_cells= list(map(tuple, np.argwhere(problem_board == 0)))

            if len(empty_cells) == 0:
                continue

            target_cell_idx = np.random.randint(len(empty_cells))
            target_row, target_col = empty_cells.pop(target_cell_idx)

            is_error_injected = False
            # 检查是否有可供注入错误的位置
            if empty_cells and np.random.random() < error_injection_prob:

                # 决定注入哪种类型的错误
                if np.random.random() < cunning_error_prob:
                    # 尝试注入“狡猾”的错误
                    board_after_injection, success = self._inject_cunning_error(problem_board.copy(),
                                                                                solution_board,
                                                                                empty_cells
                                                                                )
                    if success:
                        problem_board = board_after_injection
                        is_error_injected = True

                # 如果狡猾错误没有成功注入，或者随机决定注入简单错误
                if not is_error_injected:
                    # 注入简单错误
                    board_after_injection, success = self._inject_simple_error(problem_board.copy(),
                                                                               solution_board,
                                                                               empty_cells
                                                                               )
                    if success:
                        problem_board = board_after_injection
                        is_error_injected = True

            # 创建模型输入
            model_input = self.__create_model_input(problem_board, target_row, target_col)

            if not is_error_injected:
                # 没有注入错误，标签是目标格子的正确数字 (0-8)
                label = solution_board[target_row, target_col] - 1
            else:
                # 注入了错误，模型应该学会识别并输出“回溯”信号
                label = self.grid_size  # 标签为 9

            X_list.append(model_input)
            y_list.append(label)

        if not X_list: # 如果没有成功生成任何样本
             return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def _inject_simple_error(self,
                             board: np.ndarray,
                             solution: np.ndarray,
                             empty_cells: list[tuple[int, int]]) -> tuple[np.ndarray, bool]:
        """
        注入一个简单的、局部冲突的错误。
        该方法直接选择一个空格，并填入一个与其所在行、列或宫内已有数字冲突的数字。
        """
        if not empty_cells:
            return board, False

        # 随机打乱空格列表，以增加随机性
        np.random.shuffle(empty_cells)

        for r, c in empty_cells:
            # 找出该位置所有已使用的数字（冲突数字）
            # 检查行和列
            row_nums = set(board[r, :])
            col_nums = set(board[:, c])

            # 检查sub grid
            start_row = (r // self.sub_grid_size) * self.sub_grid_size
            start_col = (c // self.sub_grid_size) * self.sub_grid_size
            box_nums = set(
                board[start_row:start_row + self.sub_grid_size, start_col:start_col + self.sub_grid_size].flatten())

            # 合并所有冲突数字，并排除0（代表空格）
            used_nums = row_nums.union(col_nums, box_nums)
            used_nums.discard(0)

            # 如果存在可用的冲突数字，就随机选一个填入
            if used_nums:
                # 将集合转换为列表以便随机选择
                error_candidates = list(used_nums)
                error_move = np.random.choice(error_candidates)

                # 创建副本并注入错误
                board_with_error = board.copy()
                board_with_error[r, c] = error_move

                # 成功注入错误，直接返回
                return board_with_error, True

        # 如果遍历了所有空格都找不到任何可以制造冲突的数字（理论上不太可能发生），则返回失败
        return board, False

    def _inject_cunning_error(self,
                              board: np.ndarray,
                              solution: np.ndarray,
                              empty_cells: list[tuple[int, int]],
                              timeout: float = 16.0
                              ) -> tuple[np.ndarray, bool]:
        """
        尝试向棋盘注入一个“狡猾”的错误。

        Args:
            board (np.ndarray): 当前棋盘状态。
            solution (np.ndarray): 正确的解。
            empty_cells (list[tuple[int, int]]): 可供填写的空格列表。
            timeout (float): 允许寻找狡猾错误的最大时间（秒）。

        Returns:
            Tuple[np.ndarray, bool]: 修改后的棋盘和是否成功的标志。
        """
        np.random.shuffle(empty_cells)  # 随机化空格顺序
        start_time = time.monotonic()  # 记录开始时间，monotonic() 不受系统时间更改影响，适合计时

        for r, c in empty_cells:
            # 在处理下一个空格之前检查时间
            if time.monotonic() - start_time > timeout:
                return board, False

            # 找到所有局部合法的数字
            legal_moves = self.__get_legal_moves(board, r, c)

            # 找到正确答案
            correct_answer = solution[r, c]

            # 找出“合法但不正确”的数字作为狡猾错误的候选
            cunning_candidates = list(legal_moves - {correct_answer})
            np.random.shuffle(cunning_candidates)

            if not cunning_candidates:
                continue

            # 验证这些候选是否真的能破坏解
            for move in cunning_candidates:
                # 在尝试下一个候选数字之前检查时间
                if time.monotonic() - start_time > timeout:
                    return board, False

                temp_board = board.copy()
                temp_board[r, c] = move

                # 创建一个 Sudoku 实例来验证可解性
                # 注意：py-sudoku 的 board 接受 list[list[Optional[int]]]
                temp_board_list = [[int(val) if val != 0 else None for val in row] for row in temp_board]

                try:
                    # 尝试求解这个被污染的棋盘
                    puzzle_to_check = Sudoku(self.sub_grid_size, board=temp_board_list)
                    solved_puzzle = puzzle_to_check.solve(assert_solvable=True)  # assert_solvable=True 会在无解时抛出UnsolvableSodoku异常

                    # 如果能成功求解，说明这个错误不够“狡猾”，继续尝试下一个
                    if solved_puzzle.validate():
                        continue

                except UnsolvableSudoku:
                    # 如果 solve() 抛出异常，说明棋盘已不可解
                    return temp_board, True

        # 如果遍历完所有可能都无法找到狡猾错误（或中途超时退出了循环）
        return board, False

    def __get_legal_moves(self,
                          board: np.ndarray,
                          r: int,
                          c: int) -> set[int]:
        """
        获取在 (r, c) 位置所有局部合法的数字 (不与行、列、宫冲突)。
        """
        all_nums = set(range(1, self.grid_size + 1))

        # 检查行和列
        row_nums = set(board[r, :])
        col_nums = set(board[:, c])

        # 检查sub grid
        start_row = (r // self.sub_grid_size) * self.sub_grid_size
        start_col = (c // self.sub_grid_size) * self.sub_grid_size
        box_nums = set(board[start_row:start_row + self.sub_grid_size, start_col:start_col + self.sub_grid_size].flatten())

        used_nums = row_nums.union(col_nums, box_nums)

        return all_nums - used_nums

    def __create_model_input(self, board: np.ndarray, target_row: int, target_col: int) -> np.ndarray:
        """
        将棋盘状态和目标位置编码成一个 (9, 9, 10) 的张量。

        Args:
            board (np.ndarray): 9x9 的数独棋盘，0代表空格。
            target_row (int): 目标单元格的行索引。
            target_col (int): 目标单元格的列索引。

        Returns:
            np.ndarray: 形状为 (9, 9, 10) 的输入张量。
        """
        # 初始化一个全零张量
        input_tensor = np.zeros((self.grid_size, self.grid_size, self.grid_size + 1), dtype=np.float32)

        # 遍历棋盘上的每一个格子
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                num = board[r, c]
                if num != 0:
                    # 将数字 (1-9) 映射到通道索引 (0-8)
                    channel_idx = num - 1
                    input_tensor[r, c, channel_idx] = 1.0
        # 在第10个通道 (索引为9) 中标记目标单元格的位置
        input_tensor[target_row, target_col, self.grid_size] = 1.0
        return input_tensor


# --- main 函数用于测试 ---
if __name__ == "__main__":
    game_gen = GameCollector(difficulty=0.6, concentration=20)
    data_gen = DataCollector(game_collector=game_gen)

    print("生成包含狡猾错误的数据...")
    X, y = data_gen.collect_training_data(num_games=20, error_injection_prob=1.0, cunning_error_prob=1.0)

    print(f"成功生成 {len(X)} 个样本。")
    if len(X) > 0:
        error_samples_count = np.sum(y == data_gen.grid_size)
        correct_samples_count = len(y) - error_samples_count
        print(f"  - 错误样本 (label=9): {error_samples_count}")
        print(f"  - 正确样本 (label=0-8): {correct_samples_count}")

        # 检查一个错误样本
        error_indices = np.where(y == data_gen.grid_size)[0]
        if len(error_indices) > 0:
            sample_idx = error_indices[0]
            sample_X = X[sample_idx]
            print("\n--- 检查一个狡猾错误样本 ---")

            reconstructed_board = np.zeros((9, 9), dtype=int)
            for r in range(9):
                for c in range(9):
                    if np.sum(sample_X[r, c, :9]) > 0:
                        reconstructed_board[r, c] = np.argmax(sample_X[r, c, :9]) + 1

            print("注入错误后的棋盘:")
            print(reconstructed_board)

            target_pos = np.unravel_index(np.argmax(sample_X[:, :, 9]), (9, 9))
            print(f"模型被要求判断位置: {target_pos}")
            print("由于棋盘存在逻辑死锁，正确标签是 '回溯'(9)。")
