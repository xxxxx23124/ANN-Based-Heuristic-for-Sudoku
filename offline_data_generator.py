#!/usr/bin/env python3

import numpy as np
import h5py
import time
import uuid
from tqdm import tqdm
import os

from game_collector import GameCollector
from data_collector import DataCollector
from sudoku.sudoku import UnsolvableSudoku


class OfflineDataGenerator(DataCollector):
    """
    一个支持追加功能的离线数据生成器。
    如果文件不存在，则创建它；如果文件已存在，则向其中添加新数据。
    保存的条目如下：
        'initial_boards': [], 用于保存基础的游戏模板
        'modified_boards': [], 用于保存在基础模板上错误填写一个空格后的模板
        'solution_boards': [], 用于保存解
        'error_types': [], 用于保存modified_boards的错误类型
        'difficulties': [], 用于保存难度
        'ids': []
    """

    def __init__(self, game_collector: GameCollector):
        """
        初始化离线数据生成器。

        这个构造函数的主要作用是调用其父类 `DataCollector` 的构造函数，
        将 `game_collector` 实例传递上去。父类会完成所有必要的
        属性设置，如 `self.grid_size` 等。

        Args:
            game_collector (GameCollector): 一个 GameCollector 实例，用于生成原始数独游戏。
        """
        super().__init__(game_collector)

    def generate_and_save(self,
                          filepath: str,
                          num_games: int,
                          ):
        """
        生成数据并将其保存或追加到 HDF5 文件。

        Args:
            filepath (str): HDF5 文件的路径。
            num_games (int): 要生成的新游戏数量。
        """
        # 生成所有需要添加的新数据（在内存中）
        print(f"开始生成 {num_games} 个新的游戏样本...")
        new_data = self._collect_data_in_memory(num_games)

        num_generated = len(new_data['ids'])
        if num_generated == 0:
            print("警告：本次未能生成任何有效样本，文件未作修改。")
            return

        print(f"\n成功生成 {num_generated} 个新样本。现在开始写入文件...")

        # 将新数据写入 HDF5 文件（创建或追加）
        self._write_to_hdf5(filepath, new_data)

    def _collect_data_in_memory(self, num_games):
        """在内存中收集一批数据，返回一个字典。"""
        sudoku_puzzles = self.game_collector.collect_games(num_games)

        data_to_add = {
            'initial_boards': [],
            'modified_boards': [],
            'solution_boards': [],
            'error_types': [], # 2 for cunning error, 1 for simple error, 0 for no error
            'difficulties': [],
            'ids': []
        }

        # 遍历谜题，处理并生成数据
        for puzzle in tqdm(sudoku_puzzles, desc="[DataGen]"):
            try:
                solution_board_raw = puzzle.solve().board
                difficulty = puzzle.get_difficulty()
            except (UnsolvableSudoku, AttributeError):
                continue

            solution_board = np.array([[v or 0 for v in r] for r in solution_board_raw], dtype=np.int8)
            initial_board = np.array([[v or 0 for v in r] for r in puzzle.board], dtype=np.int8)

            empty_cells_for_error = list(map(tuple, np.argwhere(initial_board == 0)))
            """
            至少有两个空格剩下 
            __inject_cunning_error和__inject_simple_error这两个方法都只消耗一个空格。
            反正能保证最后一定可以有一个空格可以填写
            """
            is_error = False
            if len(empty_cells_for_error) > 1:
                modified_board = initial_board.copy()
                board, success = self._inject_cunning_error(modified_board, solution_board, empty_cells_for_error)
                if success:
                    modified_board, is_error, is_cunning_error  = board, True, True
                else:
                    board, success = self._inject_simple_error(modified_board, solution_board, empty_cells_for_error)
                    if success:
                        modified_board, is_error, is_cunning_error = board, True, False
                    else:
                        is_error, is_cunning_error = False, False
            elif len(empty_cells_for_error) == 1:
                is_cunning_error = False
                modified_board = initial_board.copy()
            else:
                continue

            data_to_add['initial_boards'].append(initial_board)
            data_to_add['modified_boards'].append(modified_board)
            data_to_add['solution_boards'].append(solution_board)
            data_to_add['error_types'].append(int(is_error)+int(is_cunning_error)) # 2就是狡猾错误，1就是简单错误，0就是没有填入错误
            data_to_add['difficulties'].append(difficulty)
            data_to_add['ids'].append(str(uuid.uuid4()))

        return data_to_add

    def _write_to_hdf5(self, filepath, new_data):
        """将数据写入HDF5，如果文件不存在则创建，如果存在则追加。"""
        file_exists = os.path.exists(filepath)

        # 使用 'a' 模式，它集创建（如果不存在）和追加（如果存在）于一身
        with h5py.File(filepath, 'a') as f:
            if not file_exists:
                # 文件是新创建的，需要初始化数据集结构
                print(f"文件 '{filepath}' 不存在，正在创建并初始化...")
                grid_size = self.grid_size
                f.create_dataset('initial_boards', shape=(0, grid_size, grid_size),
                                 maxshape=(None, grid_size, grid_size), dtype='i1')
                f.create_dataset('modified_boards', shape=(0, grid_size, grid_size),
                                 maxshape=(None, grid_size, grid_size), dtype='i1')
                f.create_dataset('solution_boards', shape=(0, grid_size, grid_size),
                                 maxshape=(None, grid_size, grid_size), dtype='i1')
                f.create_dataset('error_types', shape=(0,), maxshape=(None,), dtype='i1')
                f.create_dataset('difficulties', shape=(0,), maxshape=(None,), dtype='f4')
                # 对于字符串，maxshape 的工作方式相同
                f.create_dataset('ids', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'))

            # 追加数据
            num_new = len(new_data['ids'])

            for key, data_list in new_data.items():
                dset = f[key]
                target_dtype = dset.dtype
                current_size = dset.shape[0]
                dset.resize((current_size + num_new,) + dset.shape[1:])
                dset[current_size:] = np.array(data_list, dtype=target_dtype)

            print(f"成功向 '{filepath}' 追加了 {num_new} 条数据。")
            print(f"文件现在总共包含 {f['ids'].shape[0]} 条数据。")


# --- main 函数用于测试 ---
if __name__ == "__main__":
    game_gen = GameCollector(difficulty=0.5, concentration=10)
    data_generator = OfflineDataGenerator(game_collector=game_gen)

    HDF5_FILE = "sudoku_dataset_appendable.h5"

    # 第一次运行：生成 500 个样本
    print("--- 第一次运行 ---")
    data_generator.generate_and_save(
        filepath=HDF5_FILE,
        num_games=500
    )

    print("\n" + "=" * 50 + "\n")
    time.sleep(2)  # 稍作停顿，模拟两次分开的运行

    # 第二次运行：再生成 300 个样本，追加到同一个文件
    print("--- 第二次运行（追加模式）---")
    data_generator.generate_and_save(
        filepath=HDF5_FILE,
        num_games=300
    )

    # 验证最终文件
    print("\n--- 验证最终文件内容 ---")
    with h5py.File(HDF5_FILE, 'r') as f:
        print(f"最终文件 '{HDF5_FILE}' 包含 {f['ids'].shape[0]} 个样本。")
        # 预期的样本数应该是 500 + 300 = 800 左右（取决于每次有多少有效样本生成）