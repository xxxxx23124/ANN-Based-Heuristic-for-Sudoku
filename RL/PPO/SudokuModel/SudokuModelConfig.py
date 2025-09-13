from dataclasses import dataclass, field
from ANN.Networks.NetworkConfig import NetworkConfig

@dataclass
class SudokuModelConfig:
    grid_size: int
    input_channels: int

    # 嵌入层配置
    embed_dim: int = 512

    # 主干网络配置
    backbone_depth: int = 8

    # 策略头配置
    actor_depth: int = 8

    # 价值头配置
    critic_depth: int = 8

    d_model: int = field(init=False)
    max_seq_len: int = field(init=False)
    output_dim: int = field(init=False)

    def __post_init__(self):
        """根据基础配置计算衍生配置"""
        self.d_model = self.embed_dim
        self.max_seq_len = self.grid_size ** 2
        self.output_dim = self.grid_size ** 3

        # 主干配置
        self.backbone_args = NetworkConfig(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            timespaceblock_num=self.backbone_depth,
        )
        # 动作头的配置
        self.actor_args = NetworkConfig(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            timespaceblock_num=self.actor_depth,
        )

        # 价值头的配置
        self.critic_args = NetworkConfig(
            d_model=self.embed_dim,
            max_seq_len=self.max_seq_len,
            timespaceblock_num=self.critic_depth,
        )