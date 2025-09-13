import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

from ANN.Networks.TimeSpaceChunk import TimeSpaceChunk
from RL.SAC.SudokuModel.SudokuModelConfig import SudokuModelConfig


class SudokuValueInputEncoder(nn.Module):
    """负责将 (B, S, H, W, C)的输入编码为 (B, S, L, D) 的特征"""

    def __init__(self, input_channels: int, embed_dim: int, device: torch.device):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding='same',
            device=device
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        state: (B, S, H, W, C)
        """
        B, S, H, W, C = state.shape

        state = rearrange(state, "b s h w c -> (b s) c h w")
        x = F.gelu(self.conv(state))
        x = rearrange(x, "(b s) d h w -> b s (h w) d", s=S)
        # return: (B, S, L, D)
        return x

class ValueHead(nn.Module):
    """Value 头：TimeSpace -> 输出"""

    def __init__(self, config: NetworkConfig, device: torch.device):
        super().__init__()
        self.config = config

        # 时空处理块
        self.timespace_chunk = TimeSpaceChunk(config, device=device)

        # 输出层
        self.output_head = nn.Linear(config.d_model, 1, device=device)

    def forward(
            self,
            x: Tensor,
            H: int, W: int,
            rotary_emb: RotaryEmbedding,
            cache_list: list[Mamba2InferenceCache] | None = None
    ) -> tuple[Tensor, list[Mamba2InferenceCache]]:

        # 时空处理
        value, new_cache_list = self.timespace_chunk(
            x, H, W, rotary_emb, cache_list
        )

        # 聚合与输出
        # B, S, L, D -> B, S, L
        value = self.output_head(value).squeeze(-1)
        # B, S, L -> B, S, 1
        value = torch.mean(value, dim=-1, keepdim=True)

        return value, new_cache_list

class TimeSpaceSudokuModel(nn.Module):
    def __init__(self, config: SudokuModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # 输入编码器
        self.encoder = SudokuValueInputEncoder(config.input_channels, config.embed_dim, device)

        # 核心主干网络
        self.backbone = TimeSpaceChunk(config.backbone_args, device)

        # value 价值头
        self.value_head = ValueHead(config.value_args, device)

        # 共享的旋转位置编码 (headdim 从 block 配置中获取)
        # 假设所有Transformer共享相同的headdim
        # 配置rotary_emb，rotary_emb是根据headdim来的，headdim在Transformer的配置文件中写死为64了
        self.rotary_emb = RotaryEmbedding(self.config.backbone_args.block_args.transformer_args, device=device)

    def forward(
            self,
            state: Tensor,
            cache_list: list[list[Mamba2InferenceCache]] | None = None,
    ) -> tuple[Tensor, list[list[Mamba2InferenceCache]]]:
        B, S, H, W, C = state.shape

        # 分离缓存
        cache_backbone, cache_value = None, None
        if cache_list is not None:
            cache_backbone, cache_value = cache_list[0], cache_list[1]

        # 编码输入
        x_embed = self.encoder(state)  # (B, S, L, D)

        # 通过主干网络
        x_backbone, new_cache_backbone = self.backbone(
            x_embed, H, W, self.rotary_emb, cache_backbone
        )

        # 通过价值头
        value, new_cache_value = self.value_head(
            x_backbone, H, W, self.rotary_emb, cache_value
        )

        # 组合新的缓存
        new_cache_list = [new_cache_backbone, new_cache_value]

        return value, new_cache_list