import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

from ANN.Networks.TimeSpaceChunk import TimeSpaceChunk
from ANN.Networks.SudokuModel.SudokuModelConfig import SudokuModelConfig


class SudokuInputEncoder(nn.Module):
    """负责将 (B, S, H, W, C) 的输入编码为 (B, S, L, D) 的特征"""

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

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, S, H, W, C)
        B, S, H, W, C = x.shape
        x = rearrange(x, "b s h w c -> (b s) c h w")
        x = F.gelu(self.conv(x))
        x = rearrange(x, "(b s) d h w -> b s (h w) d", s=S)
        # return: (B, S, L, D)
        return x


class ActorHead(nn.Module):
    """策略头：TimeSpace -> 输出 -> Reshape"""

    def __init__(self, config: NetworkConfig, grid_size: int, device: torch.device):
        super().__init__()
        self.timespace_chunk = TimeSpaceChunk(config, device=device)
        self.output_head = nn.Linear(config.d_model, grid_size, device=device)

    def forward(
            self,
            x: Tensor,
            H: int,
            W: int,
            rotary_emb: RotaryEmbedding,
            cache_list: list[Mamba2InferenceCache] | None = None
    ) -> tuple[Tensor, list[Mamba2InferenceCache]]:
        action_logits, new_cache_list = self.timespace_chunk(
            x, H, W, rotary_emb, cache_list
        )

        # B, S, L, D -> B, S, L, grid_size
        action_logits = self.output_head(action_logits)
        # B, S, L, grid_size -> B, S, (L * grid_size)
        action_logits = rearrange(action_logits, "b s l d -> b s (l d)")

        return action_logits, new_cache_list


class CriticHead(nn.Module):
    """价值头：TimeSpace -> 输出"""

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
        # B, S, L, D_low -> B, S, L
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
        self.encoder = SudokuInputEncoder(config.input_channels, config.embed_dim, device)

        # 核心主干网络
        self.backbone = TimeSpaceChunk(config.backbone_args, device)

        # 策略头
        self.actor_head = ActorHead(config.actor_args, config.grid_size, device)

        # 价值头
        self.critic_head = CriticHead(config.critic_args, device)

        # 共享的旋转位置编码 (headdim 从 block 配置中获取)
        # 假设所有Transformer共享相同的headdim
        # 配置rotary_emb，rotary_emb是根据headdim来的，headdim在Transformer的配置文件中写死为64了
        self.rotary_emb = RotaryEmbedding(self.config.backbone_args.block_args.transformer_args, device=device)

    def forward(
            self,
            x: Tensor,
            cache_list: list[list[Mamba2InferenceCache]] | None = None,
    ) -> tuple[Tensor, Tensor, list[list[Mamba2InferenceCache]]]:
        B, S, H, W, C = x.shape

        # 分离缓存
        cache_backbone, cache_actor, cache_critic = None, None, None
        if cache_list is not None:
            cache_backbone, cache_actor, cache_critic = cache_list[0], cache_list[1], cache_list[2]

        # 编码输入
        x_embed = self.encoder(x)  # (B, S, L, D)

        # 通过主干网络
        x_backbone, new_cache_backbone = self.backbone(
            x_embed, H, W, self.rotary_emb, cache_backbone
        )

        # 通过策略头
        action_logits, new_cache_actor = self.actor_head(
            x_backbone, H, W, self.rotary_emb, cache_actor
        )

        # 通过价值头
        value, new_cache_critic = self.critic_head(
            x_backbone, H, W, self.rotary_emb, cache_critic
        )

        # 组合新的缓存
        new_cache_list = [new_cache_backbone, new_cache_actor, new_cache_critic]

        return action_logits, value, new_cache_list


def print_model_parameters(model: nn.Module, verbose: bool = True):
    """
    打印模型的详细参数信息，并返回参数总数。

    Args:
        model (nn.Module): PyTorch 模型。
        verbose (bool): 是否打印每一层的详细信息。
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    if verbose:
        print(f"{'=' * 130}")
        print(f"{'Parameter Name':<30} | {'Shape':<20} | {'# Params':<12} | {'Trainable'}")
        print(f"{'-' * 130}")

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        is_trainable = "Yes" if param.requires_grad else "No"
        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params

        if verbose:
            print(f"{name:>80} | {str(list(param.shape)):>16} | {num_params:>16,} | {is_trainable}")

    print(f"{'=' * 130}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练参数数量: {non_trainable_params:,}")
    print(f"{'=' * 130}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = SudokuModelConfig(grid_size=9, input_channels=10)
    actor_critic = TimeSpaceSudokuModel(model_config, device=device)
    print_model_parameters(model=actor_critic)
