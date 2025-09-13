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


class TimeSpaceSudokuActorModel(nn.Module):
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

        # 共享的旋转位置编码 (headdim 从 block 配置中获取)
        # 假设所有Transformer共享相同的headdim
        # 配置rotary_emb，rotary_emb是根据headdim来的，headdim在Transformer的配置文件中写死为64了
        self.rotary_emb = RotaryEmbedding(self.config.backbone_args.block_args.transformer_args, device=device)

    def forward(
            self,
            x: Tensor,
            cache_list: list[list[Mamba2InferenceCache]] | None = None,
    ) -> tuple[Tensor, list[list[Mamba2InferenceCache]]]:
        B, S, H, W, C = x.shape

        # 分离缓存
        cache_backbone, cache_actor = None, None
        if cache_list is not None:
            cache_backbone, cache_actor = cache_list[0], cache_list[1]

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

        # 组合新的缓存
        new_cache_list = [new_cache_backbone, new_cache_actor]

        return action_logits, new_cache_list

class TimeSpaceSudokuCriticModel(nn.Module):
    def __init__(self, config: SudokuModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # 输入编码器
        self.encoder = SudokuInputEncoder(config.input_channels, config.embed_dim, device)

        # 核心主干网络
        self.backbone = TimeSpaceChunk(config.backbone_args, device)

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
    ) -> tuple[Tensor, list[list[Mamba2InferenceCache]]]:
        B, S, H, W, C = x.shape

        # 分离缓存
        cache_backbone, cache_critic = None, None
        if cache_list is not None:
            cache_backbone, cache_critic = cache_list[0], cache_list[1]

        # 编码输入
        x_embed = self.encoder(x)  # (B, S, L, D)

        # 通过主干网络
        x_backbone, new_cache_backbone = self.backbone(
            x_embed, H, W, self.rotary_emb, cache_backbone
        )

        # 通过价值头
        value, new_cache_critic = self.critic_head(
            x_backbone, H, W, self.rotary_emb, cache_critic
        )

        # 组合新的缓存
        new_cache_list = [new_cache_backbone, new_cache_critic]

        return value, new_cache_list

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


def test_chunking_consistency(
        model: TimeSpaceSudokuModel,
        device: torch.device,
        sequence_length: int,
        chunk_size: int,
        model_config: SudokuModelConfig
):
    """
    测试模型在分块推理和完整推理下的一致性。

    Args:
        model (TimeSpaceSudokuModel): 要测试的模型实例。
        device (torch.device): 计算设备。
        sequence_length (int): 总序列长度。
        chunk_size (int): 每个块的大小。
        model_config (SudokuModelConfig): 模型配置。
    """
    print(f"\n--- Testing with Sequence Length={sequence_length}, Chunk Size={chunk_size} ---")

    # 确保模型处于评估模式
    model.eval()

    # 1. 创建随机输入数据
    B = 1  # 批次大小固定为1进行测试
    H = W = model_config.grid_size
    C = model_config.input_channels
    full_input = torch.randn(B, sequence_length, H, W, C, device=device)

    # 2. 完整推理 (Full Pass)
    with torch.no_grad():
        logits_full, value_full, _ = model(full_input, cache_list=None)

    # 3. 分块推理 (Chunked Pass)
    logits_chunks = []
    value_chunks = []
    chunked_cache = None  # 初始缓存为空

    with torch.no_grad():
        for i in range(0, sequence_length, chunk_size):
            # 获取当前块的输入
            input_chunk = full_input[:, i:i + chunk_size, ...]

            # 使用上一个块的缓存进行推理
            logits_chunk, value_chunk, chunked_cache = model(input_chunk, cache_list=chunked_cache)

            logits_chunks.append(logits_chunk)
            value_chunks.append(value_chunk)

    # 将所有块的输出拼接起来
    logits_chunked_cat = torch.cat(logits_chunks, dim=1)
    value_chunked_cat = torch.cat(value_chunks, dim=1)

    # 4. 比较结果
    # 使用 torch.allclose 来比较浮点张量，它允许有微小的容差
    are_logits_close = torch.allclose(logits_full, logits_chunked_cat, atol=1e-5)
    are_values_close = torch.allclose(value_full, value_chunked_cat, atol=1e-5)

    print(f"Logits consistency check: {'PASSED' if are_logits_close else 'FAILED'}")
    print(f"Value consistency check:  {'PASSED' if are_values_close else 'FAILED'}")

    if not are_logits_close:
        diff = torch.max(torch.abs(logits_full - logits_chunked_cat))
        print(f"  Max absolute difference in logits: {diff.item()}")
    if not are_values_close:
        diff = torch.max(torch.abs(value_full - value_chunked_cat))
        print(f"  Max absolute difference in values: {diff.item()}")

    return are_logits_close and are_values_close

def test():
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = SudokuModelConfig(grid_size=9, input_channels=10)
    actor_critic = TimeSpaceSudokuModel(model_config, device=device)

    # 打印模型参数
    print_model_parameters(model=actor_critic)

    # =========================================================
    # ============= 开始进行分块推理一致性测试 =================
    # =========================================================

    total_sequence_length = 64
    chunk_sizes_to_test = [64, 32, 16, 8, 4, 1]

    all_tests_passed = True
    for size in chunk_sizes_to_test:
        if total_sequence_length % size != 0:
            print(f"Skipping chunk size {size} as it doesn't evenly divide {total_sequence_length}")
            continue

        passed = test_chunking_consistency(
            model=actor_critic,
            device=device,
            sequence_length=total_sequence_length,
            chunk_size=size,
            model_config=model_config
        )
        if not passed:
            all_tests_passed = False

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All chunking consistency tests passed successfully!")
    else:
        print("❌ Some chunking consistency tests failed.")
    print("=" * 50)
