import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Layers.Norm_layer.RMSNorm import RMSNorm
from ANN.Layers.FeedForward_layer.SwiGLUMlp import SwiGLUFeedForward
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding
from ANN.Layers.FeedForward_layer.FeedForwardConfig import FeedForwardConfig

from ANN.Networks.TimeSpaceChunk import TimeSpaceChunk
from ANN.Networks.SpaceChunk import SpaceChunk
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
    """价值头：降维 -> TimeSpace -> SpaceFusion -> 输出"""

    def __init__(self, config: NetworkConfig, projection_config: FeedForwardConfig, device: torch.device):
        super().__init__()
        self.config = config

        # 初始降维投影
        self.projection = nn.Sequential(
            RMSNorm(projection_config.d_model, device=device),
            SwiGLUFeedForward(projection_config, device=device),
        )

        # 时空处理块
        self.timespace_chunk = TimeSpaceChunk(config, device=device)

        # 空间融合块
        self.space_chunk = SpaceChunk(config, device=device)

        # 输出层
        self.output_head = nn.Linear(config.d_model, 1, device=device)

    def forward(
            self,
            x: Tensor,
            H: int, W: int,
            rotary_emb: RotaryEmbedding,
            cache_list: list[Mamba2InferenceCache] | None = None
    ) -> tuple[Tensor, list[Mamba2InferenceCache]]:
        # 降维
        x_projected = self.projection(x)

        # 时空处理
        value, new_cache_list = self.timespace_chunk(
            x_projected, H, W, rotary_emb, cache_list
        )

        # 空间融合
        value = self.space_chunk(value, H, W, rotary_emb)

        # 聚合与输出
        # B, S, L, D_low -> B, S, L
        value = self.output_head(value).squeeze(-1)
        # B, S, L -> B, S, 1
        value = torch.mean(value, dim=-1, keepdim=True)

        return value, new_cache_list


# 假设上面的 SudokuModelConfig 和重构后的模块都已定义

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
        self.critic_head = CriticHead(config.critic_args, config.critic_projection_args, device)

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


def print_model_parameters_by_module(model):
    """
    按模块名称和层级打印参数数量。
    """
    print("--- Model Parameters by Module ---")
    total_params = 0
    for name, module in model.named_modules():
        # 我们只关心那些直接包含参数的模块 (Linear, Conv, LayerNorm, etc.)
        # 并且避免重复计算父模块的参数
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:<60} | Parameters: {params:,}")
                total_params += params

    # 手动计算可能遗漏的参数（如自定义层的参数）
    # 更稳妥的方式是直接迭代 named_parameters
    print("\n--- Parameters by Named Parameter ---")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name:<60} | Shape: {str(list(param.shape)):<20} | Count: {param.numel():,}")
        total_params += param.numel()

    print("-" * 80)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 80)


def print_model_summary(model, model_name="Model"):
    """
    打印模型的参数总量和可训练参数总量。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"--- {model_name} Summary ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * (len(model_name) + 14))


def test():
    for _ in range(1):
        """
        测试函数，修改为处理长序列（150），并使用分块、缓存和梯度累积。
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA is not available. Testing on CPU. Memory usage will not be representative.")
            return

        H, W = 9, 9
        BATCH_SIZE = 1
        TOTAL_SEQUENCE_LENGTH = 150  # 总序列长度
        INPUT_CHANNELS = 10

        print(f"--- Model Test for Long Sequence Processing ---")
        print(f"Device: {device}")
        print(f"Board size: {H}x{W}")
        print(f"Batch size: {BATCH_SIZE}, Total sequence length: {TOTAL_SEQUENCE_LENGTH}")

        # 1. 初始化模型
        try:
            config = SudokuModelConfig(grid_size=H,
                              input_channels=INPUT_CHANNELS)
            model = TimeSpaceSudokuModel(config, device)
            model.to(device)
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error during model initialization: {e}")
            return

        # 2. 创建完整的模拟输入和目标
        # 输入形状: (B, S_total, H, W, C)
        full_input_tensor = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, H, W, INPUT_CHANNELS, device=device)
        # 目标策略形状: (B, S_total, 729)
        full_target_policy = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, 729, device=device)
        # 目标价值形状: (B, S_total, 1)
        full_target_value = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, 1, device=device)

        print(f"Full input tensor shape: {full_input_tensor.shape}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        # 3. 运行前向和反向传播，并监控显存
        # 清空缓存以获得准确的初始显存读数
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
        initial_memory = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"\nInitial CUDA memory allocated: {initial_memory:.2f} MB")

        try:
            # --- 梯度累积和分块处理 ---
            optimizer.zero_grad()  # 在循环开始前清零梯度

            cache_list = None  # 初始化缓存为空
            total_loss_accumulated = 0.0

            print("\nStarting chunk processing...")
            processed_len = 0
            while processed_len < TOTAL_SEQUENCE_LENGTH:
                remaining_len = TOTAL_SEQUENCE_LENGTH - processed_len

                # 决定当前块的大小
                if remaining_len >= 64:
                    chunk_size = 64
                elif remaining_len >= 32:
                    chunk_size = 32
                elif remaining_len >= 16:
                    chunk_size = 16
                elif remaining_len >= 8:
                    chunk_size = 8
                elif remaining_len >= 4:
                    chunk_size = 4
                else:
                    chunk_size = 1  # 如果连4都不到，就使用1来处理剩余部分

                # 获取当前块的数据
                chunk_input = full_input_tensor[:, processed_len: processed_len + chunk_size]
                chunk_target_policy = full_target_policy[:, processed_len: processed_len + chunk_size]
                chunk_target_value = full_target_value[:, processed_len: processed_len + chunk_size]

                print(
                    f"  - Processing chunk: index {processed_len} to {processed_len + chunk_size - 1} (size: {chunk_size})")

                # --- 前向传播 ---
                # 使用 checkpoint 来节省显存
                # 将上一个块的 cache 传入
                policy_pred, value_pred, cache_list = model(chunk_input, cache_list)

                # --- 计算损失 ---
                policy_loss = loss_fn(policy_pred, chunk_target_policy)
                value_loss = loss_fn(value_pred, chunk_target_value)
                # 为了防止梯度累积时因计算图释放导致损失过大，可以适当缩放
                chunk_loss = (policy_loss + value_loss) * (chunk_size / TOTAL_SEQUENCE_LENGTH)

                total_loss_accumulated += chunk_loss.item() * (TOTAL_SEQUENCE_LENGTH / chunk_size)  # 累加回未缩放的损失值

                # --- 反向传播 (梯度累积) ---
                # 每个块都进行反向传播，梯度会累加到 .grad 属性上
                chunk_loss.backward()

                processed_len += chunk_size

            print("All chunks processed successfully.")

            # --- 优化器步骤 ---
            # 在所有块处理完毕后，进行一次参数更新
            print("Running optimizer step...")
            optimizer.step()
            print("Optimizer step successful.")

            # --- 显存和损失总结 ---
            final_memory = torch.cuda.memory_allocated() / 1024 ** 2
            peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # 整个过程中的峰值

            print(f"\n--- Run Summary ---")
            print(f"Final accumulated loss: {total_loss_accumulated:.4f}")
            print(f"Memory after optimizer step: {final_memory:.2f} MB")
            print(f"Peak CUDA memory allocated during the entire process: {peak_memory:.2f} MB")

        except Exception as e:
            import traceback
            print(f"\nAn error occurred during the test run: {e}")
            traceback.print_exc()

    # 打印模型参数信息
    print_model_summary(model, "TimeSpaceChessModel")


if __name__ == '__main__':
    test()