import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Blocks.TimeSpaceBlock import TimeSpaceBlock
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding


class TimeSpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # TimeSpaceBlocks
        assert self.args.timespaceblock_num > 0
        if self.args.timespaceblock_num > 0:
            self.timespace_blocks = nn.ModuleList(
                [TimeSpaceBlock(args.block_args, device) for _ in range(args.timespaceblock_num)]
            )
        else:
            self.timespace_blocks = None

    def forward(
            self,
            x: Tensor,
            H: int,
            W: int,
            rotary_emb: RotaryEmbedding | None = None,
            cache_list: list[Mamba2InferenceCache] | None = None,
    ) -> tuple[Tensor, list[Mamba2InferenceCache]]:
        # 这里的输入形状为: B, S, L, D
        new_cache_list = []
        # 处理 TimeSpaceBlocks
        if self.timespace_blocks is not None:
            for i, block in enumerate(self.timespace_blocks):
                if cache_list is not None:
                    cache = cache_list[i]
                else:
                    cache = None
                x, cache = checkpoint(block, x, H, W, rotary_emb, cache)
                new_cache_list.append(cache)

        return x, new_cache_list


def test_timespacechunk_consistency(
        batch_size: int = 2,
        seq_len: int = 64,  # 必须是偶数且 > 2
        d_model: int = 128,
        height: int = 8,
        width: int = 8,
        timespaceblock_num: int = 4,  # 测试堆叠的块数
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
        tolerance: float = 1e-5,
) -> None:
    device = torch.device(device_str)
    torch.manual_seed(42)

    print("\n======= Test: TimeSpaceChunk Consistency Check =======")
    print(f"Testing on device: {device}")
    print(f"Params: B={batch_size}, S={seq_len}, D={d_model}, H={height}, W={width}, Blocks={timespaceblock_num}")

    net_cfg = NetworkConfig(
        d_model=d_model,
        max_seq_len=seq_len,
        timespaceblock_num=timespaceblock_num
    )
    model = TimeSpaceChunk(net_cfg, device).to(device).eval()  # 切换到评估模式

    transformer_cfg = net_cfg.block_args.transformer_args
    rotary_emb = RotaryEmbedding(transformer_cfg, device=device)

    L = height * width
    # 输入形状: (B, S, L, D) - 直接是 TimeSpaceChunk 的输入格式
    input_tensor = torch.randn(batch_size, seq_len, L, d_model, device=device)

    # =========================================================================
    # 方案 A：步进模式 (chunk_size = 1，连续喂 seq_len 次)
    # =========================================================================
    print(f"Running scenario A: {seq_len} steps of chunk_size=1...")
    with torch.no_grad():
        cache_a = None
        output_a = []

        for t in range(seq_len):
            # 准备当前时间步的输入，形状 (B, 1, L, D)
            chunk = input_tensor[:, t: t + 1]
            output_chunk, cache_a = model(chunk, height, width, rotary_emb, cache_a)
            output_a.append(output_chunk)

        # 拼接回 (B, S, L, D) 维度
        output_a = torch.cat(output_a, dim=1)

    # =========================================================================
    # 方案 B：分块模式 (chunk_size = 32，连续喂 2 次)
    # =========================================================================
    chunk_size_b = 32
    num_chunks_b = seq_len // chunk_size_b
    print(f"Running scenario B: {num_chunks_b} steps of chunk_size={chunk_size_b}...")
    with torch.no_grad():
        cache_b = None
        output_b = []

        for i in range(num_chunks_b):
            start = i * chunk_size_b
            end = start + chunk_size_b
            # 准备当前块的输入，形状 (B, chunk_size_b, L, D)
            chunk = input_tensor[:, start: end]
            output_chunk, cache_b = model(chunk, height, width, rotary_emb, cache_b)
            output_b.append(output_chunk)

        # 拼接回 (B, S, L, D) 维度
        output_b = torch.cat(output_b, dim=1)

    print("Comparing outputs...")
    # 检查形状是否一致
    assert output_a.shape == output_b.shape, \
        f"Shape mismatch! A: {output_a.shape}, B: {output_b.shape}"

    diff = output_a - output_b
    max_abs_err = diff.abs().max().item()
    mse = torch.mean(diff ** 2).item()

    print(f"Comparison Result ‖ max|Δ| = {max_abs_err:.6e},  MSE = {mse:.6e}")

    if max_abs_err < tolerance:
        print(f"✅ PASSED: Outputs are consistent within tolerance ({tolerance}).")
    else:
        print(f"❌ FAILED: Outputs differ more than tolerance ({tolerance})!")
        mismatch_indices = (diff.abs() > tolerance).nonzero(as_tuple=False)
        if mismatch_indices.numel() > 0:
            first_mismatch = mismatch_indices[0].tolist()
            print(f"  - First mismatch at (B, S, L, D): {first_mismatch}")
            print(f"    - Value from step-by-step: {output_a[tuple(first_mismatch)].item():.8f}")
            print(f"    - Value from chunked:      {output_b[tuple(first_mismatch)].item():.8f}")

    print("====================================================\n")


if __name__ == '__main__':

    test_timespacechunk_consistency(
        batch_size=2,
        seq_len=64,
        d_model=128,
        height=8,
        width=8,
        timespaceblock_num=4,
        tolerance=1e-5
    )
