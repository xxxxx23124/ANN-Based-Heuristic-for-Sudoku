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
