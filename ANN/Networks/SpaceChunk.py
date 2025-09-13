import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Blocks.SpatialFusion_block import SpatialFusion_block
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

class SpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # SpaceBlocks
        assert self.args.spatialfusion_block_num is not None
        if self.args.spatialfusion_block_num is not None:
            self.space_blocks = nn.ModuleList(
                [SpatialFusion_block(args.block_args.transformer_args, device) for _ in range(args.spatialfusion_block_num)]
            )
        else:
            self.space_blocks = None

    def forward(
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: RotaryEmbedding | None = None,
            ) -> Tensor:
        # 这里的输入形状为: B, S, L, D_low
        # 输出形状为: B, S, L, D_low
        # 处理空间讯息
        if self.space_blocks is not None:
            for i, block in enumerate(self.space_blocks):
                x = checkpoint(block, x, H, W, rotary_emb)
        return x