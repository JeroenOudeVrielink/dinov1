from attn_pooling import ATTNPOOL_BASE
import torch

attn_pool = ATTNPOOL_BASE(
    in_channels=2048, patch_size=1, dim_reduced_ratio=1, num_heads=2
)

x = torch.rand(1, 2048, 8, 8)

output = attn_pool(x)
print(output.size())
