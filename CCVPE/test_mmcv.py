import torch
from mmcv.ops import MultiScaleDeformableAttention

# Define the module
embed_dims = 256  # Embedding dimension
num_heads = 8
num_levels = 1  # Single level in this example
num_points = 4

attention = MultiScaleDeformableAttention(
    embed_dims=embed_dims,
    num_heads=num_heads,
    num_levels=num_levels,
    num_points=num_points,
    batch_first=True  # Assuming batch dimension comes first
)

# Input tensors
bs = 3  # Batch size
h, w = 512, 512  # Spatial dimensions of the feature map
query = torch.rand(bs, h * w, embed_dims)  # Query reshaped to (batch, num_query, embed_dims)
key_value = torch.rand(bs, h * w, embed_dims)  # Key and Value reshaped similarly

# Spatial shapes and reference points
spatial_shapes = torch.tensor([[h, w]])  # Single level with height and width
reference_points = torch.rand(bs, query.size(1), num_levels, 2)  # Random normalized points


# Forward pass through the attention module
output = attention.forward(
    query=query, 
    key=key_value, 
    value=key_value, 
    reference_points=reference_points, 
    spatial_shapes=spatial_shapes, 
    level_start_index=torch.tensor([0])  # Start index for single level
)

# Output
print("Output shape:", output.shape)
