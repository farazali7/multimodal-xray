import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, n_heads: int):
        """cross-attention block

        Args:
            text_dim: Dimensionality of text embedding
            image_dim: Dimensionality of image embedding
            n_heads: Number of parallel heads in attention block
        """
        super(CrossAttentionBlock, self).__init__()
        # Make the text embeddings the same size of the img embeddings to be able to compare
        self.text_projection = nn.Linear(text_dim, image_dim)
        self.cross_attention = nnd  .MultiheadAttention(embed_dim=image_dim, num_heads=n_heads)

    def forward(self, text_features, image_keys, image_values):
        queries = self.text_projection(text_features)
        # text for queries, image for keys and values
        output, weights = self.cross_attention(query=queries, key=image_keys, value=image_values)
        return output