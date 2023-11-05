import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        """ Feed forward network (2-layer dense network).

        Args:
            embed_dim: Dimensionality of input sequence
            hidden_dim: Dimensionality of hidden layers in feed forward network
            dropout: Amount of dropout in feed forward network
        """
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        """ Multi-headed self-attention block

        Args:
            embed_dim: Dimensionality of input sequence
            n_heads: Number of parallel heads in attention block
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=n_heads)

    def forward(self, x):
        mha_out = self.mha(x, x, x)[0]  # Returns (output, weights)

        return mha_out


class ViTEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        """ Vision Transformer Encoder layer.

        Args:
            embed_dim: Dimensionality of input sequence
            hidden_dim: Dimensionality of hidden layers in feed forward network
            n_heads: Number of parallel heads in attention block
            dropout: Amount of dropout in feed forward network
        """
        super(ViTEncoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttentionBlock(embed_dim, n_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        x = self.layer_norm_1(x)
        mha_out = self.mha(x)
        x = mha_out + x  # Skip connection

        x = self.layer_norm_2(x)
        ffn_out = self.ffn(x)
        x = ffn_out + x  # Skip connection

        return x


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        """ Vision Transformer Encoder which returns embeddings (no cls output).

        Implements the original Vision Transformer encoder but ignores the class token to only
        output learned embeddings. Output pooling is left up to user.

        Args:
            embed_dim: Dimensionality of input sequence
            hidden_dim: Dimensionality of hidden layers in feed forward network
            n_heads: Number of heads for multi-head attention
            n_layers: Number of stacked attention layers
            dropout: Amount of dropout for all feed forward networks and on input
        """
        super(ViTEncoder, self).__init__()

        self.enc_layers = nn.Sequential(
            *[ViTEncoderLayer(embed_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expects input 'x' is already tokenized & embedded
        # x.shape == [B, T, H, W, C], where T is seq len
        x = self.dropout(x)
        x = self.enc_layers(x)

        return x
