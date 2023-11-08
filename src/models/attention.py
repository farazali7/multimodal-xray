import torch.nn as nn
import torch
from einops import repeat

from functools import partial

from src.utils.constants import EPS


def orthogonal_matrix_chunk(size, device=None) -> torch.Tensor:
    """Create a square orthogonal matrix via QR-decomposition.

    Args:
        size: Number of rows & cols
        device: Device to place resulting matrix on (ex. 'cpu')

    Returns:
        An size x size orthogonal matrix, Q, from the QR-decomposition
    """
    random_block = torch.randn((size, size), device=device)
    q, _ = torch.linalg.qr(random_block.cpu(), mode='reduced')

    return q.to(device).t()


def gaussian_orthogonal_random_matrix(n_rows, n_cols, device=None) -> torch.Tensor:
    """Create matrix of values drawn from a gaussian random distribution with entire matrix being orthogonal.

    Args:
        n_rows: Number of rows (features)
        n_cols: Number of cols (dimensionality of each feature vector)
        device: Device to place resulting matrix on (ex. 'cpu')

    Returns:
        An n_rows x n_cols orthogonal matrix of gaussian random values.
    """
    # To ascertain an orthogonal matrix, the QR-decomposition algorithm can be employed
    # This algorithm decomposes a matrix A (M x d), into matrices Q (M x M) and R (M x d),
    # where Q is orthogonal and R is upper-triangular.
    # Since it only works on square matrices, the desired orf must be made in square chunks.
    n_full_blocks = int(n_rows/n_cols)

    blocks = []

    # For each square blocks that fits inside desired matrix, perform QR decomposition and store the Q
    for _ in range(n_full_blocks):
        q = orthogonal_matrix_chunk(n_cols, device=device)
        blocks.append(q)

    # If there are some rows left over that could not be evenly divided into desired size, retrieve
    # same amount of rows from a final QR decomposition and add to end
    remaining_rows = n_rows - (n_full_blocks * n_cols)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(n_cols, device=device)
        blocks.append(q[:remaining_rows])

    # [n_rows x n_cols]
    final_matrix = torch.cat(blocks)

    # Rows drawn from normal distribution have norm that follows a chi-distribution but get unit
    # norm after QR decomposition, so need to be rescaled. This is done by left-multiplying with
    # the sum of squared normal random vectors which also follow a chi-distribution.
    multiplier = torch.diag(torch.randn((n_rows, n_cols), device=device).norm(dim=1))

    return multiplier @ final_matrix


def softmax_kernel(data, projection_matrix, is_query) -> torch.Tensor:
    """Approximate the SoftMax kernel through the kernel trick using the definition for the kernel
    provided in the Performer paper: https://arxiv.org/pdf/2009.14794v4.pdf.

    Args:
        data: Original data tensor across attention heads with shape [B, H, T, Dh]
        projection_matrix: Projection weights tensor with shape [n_features, Dh]
        is_query: Boolean for whether data represents the query in following attention operation

    Returns:
        Transformed data tensor that represents values that can be linearly multiplied against
        to approximate SoftMax operation with another tensor.
    """
    b, h, *_ = data.shape

    # The original qk_T/sqrt(d) denominator becomes a scale factor of sqrt(sqrt(d))
    # when you decompose q and k into separate arguments
    data_scale = data.shape[-1] ** -0.25

    # This scales the kernel
    scale = projection_matrix.shape[0] ** -0.5

    # As we want to do batched matmul we must extend the projection matrix to have batch and heads dims
    projection = repeat(projection_matrix, 't d -> b h t d', b=b, h=h).type_as(data)

    # Project data via an outer product
    data_projected = torch.einsum('...id,...jd -> ...ij', (data_scale * data), projection)

    # Compute the norm of the data: [B, H, T, Dh=1]
    data_norm = ((torch.sum(data ** 2, dim=-1) / 2.0) * (data_scale ** 2)).unsqueeze(dim=-1)

    # Determine idx of max value across all values for each sequence if this is a query
    # otherwise find idx of max value across all sequences for each head
    argmax_dim = (-1, -2) if is_query else -1

    # Compute feature vector as described in Performer paper
    # The following also uses a variant of the logsumexp trick to induce stability by subtracting max value
    # inside the exp function
    data_projected = scale * (
        torch.exp(data_projected - data_norm - torch.amax(data_projected, dim=argmax_dim, keepdim=True).detach()
                  + EPS)
    )

    return data_projected.type_as(data)


class FastAttentionBlock(nn.Module):
    def __init__(self, dim_heads, n_features):
        """Implements multi-headed approximation of soft-max attention from Performer
        paper: https://arxiv.org/pdf/2009.14794v4.pdf
        and
        repo: https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py#L179

        Note that this module assumes the input is already split for each head, i.e., input shape to this layer
        is (B, H, T, Dh) where B is batch size, H is number of heads, T is seq len, and Dh is dim_heads.

        Args:
            dim_heads: Dimensionality of each head
            n_features: Number of orthogonal features to use in approximation (more features -> more accurate)
        """
        super(FastAttentionBlock, self).__init__()

        self.dim_heads = dim_heads
        self.n_features = n_features

        # Create matrix of randomly sampled, orthogonal feature vectors
        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         n_rows=self.n_features, n_cols=self.dim_heads)
        projection_matrix = self.create_projection()

        # This registers a 'buffer' for the model, which is nothing but a parameter that is serialized along
        # with the model and stored in the state_dict. Buffers aren't listed in model.parameters() and thus, are
        # not updated with gradients during training.
        self.register_buffer('projection_matrix', projection_matrix)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_ = projections
        del projections

    def forward(self, q, k, v):
        # Create softmax approximation kernel
        create_sm_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix)

        # Generate query and key projections in softmax kernel subspace
        q = create_sm_kernel(q, is_query=True)
        k = create_sm_kernel(k, is_query=False)

        # Perform normal, non-causal attention
        # Since the above q and k are now linear in their matmul for approximating the softmax scores,
        # we can choose to matmul our keys with values first to lower dim from sequence len n
        # to dimensionality d x e
        # Inner product of keys and values
        context = torch.einsum('...nd,...ne -> ...de', k, v)

        # Compute norm factor for softmax
        D_inv = 1. / torch.einsum('...nd,...d -> ...n', q, k.sum(dim=-2).type_as(q))

        # Outer product of queries and values with normalization also done
        out = torch.einsum('...de,...nd,...n -> ...ne', context, q, D_inv)

        return out


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        """ Multi-headed attention block

        Args:
            embed_dim: Dimensionality of input sequence
            n_heads: Number of parallel heads in attention block
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=n_heads)

    def forward(self, q, k, v):
        mha_out = self.mha(q, k, v)[0]  # Returns (output, weights)

        return mha_out
