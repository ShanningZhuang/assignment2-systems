"""
Naive attention implementation for benchmarking.
This implements a standard scaled dot-product attention without any optimizations.
"""

import math
import torch
from torch import Tensor
from jaxtyping import Float


def naive_attention(
    Q: Float[Tensor, "batch seq_len d_model"],
    K: Float[Tensor, "batch seq_len d_model"],
    V: Float[Tensor, "batch seq_len d_model"],
) -> Float[Tensor, "batch seq_len d_model"]:
    """
    Naive implementation of scaled dot-product attention.

    This is the standard attention mechanism from "Attention is All You Need":
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q: Query tensor of shape (batch, seq_len, d_model)
        K: Key tensor of shape (batch, seq_len, d_model)
        V: Value tensor of shape (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    # Get the dimensionality for scaling
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Shape: (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply softmax to get attention weights
    # Shape: (batch, seq_len, seq_len)
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention weights to values
    # Shape: (batch, seq_len, d_model)
    output = torch.matmul(attn_weights, V)

    return output
