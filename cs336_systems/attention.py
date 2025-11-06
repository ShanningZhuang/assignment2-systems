"""
Naive attention implementation for benchmarking.
This implements a standard scaled dot-product attention without any optimizations.
"""

import math
import torch
from torch import Tensor
from jaxtyping import Float
import einops


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

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        '''
        Forward pass of FlashAttention.
        Q ... Nq d
        K ... Nk d
        V ... Nk d
        is_causal bool

        Returns:
        O Nq d
        L Nq
        '''
        ctx.is_causal = is_causal
        BATCH_SIZE, Nq, D = Q.shape
        BLOCK_SIZE = 64
        
        O = torch.zeros_like(Q)
        # L = torch.zeros(BATCH_SIZE, Nq, device=Q.device, dtype=Q.dtype)
        for i in range(0, Nq, BLOCK_SIZE):
            q_end = min(i + BLOCK_SIZE, Nq)
            Q_i = Q[:, i:q_end, :]
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros((BATCH_SIZE, q_end - i), device=Q.device, dtype=Q.dtype)
            m_i = - torch.inf * torch.ones((BATCH_SIZE, q_end - i), device=Q.device, dtype=Q.dtype)
            for j in range(0,Nq,BLOCK_SIZE):
                k_end = min(j + BLOCK_SIZE, Nq)
                K_j = K[:, j:k_end, :]
                V_j = V[:, j:k_end, :]
                S_i = einops.einsum(Q_i, K_j, 'b nq d, b nk d -> b nq nk') / math.sqrt(D) # (B, q_end - i, k_end - j)
                m_i_new = torch.max(m_i, torch.max(S_i, dim=-1).values) # (B, q_end - i)
                P_i = torch.exp(S_i - m_i_new.unsqueeze(-1))
                l_i_new = torch.exp(m_i-m_i_new) * l_i + torch.sum(P_i, dim=-1)
                O_i_new = O_i * torch.exp(m_i-m_i_new).unsqueeze(-1) + einops.einsum(P_i, V_j, 'b nq nk, b nk d -> b nq d')
                l_i = l_i_new
                m_i = m_i_new
                O_i = O_i_new
            O_i = O_i / l_i.unsqueeze(-1)
            O[:, i:q_end, :] = O_i
            # L[:, i:q_end] = l_i
        return O
            
        

    @staticmethod
    def backward(ctx, dO):
        pass