"""
Naive attention implementation for benchmarking.
This implements a standard scaled dot-product attention without any optimizations.
"""

from imaplib import IMAP4
import math
import torch
from torch import Tensor
from jaxtyping import Float
import einops

from torch._dynamo.exc import unimplemented
import triton
import triton.language as tl


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
        if is_causal is True:
            raise NotImplemented
        BATCH_SIZE, Nq, D = Q.shape
        _, Nk, _ = K.shape
        
        Q_BLOCK_SIZE = 64
        K_BLOCK_SIZE = 64
        
        O = torch.zeros_like(Q)
        L = torch.zeros((BATCH_SIZE, Nq), device=Q.device, dtype=Q.dtype)
        
        for i in range(0, Nq, Q_BLOCK_SIZE):
            q_end = min(i + Q_BLOCK_SIZE, Nq)
            Q_i = Q[:, i:q_end, :]
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros((BATCH_SIZE, q_end - i), device=Q.device, dtype=Q.dtype)
            m_i = - torch.inf * torch.ones((BATCH_SIZE, q_end - i), device=Q.device, dtype=Q.dtype) # Maximum value of each query
            for j in range(0, Nk, K_BLOCK_SIZE):
                k_end = min(j + K_BLOCK_SIZE, Nk)
                K_j = K[:, j:k_end, :]
                V_j = V[:, j:k_end, :]
                S_i = einops.einsum(Q_i, K_j, 'b nq d, b nk d -> b nq nk') / math.sqrt(D)
                m_i_new = torch.max(m_i, torch.max(S_i, dim=-1).values)
                P_i = torch.exp(S_i - m_i_new.unsqueeze(-1))
                l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(P_i, dim = -1)
                O_i_new = O_i * torch.exp(m_i - m_i_new).unsqueeze(-1) + einops.einsum(P_i, V_j, 'b nq nk, b nk d -> b nq d')
                l_i = l_i_new
                m_i = m_i_new
                O_i = O_i_new
            O_i = O_i / l_i.unsqueeze(-1)
            O[:, i:q_end, :] = O_i
            L[:, i:q_end] = m_i + torch.log(l_i)
            
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        BATCH_SIZE, Nq, dim = Q.shape
        _, Nk, _ = K.shape
        
        # Step 1: Compute D vector (Equation 13, line 1)
        # D = rowsum(O ⊙ dO) = sum along last dimension
        D = (O * dO).sum(dim=-1)  # Shape: [batch, Nq]
        
        # Initialize gradients
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Use block sizes for tiled computation
        Q_BLOCK_SIZE = 64
        K_BLOCK_SIZE = 64
        
        for i in range(0, Nq, Q_BLOCK_SIZE):
            q_end = min(i + Q_BLOCK_SIZE, Nq)
            Q_i = Q[:, i:q_end, :]      # [batch, Bq, d]
            O_i = O[:, i:q_end, :]      # [batch, Bq, d]
            dO_i = dO[:, i:q_end, :]    # [batch, Bq, d]
            L_i = L[:, i:q_end]         # [batch, Bq]
            D_i = D[:, i:q_end]         # [batch, Bq]
            
            dQ_i = torch.zeros_like(Q_i)

            for j in range(0, Nk, K_BLOCK_SIZE):
                k_end = min(j + K_BLOCK_SIZE, Nk)
                K_j = K[:, j:k_end, :]  # [batch, Bk, d]
                V_j = V[:, j:k_end, :]  # [batch, Bk, d]
                
                # Equation 13, line 2: S = Q K^T / sqrt(d)
                S_ij = torch.einsum('bqd,bkd->bqk', Q_i, K_j) / math.sqrt(dim)
                
                # Equation 14: P_ij = exp(S_ij - L_i)
                # L_i is the log-sum-exp, so this recomputes P without storing it
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))
                
                # Apply causal mask if needed
                if is_causal:
                    # Create causal mask for this block
                    mask = torch.arange(i, q_end, device=Q.device).unsqueeze(-1) >= \
                        torch.arange(j, k_end, device=Q.device).unsqueeze(0)
                    P_ij = P_ij * mask.unsqueeze(0)
                    
                # Equation 15: dV = P^T dO
                dV[:, j:k_end, :] += torch.einsum('bqk,bqd->bkd', P_ij, dO_i)
                
                # Equation 16: dP = dO V^T
                dP_ij = torch.einsum('bqd,bkd->bqk', dO_i, V_j)
                
                # Equation 17: dS_ij = P_ij ⊙ (dP_ij - D_i)
                # This is the key step where D is used!
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))
                
                # Equation 18: dQ = dS K / sqrt(d)
                dQ_i += torch.einsum('bqk,bkd->bqd', dS_ij, K_j) / math.sqrt(dim)
                
                # Equation 19: dK = dS^T Q / sqrt(d)
                dK[:, j:k_end, :] += torch.einsum('bqk,bqd->bkd', dS_ij, Q_i) / math.sqrt(dim)
            
            dQ[:, i:q_end, :] = dQ_i
        
        return dQ, dK, dV, None  # None for is_causal (non-differentiable)
        
    
class FlashAttentionTriton(torch.autograd.Function):
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
        _, Nk, _ = K.shape
        
        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((BATCH_SIZE, Nq), device=Q.device, dtype=Q.dtype)
        
        # Define tile sizes
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        
        grid = (triton.cdiv(Nq, Q_TILE_SIZE), BATCH_SIZE)
        
        # Launch the Triton kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        # You'll need to implement backward pass with Triton kernel too
        # For now, can return None or raise NotImplementedError
        raise NotImplementedError("Backward pass not yet implemented")
        

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Launch grid: (Tq, batch_size)
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),              # 1D tensor: just queries dimension
        strides=(stride_lq,),            
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),      
        order=(0,),
    )
    
    m_i = tl.full([Q_TILE_SIZE], value=float('-inf'), dtype=tl.float32) # Q_TILE_SIZE
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    Q_tile_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # Q_TILE_SIZE D
    
    q_start_index = query_tile_index * Q_TILE_SIZE
    q_indices = q_start_index + tl.arange(0, Q_TILE_SIZE) # Q_TILE_SIZE
    
    # The parallel is for the Nq//Q_TILE_SIZE dim 0 and the BATCH_SIZE dim 1
    # For each query dimension it is actually independent so that we can parallel them
    # Loop over remaining tiles in the D dimension
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # K_TILE_SIZE D
        V_tile_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # K_TILE_SIZE D
        S_ij = tl.dot(Q_tile_i, tl.trans(K_tile_j))  # Q_TILE_SIZE K_TILE_SIZE
        S_ij = S_ij.to(tl.float32) * scale  # Ensure FP32 for scores
        
        if is_causal:
            k_start_index = j * K_TILE_SIZE
            k_indices = k_start_index + tl.arange(0, K_TILE_SIZE) # K_TILE_SIZE
            
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, float("-inf"))
        
        
        # Update running max
        m_ij = tl.max(S_ij, axis=-1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute probabilities
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        
        # Update running sum
        l_ij = tl.sum(P_ij, axis=-1)
        correction = tl.exp(m_i - m_i_new)
        l_i_new = correction * l_i + l_ij
        
        # Update output: cast P to V's dtype, use acc argument
        P_ij_cast = P_ij.to(V_tile_j.dtype)  # Cast P to match V's dtype
        O_i_scaled = correction[:, None] * O_i
        O_i_new = tl.dot(P_ij_cast, V_tile_j, acc=O_i_scaled)  # Use acc argument
        
        l_i = l_i_new
        m_i = m_i_new
        O_i = O_i_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))  # Move K_TILE_SIZE rows down
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))  # Move K_TILE_SIZE rows down
    
    O_i = O_i / l_i[:, None]
    
    # Cast O to output dtype before storing
    output_dtype = O_block_ptr.type.element_ty
    O_i = O_i.to(output_dtype)
    
    # Compute log-sum-exp
    L_i = m_i + tl.log(l_i)
    
    # Cast L to output dtype if needed
    L_dtype = L_block_ptr.type.element_ty
    L_i = L_i.to(L_dtype)
    
    # Store results
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))
    
    return