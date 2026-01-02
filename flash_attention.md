The goal of flash attention

Calculate:

forward:
S = Q K^T
P = softmax(s)
O = PV

Backward:

Here are the equations in markdown format:

$$dV = P^T dO \in \mathbb{R}^{N \times d}$$

$$dP = dOV^T \in \mathbb{R}^{N \times N}$$

$$dS = \text{dsoftmax}(dP) \in \mathbb{R}^{N \times N}$$

$$dQ = dSK \in \mathbb{R}^{N \times d}$$

$$dK = QdS^T \in \mathbb{R}^{N \times d}$$

Or if you prefer inline format:

$dV = P^T dO \in \mathbb{R}^{N \times d}$

$dP = dOV^T \in \mathbb{R}^{N \times N}$

$dS = \text{dsoftmax}(dP) \in \mathbb{R}^{N \times N}$

$dQ = dSK \in \mathbb{R}^{N \times d}$

$dK = QdS^T \in \mathbb{R}^{N \times d}$

Tile:

B is the block_size, S is the sequence length, D is the dimension of hidden layer, in long context S>>D
using tile to seperate a big matrix Q(S,D) into Q_i(B,D)
```
Tiling Q*K^T Computation:

Q (S×D) tiled into blocks of (B×D):

         D
    ┌─────────┐
  B │   Q₁    │
    ├─────────┤
  B │   Q₂    │
    ├─────────┤
  B │   Q₃    │
    ├─────────┤
    │   ...   │
    ├─────────┤
  B │   Qₙ    │
    └─────────┘
    
K^T (D×S) tiled into blocks of (D×B):

         B    B    B        B
    ┌────┬────┬────┬───┬────┐
  D │K₁ᵀ │K₂ᵀ │K₃ᵀ │...│Kₙᵀ │
    └────┴────┴────┴───┴────┘


Q*K^T Result (S×S) - Block Matrix:

              K₁ᵀ      K₂ᵀ      K₃ᵀ           Kₙᵀ
         ┌─────────┬─────────┬─────────┬───┬─────────┐
      Q₁ │ Q₁K₁ᵀ   │ Q₁K₂ᵀ   │ Q₁K₃ᵀ   │...│ Q₁Kₙᵀ   │ B
         ├─────────┼─────────┼─────────┼───┼─────────┤
      Q₂ │ Q₂K₁ᵀ   │ Q₂K₂ᵀ   │ Q₂K₃ᵀ   │...│ Q₂Kₙᵀ   │ B
         ├─────────┼─────────┼─────────┼───┼─────────┤
      Q₃ │ Q₃K₁ᵀ   │ Q₃K₂ᵀ   │ Q₃K₃ᵀ   │...│ Q₃Kₙᵀ   │ B
         ├─────────┼─────────┼─────────┼───┼─────────┤
     ... │   ...   │   ...   │   ...   │...│   ...   │
         ├─────────┼─────────┼─────────┼───┼─────────┤
      Qₙ │ QₙK₁ᵀ   │ QₙK₂ᵀ   │ QₙK₃ᵀ   │...│ QₙKₙᵀ   │ B
         └─────────┴─────────┴─────────┴───┴─────────┘
            B         B         B              B

Each block QᵢKⱼᵀ is (B×B)
Total matrix is (S×S) where S = n×B


Softmax(Q*K^T) Result (S×S) - Block Matrix P:

After applying softmax row-wise to Q*K^T:

              K₁ᵀ          K₂ᵀ          K₃ᵀ               Kₙᵀ
         ┌───────────┬───────────┬───────────┬───┬───────────┐
      Q₁ │softmax(Q₁K₁ᵀ)│softmax(Q₁K₂ᵀ)│softmax(Q₁K₃ᵀ)│...│softmax(Q₁Kₙᵀ)│ B
         ├───────────┼───────────┼───────────┼───┼───────────┤
      Q₂ │softmax(Q₂K₁ᵀ)│softmax(Q₂K₂ᵀ)│softmax(Q₂K₃ᵀ)│...│softmax(Q₂Kₙᵀ)│ B
         ├───────────┼───────────┼───────────┼───┼───────────┤
      Q₃ │softmax(Q₃K₁ᵀ)│softmax(Q₃K₂ᵀ)│softmax(Q₃K₃ᵀ)│...│softmax(Q₃Kₙᵀ)│ B
         ├───────────┼───────────┼───────────┼───┼───────────┤
     ... │    ...    │    ...    │    ...    │...│    ...    │
         ├───────────┼───────────┼───────────┼───┼───────────┤
      Qₙ │softmax(QₙK₁ᵀ)│softmax(QₙK₂ᵀ)│softmax(QₙK₃ᵀ)│...│softmax(QₙKₙᵀ)│ B
         └───────────┴───────────┴───────────┴───┴───────────┘
            B           B           B                  B

Note: Softmax is applied row-wise across the entire row (not per block)
Each row sums to 1: Σⱼ P[i,j] = 1


V (S×D) tiled into blocks of (B×D):

         D
    ┌─────────┐
  B │   V₁    │
    ├─────────┤
  B │   V₂    │
    ├─────────┤
  B │   V₃    │
    ├─────────┤
    │   ...   │
    ├─────────┤
  B │   Vₙ    │
    └─────────┘


P*V Computation - Output O (S×D):

P (S×S) × V (S×D) = O (S×D)

For each output block Oᵢ (B×D):

Oᵢ = Σⱼ Pᵢⱼ × Vⱼ

Where:
- Pᵢⱼ is the (B×B) block from row block i, column block j of P
- Vⱼ is the (B×D) block j of V
- Each Oᵢ is accumulated across all j blocks

Visual representation:

         D
    ┌─────────┐
  B │   O₁    │ = P₁₁V₁ + P₁₂V₂ + P₁₃V₃ + ... + P₁ₙVₙ
    ├─────────┤
  B │   O₂    │ = P₂₁V₁ + P₂₂V₂ + P₂₃V₃ + ... + P₂ₙVₙ
    ├─────────┤
  B │   O₃    │ = P₃₁V₁ + P₃₂V₂ + P₃₃V₃ + ... + P₃ₙVₙ
    ├─────────┤
    │   ...   │
    ├─────────┤
  B │   Oₙ    │ = Pₙ₁V₁ + Pₙ₂V₂ + Pₙ₃V₃ + ... + PₙₙVₙ
    └─────────┘

Each Oᵢ block (B×D) is computed by:
- Taking row block i from P (which spans B×S)
- Multiplying with all V blocks (S×D)
- Result is (B×D)

Final Output O is (S×D) where S = n×B
```

And the matrix multiplication part is easy, the hard part is softmax.

Flash attention uses a incremental softmax method.

Actually we couldn't get softmax answer until we finish 1 row computation, so we need to maintain m and l to restore the max input x and the sum of the exponentials

## Incremental Softmax (Online Softmax)

The key insight is that we can compute softmax incrementally as we process tiles, without materializing the full attention matrix.

### State Variables

For each query block, we maintain:
- **mᵢ** = running maximum of attention scores
- **ℓᵢ** = running sum of exponentials (denominator of softmax)
- **Oᵢ** = partial output (weighted sum of values)

### Mathematical Foundation

Standard softmax for a row x = [x₁, x₂, ..., xₙ]:

\[
\text{softmax}(x_j) = \frac{e^{x_j}}{\sum_{k=1}^{n} e^{x_k}} = \frac{e^{x_j - m}}{e^{-m}\sum_{k=1}^{n} e^{x_k}} = \frac{e^{x_j - m}}{\sum_{k=1}^{n} e^{x_k - m}}
\]

where \(m = \max(x)\) for numerical stability.

### Incremental Update Rules

When processing tiles sequentially, suppose we have:
- Current state: \((m^{(old)}, ℓ^{(old)}, O^{(old)})\)
- New tile with scores \(S^{(new)}\)

**Step 1: Update maximum**

\[
m^{(new)} = \max(m^{(old)}, \max(S^{(new)}))
\]

**Step 2: Update sum of exponentials**

\[
ℓ^{(new)} = e^{m^{(old)} - m^{(new)}} \cdot ℓ^{(old)} + \sum_{j \in \text{new tile}} e^{S_j^{(new)} - m^{(new)}}
\]

The correction factor \(e^{m^{(old)} - m^{(new)}}\) rescales the old sum when the maximum changes.

**Step 3: Update output**

\[
O^{(new)} = e^{m^{(old)} - m^{(new)}} \cdot O^{(old)} + \sum_{j \in \text{new tile}} e^{S_j^{(new)} - m^{(new)}} \cdot V_j
\]

Again, we rescale the old output and add the contribution from the new tile.

**Step 4: Final normalization**

After processing all tiles:

\[
O^{(final)} = \frac{O^{(final)}}{ℓ^{(final)}}
\]

## Flash Attention v2: Tile-by-Tile Example

Let's walk through a concrete example with 2 query blocks and 3 key/value blocks.

### Setup

```
Block size B = 2
Sequence length S = 6 (so we have 3 blocks)

Q = [Q₁]  (B×D)  - Processing one query block at a time
    [Q₂]

K = [K₁]  (B×D)
    [K₂]
    [K₃]

V = [V₁]  (B×D)
    [V₂]
    [V₃]
```

### Processing Query Block Q₁

We compute attention for the first B rows of Q.

#### Iteration 1: Process K₁, V₁

Compute attention scores for this tile:

\[
S₁ = Q₁K₁^T \in \mathbb{R}^{B \times B}
\]

Initialize state:

\[
m₁ = \max(S₁) \text{ (row-wise max)}
\]

\[
ℓ₁ = \sum_j e^{S₁[i,j] - m₁[i]} \text{ (row-wise sum)}
\]

\[
P₁ = e^{S₁ - m₁} \text{ (element-wise, broadcast m₁)}
\]

\[
O₁ = P₁V₁ \in \mathbb{R}^{B \times D}
\]

State after iteration 1: \((m₁, ℓ₁, O₁)\)

#### Iteration 2: Process K₂, V₂

Compute new attention scores:

\[
S₂ = Q₁K₂^T \in \mathbb{R}^{B \times B}
\]

Update maximum:

\[
m₂ = \max(m₁, \max(S₂))
\]

Compute new exponentials:

\[
P₂ = e^{S₂ - m₂}
\]

Update sum of exponentials with correction:

\[
ℓ₂ = e^{m₁ - m₂} \cdot ℓ₁ + \sum_j P₂[i,j]
\]

Update output with correction:

\[
O₂ = e^{m₁ - m₂} \cdot O₁ + P₂V₂
\]

State after iteration 2: \((m₂, ℓ₂, O₂)\)

#### Iteration 3: Process K₃, V₃

Compute new attention scores:

\[
S₃ = Q₁K₃^T \in \mathbb{R}^{B \times B}
\]

Update maximum:

\[
m₃ = \max(m₂, \max(S₃))
\]

Compute new exponentials:

\[
P₃ = e^{S₃ - m₃}
\]

Update sum of exponentials:

\[
ℓ₃ = e^{m₂ - m₃} \cdot ℓ₂ + \sum_j P₃[i,j]
\]

Update output:

\[
O₃ = e^{m₂ - m₃} \cdot O₂ + P₃V₃
\]

#### Final Output for Q₁

Normalize:

\[
O_{Q₁} = \frac{O₃}{ℓ₃}
\]

This is the final attention output for query block Q₁.

### Processing Query Block Q₂

Repeat the same process for Q₂:
- Initialize state with K₁, V₁
- Update with K₂, V₂
- Update with K₃, V₃
- Normalize

### Key Advantages of Flash Attention v2

1. **Memory Efficiency**: We never materialize the full \(S \times S\) attention matrix
   - Only store \(B \times B\) tiles at a time
   - Memory usage: \(O(S \cdot D + B^2)\) instead of \(O(S^2)\)

2. **Numerical Stability**: The running maximum \(m\) prevents overflow/underflow
   - All exponentials are computed relative to the current maximum
   - Correction factors keep values in a stable range

3. **I/O Efficiency**: Minimizes HBM ↔ SRAM transfers
   - Load Q, K, V blocks once
   - Compute in fast SRAM
   - Write final output back to HBM

4. **Exact Computation**: Despite being incremental, this computes the exact same result as standard attention
   - The mathematical equivalence is guaranteed by the update rules

## Pseudocode for Flash Attention v2 Forward Pass

```python
def flash_attention_v2_forward(Q, K, V, block_size):
    """
    Q, K, V: (S, D) tensors
    block_size: B
    Returns: O (S, D) - attention output
    """
    S, D = Q.shape
    num_blocks = S // block_size
    O = zeros(S, D)
    
    # Process each query block
    for i in range(num_blocks):
        Q_i = Q[i*block_size : (i+1)*block_size]  # (B, D)
        
        # Initialize state
        m_i = -inf * ones(block_size)  # (B,)
        l_i = zeros(block_size)        # (B,)
        O_i = zeros(block_size, D)     # (B, D)
        
        # Process each key/value block
        for j in range(num_blocks):
            K_j = K[j*block_size : (j+1)*block_size]  # (B, D)
            V_j = V[j*block_size : (j+1)*block_size]  # (B, D)
            
            # Compute attention scores for this tile
            S_ij = Q_i @ K_j.T  # (B, B)
            
            # Update maximum
            m_i_new = max(m_i, row_max(S_ij))  # (B,)
            
            # Compute exponentials
            P_ij = exp(S_ij - m_i_new)  # (B, B)
            
            # Update sum with correction
            correction = exp(m_i - m_i_new)  # (B,)
            l_i_new = correction * l_i + row_sum(P_ij)  # (B,)
            
            # Update output with correction
            O_i = diag(correction) @ O_i + P_ij @ V_j  # (B, D)
            
            # Update state
            m_i = m_i_new
            l_i = l_i_new
        
        # Final normalization
        O_i = diag(1 / l_i) @ O_i  # (B, D)
        
        # Write back to output
        O[i*block_size : (i+1)*block_size] = O_i
    
    return O
```

## Visualization of State Evolution

For a single query row processing 3 key blocks:

```
Initial:  m = -∞,  ℓ = 0,  O = 0

After K₁: m = 5.2,  ℓ = 2.3,  O = [weighted V₁]
          ↓ (process K₂, max increases)
After K₂: m = 6.1,  ℓ = 2.3*e^(-0.9) + 1.8 = 2.7,  O = [rescaled O + weighted V₂]
          ↓ (process K₃, max stays same)
After K₃: m = 6.1,  ℓ = 2.7 + 1.5 = 4.2,  O = [O + weighted V₃]
          ↓ (normalize)
Final:    O_final = O / 4.2
```

The rescaling factor \(e^{m_{old} - m_{new}}\) ensures that when we discover a larger maximum, we correctly downweight the contributions from previous tiles.

## Why This Works: Mathematical Proof

Consider computing softmax over concatenated tiles: \(x = [x^{(1)}, x^{(2)}]\)

After processing tile 1:
\[
m^{(1)} = \max(x^{(1)})
\]
\[
ℓ^{(1)} = \sum_{i \in \text{tile 1}} e^{x_i - m^{(1)}}
\]

After processing tile 2:
\[
m^{(2)} = \max(m^{(1)}, \max(x^{(2)}))
\]

The true softmax denominator is:
\[
\sum_{i \in \text{all}} e^{x_i - m^{(2)}} = \sum_{i \in \text{tile 1}} e^{x_i - m^{(2)}} + \sum_{i \in \text{tile 2}} e^{x_i - m^{(2)}}
\]

For tile 1 terms:
\[
\sum_{i \in \text{tile 1}} e^{x_i - m^{(2)}} = \sum_{i \in \text{tile 1}} e^{x_i - m^{(1)}} \cdot e^{m^{(1)} - m^{(2)}} = ℓ^{(1)} \cdot e^{m^{(1)} - m^{(2)}}
\]

For tile 2 terms:
\[
\sum_{i \in \text{tile 2}} e^{x_i - m^{(2)}} = ℓ^{(2)}_{\text{new}}
\]

Therefore:
\[
ℓ^{(2)} = e^{m^{(1)} - m^{(2)}} \cdot ℓ^{(1)} + ℓ^{(2)}_{\text{new}}
\]

This proves the correctness of the incremental update rule! The same logic applies to the output \(O\).

