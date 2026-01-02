"""
Triton Tutorial: Weighted Sum Kernel

This script walks through implementing a custom weighted sum operation in Triton,
based on the CS336 Assignment 2 example.

What we'll learn:
1. How to write a Triton kernel with the @triton.jit decorator
2. How to use block pointers for memory access
3. How to implement forward and backward passes
4. How to integrate Triton kernels with PyTorch's autograd

The Operation:
Given:
- Input matrix X with shape [..., D] (can be batched)
- Weight vector w with shape [D]

Compute: (w * X).sum(axis=-1) - a weighted sum along the last dimension
"""

import torch
import triton
import triton.language as tl
import time

# Helper function for ceiling division
def cdiv(a, b):
    return (a + b - 1) // b

print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# ==============================================================================
# Step 1: PyTorch Reference Implementation
# ==============================================================================

def weighted_sum_pytorch(x, weight):
    """Reference implementation in PyTorch"""
    # x has shape [..., D], weight has shape [D]
    return (weight * x).sum(axis=-1)

print("=" * 80)
print("STEP 1: PyTorch Reference Implementation")
print("=" * 80)

# Test it
D = 8
n_rows = 4
x = torch.randn(n_rows, D, device='cuda')
weight = torch.randn(D, device='cuda')

result = weighted_sum_pytorch(x, weight)
print(f"Input shape: {x.shape}")
print(f"Weight shape: {weight.shape}")
print(f"Output shape: {result.shape}")
print(f"Output: {result}")
print()

# ==============================================================================
# Step 2: Triton Forward Kernel
# ==============================================================================

print("=" * 80)
print("STEP 2: Triton Forward Kernel")
print("=" * 80)

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row, x_stride_dim,  # Strides for x
    weight_stride_dim,  # Stride for weight
    output_stride_row,  # Stride for output
    ROWS, D,  # Dimensions
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile sizes (compile-time constants)
):
    """
    Triton kernel for weighted sum forward pass.
    
    Each thread block processes a tile of rows:
    - row_tile_idx determines which tile of rows this block handles
    - We loop over D in tiles, accumulating the weighted sum
    """
    # Each thread block processes a tile of rows
    row_tile_idx = tl.program_id(0)
    
    # Create block pointer for x (2D: rows × D)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),  # Start at our tile
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),  # Column-major within block
    )
    
    # Create block pointer for weight (1D: D)
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    
    # Create block pointer for output (1D: rows)
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    
    # Load first tile to determine dtype, then initialize accumulator
    x_tile_first = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    weight_tile_first = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Compute first partial sum
    weighted_first = x_tile_first * weight_tile_first[None, :]
    acc = tl.sum(weighted_first, axis=1)
    
    # Advance pointers for next iteration
    x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
    weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
    
    # Loop over remaining tiles in the D dimension
    for i in range(1, tl.cdiv(D, D_TILE_SIZE)):
        # Load tiles
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # Compute weighted sum for this tile
        # x_tile: (ROWS_TILE_SIZE, D_TILE_SIZE)
        # weight_tile: (D_TILE_SIZE,)
        weighted = x_tile * weight_tile[None, :]  # Broadcast weight
        acc += tl.sum(weighted, axis=1)  # Sum along D dimension
        
        # Advance pointers to next tile
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
    
    # Store result
    tl.store(output_block_ptr, acc, boundary_check=(0,))

print("✓ Forward kernel defined!")
print()

# ==============================================================================
# Step 3: Triton Backward Kernel
# ==============================================================================

print("=" * 80)
print("STEP 3: Triton Backward Kernel")
print("=" * 80)

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,  # Inputs from forward pass
    grad_output_ptr,  # Gradient w.r.t. output
    grad_x_ptr, partial_grad_weight_ptr,  # Gradient outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    """
    Triton kernel for weighted sum backward pass.
    
    Computes:
    - grad_x[i,j] = weight[j] * grad_output[i] (outer product)
    - grad_weight[j] = sum_i(x[i,j] * grad_output[i]) (reduction)
    
    For grad_weight, we compute partial sums per tile and reduce later.
    """
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    
    # Block pointer for grad_output (1D)
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    
    # Block pointer for x (2D)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # Block pointer for weight (1D)
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    
    # Block pointer for grad_x (2D)
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # Block pointer for partial grad_weight (2D: n_tiles × D)
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # Load grad_output once (same for all D tiles)
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Loop over D dimension
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Compute grad_x: outer product of grad_output and weight
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_tile = grad_output[:, None] * weight[None, :]  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))
        
        # Compute partial grad_weight: reduce over rows in this tile
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_tile = tl.sum(x_tile * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_tile, boundary_check=(1,))
        
        # Advance pointers
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, D_TILE_SIZE))

print("✓ Backward kernel defined!")
print()

# ==============================================================================
# Step 4: PyTorch Autograd Integration
# ==============================================================================

print("=" * 80)
print("STEP 4: PyTorch Autograd Integration")
print("=" * 80)

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Save dimensions
        D = x.shape[-1]
        output_dims = x.shape[:-1]
        
        # Reshape to 2D for kernel
        input_shape = x.shape
        x = x.reshape(-1, D)
        
        # Validation
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "x must be contiguous"
        
        # Save for backward
        ctx.save_for_backward(x, weight)
        
        # Choose tile sizes
        ctx.D_TILE_SIZE = min(triton.next_power_of_2(D) // 16, 128)
        ctx.D_TILE_SIZE = max(ctx.D_TILE_SIZE, 1)
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape
        
        # Allocate output
        n_rows = x.shape[0]
        y = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
        weighted_sum_fwd[grid](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        
        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape
        
        # Flatten grad_output
        grad_output = grad_output.reshape(-1).contiguous()
        
        # Allocate outputs
        grad_x = torch.empty_like(x)
        n_tiles = cdiv(n_rows, ROWS_TILE_SIZE)
        partial_grad_weight = torch.empty((n_tiles, D), device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (n_tiles,)
        weighted_sum_backward[grid](
            x, weight,
            grad_output,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_output.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        
        # Reduce partial gradients
        grad_weight = partial_grad_weight.sum(axis=0)
        
        # Reshape grad_x back to original shape
        grad_x = grad_x.view(ctx.input_shape)
        
        return grad_x, grad_weight

# Create the function
weighted_sum_triton = WeightedSumFunc.apply

print("✓ Autograd function created!")
print()

# ==============================================================================
# Step 5: Test Forward Pass
# ==============================================================================

print("=" * 80)
print("STEP 5: Test Forward Pass")
print("=" * 80)

torch.manual_seed(42)
n_rows, D = 32, 64
x = torch.randn(n_rows, D, device='cuda', requires_grad=True)
weight = torch.randn(D, device='cuda', requires_grad=True)

# PyTorch reference
output_pytorch = weighted_sum_pytorch(x, weight)

# Triton implementation
output_triton = weighted_sum_triton(x, weight)

# Compare
print(f"PyTorch output shape: {output_pytorch.shape}")
print(f"Triton output shape: {output_triton.shape}")
print(f"\nMax absolute difference: {(output_pytorch - output_triton).abs().max().item():.2e}")
print(f"Mean absolute difference: {(output_pytorch - output_triton).abs().mean().item():.2e}")
print(f"\n✓ Outputs match: {torch.allclose(output_pytorch, output_triton, rtol=1e-4, atol=1e-4)}")
print()

# ==============================================================================
# Step 6: Test Backward Pass
# ==============================================================================

print("=" * 80)
print("STEP 6: Test Backward Pass")
print("=" * 80)

torch.manual_seed(42)
n_rows, D = 32, 64
x_pt = torch.randn(n_rows, D, device='cuda', requires_grad=True, dtype=torch.float64)
weight_pt = torch.randn(D, device='cuda', requires_grad=True, dtype=torch.float64)

x_tr = x_pt.clone().detach().requires_grad_(True)
weight_tr = weight_pt.clone().detach().requires_grad_(True)

# Forward pass
output_pt = weighted_sum_pytorch(x_pt, weight_pt)
output_tr = weighted_sum_triton(x_tr, weight_tr)

# Backward pass
grad_output = torch.randn_like(output_pt)
output_pt.backward(grad_output)
output_tr.backward(grad_output)

# Compare gradients
print("Gradient w.r.t. x:")
print(f"  Max absolute difference: {(x_pt.grad - x_tr.grad).abs().max().item():.2e}")
print(f"  ✓ Gradients match: {torch.allclose(x_pt.grad, x_tr.grad, rtol=1e-4, atol=1e-4)}")

print("\nGradient w.r.t. weight:")
print(f"  Max absolute difference: {(weight_pt.grad - weight_tr.grad).abs().max().item():.2e}")
print(f"  ✓ Gradients match: {torch.allclose(weight_pt.grad, weight_tr.grad, rtol=1e-4, atol=1e-4)}")
print()

# ==============================================================================
# Step 7: Test with Different Shapes
# ==============================================================================

print("=" * 80)
print("STEP 7: Test with Different Shapes")
print("=" * 80)

test_shapes = [
    (16, 32),      # Small
    (128, 256),    # Medium
    (1024, 512),   # Large
    (8, 16, 64),   # Batched (batch_size=8, seq_len=16, D=64)
]

for shape in test_shapes:
    D = shape[-1]
    x = torch.randn(*shape, device='cuda')
    weight = torch.randn(D, device='cuda')
    
    output_pt = weighted_sum_pytorch(x, weight)
    output_tr = weighted_sum_triton(x, weight)
    
    matches = torch.allclose(output_pt, output_tr, rtol=1e-4, atol=1e-4)
    max_diff = (output_pt - output_tr).abs().max().item()
    
    status = "✓" if matches else "✗"
    print(f"{status} Shape {shape}: max diff = {max_diff:.2e}")

print()

# ==============================================================================
# Step 8: Benchmark Performance
# ==============================================================================

print("=" * 80)
print("STEP 8: Benchmark Performance")
print("=" * 80)

def benchmark(fn, x, weight, n_iters=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        fn(x, weight)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_iters):
        fn(x, weight)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / n_iters * 1000  # ms

# Test different sizes
sizes = [(1024, 512), (2048, 1024), (4096, 2048)]

print("\nPerformance Comparison (forward pass only):")
print(f"{'Shape':<20} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
print("-" * 60)

for n_rows, D in sizes:
    x = torch.randn(n_rows, D, device='cuda')
    weight = torch.randn(D, device='cuda')
    
    time_pt = benchmark(weighted_sum_pytorch, x, weight)
    time_tr = benchmark(weighted_sum_triton, x, weight)
    speedup = time_pt / time_tr
    
    print(f"{str((n_rows, D)):<20} {time_pt:<15.4f} {time_tr:<15.4f} {speedup:<10.2f}x")

print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Block Pointers: tl.make_block_ptr() simplifies memory access by handling 
   pointer arithmetic automatically.

2. Tiling: We process data in tiles to maximize memory locality and parallelism.
   Each thread block handles a tile of rows.

3. Program ID: tl.program_id(0) lets different thread blocks work on different 
   tiles in parallel.

4. Boundary Checking: boundary_check and padding_option handle edge cases when
   tiles don't evenly divide the input.

5. Autograd Integration: torch.autograd.Function connects Triton kernels to 
   PyTorch's autograd system.

6. Backward Pass Strategy: For reductions (like grad_weight), we compute partial
   results per tile and reduce outside the kernel.

EXPERIMENT FURTHER:
- Try modifying tile sizes (ROWS_TILE_SIZE, D_TILE_SIZE)
- Test with different input shapes and data types
- Modify the operation (e.g., add a bias term, use different aggregation)
- This pattern extends to more complex kernels like attention, layer norm, etc.!
""")

