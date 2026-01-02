#!/usr/bin/env python3
"""
Benchmark FlashAttention-2 (Triton) vs Regular PyTorch Attention

This script compares:
- FlashAttention-2 (partially Triton implementation)
- Regular PyTorch attention (naive implementation)

Configuration:
- Batch size: 1 (default, configurable)
- Causal masking: Configurable (default: both True and False)
- Sequence lengths: powers of 2 from 128 to 65536 (configurable)
- Embedding dimensions: powers of 2 from 16 to 128 (configurable)
- Precisions: torch.bfloat16 and torch.float32 (configurable)

Measures:
- Forward pass latency (always)
- Backward pass latency (optional, if implemented)
- End-to-end forward+backward latency (optional, if implemented)

Usage:
    # Full benchmark with all options
    python benchmarks/benchmark_flash_attention.py
    
    # Quick test without backward pass
    python benchmarks/benchmark_flash_attention.py --no-backward
    
    # Test only non-causal attention
    python benchmarks/benchmark_flash_attention.py --no-causal
    
    # Test only causal attention
    python benchmarks/benchmark_flash_attention.py --causal-only
    
    # Reduced configuration for quick testing
    python benchmarks/benchmark_flash_attention.py --quick
"""

import torch
import triton
import triton.testing
import sys
import itertools
import math
import argparse
from typing import Dict, List, Optional
import pandas as pd
from cs336_systems.attention import FlashAttentionTriton, FlashAttentionPytorch, naive_attention


# Default configuration parameters
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_EMBEDDING_DIMS = [16, 32, 64, 128]
DEFAULT_DTYPES = [torch.bfloat16, torch.float32]

# Quick test configuration (smaller subset)
QUICK_SEQ_LENGTHS = [128, 512, 2048, 8192]
QUICK_EMBEDDING_DIMS = [32, 64, 128]
QUICK_DTYPES = [torch.float32]


def get_tile_sizes(seq_len: int, d_model: int) -> tuple:
    """
    Adjust tile sizes based on input dimensions.
    
    Larger tiles for smaller sequences, smaller tiles for larger sequences
    to balance memory usage and parallelism.
    """
    if seq_len <= 512:
        q_tile = 128
        k_tile = 128
    elif seq_len <= 2048:
        q_tile = 64
        k_tile = 64
    elif seq_len <= 8192:
        q_tile = 64
        k_tile = 64
    else:
        q_tile = 32
        k_tile = 32
    
    # Adjust for very small d_model
    if d_model <= 32:
        q_tile = min(q_tile, 64)
        k_tile = min(k_tile, 64)
    
    return q_tile, k_tile


class NaiveAttentionWithCausal:
    """Wrapper for naive attention with causal masking support."""
    
    def __call__(self, Q, K, V, is_causal=False):
        """
        Naive attention with optional causal masking.
        
        Args:
            Q: Query tensor [batch, seq_len, d_model]
            K: Key tensor [batch, seq_len, d_model]
            V: Value tensor [batch, seq_len, d_model]
            is_causal: Whether to apply causal masking
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        d_k = Q.shape[-1]
        seq_len = Q.shape[1]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply causal mask if needed
        if is_causal:
            # Create causal mask: upper triangular matrix of -inf
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=Q.device, dtype=Q.dtype),
                diagonal=1
            )
            scores = scores + mask
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        
        return output


def benchmark_forward(fn, Q, K, V, is_causal=True, quantiles=[0.5, 0.2, 0.8]):
    """
    Benchmark forward pass using triton.testing.do_bench.
    
    Args:
        fn: Function to benchmark
        Q, K, V: Input tensors
        is_causal: Whether to use causal masking
        quantiles: Quantiles to measure
    
    Returns:
        Median latency in milliseconds
    """
    def forward_fn():
        return fn(Q, K, V, is_causal)
    
    # Use triton's do_bench for accurate timing
    ms = triton.testing.do_bench(forward_fn, quantiles=quantiles)
    return ms[0]  # Return median


def benchmark_backward(fn, Q, K, V, is_causal=True, quantiles=[0.5, 0.2, 0.8]):
    """
    Benchmark backward pass using triton.testing.do_bench.
    
    Args:
        fn: Function to benchmark
        Q, K, V: Input tensors
        is_causal: Whether to use causal masking
        quantiles: Quantiles to measure
    
    Returns:
        Median latency in milliseconds, or None if backward not supported
    """
    try:
        def backward_fn():
            Q.grad = None
            K.grad = None
            V.grad = None
            output = fn(Q, K, V, is_causal)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
        
        # Use triton's do_bench for accurate timing
        ms = triton.testing.do_bench(backward_fn, quantiles=quantiles)
        return ms[0]  # Return median
    except (NotImplementedError, RuntimeError) as e:
        # Backward not implemented
        return None


def benchmark_forward_backward(fn, Q, K, V, is_causal=True, quantiles=[0.5, 0.2, 0.8]):
    """
    Benchmark end-to-end forward+backward pass using triton.testing.do_bench.
    
    Args:
        fn: Function to benchmark
        Q, K, V: Input tensors
        is_causal: Whether to use causal masking
        quantiles: Quantiles to measure
    
    Returns:
        Median latency in milliseconds, or None if backward not supported
    """
    try:
        def forward_backward_fn():
            Q.grad = None
            K.grad = None
            V.grad = None
            output = fn(Q, K, V, is_causal)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
        
        # Use triton's do_bench for accurate timing
        ms = triton.testing.do_bench(forward_backward_fn, quantiles=quantiles)
        return ms[0]  # Return median
    except (NotImplementedError, RuntimeError) as e:
        # Backward not implemented
        return None


def benchmark_configuration(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    is_causal: bool,
    test_backward: bool,
    config_num: int,
    total_configs: int
) -> Dict:
    """
    Benchmark a single configuration for both implementations.
    
    Args:
        seq_len: Sequence length
        d_model: Embedding dimension
        dtype: Data type (torch.bfloat16 or torch.float32)
        is_causal: Whether to use causal masking
        test_backward: Whether to test backward pass
        config_num: Current configuration number
        total_configs: Total number of configurations
    
    Returns:
        Dictionary with benchmark results
    """
    dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    causal_str = "causal" if is_causal else "non-causal"
    print(f"\n[{config_num}/{total_configs}] seq_len={seq_len}, d_model={d_model}, dtype={dtype_name}, {causal_str}")
    
    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Generate inputs
        print(f"  Generating inputs...")
        Q = torch.randn(DEFAULT_BATCH_SIZE, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
        K = torch.randn(DEFAULT_BATCH_SIZE, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
        V = torch.randn(DEFAULT_BATCH_SIZE, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
        
        # Get tile sizes for this configuration
        q_tile, k_tile = get_tile_sizes(seq_len, d_model)
        print(f"  Using tile sizes: Q_TILE={q_tile}, K_TILE={k_tile}")
        
        results = {
            'seq_len': seq_len,
            'd_model': d_model,
            'dtype': dtype_name,
            'batch_size': DEFAULT_BATCH_SIZE,
            'is_causal': is_causal,
            'test_backward': test_backward,
        }
        
        # Benchmark PyTorch naive attention
        print(f"  Benchmarking PyTorch naive attention...")
        naive_attn = NaiveAttentionWithCausal()
        
        try:
            # Warmup
            for _ in range(5):
                output = naive_attn(Q.clone(), K.clone(), V.clone(), is_causal=is_causal)
                torch.cuda.synchronize()
            
            # Benchmark forward
            print(f"    Forward pass...")
            Q_clone = Q.clone().detach().requires_grad_(True)
            K_clone = K.clone().detach().requires_grad_(True)
            V_clone = V.clone().detach().requires_grad_(True)
            fwd_time = benchmark_forward(naive_attn, Q_clone, K_clone, V_clone, is_causal)
            results['pytorch_forward_ms'] = fwd_time
            print(f"      {fwd_time:.3f} ms")
            
            # Benchmark backward (if requested)
            if test_backward:
                print(f"    Backward pass...")
                Q_clone = Q.clone().detach().requires_grad_(True)
                K_clone = K.clone().detach().requires_grad_(True)
                V_clone = V.clone().detach().requires_grad_(True)
                bwd_time = benchmark_backward(naive_attn, Q_clone, K_clone, V_clone, is_causal)
                if bwd_time is not None:
                    results['pytorch_backward_ms'] = bwd_time
                    print(f"      {bwd_time:.3f} ms")
                else:
                    print(f"      Not supported")
                    results['pytorch_backward_ms'] = None
                
                # Benchmark forward+backward
                print(f"    Forward+Backward pass...")
                Q_clone = Q.clone().detach().requires_grad_(True)
                K_clone = K.clone().detach().requires_grad_(True)
                V_clone = V.clone().detach().requires_grad_(True)
                fwd_bwd_time = benchmark_forward_backward(naive_attn, Q_clone, K_clone, V_clone, is_causal)
                if fwd_bwd_time is not None:
                    results['pytorch_fwd_bwd_ms'] = fwd_bwd_time
                    print(f"      {fwd_bwd_time:.3f} ms")
                else:
                    print(f"      Not supported")
                    results['pytorch_fwd_bwd_ms'] = None
            else:
                print(f"    Backward pass skipped (--no-backward)")
                results['pytorch_backward_ms'] = None
                results['pytorch_fwd_bwd_ms'] = None
            
            results['pytorch_status'] = 'success'
            
        except Exception as e:
            print(f"    ❌ PyTorch failed: {e}")
            results['pytorch_status'] = 'error'
            results['pytorch_error'] = str(e)
        
        # Benchmark FlashAttention-2 (Triton)
        print(f"  Benchmarking FlashAttention-2 (Triton)...")
        
        try:
            # Warmup
            for _ in range(5):
                output = FlashAttentionTriton.apply(Q.clone(), K.clone(), V.clone(), is_causal)
                torch.cuda.synchronize()
            
            # Benchmark forward
            print(f"    Forward pass...")
            Q_clone = Q.clone().detach().requires_grad_(True)
            K_clone = K.clone().detach().requires_grad_(True)
            V_clone = V.clone().detach().requires_grad_(True)
            fwd_time = benchmark_forward(FlashAttentionTriton.apply, Q_clone, K_clone, V_clone, is_causal)
            results['triton_forward_ms'] = fwd_time
            print(f"      {fwd_time:.3f} ms")
            
            # Try backward if requested
            if test_backward:
                print(f"    Backward pass...")
                Q_clone = Q.clone().detach().requires_grad_(True)
                K_clone = K.clone().detach().requires_grad_(True)
                V_clone = V.clone().detach().requires_grad_(True)
                bwd_time = benchmark_backward(FlashAttentionTriton.apply, Q_clone, K_clone, V_clone, is_causal)
                if bwd_time is not None:
                    results['triton_backward_ms'] = bwd_time
                    print(f"      {bwd_time:.3f} ms")
                    
                    # Try forward+backward
                    print(f"    Forward+Backward pass...")
                    Q_clone = Q.clone().detach().requires_grad_(True)
                    K_clone = K.clone().detach().requires_grad_(True)
                    V_clone = V.clone().detach().requires_grad_(True)
                    fwd_bwd_time = benchmark_forward_backward(FlashAttentionTriton.apply, Q_clone, K_clone, V_clone, is_causal)
                    results['triton_fwd_bwd_ms'] = fwd_bwd_time
                    if fwd_bwd_time is not None:
                        print(f"      {fwd_bwd_time:.3f} ms")
                    results['triton_status'] = 'success'
                else:
                    print(f"      Not implemented")
                    results['triton_backward_ms'] = None
                    results['triton_fwd_bwd_ms'] = None
                    results['triton_status'] = 'forward_only'
            else:
                print(f"    Backward pass skipped (--no-backward)")
                results['triton_backward_ms'] = None
                results['triton_fwd_bwd_ms'] = None
                results['triton_status'] = 'forward_only'
            
        except Exception as e:
            print(f"    ❌ Triton failed: {e}")
            results['triton_status'] = 'error'
            results['triton_error'] = str(e)
        
        # Benchmark FlashAttention-2 (PyTorch)
        print(f"  Benchmarking FlashAttention-2 (PyTorch)...")
        
        try:
            # Warmup
            for _ in range(5):
                output = FlashAttentionPytorch.apply(Q.clone(), K.clone(), V.clone(), is_causal)
                torch.cuda.synchronize()
            
            # Benchmark forward
            print(f"    Forward pass...")
            Q_clone = Q.clone().detach().requires_grad_(True)
            K_clone = K.clone().detach().requires_grad_(True)
            V_clone = V.clone().detach().requires_grad_(True)
            fwd_time = benchmark_forward(FlashAttentionPytorch.apply, Q_clone, K_clone, V_clone, is_causal)
            results['flash_pytorch_forward_ms'] = fwd_time
            print(f"      {fwd_time:.3f} ms")
            
            # Benchmark backward (if requested)
            if test_backward:
                print(f"    Backward pass...")
                Q_clone = Q.clone().detach().requires_grad_(True)
                K_clone = K.clone().detach().requires_grad_(True)
                V_clone = V.clone().detach().requires_grad_(True)
                bwd_time = benchmark_backward(FlashAttentionPytorch.apply, Q_clone, K_clone, V_clone, is_causal)
                if bwd_time is not None:
                    results['flash_pytorch_backward_ms'] = bwd_time
                    print(f"      {bwd_time:.3f} ms")
                else:
                    print(f"      Not supported")
                    results['flash_pytorch_backward_ms'] = None
                
                # Benchmark forward+backward
                print(f"    Forward+Backward pass...")
                Q_clone = Q.clone().detach().requires_grad_(True)
                K_clone = K.clone().detach().requires_grad_(True)
                V_clone = V.clone().detach().requires_grad_(True)
                fwd_bwd_time = benchmark_forward_backward(FlashAttentionPytorch.apply, Q_clone, K_clone, V_clone, is_causal)
                if fwd_bwd_time is not None:
                    results['flash_pytorch_fwd_bwd_ms'] = fwd_bwd_time
                    print(f"      {fwd_bwd_time:.3f} ms")
                else:
                    print(f"      Not supported")
                    results['flash_pytorch_fwd_bwd_ms'] = None
            else:
                print(f"    Backward pass skipped (--no-backward)")
                results['flash_pytorch_backward_ms'] = None
                results['flash_pytorch_fwd_bwd_ms'] = None
            
            results['flash_pytorch_status'] = 'success'
            
        except Exception as e:
            print(f"    ❌ FlashAttention PyTorch failed: {e}")
            results['flash_pytorch_status'] = 'error'
            results['flash_pytorch_error'] = str(e)
        
        # Calculate speedups if both succeeded
        if results.get('pytorch_status') == 'success' and results.get('flash_pytorch_status') == 'success':
            # Forward speedup (always available)
            if results.get('pytorch_forward_ms') and results.get('flash_pytorch_forward_ms'):
                results['flash_fwd_speedup'] = results['pytorch_forward_ms'] / results['flash_pytorch_forward_ms']
            else:
                results['flash_fwd_speedup'] = None
            
            # Backward speedup (only if both have backward)
            if results.get('pytorch_backward_ms') and results.get('flash_pytorch_backward_ms'):
                results['flash_bwd_speedup'] = results['pytorch_backward_ms'] / results['flash_pytorch_backward_ms']
            else:
                results['flash_bwd_speedup'] = None
            
            # Forward+Backward speedup (only if both have it)
            if results.get('pytorch_fwd_bwd_ms') and results.get('flash_pytorch_fwd_bwd_ms'):
                results['flash_fwd_bwd_speedup'] = results['pytorch_fwd_bwd_ms'] / results['flash_pytorch_fwd_bwd_ms']
            else:
                results['flash_fwd_bwd_speedup'] = None
            
            print(f"  Speedups (FlashAttention PyTorch vs Naive):")
            if results['flash_fwd_speedup']:
                print(f"    Forward: {results['flash_fwd_speedup']:.2f}x")
            if results['flash_bwd_speedup']:
                print(f"    Backward: {results['flash_bwd_speedup']:.2f}x")
            if results['flash_fwd_bwd_speedup']:
                print(f"    Forward+Backward: {results['flash_fwd_bwd_speedup']:.2f}x")
        
        if results.get('pytorch_status') == 'success' and results.get('triton_status') in ['forward_only', 'success']:
            if results.get('pytorch_forward_ms') and results.get('triton_forward_ms'):
                results['triton_fwd_speedup'] = results['pytorch_forward_ms'] / results['triton_forward_ms']
                print(f"  Speedup (Triton forward vs Naive forward): {results['triton_fwd_speedup']:.2f}x")
            else:
                results['triton_fwd_speedup'] = None
        
        return results
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OUT OF MEMORY")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'seq_len': seq_len,
            'd_model': d_model,
            'dtype': dtype_name,
            'status': 'OOM',
            'error': str(e),
        }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'seq_len': seq_len,
            'd_model': d_model,
            'dtype': dtype_name,
            'status': 'error',
            'error': str(e),
        }


def print_results_table(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "=" * 180)
    print("BENCHMARK RESULTS")
    print("=" * 180)
    
    # Create DataFrame for easier formatting
    df = pd.DataFrame(results)
    
    # Print configuration columns
    print("\nConfiguration:")
    print(df[['seq_len', 'd_model', 'dtype']].to_string(index=False))
    
    # Print PyTorch naive results
    print("\n" + "-" * 180)
    print("PyTorch Naive Attention:")
    print("-" * 180)
    cols = ['seq_len', 'd_model', 'dtype', 'pytorch_forward_ms', 'pytorch_backward_ms', 'pytorch_fwd_bwd_ms', 'pytorch_status']
    if all(col in df.columns for col in cols):
        print(df[cols].to_string(index=False))
    
    # Print FlashAttention PyTorch results
    print("\n" + "-" * 180)
    print("FlashAttention-2 (PyTorch):")
    print("-" * 180)
    cols = ['seq_len', 'd_model', 'dtype', 'flash_pytorch_forward_ms', 'flash_pytorch_backward_ms', 'flash_pytorch_fwd_bwd_ms', 'flash_pytorch_status']
    if all(col in df.columns for col in cols):
        print(df[cols].to_string(index=False))
    
    # Print Triton results
    print("\n" + "-" * 180)
    print("FlashAttention-2 (Triton - Forward Only):")
    print("-" * 180)
    cols = ['seq_len', 'd_model', 'dtype', 'triton_forward_ms', 'triton_status']
    if all(col in df.columns for col in cols):
        print(df[cols].to_string(index=False))
    
    # Print speedups
    print("\n" + "-" * 180)
    print("Speedups (FlashAttention vs Naive):")
    print("-" * 180)
    cols = ['seq_len', 'd_model', 'dtype', 'flash_fwd_speedup', 'flash_bwd_speedup', 'flash_fwd_bwd_speedup']
    if all(col in df.columns for col in cols):
        speedup_df = df[cols].copy()
        for col in ['flash_fwd_speedup', 'flash_bwd_speedup', 'flash_fwd_bwd_speedup']:
            if col in speedup_df.columns:
                speedup_df[col] = speedup_df[col].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
        print(speedup_df.to_string(index=False))
    
    print("=" * 180)


def save_results_csv(results: List[Dict], filename: str = "flash_attention_benchmark_results.csv"):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark FlashAttention-2 vs Regular PyTorch Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark with all options
  python benchmarks/benchmark_flash_attention.py
  
  # Quick test without backward pass
  python benchmarks/benchmark_flash_attention.py --no-backward
  
  # Test only non-causal attention
  python benchmarks/benchmark_flash_attention.py --no-causal
  
  # Test only causal attention  
  python benchmarks/benchmark_flash_attention.py --causal-only
  
  # Reduced configuration for quick testing
  python benchmarks/benchmark_flash_attention.py --quick
  
  # Combine options
  python benchmarks/benchmark_flash_attention.py --quick --no-backward --no-causal
        """
    )
    
    # Causal masking options
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument(
        '--causal-only',
        action='store_true',
        help='Test only causal attention (default: test both causal and non-causal)'
    )
    causal_group.add_argument(
        '--no-causal',
        action='store_true',
        help='Skip causal attention, test only non-causal'
    )
    
    # Backward pass option
    parser.add_argument(
        '--no-backward',
        action='store_true',
        help='Skip backward pass testing (only test forward pass)'
    )
    
    # Quick test option
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with reduced configurations'
    )
    
    # Custom configurations
    parser.add_argument(
        '--seq-lengths',
        type=int,
        nargs='+',
        help='Custom sequence lengths (e.g., --seq-lengths 128 512 2048)'
    )
    parser.add_argument(
        '--dims',
        type=int,
        nargs='+',
        help='Custom embedding dimensions (e.g., --dims 32 64 128)'
    )
    parser.add_argument(
        '--dtypes',
        choices=['float32', 'bfloat16', 'both'],
        default='both',
        help='Data types to test (default: both)'
    )
    
    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Determine configurations based on arguments
    if args.quick:
        seq_lengths = QUICK_SEQ_LENGTHS
        embedding_dims = QUICK_EMBEDDING_DIMS
        dtypes = QUICK_DTYPES
    else:
        seq_lengths = args.seq_lengths if args.seq_lengths else DEFAULT_SEQ_LENGTHS
        embedding_dims = args.dims if args.dims else DEFAULT_EMBEDDING_DIMS
        
        if args.dtypes == 'float32':
            dtypes = [torch.float32]
        elif args.dtypes == 'bfloat16':
            dtypes = [torch.bfloat16]
        else:
            dtypes = DEFAULT_DTYPES
    
    # Determine causal masking options
    if args.causal_only:
        causal_options = [True]
    elif args.no_causal:
        causal_options = [False]
    else:
        causal_options = [False, True]  # Test both
    
    # Determine backward testing
    test_backward = not args.no_backward
    
    print("=" * 180)
    print("FLASHATTENTION-2 vs PYTORCH NAIVE ATTENTION BENCHMARK")
    print("=" * 180)
    print(f"Configuration:")
    print(f"  Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"  Causal masking: {causal_options}")
    print(f"  Test backward: {test_backward}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Embedding dimensions: {embedding_dims}")
    print(f"  Data types: {[str(dt) for dt in dtypes]}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 180)
    
    # Run benchmarks for all configurations
    results = []
    configs = list(itertools.product(seq_lengths, embedding_dims, dtypes, causal_options))
    total_configs = len(configs)
    
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"  = {len(seq_lengths)} seq_lengths × {len(embedding_dims)} dims × {len(dtypes)} dtypes × {len(causal_options)} causal_options")
    
    for idx, (seq_len, d_model, dtype, is_causal) in enumerate(configs, 1):
        result = benchmark_configuration(
            seq_len, d_model, dtype, is_causal, test_backward, idx, total_configs
        )
        results.append(result)
        
        # Save intermediate results every 10 configs
        if idx % 10 == 0:
            save_results_csv(results, f"flash_attention_benchmark_intermediate_{idx}.csv")
    
    # Print final results
    print_results_table(results)
    
    # Save final results
    save_results_csv(results, "flash_attention_benchmark_final.csv")
    
    # Print summary statistics
    print("\n" + "=" * 180)
    print("SUMMARY STATISTICS")
    print("=" * 180)
    
    df = pd.DataFrame(results)
    
    # Filter successful results
    success_df = df[
        (df.get('pytorch_status') == 'success') & 
        (df.get('flash_pytorch_status') == 'success')
    ]
    
    if not success_df.empty:
        print(f"\nSuccessful configurations: {len(success_df)}/{len(results)}")
        print(f"\nAverage speedups (FlashAttention PyTorch vs Naive):")
        print(f"  Forward: {success_df['flash_fwd_speedup'].mean():.2f}x (std: {success_df['flash_fwd_speedup'].std():.2f})")
        print(f"  Backward: {success_df['flash_bwd_speedup'].mean():.2f}x (std: {success_df['flash_bwd_speedup'].std():.2f})")
        print(f"  Forward+Backward: {success_df['flash_fwd_bwd_speedup'].mean():.2f}x (std: {success_df['flash_fwd_bwd_speedup'].std():.2f})")
        
        print(f"\nMax speedups:")
        print(f"  Forward: {success_df['flash_fwd_speedup'].max():.2f}x")
        print(f"  Backward: {success_df['flash_bwd_speedup'].max():.2f}x")
        print(f"  Forward+Backward: {success_df['flash_fwd_bwd_speedup'].max():.2f}x")
        
        print(f"\nBest configuration (highest forward+backward speedup):")
        best_idx = success_df['flash_fwd_bwd_speedup'].idxmax()
        best = success_df.loc[best_idx]
        print(f"  Sequence length: {best['seq_len']}")
        print(f"  Embedding dim: {best['d_model']}")
        print(f"  Data type: {best['dtype']}")
        print(f"  Speedup: {best['flash_fwd_bwd_speedup']:.2f}x")
    
    # Report OOM cases
    oom_df = df[df.get('status') == 'OOM']
    if not oom_df.empty:
        print(f"\nOut of Memory configurations: {len(oom_df)}")
        print(f"First OOM at: seq_len={oom_df.iloc[0]['seq_len']}, d_model={oom_df.iloc[0]['d_model']}, dtype={oom_df.iloc[0]['dtype']}")
    
    print("=" * 180)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)
    
    main()

