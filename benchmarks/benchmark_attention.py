#!/usr/bin/env python3
"""
Benchmark naive attention implementation at different scales.

This script benchmarks the naive attention implementation with:
- Fixed batch size of 8
- No multihead attention (single head)
- Cartesian product of:
  - Head embedding dimensions: [16, 32, 64, 128]
  - Sequence lengths: [256, 1024, 4096, 8192, 16384]

Measures:
- Forward pass timing (100 iterations)
- Memory usage before backward pass
- Backward pass timing (100 iterations)

Usage:
    python benchmarks/benchmark_attention.py
"""

import torch
import numpy as np
import sys
import itertools
from cs336_systems.attention import naive_attention


# Configuration parameters as per assignment
BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]
NUM_ITERATIONS = 100


def get_memory_usage():
    """Get current CUDA memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0


def benchmark_attention(d_model, seq_length):
    """
    Benchmark attention for a given configuration.

    Args:
        d_model: Head embedding dimension
        seq_length: Sequence length

    Returns:
        Dictionary with benchmark results or None if OOM
    """
    try:
        # Clear cache before each benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"\nBenchmarking d_model={d_model}, seq_length={seq_length}")
        print(f"  Creating inputs...")

        # Create random inputs Q, K, V
        # Shape: (batch_size, seq_length, d_model)
        Q = torch.randn(
            BATCH_SIZE, seq_length, d_model, device="cuda", requires_grad=True
        )
        K = torch.randn(
            BATCH_SIZE, seq_length, d_model, device="cuda", requires_grad=True
        )
        V = torch.randn(
            BATCH_SIZE, seq_length, d_model, device="cuda", requires_grad=True
        )

        # Warmup
        print(f"  Warming up...")
        for _ in range(10):
            output = naive_attention(Q, K, V)
            torch.cuda.synchronize()

        # Benchmark forward pass
        print(f"  Timing forward pass ({NUM_ITERATIONS} iterations)...")
        forward_times = []

        for i in range(NUM_ITERATIONS):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = naive_attention(Q, K, V)
            end.record()

            torch.cuda.synchronize()
            forward_times.append(start.elapsed_time(end))  # in milliseconds

        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)

        print(f"  Forward pass: {forward_mean:.3f} ± {forward_std:.3f} ms")

        # Measure memory before backward pass
        memory_before_backward = get_memory_usage()
        print(f"  Memory before backward: {memory_before_backward:.2f} MB")

        # Create gradient for backward pass
        grad_output = torch.randn_like(output)

        # Warmup backward pass
        print(f"  Warming up backward pass...")
        for _ in range(10):
            # Zero gradients
            if Q.grad is not None:
                Q.grad.zero_()
            if K.grad is not None:
                K.grad.zero_()
            if V.grad is not None:
                V.grad.zero_()

            output = naive_attention(Q, K, V)
            output.backward(grad_output)
            torch.cuda.synchronize()

        # Benchmark backward pass
        print(f"  Timing backward pass ({NUM_ITERATIONS} iterations)...")
        backward_times = []

        for i in range(NUM_ITERATIONS):
            # Zero gradients
            if Q.grad is not None:
                Q.grad.zero_()
            if K.grad is not None:
                K.grad.zero_()
            if V.grad is not None:
                V.grad.zero_()

            # Forward pass
            output = naive_attention(Q, K, V)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output.backward(grad_output)
            end.record()

            torch.cuda.synchronize()
            backward_times.append(start.elapsed_time(end))  # in milliseconds

        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)

        print(f"  Backward pass: {backward_mean:.3f} ± {backward_std:.3f} ms")

        # Get peak memory usage
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  Peak memory: {peak_memory:.2f} MB")

        return {
            "d_model": d_model,
            "seq_length": seq_length,
            "forward_mean_ms": forward_mean,
            "forward_std_ms": forward_std,
            "backward_mean_ms": backward_mean,
            "backward_std_ms": backward_std,
            "memory_before_backward_mb": memory_before_backward,
            "peak_memory_mb": peak_memory,
            "status": "success",
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OUT OF MEMORY")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "d_model": d_model,
            "seq_length": seq_length,
            "status": "OOM",
            "error": str(e),
        }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return {
            "d_model": d_model,
            "seq_length": seq_length,
            "status": "error",
            "error": str(e),
        }


def print_summary_table(results):
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 140)
    print("BENCHMARK SUMMARY")
    print("=" * 140)
    print(
        f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<20} {'Backward (ms)':<20} {'Mem Before (MB)':<18} {'Peak Mem (MB)':<18} {'Status':<10}"
    )
    print("-" * 140)

    for result in results:
        d_model = result["d_model"]
        seq_length = result["seq_length"]

        if result["status"] == "success":
            fwd = f"{result['forward_mean_ms']:.2f} ± {result['forward_std_ms']:.2f}"
            bwd = f"{result['backward_mean_ms']:.2f} ± {result['backward_std_ms']:.2f}"
            mem_before = f"{result['memory_before_backward_mb']:.2f}"
            peak_mem = f"{result['peak_memory_mb']:.2f}"
            status = "✓"
        elif result["status"] == "OOM":
            fwd = "N/A"
            bwd = "N/A"
            mem_before = "N/A"
            peak_mem = "N/A"
            status = "OOM"
        else:
            fwd = "N/A"
            bwd = "N/A"
            mem_before = "N/A"
            peak_mem = "N/A"
            status = "ERROR"

        print(
            f"{d_model:<10} {seq_length:<10} {fwd:<20} {bwd:<20} {mem_before:<18} {peak_mem:<18} {status:<10}"
        )

    print("=" * 140)


def analyze_memory_scaling(results):
    """Analyze how memory usage scales with sequence length."""
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    # Group by d_model
    by_d_model = {}
    for result in results:
        if result["status"] == "success":
            d = result["d_model"]
            if d not in by_d_model:
                by_d_model[d] = []
            by_d_model[d].append(
                (result["seq_length"], result["memory_before_backward_mb"])
            )

    for d_model in sorted(by_d_model.keys()):
        data = sorted(by_d_model[d_model], key=lambda x: x[0])
        print(f"\nd_model = {d_model}:")
        print(f"  {'Seq Length':<12} {'Memory (MB)':<15} {'Memory/seq^2':<15}")
        print("  " + "-" * 45)

        for seq_len, mem in data:
            # Memory should scale as O(seq_len^2) for attention
            mem_per_seq2 = mem / (seq_len**2) * 1e6  # normalized
            print(f"  {seq_len:<12} {mem:<15.2f} {mem_per_seq2:<15.6f}")

    print("\nNote: For attention, memory usage should scale as O(seq_len^2) due to")
    print(
        "the attention matrix (batch_size, seq_len, seq_len) saved for backward pass."
    )
    print("=" * 80)


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("ATTENTION BENCHMARK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Head dimensions: {HEAD_DIMS}")
    print(f"  Sequence lengths: {SEQ_LENGTHS}")
    print(f"  Iterations per benchmark: {NUM_ITERATIONS}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    print("=" * 80)

    # Run benchmarks for all configurations
    results = []
    total_configs = len(HEAD_DIMS) * len(SEQ_LENGTHS)
    current = 0

    for d_model, seq_length in itertools.product(HEAD_DIMS, SEQ_LENGTHS):
        current += 1
        print(
            f"\n[{current}/{total_configs}] Configuration: d_model={d_model}, seq_length={seq_length}"
        )
        result = benchmark_attention(d_model, seq_length)
        results.append(result)

    # Print summary
    print_summary_table(results)

    # Analyze memory scaling
    analyze_memory_scaling(results)

    # Find first OOM configuration
    oom_results = [r for r in results if r["status"] == "OOM"]
    if oom_results:
        print("\n" + "=" * 80)
        print("OUT OF MEMORY ANALYSIS")
        print("=" * 80)
        first_oom = min(oom_results, key=lambda x: (x["seq_length"], x["d_model"]))
        print(f"\nFirst OOM configuration:")
        print(f"  d_model: {first_oom['d_model']}")
        print(f"  seq_length: {first_oom['seq_length']}")
        print(f"\nMemory accounting for this configuration:")

        batch = BATCH_SIZE
        seq = first_oom["seq_length"]
        d = first_oom["d_model"]

        # Memory calculations (in bytes, assuming float32 = 4 bytes)
        bytes_per_float = 4

        q_k_v_memory = 3 * batch * seq * d * bytes_per_float
        attention_scores_memory = batch * seq * seq * bytes_per_float
        attention_weights_memory = batch * seq * seq * bytes_per_float
        output_memory = batch * seq * d * bytes_per_float

        # Saved for backward: attention scores/weights and possibly inputs
        backward_memory = attention_weights_memory + q_k_v_memory

        print(f"\nForward pass memory (approximate):")
        print(f"  Q, K, V: {q_k_v_memory / (1024**2):.2f} MB")
        print(f"  Attention scores: {attention_scores_memory / (1024**2):.2f} MB")
        print(f"  Attention weights: {attention_weights_memory / (1024**2):.2f} MB")
        print(f"  Output: {output_memory / (1024**2):.2f} MB")
        print(
            f"  Total forward: {(q_k_v_memory + attention_scores_memory + attention_weights_memory + output_memory) / (1024**2):.2f} MB"
        )
        print(f"\nSaved for backward (approximate):")
        print(f"  Attention weights: {attention_weights_memory / (1024**2):.2f} MB")
        print(f"  Inputs (Q, K, V): {q_k_v_memory / (1024**2):.2f} MB")
        print(f"  Total backward: {backward_memory / (1024**2):.2f} MB")
        print(
            f"\nTotal estimated: {(q_k_v_memory + attention_scores_memory + attention_weights_memory + output_memory + backward_memory) / (1024**2):.2f} MB"
        )

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print("\nTo eliminate the O(seq_len^2) memory cost for backward pass:")
        print("1. Flash Attention: Recompute attention on-the-fly during backward pass")
        print("   - Trade computation for memory")
        print(
            "   - Only store softmax statistics (logsumexp), not full attention matrix"
        )
        print("2. Gradient checkpointing: Recompute activations during backward")
        print(
            "3. Sparse attention: Use local/strided patterns to reduce O(n^2) to O(n)"
        )
        print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    main()
