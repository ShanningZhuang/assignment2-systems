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

# Create compiled version of attention (using default mode as per homework instructions)
compiled_attention = torch.compile(naive_attention)


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


def benchmark_attention(
    d_model, seq_length, use_compiled=False, version_name="uncompiled"
):
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

        print(
            f"\nBenchmarking {version_name} d_model={d_model}, seq_length={seq_length}"
        )
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

        # Select attention function
        attention_fn = compiled_attention if use_compiled else naive_attention

        # Warmup
        print(f"  Warming up...")
        for _ in range(10):
            output = attention_fn(Q, K, V)
            torch.cuda.synchronize()

        # Benchmark forward pass
        print(f"  Timing forward pass ({NUM_ITERATIONS} iterations)...")
        forward_times = []

        for _ in range(NUM_ITERATIONS):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = attention_fn(Q, K, V)
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

            output = attention_fn(Q, K, V)
            output.backward(grad_output)
            torch.cuda.synchronize()

        # Benchmark backward pass
        print(f"  Timing backward pass ({NUM_ITERATIONS} iterations)...")
        backward_times = []

        for _ in range(NUM_ITERATIONS):
            # Zero gradients
            if Q.grad is not None:
                Q.grad.zero_()
            if K.grad is not None:
                K.grad.zero_()
            if V.grad is not None:
                V.grad.zero_()

            # Forward pass
            output = attention_fn(Q, K, V)

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
            "version": version_name,
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
            "version": version_name,
            "status": "OOM",
            "error": str(e),
        }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return {
            "d_model": d_model,
            "seq_length": seq_length,
            "version": version_name,
            "status": "error",
            "error": str(e),
        }


def print_summary_table(results):
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 160)
    print("BENCHMARK SUMMARY")
    print("=" * 160)
    print(
        f"{'Version':<12} {'d_model':<10} {'seq_len':<10} {'Forward (ms)':<20} {'Backward (ms)':<20} {'Mem Before (MB)':<18} {'Peak Mem (MB)':<18} {'Status':<10}"
    )
    print("-" * 160)

    for result in results:
        version = result["version"]
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
            f"{version:<12} {d_model:<10} {seq_length:<10} {fwd:<20} {bwd:<20} {mem_before:<18} {peak_mem:<18} {status:<10}"
        )

    print("=" * 160)


def print_performance_comparison(results):
    """Print performance comparison between compiled and uncompiled versions."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON ANALYSIS")
    print("=" * 80)

    # Group results by configuration
    config_pairs = {}
    for result in results:
        if result["status"] == "success":
            key = (result["d_model"], result["seq_length"])
            if key not in config_pairs:
                config_pairs[key] = {}
            config_pairs[key][result["version"]] = result

    print(
        f"{'Config':<20} {'Forward Speedup':<18} {'Backward Speedup':<19} {'Memory Diff (MB)':<18}"
    )
    print("-" * 80)

    forward_speedups = []
    backward_speedups = []

    for (d_model, seq_length), versions in sorted(config_pairs.items()):
        if "uncompiled" in versions and "compiled" in versions:
            uncompiled = versions["uncompiled"]
            compiled = versions["compiled"]

            fwd_speedup = uncompiled["forward_mean_ms"] / compiled["forward_mean_ms"]
            bwd_speedup = uncompiled["backward_mean_ms"] / compiled["backward_mean_ms"]
            mem_diff = compiled["peak_memory_mb"] - uncompiled["peak_memory_mb"]

            forward_speedups.append(fwd_speedup)
            backward_speedups.append(bwd_speedup)

            config_str = f"d={d_model}, seq={seq_length}"
            print(
                f"{config_str:<20} {fwd_speedup:<18.2f} {bwd_speedup:<18.2f} {mem_diff:<18.2f}"
            )

    if forward_speedups:
        print("-" * 80)
        print(
            f"Average forward speedup: {np.mean(forward_speedups):.2f}x (std: {np.std(forward_speedups):.2f})"
        )
        print(
            f"Average backward speedup: {np.mean(backward_speedups):.2f}x (std: {np.std(backward_speedups):.2f})"
        )
        print(f"Max forward speedup: {max(forward_speedups):.2f}x")
        print(f"Max backward speedup: {max(backward_speedups):.2f}x")

    print("=" * 80)


def analyze_memory_scaling(results):
    """Analyze how memory usage scales with sequence length."""
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    # Group by d_model and version
    by_d_model = {}
    for result in results:
        if result["status"] == "success":
            d = result["d_model"]
            version = result["version"]
            key = f"{d} ({version})"
            if key not in by_d_model:
                by_d_model[key] = []
            by_d_model[key].append(
                (result["seq_length"], result["memory_before_backward_mb"])
            )

    for d_model_version in sorted(by_d_model.keys()):
        data = sorted(by_d_model[d_model_version], key=lambda x: x[0])
        print(f"\n{d_model_version}:")
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


def compare_attention_versions(d_model, seq_length, config_num, total_configs):
    """Compare compiled vs uncompiled attention for a single configuration."""
    print(
        f"\n[{config_num}/{total_configs}] Configuration: d_model={d_model}, seq_length={seq_length}"
    )

    results = []

    # Benchmark uncompiled version
    print("  Testing uncompiled version...")
    result_uncompiled = benchmark_attention(
        d_model, seq_length, use_compiled=False, version_name="uncompiled"
    )
    results.append(result_uncompiled)

    # Only benchmark compiled version if uncompiled succeeded
    if result_uncompiled["status"] == "success":
        print("  Testing compiled version...")
        result_compiled = benchmark_attention(
            d_model, seq_length, use_compiled=True, version_name="compiled"
        )
        results.append(result_compiled)

        # Print speedup comparison
        if result_compiled["status"] == "success":
            fwd_speedup = (
                result_uncompiled["forward_mean_ms"]
                / result_compiled["forward_mean_ms"]
            )
            bwd_speedup = (
                result_uncompiled["backward_mean_ms"]
                / result_compiled["backward_mean_ms"]
            )
            print(f"    Forward speedup: {fwd_speedup:.2f}x")
            print(f"    Backward speedup: {bwd_speedup:.2f}x")
    else:
        print("  Skipping compiled version due to uncompiled failure")
        # Add placeholder result for compiled version
        result_compiled = {
            "d_model": d_model,
            "seq_length": seq_length,
            "version": "compiled",
            "status": "skipped",
            "error": "Uncompiled version failed",
        }
        results.append(result_compiled)

    return results


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("ATTENTION BENCHMARK - COMPILED vs UNCOMPILED")
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
    print("  Compilation mode: default")
    print("=" * 80)

    # Run benchmarks for all configurations
    results = []
    total_configs = len(HEAD_DIMS) * len(SEQ_LENGTHS)
    current = 0

    for d_model, seq_length in itertools.product(HEAD_DIMS, SEQ_LENGTHS):
        current += 1
        config_results = compare_attention_versions(
            d_model, seq_length, current, total_configs
        )
        results.extend(config_results)

    # Print summary
    print_summary_table(results)

    # Analyze memory scaling
    analyze_memory_scaling(results)

    # Performance comparison analysis
    print_performance_comparison(results)

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
        print(f"  version: {first_oom['version']}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    main()
