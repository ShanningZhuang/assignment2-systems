#!/usr/bin/env python3
"""
Benchmark the model sizes from Table 1 (§1.1.2).
Measures forward and backward pass timing with 5 warmup steps and 10 measurement steps.
Uses vocabulary size of 10,000 and batch size of 4 for all models.
"""

import torch
import timeit
import numpy as np
from cs336_basics.model import BasicsTransformerLM

# Model configurations from Table 1
MODELS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 1024,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "context_length": 1024,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "context_length": 2048,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "context_length": 2048,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
        "context_length": 2048,
    },
}

# Fixed parameters (as per homework requirements)
VOCAB_SIZE = 10000
BATCH_SIZE = 4
WARMUP_STEPS = 5
MEASUREMENT_STEPS = 10


def benchmark_model(name, config, seq_length=512):
    """Benchmark a single model configuration."""
    print(f"\n{'='*60}\n{name.upper()} MODEL\n{'='*60}")

    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).cuda()

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()
    timer = timeit.default_timer

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {config['context_length']}, Sequence length: {seq_length}")

    # Benchmark forward pass
    print("\nForward pass:")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model(input_ids)
        torch.cuda.synchronize()

    times = []
    for _ in range(MEASUREMENT_STEPS):
        start = timer()
        with torch.no_grad():
            model(input_ids)
        torch.cuda.synchronize()
        times.append(timer() - start)

    fwd_mean, fwd_std = np.mean(times), np.std(times)
    print(f"  {fwd_mean:.6f} ± {fwd_std:.6f} seconds (CV: {fwd_std/fwd_mean*100:.2f}%)")

    # Benchmark forward + backward pass
    print("Forward + Backward pass:")
    model.train()
    for _ in range(WARMUP_STEPS):
        model(input_ids).sum().backward()
        model.zero_grad()
        torch.cuda.synchronize()

    times = []
    for _ in range(MEASUREMENT_STEPS):
        start = timer()
        model(input_ids).sum().backward()
        model.zero_grad()
        torch.cuda.synchronize()
        times.append(timer() - start)

    bwd_mean, bwd_std = np.mean(times), np.std(times)
    bwd_only = bwd_mean - fwd_mean

    print(f"  {bwd_mean:.6f} ± {bwd_std:.6f} seconds (CV: {bwd_std/bwd_mean*100:.2f}%)")
    print(f"  Backward-only: {bwd_only:.6f} seconds ({bwd_only/fwd_mean:.2f}x forward)")

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "forward_mean": fwd_mean,
        "forward_std": fwd_std,
        "backward_mean": bwd_only,
        "backward_std": bwd_std,
    }


def main():
    print("=" * 60)
    print("BENCHMARKING MODEL SIZES FROM TABLE 1")
    print("=" * 60)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measurement steps: {MEASUREMENT_STEPS}")

    results = []
    for name, config in MODELS.items():
        # Use smaller sequence length for 2.7B model to fit in memory
        seq = 256 if name == "2.7B" else 512
        result = benchmark_model(name, config, seq_length=seq)
        results.append(result)

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Model':<10} {'Params':<12} {'Forward (s)':<15} {'Backward (s)':<15}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<10} {r['params']:>11,} "
            f"{r['forward_mean']:>8.6f}±{r['forward_std']:.6f} "
            f"{r['backward_mean']:>8.6f}±{r['backward_std']:.6f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
