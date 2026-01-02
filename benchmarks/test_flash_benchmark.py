#!/usr/bin/env python3
"""
Quick test script for FlashAttention benchmark.

This runs a minimal subset of configurations to verify everything works
before running the full benchmark suite.
"""

import torch
import sys
import argparse
from benchmark_flash_attention import (
    benchmark_configuration,
    print_results_table,
    save_results_csv
)


def main():
    """Run a quick test with minimal configurations."""
    parser = argparse.ArgumentParser(description="Quick test for FlashAttention benchmark")
    parser.add_argument('--no-backward', action='store_true', help='Skip backward pass testing')
    parser.add_argument('--no-causal', action='store_true', help='Skip causal masking')
    parser.add_argument('--causal-only', action='store_true', help='Test only causal masking')
    args = parser.parse_args()
    
    print("=" * 80)
    print("FLASHATTENTION BENCHMARK - QUICK TEST")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Determine causal options
    if args.causal_only:
        causal_options = [True]
    elif args.no_causal:
        causal_options = [False]
    else:
        causal_options = [False, True]
    
    test_backward = not args.no_backward
    
    print(f"Test backward: {test_backward}")
    print(f"Causal options: {causal_options}")
    print("=" * 80)
    
    # Test configurations: small subset
    base_configs = [
        (128, 32, torch.float32),
        (512, 64, torch.float32),
        (1024, 64, torch.bfloat16),
    ]
    
    # Expand with causal options
    test_configs = []
    for seq_len, d_model, dtype in base_configs:
        for is_causal in causal_options:
            test_configs.append((seq_len, d_model, dtype, is_causal))
    
    print(f"\nTesting {len(test_configs)} configurations...")
    
    results = []
    for idx, (seq_len, d_model, dtype, is_causal) in enumerate(test_configs, 1):
        result = benchmark_configuration(
            seq_len, d_model, dtype, is_causal, test_backward, idx, len(test_configs)
        )
        results.append(result)
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results_csv(results, "flash_attention_benchmark_test.csv")
    
    # Check if any succeeded
    success_count = sum(1 for r in results if r.get('pytorch_status') == 'success')
    print(f"\n✓ Test completed: {success_count}/{len(test_configs)} configurations succeeded")
    
    if success_count > 0:
        print("\n✓ Benchmark is working! You can now run the full benchmark:")
        print("  python benchmarks/benchmark_flash_attention.py")
        if args.no_backward:
            print("  (Add --no-backward if backward pass is not implemented)")
        if args.no_causal:
            print("  (Add --no-causal if causal masking is not implemented)")
    else:
        print("\n✗ All configurations failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

