#!/usr/bin/env python3
"""
Benchmark the model sizes from Table 1 (§1.1.2).
Measures forward and backward pass timing, as well as complete training steps with AdamW optimizer.
Uses configurable warmup steps and 10 measurement steps.
Uses vocabulary size of 10,000 and batch size of 4 for all models.

Usage:
    python benchmark.py                          # Run all benchmarks (forward, backward, training)
    python benchmark.py 10                       # Run all with 10 warmup steps
    python benchmark.py small medium             # Run only small and medium models
    python benchmark.py 10 small medium          # Run small and medium with 10 warmup steps

    # Mixed precision (BF16):
    python benchmark.py --bf16                   # Run all models with BF16 mixed precision
    python benchmark.py --bf16 small medium      # Run specific models with BF16
    python benchmark.py 10 --bf16 small          # Run with 10 warmup steps and BF16

    # Compilation:
    python benchmark.py --compile                # Run all models with torch.compile
    python benchmark.py --compile --bf16 small   # Run with compilation and BF16
    python benchmark.py 10 --compile small       # Run with 10 warmup steps and compilation

    # Selective benchmarks:
    python benchmark.py --forward-only small     # Run only forward pass
    python benchmark.py --backward-only small    # Run only forward+backward pass
    python benchmark.py --training-only small    # Run only training step with optimizer
    python benchmark.py --forward-only --backward-only small  # Run forward and backward (skip training)

    # Memory profiling:
    python benchmark.py --profile-memory small   # Run with CUDA memory profiling
    python benchmark.py --profile-memory --bf16 --forward-only small  # Profile BF16 forward pass

For profiling with Nsight Systems:
    nsys profile -o profile_training --trace=cuda,nvtx python benchmark.py small

For memory profiling:
    python benchmark.py --profile-memory small
    # Upload the generated memory_snapshot_*.pickle file to https://pytorch.org/memory_viz
"""

import torch
import timeit
import numpy as np
import sys
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from cs336_systems.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# Model configurations from Table 1
MODELS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 1024,
        "seq_lengths": [128, 256, 512, 1024],
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "context_length": 1024,
        "seq_lengths": [128, 256, 512, 1024],
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "context_length": 1024,
        "seq_lengths": [128, 256, 512, 1024],
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "context_length": 1024,
        "seq_lengths": [128, 256, 512],
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
        "context_length": 1024,
        "seq_lengths": [128, 256, 512],
    },
}

# Fixed parameters (as per homework requirements)
VOCAB_SIZE = 10000
BATCH_SIZE = 4
WARMUP_STEPS = 5  # Can be overridden via command line
MEASUREMENT_STEPS = 10


def benchmark_forward_only(
    name, config, seq_length=512, warmup_steps=WARMUP_STEPS, use_bf16=False, use_compile=False
):
    """Benchmark forward pass only.

    Args:
        name: Model name for display
        config: Model configuration dict
        seq_length: Sequence length to benchmark
        warmup_steps: Number of warmup iterations
        use_bf16: If True, use BF16 mixed precision via torch.autocast
        use_compile: If True, compile model with torch.compile
    """
    precision_str = "BF16" if use_bf16 else "FP32"
    compile_str = " (Compiled)" if use_compile else ""
    print(
        f"\n{'='*60}\n{name.upper()} MODEL - FORWARD ONLY ({precision_str}{compile_str})\n{'='*60}"
    )

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
    
    # Compile model if requested
    if use_compile:
        model = torch.compile(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()
    timer = timeit.default_timer

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {config['context_length']}, Sequence length: {seq_length}")
    print(f"Precision: {precision_str}")
    print(f"Compilation: {'Enabled' if use_compile else 'Disabled'}")

    # Create autocast context for mixed precision or nullcontext for full precision
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    # Benchmark forward pass
    print("\nForward pass:")
    print("  Warmup steps:")
    with nvtx.range(f"{name}_forward_warmup"):
        for i in range(warmup_steps):
            start = timer()
            with nvtx.range(f"warmup_step_{i+1}"):
                with torch.no_grad():
                    with autocast_context:
                        model(input_ids)
                torch.cuda.synchronize()
            warmup_time = timer() - start
            print(f"    Step {i+1}: {warmup_time:.6f} seconds")

    print("  Measurement steps:")
    times = []
    with nvtx.range(f"{name}_forward_measurement"):
        for i in range(MEASUREMENT_STEPS):
            start = timer()
            with nvtx.range(f"measurement_step_{i+1}"):
                with torch.no_grad():
                    with autocast_context:
                        model(input_ids)
                torch.cuda.synchronize()
            step_time = timer() - start
            times.append(step_time)
            print(f"    Step {i+1}: {step_time:.6f} seconds")

    fwd_mean, fwd_std = np.mean(times), np.std(times)
    print(
        f"  Mean: {fwd_mean:.6f} ± {fwd_std:.6f} seconds (CV: {fwd_std/fwd_mean*100:.2f}%)"
    )

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "precision": precision_str,
        "compiled": use_compile,
        "forward_mean": fwd_mean,
        "forward_std": fwd_std,
    }


def benchmark_backward_only(
    name, config, seq_length=512, warmup_steps=WARMUP_STEPS, use_bf16=False, use_compile=False
):
    """Benchmark forward + backward pass.

    Args:
        name: Model name for display
        config: Model configuration dict
        seq_length: Sequence length to benchmark
        warmup_steps: Number of warmup iterations
        use_bf16: If True, use BF16 mixed precision via torch.autocast
        use_compile: If True, compile model with torch.compile
    """
    precision_str = "BF16" if use_bf16 else "FP32"
    compile_str = " (Compiled)" if use_compile else ""
    print(
        f"\n{'='*60}\n{name.upper()} MODEL - FORWARD + BACKWARD ({precision_str}{compile_str})\n{'='*60}"
    )

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
    
    # Compile model if requested
    if use_compile:
        model = torch.compile(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()
    timer = timeit.default_timer

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {config['context_length']}, Sequence length: {seq_length}")
    print(f"Precision: {precision_str}")
    print(f"Compilation: {'Enabled' if use_compile else 'Disabled'}")

    # Create autocast context for mixed precision or nullcontext for full precision
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    # Benchmark forward + backward pass
    print("\nForward + Backward pass:")
    model.train()
    print("  Warmup steps:")
    with nvtx.range(f"{name}_forward_backward_warmup"):
        for i in range(warmup_steps):
            start = timer()
            with nvtx.range(f"warmup_step_{i+1}"):
                with nvtx.range("forward"):
                    with autocast_context:
                        output = model(input_ids)
                with nvtx.range("backward"):
                    output.sum().backward()
                model.zero_grad()
                torch.cuda.synchronize()
            warmup_time = timer() - start
            print(f"    Step {i+1}: {warmup_time:.6f} seconds")

    print("  Measurement steps:")
    times = []
    with nvtx.range(f"{name}_forward_backward_measurement"):
        for i in range(MEASUREMENT_STEPS):
            start = timer()
            with nvtx.range(f"measurement_step_{i+1}"):
                with nvtx.range("forward"):
                    with autocast_context:
                        output = model(input_ids)
                with nvtx.range("backward"):
                    output.sum().backward()
                model.zero_grad()
                torch.cuda.synchronize()
            step_time = timer() - start
            times.append(step_time)
            print(f"    Step {i+1}: {step_time:.6f} seconds")

    bwd_mean, bwd_std = np.mean(times), np.std(times)

    print(
        f"  Mean: {bwd_mean:.6f} ± {bwd_std:.6f} seconds (CV: {bwd_std/bwd_mean*100:.2f}%)"
    )

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "precision": precision_str,
        "compiled": use_compile,
        "forward_backward_mean": bwd_mean,
        "forward_backward_std": bwd_std,
    }


def benchmark_model(
    name, config, seq_length=512, warmup_steps=WARMUP_STEPS, use_bf16=False, use_compile=False
):
    """Benchmark a single model configuration (forward + backward, legacy function).

    Args:
        name: Model name for display
        config: Model configuration dict
        seq_length: Sequence length to benchmark
        warmup_steps: Number of warmup iterations
        use_bf16: If True, use BF16 mixed precision via torch.autocast
        use_compile: If True, compile model with torch.compile
    """
    precision_str = "BF16" if use_bf16 else "FP32"
    compile_str = " (Compiled)" if use_compile else ""
    print(f"\n{'='*60}\n{name.upper()} MODEL ({precision_str}{compile_str})\n{'='*60}")

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
    
    # Compile model if requested
    if use_compile:
        model = torch.compile(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()
    timer = timeit.default_timer

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {config['context_length']}, Sequence length: {seq_length}")
    print(f"Precision: {precision_str}")
    print(f"Compilation: {'Enabled' if use_compile else 'Disabled'}")

    # Create autocast context for mixed precision or nullcontext for full precision
    # torch.autocast automatically converts operations to BF16 where appropriate
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    # Benchmark forward pass
    print("\nForward pass:")
    print("  Warmup steps:")
    with nvtx.range(f"{name}_forward_warmup"):
        for i in range(warmup_steps):
            start = timer()
            with nvtx.range(f"warmup_step_{i+1}"):
                with torch.no_grad():
                    # Use autocast context for mixed precision
                    with autocast_context:
                        model(input_ids)
                torch.cuda.synchronize()
            warmup_time = timer() - start
            print(f"    Step {i+1}: {warmup_time:.6f} seconds")

    print("  Measurement steps:")
    times = []
    with nvtx.range(f"{name}_forward_measurement"):
        for i in range(MEASUREMENT_STEPS):
            start = timer()
            with nvtx.range(f"measurement_step_{i+1}"):
                with torch.no_grad():
                    # Use autocast context for mixed precision
                    with autocast_context:
                        model(input_ids)
                torch.cuda.synchronize()
            step_time = timer() - start
            times.append(step_time)
            print(f"    Step {i+1}: {step_time:.6f} seconds")

    fwd_mean, fwd_std = np.mean(times), np.std(times)
    print(
        f"  Mean: {fwd_mean:.6f} ± {fwd_std:.6f} seconds (CV: {fwd_std/fwd_mean*100:.2f}%)"
    )

    # Benchmark forward + backward pass
    print("\nForward + Backward pass:")
    model.train()
    print("  Warmup steps:")
    with nvtx.range(f"{name}_forward_backward_warmup"):
        for i in range(warmup_steps):
            start = timer()
            with nvtx.range(f"warmup_step_{i+1}"):
                with nvtx.range("forward"):
                    # Use autocast for forward pass in mixed precision
                    with autocast_context:
                        output = model(input_ids)
                with nvtx.range("backward"):
                    # Backward pass automatically handles mixed precision gradients
                    output.sum().backward()
                model.zero_grad()
                torch.cuda.synchronize()
            warmup_time = timer() - start
            print(f"    Step {i+1}: {warmup_time:.6f} seconds")

    print("  Measurement steps:")
    times = []
    with nvtx.range(f"{name}_forward_backward_measurement"):
        for i in range(MEASUREMENT_STEPS):
            start = timer()
            with nvtx.range(f"measurement_step_{i+1}"):
                with nvtx.range("forward"):
                    # Use autocast for forward pass in mixed precision
                    with autocast_context:
                        output = model(input_ids)
                with nvtx.range("backward"):
                    # Backward pass automatically handles mixed precision gradients
                    output.sum().backward()
                model.zero_grad()
                torch.cuda.synchronize()
            step_time = timer() - start
            times.append(step_time)
            print(f"    Step {i+1}: {step_time:.6f} seconds")

    bwd_mean, bwd_std = np.mean(times), np.std(times)
    bwd_only = bwd_mean - fwd_mean

    print(
        f"  Mean: {bwd_mean:.6f} ± {bwd_std:.6f} seconds (CV: {bwd_std/bwd_mean*100:.2f}%)"
    )
    print(f"  Backward-only: {bwd_only:.6f} seconds ({bwd_only/fwd_mean:.2f}x forward)")

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "precision": precision_str,
        "compiled": use_compile,
        "forward_mean": fwd_mean,
        "forward_std": fwd_std,
        "backward_mean": bwd_only,
        "backward_std": bwd_std,
    }


def benchmark_training_step(
    name, config, seq_length=512, warmup_steps=WARMUP_STEPS, use_bf16=False, use_compile=False
):
    """Benchmark a complete training step with AdamW optimizer (forward + loss + backward + optimizer step).

    Args:
        name: Model name for display
        config: Model configuration dict
        seq_length: Sequence length to benchmark
        warmup_steps: Number of warmup iterations
        use_bf16: If True, use BF16 mixed precision via torch.autocast
        use_compile: If True, compile model with torch.compile
    """
    precision_str = "BF16" if use_bf16 else "FP32"
    compile_str = " (Compiled)" if use_compile else ""
    print(
        f"\n{'='*60}\n{name.upper()} MODEL - TRAINING STEP WITH ADAMW ({precision_str}{compile_str})\n{'='*60}"
    )

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
    
    # Compile model if requested
    if use_compile:
        model = torch.compile(model)

    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()
    # Create target labels for loss computation
    target_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length)).cuda()

    timer = timeit.default_timer

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {config['context_length']}, Sequence length: {seq_length}")
    print(f"Precision: {precision_str}")
    print(f"Compilation: {'Enabled' if use_compile else 'Disabled'}")

    # Create autocast context for mixed precision or nullcontext for full precision
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    # Benchmark complete training step (forward + loss + backward + optimizer step)
    print("\nComplete training step (forward + loss + backward + optimizer):")
    model.train()

    print("  Warmup steps:")
    with nvtx.range(f"{name}_training_step_warmup"):
        for i in range(warmup_steps):
            start = timer()
            with nvtx.range(f"warmup_step_{i+1}"):
                with nvtx.range("forward"):
                    # Use autocast for forward pass in mixed precision
                    with autocast_context:
                        logits = model(input_ids)
                with nvtx.range("loss"):
                    # Compute cross-entropy loss (also benefits from autocast)
                    with autocast_context:
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, VOCAB_SIZE), target_ids.view(-1)
                        )
                with nvtx.range("backward"):
                    # Backward pass automatically handles mixed precision gradients
                    loss.backward()
                with nvtx.range("optimizer_step"):
                    # Optimizer operates in FP32 on the gradients
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
            warmup_time = timer() - start
            print(f"    Step {i+1}: {warmup_time:.6f} seconds")

    print("  Measurement steps:")
    times = []
    with nvtx.range(f"{name}_training_step_measurement"):
        for i in range(MEASUREMENT_STEPS):
            start = timer()
            with nvtx.range(f"measurement_step_{i+1}"):
                with nvtx.range("forward"):
                    # Use autocast for forward pass in mixed precision
                    with autocast_context:
                        logits = model(input_ids)
                with nvtx.range("loss"):
                    # Compute cross-entropy loss (also benefits from autocast)
                    with autocast_context:
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, VOCAB_SIZE), target_ids.view(-1)
                        )
                with nvtx.range("backward"):
                    # Backward pass automatically handles mixed precision gradients
                    loss.backward()
                with nvtx.range("optimizer_step"):
                    # Optimizer operates in FP32 on the gradients
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
            step_time = timer() - start
            times.append(step_time)
            print(f"    Step {i+1}: {step_time:.6f} seconds")

    train_mean, train_std = np.mean(times), np.std(times)
    print(
        f"  Mean: {train_mean:.6f} ± {train_std:.6f} seconds (CV: {train_std/train_mean*100:.2f}%)"
    )

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "precision": precision_str,
        "compiled": use_compile,
        "training_step_mean": train_mean,
        "training_step_std": train_std,
    }


def main():
    # Parse command line arguments
    warmup_steps = WARMUP_STEPS
    models_to_run = None  # None means run all models
    use_bf16 = False
    use_compile = False
    profile_memory = False

    # Benchmark selection flags (default: run all)
    run_forward = False
    run_backward = False
    run_training = False

    # Parse arguments
    args = sys.argv[1:]

    # Check for --bf16 flag
    if "--bf16" in args:
        use_bf16 = True
        args.remove("--bf16")

    # Check for --compile flag
    if "--compile" in args:
        use_compile = True
        args.remove("--compile")

    # Check for --profile-memory flag
    if "--profile-memory" in args:
        profile_memory = True
        args.remove("--profile-memory")

    # Check for benchmark selection flags
    if "--forward-only" in args:
        run_forward = True
        args.remove("--forward-only")

    if "--backward-only" in args:
        run_backward = True
        args.remove("--backward-only")

    if "--training-only" in args:
        run_training = True
        args.remove("--training-only")

    # If no specific benchmark selected, run all
    if not (run_forward or run_backward or run_training):
        run_forward = run_backward = run_training = True

    if len(args) > 0:
        # Check if first argument is a number (warmup steps)
        if args[0].isdigit():
            warmup_steps = int(args[0])
            # Second argument onwards: specific models to run (optional)
            if len(args) > 1:
                models_to_run = args[1:]
        else:
            # If first arg is not a number, treat all args as model names
            models_to_run = args
            warmup_steps = WARMUP_STEPS

    # Validate model names if specified
    if models_to_run:
        invalid_models = [m for m in models_to_run if m not in MODELS]
        if invalid_models:
            print(f"Error: Invalid model name(s): {', '.join(invalid_models)}")
            print(f"Available models: {', '.join(MODELS.keys())}")
            sys.exit(1)
        models_dict = {name: MODELS[name] for name in models_to_run}
    else:
        models_dict = MODELS

    print("=" * 60)
    print("BENCHMARKING MODEL SIZES FROM TABLE 1")
    print("=" * 60)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Measurement steps: {MEASUREMENT_STEPS}")
    print(
        f"Precision: {'BF16 (Mixed Precision)' if use_bf16 else 'FP32 (Full Precision)'}"
    )
    print(f"Compilation: {'Enabled' if use_compile else 'Disabled'}")
    print(f"Memory profiling: {'Enabled' if profile_memory else 'Disabled'}")

    # Show which benchmarks will run
    benchmarks_to_run = []
    if run_forward:
        benchmarks_to_run.append("Forward")
    if run_backward:
        benchmarks_to_run.append("Forward+Backward")
    if run_training:
        benchmarks_to_run.append("Training")
    print(f"Benchmarks: {', '.join(benchmarks_to_run)}")
    print(f"Models to run: {', '.join(models_dict.keys())}")

    forward_results = []
    backward_results = []
    training_results = []

    # Start memory profiling if requested
    if profile_memory:
        print("\n" + "=" * 60)
        print("Starting CUDA memory profiling...")
        print("=" * 60)
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    for name, config in models_dict.items():
        for seq in config["seq_lengths"]:
            model_name = f"{name}_seq{seq}"

            # Run forward-only benchmark if requested
            if run_forward:
                result = benchmark_forward_only(
                    model_name,
                    config,
                    seq_length=seq,
                    warmup_steps=warmup_steps,
                    use_bf16=use_bf16,
                    use_compile=use_compile,
                )
                forward_results.append(result)

            # Run forward+backward benchmark if requested
            if run_backward:
                result = benchmark_backward_only(
                    model_name,
                    config,
                    seq_length=seq,
                    warmup_steps=warmup_steps,
                    use_bf16=use_bf16,
                    use_compile=use_compile,
                )
                backward_results.append(result)

            # Run complete training step with AdamW if requested
            if run_training:
                training_result = benchmark_training_step(
                    model_name,
                    config,
                    seq_length=seq,
                    warmup_steps=warmup_steps,
                    use_bf16=use_bf16,
                    use_compile=use_compile,
                )
                training_results.append(training_result)

    # Save memory snapshot if profiling was enabled
    if profile_memory:
        precision_str = "bf16" if use_bf16 else "fp32"
        snapshot_file = f"memory_snapshot_{precision_str}.pickle"
        print("\n" + "=" * 60)
        print(f"Saving memory snapshot to {snapshot_file}...")
        print("=" * 60)
        torch.cuda.memory._dump_snapshot(snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None)
        print("✓ Memory snapshot saved!")
        print("  Upload to: https://pytorch.org/memory_viz")

    # Summary sections
    if forward_results:
        print(f"\n{'='*60}\nFORWARD PASS SUMMARY\n{'='*60}")
        print(f"{'Model':<15} {'Precision':<10} {'Params':<12} {'Forward (s)':<15}")
        print("-" * 60)
        for r in forward_results:
            print(
                f"{r['name']:<15} {r['precision']:<10} {r['params']:>11,} "
                f"{r['forward_mean']:>8.6f}±{r['forward_std']:.6f}"
            )
        print("=" * 60)

    if backward_results:
        print(f"\n{'='*60}\nFORWARD + BACKWARD SUMMARY\n{'='*60}")
        print(f"{'Model':<15} {'Precision':<10} {'Params':<12} {'Fwd+Bwd (s)':<15}")
        print("-" * 60)
        for r in backward_results:
            print(
                f"{r['name']:<15} {r['precision']:<10} {r['params']:>11,} "
                f"{r['forward_backward_mean']:>8.6f}±{r['forward_backward_std']:.6f}"
            )
        print("=" * 60)

    if training_results:
        print(f"\n{'='*60}\nTRAINING STEP SUMMARY (with AdamW)\n{'='*60}")
        print(
            f"{'Model':<15} {'Precision':<10} {'Params':<12} {'Training Step (s)':<20}"
        )
        print("-" * 60)
        for r in training_results:
            print(
                f"{r['name']:<15} {r['precision']:<10} {r['params']:>11,} "
                f"{r['training_step_mean']:>8.6f}±{r['training_step_std']:.6f}"
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
