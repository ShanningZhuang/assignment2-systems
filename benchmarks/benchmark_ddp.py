#!/usr/bin/env python3
"""
Benchmark DDP training with the naive DDPIndividualParameters implementation.

This script measures:
- Total time per training step
- Time spent on gradient communication (all-reduce operations)
- Proportion of communication time vs total time
- Breakdown of forward pass, backward pass, communication, and optimizer step

The naive DDP implementation (DDPIndividualParameters) issues a separate all-reduce
operation for each parameter tensor after the backward pass. This benchmark helps
understand the overhead of data parallel training.

Usage:
    python benchmark_ddp.py                    # Default: 2 GPUs, XL model, 20 iterations
    python benchmark_ddp.py --world-size 4     # Use 4 GPUs
    python benchmark_ddp.py --num-iterations 50 # Run 50 iterations
    python benchmark_ddp.py --model-size large # Use large model instead of XL
    python benchmark_ddp.py --seq-length 256   # Use sequence length of 256

For the assignment requirements (1 node x 2 GPUs, XL model):
    python benchmark_ddp.py --model-size xl --world-size 2
"""

import os
import sys
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity, schedule

from cs336_systems.model import BasicsTransformerLM
from cs336_systems.ddp import DDPIndividualParameters, DDPNaive, DDPBucketed
from cs336_basics.optimizer import AdamW


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
        "context_length": 1024,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "context_length": 1024,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
        "context_length": 1024,
    },
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4
ROPE_THETA = 10000.0


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def benchmark_ddp_training(rank, world_size, model_size, seq_length, num_warmup, num_iterations,
                          enable_profiler=False, ddp_mode="overlap", bucket_size_mb=25.0):
    """
    Benchmark DDP training with timing breakdown.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_size: Model size name (e.g., 'xl')
        seq_length: Sequence length for input
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        enable_profiler: Enable PyTorch profiler
        ddp_mode: DDP implementation to use:
            - "overlap": DDPIndividualParameters with async all-reduce (overlap)
            - "naive": DDPNaive with sync all-reduce (no overlap)
            - "bucketed": DDPBucketed with gradient bucketing + overlap
            - "native": PyTorch native DDP (bucketing + overlap)
        bucket_size_mb: Bucket size in MB (only used for "bucketed" mode)
    """
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    # Get model configuration
    config = MODELS[model_size]
    
    # Create model
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=ROPE_THETA,
    ).to(device)

    # Wrap model with DDP based on mode
    if ddp_mode == "native":
        from torch.nn.parallel import DistributedDataParallel as NativeDDP
        ddp_model = NativeDDP(model, device_ids=[rank])
    elif ddp_mode == "naive":
        ddp_model = DDPNaive(model)
    elif ddp_mode == "bucketed":
        ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    else:  # "overlap" (default)
        ddp_model = DDPIndividualParameters(model)
    
    # Create optimizer
    optimizer = AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=0.1)
    
    # Create dummy input data
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length), device=device)
    target_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_length), device=device)
    
    # Warmup runs
    for _ in range(num_warmup):
        if ddp_mode == "bucketed":
            ddp_model.reset_buckets()
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB_SIZE), target_ids.view(-1)
        )
        loss.backward()
        if ddp_mode != "native":
            ddp_model.finish_gradient_synchronization()
        optimizer.step()
    
    # Synchronize all processes before timing
    dist.barrier()
    torch.cuda.synchronize()
    
    # Timed runs
    total_times = []
    forward_times = []
    backward_times = []
    comm_times = []
    optimizer_times = []

    # Setup PyTorch profiler if enabled
    prof = None
    if enable_profiler and rank == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"ddp_trace_rank{rank}.json"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    for iter_idx in range(num_iterations):
        with nvtx.range(f"iteration_{iter_idx}"):
            # Reset buckets for bucketed mode
            if ddp_mode == "bucketed":
                ddp_model.reset_buckets()

            # Zero gradients
            with nvtx.range("zero_grad"):
                optimizer.zero_grad()

            # Synchronize before each iteration
            dist.barrier()
            torch.cuda.synchronize()

            # Measure forward pass
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            iter_start = torch.cuda.Event(enable_timing=True)
            iter_start.record()

            start_event.record()
            with nvtx.range("forward"):
                logits = ddp_model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), target_ids.view(-1)
                )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)

            # Measure backward pass (includes async all-reduce launches)
            start_event.record()
            with nvtx.range("backward"):
                loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)

            # Measure communication time (waiting for all-reduces)
            start_event.record()
            if ddp_mode != "native":
                ddp_model.finish_gradient_synchronization()
            end_event.record()
            torch.cuda.synchronize()
            comm_time = start_event.elapsed_time(end_event)

            # Measure optimizer step
            start_event.record()
            with nvtx.range("optimizer_step"):
                optimizer.step()
            end_event.record()
            torch.cuda.synchronize()
            optimizer_time = start_event.elapsed_time(end_event)

            iter_end = torch.cuda.Event(enable_timing=True)
            iter_end.record()
            torch.cuda.synchronize()
            total_time = iter_start.elapsed_time(iter_end)

            total_times.append(total_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            comm_times.append(comm_time)
            optimizer_times.append(optimizer_time)

        if prof:
            prof.step()

    if prof:
        prof.stop()
    
    # Gather timings from all ranks
    all_total_times = [None] * world_size
    all_forward_times = [None] * world_size
    all_backward_times = [None] * world_size
    all_comm_times = [None] * world_size
    all_optimizer_times = [None] * world_size
    
    dist.all_gather_object(all_total_times, total_times)
    dist.all_gather_object(all_forward_times, forward_times)
    dist.all_gather_object(all_backward_times, backward_times)
    dist.all_gather_object(all_comm_times, comm_times)
    dist.all_gather_object(all_optimizer_times, optimizer_times)
    
    # Report results from rank 0
    if rank == 0:
        # Convert to numpy arrays for easier analysis
        all_total_times = np.array(all_total_times)  # Shape: (world_size, num_iterations)
        all_forward_times = np.array(all_forward_times)
        all_backward_times = np.array(all_backward_times)
        all_comm_times = np.array(all_comm_times)
        all_optimizer_times = np.array(all_optimizer_times)
        
        # Calculate statistics across all ranks
        avg_total = all_total_times.mean()
        std_total = all_total_times.std()
        avg_forward = all_forward_times.mean()
        avg_backward = all_backward_times.mean()
        avg_comm = all_comm_times.mean()
        avg_optimizer = all_optimizer_times.mean()
        
        # Calculate proportion of communication time
        comm_proportion = (avg_comm / avg_total) * 100
        
        # Calculate model parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_params_mb = num_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        print("\n" + "=" * 70)
        print(f"DDP Training Benchmark Results")
        print("=" * 70)
        print(f"Model size: {model_size.upper()}")
        print(f"  d_model: {config['d_model']}, num_layers: {config['num_layers']}")
        print(f"  num_heads: {config['num_heads']}, d_ff: {config['d_ff']}")
        print(f"  Total parameters: {num_params:,} ({num_params_mb:.2f} MB)")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Sequence length: {seq_length}")
        print(f"Number of iterations: {num_iterations}")
        print(f"\nTiming breakdown (averaged across all ranks and iterations):")
        print(f"  Total time per step:     {avg_total:.3f} ± {std_total:.3f} ms")
        print(f"    Forward pass:          {avg_forward:.3f} ms ({avg_forward/avg_total*100:.1f}%)")
        print(f"    Backward pass:         {avg_backward:.3f} ms ({avg_backward/avg_total*100:.1f}%)")
        print(f"    Gradient communication: {avg_comm:.3f} ms ({comm_proportion:.1f}%)")
        print(f"    Optimizer step:        {avg_optimizer:.3f} ms ({avg_optimizer/avg_total*100:.1f}%)")
        print(f"\nCommunication overhead: {comm_proportion:.2f}% of total training time")
        
        # Per-rank statistics
        print(f"\nPer-rank average total time:")
        for r in range(world_size):
            rank_avg = all_total_times[r].mean()
            rank_std = all_total_times[r].std()
            print(f"  Rank {r}: {rank_avg:.3f} ± {rank_std:.3f} ms")
        
        print(f"\nPer-rank average communication time:")
        for r in range(world_size):
            rank_avg = all_comm_times[r].mean()
            rank_std = all_comm_times[r].std()
            print(f"  Rank {r}: {rank_avg:.3f} ± {rank_std:.3f} ms")
        
        print("=" * 70)
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark DDP training with naive individual parameter all-reduce"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large", "xl", "2.7B"],
        default="xl",
        help="Model size to benchmark (default: xl)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for input (default: 512)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler (generates JSON trace)"
    )
    parser.add_argument(
        "--ddp-mode",
        type=str,
        choices=["overlap", "naive", "bucketed", "native"],
        default="overlap",
        help="DDP implementation: overlap (async all-reduce), naive (sync all-reduce), bucketed (bucketed async), native (PyTorch DDP)"
    )
    parser.add_argument(
        "--bucket-size-mb",
        type=float,
        default=25.0,
        help="Bucket size in MB for bucketed DDP mode (default: 25.0)"
    )
    args = parser.parse_args()
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. DDP benchmarking requires GPUs.")
        sys.exit(1)
    
    if torch.cuda.device_count() < args.world_size:
        print(f"Error: Requested {args.world_size} GPUs, but only {torch.cuda.device_count()} available.")
        sys.exit(1)
    
    # Validate sequence length
    model_config = MODELS[args.model_size]
    if args.seq_length > model_config["context_length"]:
        print(f"Error: Sequence length {args.seq_length} exceeds model context length {model_config['context_length']}")
        sys.exit(1)
    
    ddp_types = {
        "overlap": "Custom DDP (Individual Parameters + Async Overlap)",
        "naive": "Custom DDP (Naive Sync, No Overlap)",
        "bucketed": f"Custom DDP (Bucketed {args.bucket_size_mb}MB + Async Overlap)",
        "native": "Native PyTorch DDP (Bucketing + Overlap)",
    }
    print(f"Starting DDP benchmark with {args.world_size} GPUs...")
    print(f"Model: {args.model_size}, Sequence length: {args.seq_length}")
    print(f"DDP Implementation: {ddp_types[args.ddp_mode]}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iterations}")
    print(f"Profiling: {'Enabled' if args.profile else 'Disabled'}")

    # Spawn processes for distributed training
    mp.spawn(
        fn=benchmark_ddp_training,
        args=(args.world_size, args.model_size, args.seq_length, args.num_warmup, args.num_iterations,
              args.profile, args.ddp_mode, args.bucket_size_mb),
        nprocs=args.world_size,
        join=True
    )

