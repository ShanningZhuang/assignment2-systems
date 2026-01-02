import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def benchmark_all_reduce(rank, world_size, tensor_size_mb, backend, num_warmup=5, num_iterations=20):
    """Benchmark all-reduce operation with timing."""
    setup(rank, world_size, backend)
    
    # Determine device based on backend
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    # Calculate number of float32 elements needed for the desired size
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)
    data = torch.rand(num_elements, dtype=torch.float32, device=device)
    
    # Warmup runs
    for _ in range(num_warmup):
        dist.all_reduce(data.clone(), async_op=False)
    
    # Synchronize all processes before timing
    dist.barrier()
    
    # Timed runs - collect timings from all ranks
    timings = []
    for _ in range(num_iterations):
        test_data = data.clone()
        
        # Synchronize before each iteration
        dist.barrier()
        
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            dist.all_reduce(test_data, async_op=False)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            timings.append(elapsed_ms)
        else:
            import time
            start_time = time.perf_counter()
            dist.all_reduce(test_data, async_op=False)
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            timings.append(elapsed_ms)
    
    # Gather timings from all ranks
    all_timings = [None] * world_size
    dist.all_gather_object(all_timings, timings)
    
    # Report results from rank 0
    if rank == 0:
        import numpy as np
        
        # Convert to numpy array for easier analysis
        all_timings_array = np.array(all_timings)  # Shape: (world_size, num_iterations)
        
        # Calculate statistics across all ranks
        avg_time_per_rank = all_timings_array.mean(axis=1)
        overall_avg_time = all_timings_array.mean()
        overall_min_time = all_timings_array.min()
        overall_max_time = all_timings_array.max()
        std_time = all_timings_array.std()
        
        # Calculate bandwidth (GB/s)
        # All-reduce transfers (world_size - 1) / world_size * 2 * data_size
        # Using ring all-reduce algorithm assumption
        data_size_gb = tensor_size_mb / 1024
        bandwidth_gbps = (data_size_gb * 2 * (world_size - 1) / world_size) / (overall_avg_time / 1000)
        
        print("\n" + "=" * 60)
        print(f"All-Reduce Benchmark Results")
        print("=" * 60)
        print(f"Backend: {backend.upper()}")
        print(f"Device: {'GPU (CUDA)' if use_cuda else 'CPU'}")
        print(f"World size: {world_size}")
        print(f"Tensor size: {tensor_size_mb} MB ({num_elements} float32 elements)")
        print(f"Number of iterations: {num_iterations}")
        print(f"\nAggregated statistics across all ranks:")
        print(f"  Overall average time: {overall_avg_time:.3f} ms")
        print(f"  Overall min time: {overall_min_time:.3f} ms")
        print(f"  Overall max time: {overall_max_time:.3f} ms")
        print(f"  Standard deviation: {std_time:.3f} ms")
        print(f"\nPer-rank average times:")
        for r in range(world_size):
            print(f"  Rank {r}: {avg_time_per_rank[r]:.3f} ms")
        print(f"\nEffective bandwidth: {bandwidth_gbps:.2f} GB/s")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all-reduce with different tensor sizes")
    parser.add_argument(
        "--size",
        type=str,
        choices=["1MB", "10MB", "100MB", "1GB"],
        default="1MB",
        help="Size of tensor to all-reduce (default: 1MB)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of processes to spawn (default: 4)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gloo", "nccl"],
        default="gloo",
        help="Backend to use: 'gloo' for CPU or 'nccl' for GPU (default: gloo)"
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
    args = parser.parse_args()
    
    # Validate backend choice
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            print("Error: NCCL backend requires CUDA, but CUDA is not available.")
            print("Please use --backend gloo for CPU-based all-reduce.")
            exit(1)
        if torch.cuda.device_count() < args.world_size:
            print(f"Error: NCCL backend requires at least {args.world_size} GPUs,")
            print(f"but only {torch.cuda.device_count()} GPU(s) available.")
            exit(1)
    
    # Convert size string to MB
    size_map = {
        "1MB": 1,
        "10MB": 10,
        "100MB": 100,
        "1GB": 1024
    }
    tensor_size_mb = size_map[args.size]
    
    world_size = args.world_size
    mp.spawn(
        fn=benchmark_all_reduce,
        args=(world_size, tensor_size_mb, args.backend, args.num_warmup, args.num_iterations),
        nprocs=world_size,
        join=True
    )