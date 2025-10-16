import timeit
from typing import Callable, Any, Optional, Literal
import torch
import argparse


class Profiler:
    """A profiler class that uses timeit.default_timer() to profile functions or code snippets."""

    def __init__(self):
        """Initialize the Profiler."""
        self.timer = timeit.default_timer

    def profile_function(self, func: Callable, *args, **kwargs) -> dict:
        """
        Profile a function with given arguments.

        Args:
            func: The function to profile
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary containing timing statistics
        """
        start_time = self.timer()
        result = func(*args, **kwargs)
        end_time = self.timer()
        elapsed_time = end_time - start_time

        return {
            "elapsed_time": elapsed_time,
            "result": result,
            "start_time": start_time,
            "end_time": end_time,
        }

    def profile_code(self, func: Callable) -> dict:
        """
        Profile a callable (function or lambda).

        Args:
            func: The callable to profile

        Returns:
            Dictionary containing timing statistics
        """
        start_time = self.timer()
        result = func()
        end_time = self.timer()
        elapsed_time = end_time - start_time

        return {
            "elapsed_time": elapsed_time,
            "result": result,
            "start_time": start_time,
            "end_time": end_time,
        }

    def benchmark_model(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        num_steps: int,
        warmup_steps: int,
        mode: Literal["forward", "forward_backward"] = "forward",
        device: str = "cuda",
    ) -> dict:
        """
        Benchmark a model's forward and/or backward passes.

        Args:
            model: The PyTorch model to benchmark
            input_ids: Input tensor for the model
            num_steps: Number of steps to measure timing for
            warmup_steps: Number of warm-up steps before measuring
            mode: Either "forward" for forward pass only, or "forward_backward" for both
            device: Device to run on ("cuda" or "cpu")

        Returns:
            Dictionary containing timing statistics including:
            - total_time: Total time for all measured steps
            - avg_time_per_step: Average time per step
            - steps: Number of steps measured
            - warmup_steps: Number of warm-up steps run
            - mode: The mode used (forward or forward_backward)
        """
        use_cuda = device == "cuda" and torch.cuda.is_available()

        # Warm-up steps
        for _ in range(warmup_steps):
            if mode == "forward":
                outputs = model(input_ids)
            elif mode == "forward_backward":
                outputs = model(input_ids)
                loss = outputs.sum()
                loss.backward()
                model.zero_grad()

            if use_cuda:
                torch.cuda.synchronize()

        # Measured steps
        start_time = self.timer()

        for _ in range(num_steps):
            if mode == "forward":
                outputs = model(input_ids)
            elif mode == "forward_backward":
                outputs = model(input_ids)
                loss = outputs.sum()
                loss.backward()
                model.zero_grad()

            if use_cuda:
                torch.cuda.synchronize()

        end_time = self.timer()
        total_time = end_time - start_time
        avg_time = total_time / num_steps

        return {
            "total_time": total_time,
            "avg_time_per_step": avg_time,
            "steps": num_steps,
            "warmup_steps": warmup_steps,
            "mode": mode,
            "device": device,
        }


def benchmark_transformer(
    vocab_size: int = 10000,
    context_length: int = 512,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    rope_theta: float = 10000.0,
    batch_size: int = 8,
    seq_length: int = 512,
    num_steps: int = 20,
    warmup_steps: int = 5,
    mode: Literal["forward", "forward_backward"] = "forward",
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Initialize a Transformer model, create random batch data, and benchmark it.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length for the model
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta parameter
        batch_size: Batch size for random data
        seq_length: Sequence length for random data
        num_steps: Number of steps to measure timing for
        warmup_steps: Number of warm-up steps before measuring
        mode: Either "forward" for forward pass only, or "forward_backward" for both
        device: Device to run on ("cuda" or "cpu")
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing timing statistics and model information
    """
    from cs336_basics.model import BasicsTransformerLM

    # Set random seed
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize model
    print(f"Initializing Transformer model with {num_layers} layers...")
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    # Move model to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to CUDA")
    else:
        device = "cpu"
        print(f"Using CPU (CUDA not available or not requested)")

    # Enable training mode if doing backward pass
    if mode == "forward_backward":
        model.train()
    else:
        model.eval()

    # Generate random batch of data
    print(f"Generating random batch: batch_size={batch_size}, seq_length={seq_length}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    if device == "cuda":
        input_ids = input_ids.cuda()

    # Create profiler and benchmark
    profiler = Profiler()
    print(
        f"\nRunning benchmark: {warmup_steps} warmup steps, {num_steps} measured steps, mode={mode}"
    )
    results = profiler.benchmark_model(
        model=model,
        input_ids=input_ids,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        mode=mode,
        device=device,
    )

    # Add model and data info to results
    results.update(
        {
            "model_config": {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "rope_theta": rope_theta,
            },
            "data_config": {
                "batch_size": batch_size,
                "seq_length": seq_length,
            },
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
    )

    return results


def main():
    """Command-line interface for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark a Transformer language model's forward and backward passes"
    )

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--context_length", type=int, default=512, help="Maximum context length"
    )
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="RoPE theta parameter"
    )

    # Data configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")

    # Benchmarking configuration
    parser.add_argument(
        "--num_steps", type=int, default=20, help="Number of measured steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Number of warm-up steps"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=["forward", "forward_backward"],
        help="Benchmark mode: 'forward' or 'forward_backward'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        mode=args.mode,
        device=args.device,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  - Layers: {results['model_config']['num_layers']}")
    print(f"  - d_model: {results['model_config']['d_model']}")
    print(f"  - Heads: {results['model_config']['num_heads']}")
    print(f"  - d_ff: {results['model_config']['d_ff']}")
    print(f"  - Vocab size: {results['model_config']['vocab_size']}")
    print(f"  - Total parameters: {results['num_parameters']:,}")

    print(f"\nData Configuration:")
    print(f"  - Batch size: {results['data_config']['batch_size']}")
    print(f"  - Sequence length: {results['data_config']['seq_length']}")

    print(f"\nBenchmark Configuration:")
    print(f"  - Mode: {results['mode']}")
    print(f"  - Device: {results['device']}")
    print(f"  - Warmup steps: {results['warmup_steps']}")
    print(f"  - Measured steps: {results['steps']}")

    print(f"\nTiming Results:")
    print(f"  - Total time: {results['total_time']:.4f} seconds")
    print(f"  - Average time per step: {results['avg_time_per_step']:.4f} seconds")
    print(f"  - Throughput: {1.0 / results['avg_time_per_step']:.2f} steps/second")
    print(
        f"  - Tokens/second: {results['data_config']['batch_size'] * results['data_config']['seq_length'] / results['avg_time_per_step']:.2f}"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
