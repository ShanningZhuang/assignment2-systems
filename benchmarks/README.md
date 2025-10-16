# Benchmarks

This directory contains profiling and benchmarking scripts for Transformer models.

## Structure

```
benchmarks/
├── benchmark_model_sizes.py    # Script to benchmark models from Table 1 (§1.1.2)
└── results/
    ├── benchmark_results.json  # Raw benchmark data (JSON format)
    └── BENCHMARK_RESULTS.txt   # Human-readable summary with answers
```

## Running Benchmarks

```bash
# Benchmark all model sizes from Table 1
uv run python benchmarks/benchmark_model_sizes.py
```

## Results

Results are automatically saved to `benchmarks/results/`:
- `benchmark_results.json` - Machine-readable format
- `BENCHMARK_RESULTS.txt` - Human-readable summary with analysis

## Profiler Module

The profiler implementation is in `cs336_systems/profiler.py` and can be used as:

```bash
# Command line
uv run python -m cs336_systems.profiler --help

# Python API
from cs336_systems.profiler import benchmark_transformer
```

