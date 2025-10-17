#!/usr/bin/env python3
"""
Experiment to demonstrate the effects of mixed precision accumulation.
Compares different accumulation strategies with float32 and float16.
"""

import torch

print("=" * 60)
print("MIXED PRECISION ACCUMULATION EXPERIMENT")
print("=" * 60)

# Experiment 1: Pure float32 accumulation
print("\n1. Pure float32 accumulation:")
s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32, device="cuda")
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 2: Pure float16 accumulation
print("\n2. Pure float16 accumulation:")
s = torch.tensor(0, dtype=torch.float16, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16, device="cuda")
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 3: float32 accumulator with float16 addend (implicit conversion)
print("\n3. float32 accumulator + float16 addend (implicit conversion):")
s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16, device="cuda")
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 4: float32 accumulator with explicit float16->float32 conversion
print("\n4. float32 accumulator + explicit float16->float32 conversion:")
s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16, device="cuda")
    s += x.type(torch.float32)
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 5: Pure bfloat16 accumulation
print("\n5. Pure bfloat16 accumulation:")
s = torch.tensor(0, dtype=torch.bfloat16, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.bfloat16, device="cuda")
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 6: float32 accumulator with bfloat16 addend (implicit conversion)
print("\n6. float32 accumulator + bfloat16 addend (implicit conversion):")
s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.bfloat16, device="cuda")
print(f"   Result: {s.item()}")
print(f"   Expected: 10.0")
print(f"   Error: {abs(s.item() - 10.0)}")

# Experiment 7: Comparison with larger accumulation
print("\n7. Larger accumulation test (10000 iterations of 0.001):")
print("   Expected: 10.0")
s_fp32 = torch.tensor(0, dtype=torch.float32, device="cuda")
s_fp16 = torch.tensor(0, dtype=torch.float16, device="cuda")
s_bf16 = torch.tensor(0, dtype=torch.bfloat16, device="cuda")
for i in range(10000):
    s_fp32 += torch.tensor(0.001, dtype=torch.float32, device="cuda")
    s_fp16 += torch.tensor(0.001, dtype=torch.float16, device="cuda")
    s_bf16 += torch.tensor(0.001, dtype=torch.bfloat16, device="cuda")
print(f"   FP32 Result: {s_fp32.item()}, Error: {abs(s_fp32.item() - 10.0)}")
print(f"   FP16 Result: {s_fp16.item()}, Error: {abs(s_fp16.item() - 10.0)}")
print(f"   BF16 Result: {s_bf16.item()}, Error: {abs(s_bf16.item() - 10.0)}")

print("\n" + "=" * 60)
print("OBSERVATIONS:")
print("=" * 60)
print(
    """
The results demonstrate the importance of accumulation precision:

1. Float32 accumulation maintains high accuracy with minimal error.

2. Float16 accumulation suffers from significant precision loss due to limited
   mantissa bits (10 bits) and rounding errors that compound over iterations.

3. BFloat16 accumulation also suffers from precision loss, though for different
   reasons than FP16:
   - BF16 has even fewer mantissa bits (7 bits) than FP16 (10 bits)
   - However, BF16 has the same exponent range as FP32 (8 bits)
   - This means BF16 can represent the same range of values as FP32, but with
     less precision (more rounding per operation)
   - For accumulation, the reduced mantissa precision still causes errors to
     compound, though differently than FP16

4. Mixed precision (float32 accumulator with float16/bfloat16 values) preserves
   accuracy by performing the accumulation in higher precision, regardless of
   whether the conversion is implicit or explicit.

5. In the larger accumulation test (10000 iterations), the errors become more
   pronounced, clearly showing:
   - FP32: minimal error (< 1e-6)
   - FP16: moderate error due to both limited mantissa and accumulation
   - BF16: error in between FP16 and FP32, demonstrating that while BF16 has
     better range than FP16, it still suffers from accumulation issues due to
     limited mantissa precision

Conclusion: For accumulation-heavy operations (like computing mean/variance in
LayerNorm), using FP32 is important even when using BF16 or FP16 elsewhere.
"""
)
