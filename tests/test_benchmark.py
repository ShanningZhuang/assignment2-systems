#!/usr/bin/env python3
"""
Test suite for benchmark.py functionality.
Tests model benchmarking, mixed precision (BF16), and optimizer integration.

Usage:
    pytest tests/test_benchmark.py -v
    or
    python tests/test_benchmark.py
"""

import torch
import torch.nn.functional as F
from contextlib import nullcontext
import pytest
from cs336_systems.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


class TestBenchmarkBasics:
    """Test basic benchmark functionality."""

    def test_model_initialization(self):
        """Test that model can be initialized with benchmark configs."""
        model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=1024,
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
            rope_theta=10000.0,
        ).cuda()

        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_forward_pass(self):
        """Test basic forward pass."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        input_ids = torch.randint(0, 1000, (4, 128)).cuda()
        model.eval()

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (4, 128, 1000)
        assert output.dtype == torch.float32

    def test_backward_pass(self):
        """Test forward + backward pass."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        input_ids = torch.randint(0, 1000, (4, 128)).cuda()
        model.train()

        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.dtype == torch.float32


class TestMixedPrecision:
    """Test suite for BF16 mixed precision with autocast."""

    def test_autocast_basic(self):
        """Test basic autocast functionality with simple operations."""
        x = torch.randn(4, 128, 768, device="cuda")
        y = torch.randn(4, 128, 768, device="cuda")

        # Test without autocast (FP32)
        result_fp32 = torch.matmul(x, y.transpose(-1, -2))
        assert result_fp32.dtype == torch.float32

        # Test with autocast (BF16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result_bf16 = torch.matmul(x, y.transpose(-1, -2))
            assert result_bf16.dtype == torch.bfloat16

        # Check numerical similarity
        diff = (result_fp32 - result_bf16.float()).abs().mean()
        assert diff < 1.0, f"Difference too large: {diff}"

    def test_model_forward_bf16(self):
        """Test model forward pass with FP32 and BF16."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        input_ids = torch.randint(0, 1000, (2, 64)).cuda()

        # Forward pass without autocast (FP32)
        model.eval()
        with torch.no_grad():
            output_fp32 = model(input_ids)
        assert output_fp32.dtype == torch.float32

        # Forward pass with autocast (BF16)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_bf16 = model(input_ids)
        assert output_bf16.dtype == torch.bfloat16

        # Check numerical similarity
        diff = (output_fp32 - output_bf16.float()).abs().mean()
        assert diff < 1.0, f"Output difference too large: {diff}"

    def test_backward_pass_bf16(self):
        """Test backward pass with FP32 and BF16."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        input_ids = torch.randint(0, 1000, (2, 64)).cuda()

        # Test FP32 backward
        model.train()
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        sample_param = next(model.parameters())
        assert sample_param.grad is not None
        assert sample_param.grad.dtype == torch.float32
        grad_fp32_norm = sample_param.grad.norm()
        model.zero_grad()

        # Test BF16 backward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids)
            loss = output.sum()
        loss.backward()

        assert sample_param.grad is not None
        assert sample_param.grad.dtype == torch.float32
        grad_bf16_norm = sample_param.grad.norm()

        # Check gradients are reasonable
        assert grad_bf16_norm > 0
        assert abs(grad_fp32_norm - grad_bf16_norm) / grad_fp32_norm < 0.5

    def test_nullcontext(self):
        """Test that nullcontext behaves correctly as a no-op."""
        x = torch.randn(2, 64, 256, device="cuda")

        # Using nullcontext (should be FP32)
        with nullcontext():
            result = torch.matmul(x, x.transpose(-1, -2))
        assert result.dtype == torch.float32

        # Using autocast (should be BF16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = torch.matmul(x, x.transpose(-1, -2))
        assert result.dtype == torch.bfloat16

    def test_nested_autocast(self):
        """Test nested autocast contexts."""
        x = torch.randn(2, 64, 256, device="cuda")

        # Using autocast inside no_grad
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = torch.matmul(x, x.transpose(-1, -2))
        assert result.dtype == torch.bfloat16
        assert not result.requires_grad


class TestOptimizerIntegration:
    """Test optimizer integration with benchmarks."""

    def test_adamw_training_step_fp32(self):
        """Test complete training step with AdamW in FP32."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        input_ids = torch.randint(0, 1000, (2, 64)).cuda()
        target_ids = torch.randint(0, 1000, (2, 64)).cuda()

        # Training step
        model.train()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
        assert loss.dtype == torch.float32

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without errors
        assert True

    def test_adamw_training_step_bf16(self):
        """Test complete training step with AdamW in BF16."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        input_ids = torch.randint(0, 1000, (2, 64)).cuda()
        target_ids = torch.randint(0, 1000, (2, 64)).cuda()

        # Training step with BF16
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
        assert loss.dtype == torch.bfloat16

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without errors
        assert True

    def test_multiple_training_steps(self):
        """Test multiple consecutive training steps."""
        model = BasicsTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            rope_theta=10000.0,
        ).cuda()

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        model.train()
        for _ in range(5):
            input_ids = torch.randint(0, 1000, (2, 64)).cuda()
            target_ids = torch.randint(0, 1000, (2, 64)).cuda()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should complete without errors
        assert True


def run_manual_tests():
    """Run tests manually with verbose output."""
    print("\n" + "=" * 60)
    print("BENCHMARK TEST SUITE")
    print("=" * 60)

    # Check if BF16 is supported
    if not torch.cuda.is_bf16_supported():
        print("⚠ WARNING: BF16 is not supported on this GPU!")
        print("BF16 tests may fail or produce incorrect results.\n")
    else:
        print("✓ BF16 is supported on this GPU!\n")

    test_classes = [
        ("Basic Benchmark", TestBenchmarkBasics()),
        ("Mixed Precision (BF16)", TestMixedPrecision()),
        ("Optimizer Integration", TestOptimizerIntegration()),
    ]

    total_passed = 0
    total_failed = 0

    for class_name, test_instance in test_classes:
        print(f"\n{'='*60}")
        print(f"{class_name}")
        print("=" * 60)

        # Get all test methods
        test_methods = [
            (name, getattr(test_instance, name))
            for name in dir(test_instance)
            if name.startswith("test_")
        ]

        for test_name, test_func in test_methods:
            try:
                print(f"  {test_name}...", end=" ")
                test_func()
                print("✓ PASSED")
                total_passed += 1
            except Exception as e:
                print(f"✗ FAILED")
                print(f"    Error: {e}")
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed == 0:
        print("\n✓ All tests passed! Ready to run benchmarks.")
        print("\nRun benchmarks with:")
        print("  python benchmarks/benchmark.py small")
        print("  python benchmarks/benchmark.py --bf16 small")
    else:
        print("\n✗ Some tests failed. Debug the errors before running benchmarks.")


if __name__ == "__main__":
    run_manual_tests()
