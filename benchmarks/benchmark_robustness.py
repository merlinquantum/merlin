# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Comprehensive benchmark suite for robustness and stability testing.
Tests performance under stress conditions and edge cases.
"""

import json
import os
import time
from typing import Any

import pytest
import torch
import torch.nn as nn

import merlin as ML


class RobustnessBenchmarkRunner:
    """Utility class for running and validating robustness benchmarks."""

    def __init__(self):
        self.results = []

    def validate_robustness_output_correctness(
        self, output: torch.Tensor, expected_shape: tuple[int, ...]
    ) -> bool:
        """Validate that the robustness test output is correct."""
        # Check output shape
        if output.shape != expected_shape:
            return False

        # Check that all values are finite (critical for robustness)
        if not torch.all(torch.isfinite(output)):
            return False

        # Check reasonable bounds for quantum outputs under stress
        if torch.any(output < -1e8) or torch.any(output > 1e8):
            return False

        return True


# Test configurations for different stress levels
BENCHMARK_CONFIGS = [
    {
        "n_modes": 6,
        "n_photons": 2,
        "input_size": 4,
        "output_size": 10,
        "name": "moderate_stress",
    },
    {
        "n_modes": 8,
        "n_photons": 3,
        "input_size": 6,
        "output_size": 15,
        "name": "high_stress",
    },
    {
        "n_modes": 10,
        "n_photons": 4,
        "input_size": 8,
        "output_size": 20,
        "name": "extreme_stress",
    },
]

LARGE_BATCH_CONFIGS = [64, 128, 256, 512]

DEVICE_CONFIGS = ["cpu"]

benchmark_runner = RobustnessBenchmarkRunner()


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("batch_size", LARGE_BATCH_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_large_batch_robustness_benchmark(
    benchmark, config: dict, batch_size: int, device: str
):
    """Benchmark robustness with large batch sizes."""
    experiment = ML.PhotonicBackend(
        circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
        n_modes=config["n_modes"],
        n_photons=config["n_photons"],
    )

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=config["input_size"],
        output_size=config["output_size"],
    )

    layer = ML.QuantumLayer(input_size=config["input_size"], ansatz=ansatz)

    # Large batch for stress testing
    x = torch.rand(batch_size, config["input_size"])

    def large_batch_forward():
        return layer(x)

    # Run benchmark
    result = benchmark(large_batch_forward)

    # Validate correctness
    expected_shape = (batch_size, config["output_size"])
    assert benchmark_runner.validate_robustness_output_correctness(
        result, expected_shape
    )


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_extreme_values_robustness_benchmark(benchmark, config: dict, device: str):
    """Benchmark robustness with extreme input values."""
    experiment = ML.PhotonicBackend(
        circuit_type=ML.CircuitType.SERIES,
        n_modes=config["n_modes"],
        n_photons=config["n_photons"],
    )

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=config["input_size"],
        output_size=config["output_size"],
    )

    layer = ML.QuantumLayer(input_size=config["input_size"], ansatz=ansatz)

    def test_extreme_inputs():
        results = []

        # Test different extreme value scenarios
        test_inputs = [
            torch.zeros(1, config["input_size"]),  # All zeros
            torch.ones(1, config["input_size"]),  # All ones
            torch.full((1, config["input_size"]), -1.0),  # All negative ones
            torch.full((1, config["input_size"]), 10.0),  # Large positive values
            torch.full((1, config["input_size"]), -10.0),  # Large negative values
            torch.full((1, config["input_size"]), 1e-6),  # Very small positive values
            torch.full((1, config["input_size"]), -1e-6),  # Very small negative values
        ]

        for test_input in test_inputs:
            output = layer(test_input)
            results.append(output)

        return results

    # Run benchmark
    results = benchmark(test_extreme_inputs)

    # Validate all results
    assert len(results) == 7
    for result in results:
        expected_shape = (1, config["output_size"])
        assert benchmark_runner.validate_robustness_output_correctness(
            result, expected_shape
        )


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])  # Only test smaller configs
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_numerical_stability_benchmark(benchmark, config: dict, device: str):
    """Benchmark numerical stability over multiple iterations."""
    experiment = ML.PhotonicBackend(
        circuit_type=ML.CircuitType.PARALLEL,
        n_modes=config["n_modes"],
        n_photons=config["n_photons"],
    )

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=config["input_size"],
        output_size=config["output_size"],
    )

    layer = ML.QuantumLayer(input_size=config["input_size"], ansatz=ansatz)

    def stability_test():
        x = torch.rand(32, config["input_size"])
        results = []

        # Run multiple iterations to test stability
        for _i in range(20):
            with torch.no_grad():
                output = layer(x)
                results.append(output)

        return results

    # Run benchmark
    results = benchmark(stability_test)

    # Validate numerical stability
    assert len(results) == 20
    expected_shape = (32, config["output_size"])

    for result in results:
        assert benchmark_runner.validate_robustness_output_correctness(
            result, expected_shape
        )

    # Check that results are consistent (deterministic)
    first_result = results[0]
    for result in results[1:]:
        assert torch.allclose(first_result, result, atol=1e-5)


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_memory_efficiency_benchmark(benchmark, config: dict, device: str):
    """Benchmark memory efficiency over many iterations."""
    experiment = ML.PhotonicBackend(
        circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
        n_modes=config["n_modes"],
        n_photons=config["n_photons"],
    )

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=config["input_size"],
        output_size=config["output_size"],
    )

    layer = ML.QuantumLayer(input_size=config["input_size"], ansatz=ansatz)

    def memory_efficiency_test():
        results = []

        # Run many forward passes to test memory efficiency
        for _i in range(100):
            x = torch.rand(16, config["input_size"])
            with torch.no_grad():
                output = layer(x)
                results.append(output.mean().item())  # Store only scalar to save memory
                del output, x  # Explicit cleanup

        return results

    # Run benchmark
    results = benchmark(memory_efficiency_test)

    # Validate that all computations completed successfully
    assert len(results) == 100
    assert all(
        isinstance(r, float) and torch.isfinite(torch.tensor(r)) for r in results
    )


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_hybrid_model_stress_benchmark(benchmark, config: dict, device: str):
    """Benchmark complex hybrid model under stress conditions."""

    class StressTestHybridModel(nn.Module):
        def __init__(self, n_modes, n_photons, input_size, output_size):
            super().__init__()

            # Classical preprocessing with multiple layers
            self.pre_classical = nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.ReLU(),
                nn.Linear(input_size * 2, input_size),
                nn.ReLU(),
            )

            # Quantum layer
            experiment = ML.PhotonicBackend(
                circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
                n_modes=n_modes,
                n_photons=n_photons,
            )
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=input_size,
                output_size=output_size,
            )
            self.quantum = ML.QuantumLayer(input_size=input_size, ansatz=ansatz)

            # Classical postprocessing
            self.post_classical = nn.Sequential(
                nn.Linear(output_size, output_size // 2),
                nn.ReLU(),
                nn.Linear(output_size // 2, output_size),
            )

        def forward(self, x):
            x = self.pre_classical(x)
            x = torch.sigmoid(x)  # Normalize for quantum layer
            x = self.quantum(x)
            x = self.post_classical(x)
            return x

    model = StressTestHybridModel(
        config["n_modes"],
        config["n_photons"],
        config["input_size"],
        config["output_size"],
    )

    def stress_test_hybrid():
        # Test with various batch sizes and input patterns
        results = []
        batch_sizes = [16, 32, 64]

        for batch_size in batch_sizes:
            # Test normal inputs
            x = torch.rand(batch_size, config["input_size"])
            output = model(x)
            results.append(output)

            # Test boundary inputs
            x_boundary = torch.zeros(batch_size, config["input_size"])
            output_boundary = model(x_boundary)
            results.append(output_boundary)

        return results

    # Run benchmark
    results = benchmark(stress_test_hybrid)

    # Validate all results
    expected_results = len([16, 32, 64]) * 2  # 2 tests per batch size
    assert len(results) == expected_results

    batch_sizes = [16, 32, 64]
    for i, batch_size in enumerate(batch_sizes):
        # Normal input result
        normal_result = results[i * 2]
        expected_shape = (batch_size, config["output_size"])
        assert benchmark_runner.validate_robustness_output_correctness(
            normal_result, expected_shape
        )

        # Boundary input result
        boundary_result = results[i * 2 + 1]
        assert benchmark_runner.validate_robustness_output_correctness(
            boundary_result, expected_shape
        )


# Performance regression tests
class TestRobustnessPerformanceRegression:
    """Test suite for detecting robustness performance regressions."""

    def test_large_batch_performance_bounds(self):
        """Test that large batch processing stays within reasonable time bounds."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=8, n_photons=3
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=6, output_size=15
        )

        layer = ML.QuantumLayer(input_size=6, ansatz=ansatz)

        # Large batch stress test
        large_batch_size = 256
        x = torch.rand(large_batch_size, 6)

        start_time = time.time()
        output = layer(x)
        batch_time = time.time() - start_time

        # Assert reasonable performance bounds
        assert batch_time < 10.0, (
            f"Large batch processing took {batch_time:.3f}s, expected < 10.0s"
        )

        expected_shape = (large_batch_size, 15)
        assert benchmark_runner.validate_robustness_output_correctness(
            output, expected_shape
        )

    def test_extreme_values_performance_bounds(self):
        """Test that extreme value handling stays within reasonable time bounds."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.SERIES, n_modes=6, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=4, output_size=8
        )

        layer = ML.QuantumLayer(input_size=4, ansatz=ansatz)

        # Test extreme values
        extreme_inputs = [
            torch.full((16, 4), 100.0),  # Very large values
            torch.full((16, 4), -100.0),  # Very large negative values
            torch.full((16, 4), 1e-8),  # Very small values
        ]

        start_time = time.time()
        for extreme_input in extreme_inputs:
            output = layer(extreme_input)
            assert benchmark_runner.validate_robustness_output_correctness(
                output, (16, 8)
            )
        extreme_time = time.time() - start_time

        # Assert reasonable performance bounds
        assert extreme_time < 5.0, (
            f"Extreme value handling took {extreme_time:.3f}s, expected < 5.0s"
        )


# Utility function to save benchmark results
def save_benchmark_results(
    results: list[dict[str, Any]], output_path: str = "robustness-results.json"
):
    """Save benchmark results in the format expected by github-action-benchmark."""
    formatted_results = []

    for result in results:
        formatted_results.append({
            "name": result.get("name", "unknown"),
            "unit": "seconds",
            "value": result.get("mean", 0),
            "extra": {
                "min": result.get("min", 0),
                "max": result.get("max", 0),
                "stddev": result.get("stddev", 0),
                "iterations": result.get("rounds", 1),
            },
        })

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(formatted_results, f, indent=2)


if __name__ == "__main__":
    print("Running robustness benchmarks...")

    experiment = ML.PhotonicBackend(
        circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=6, n_photons=2
    )

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment, input_size=4, output_size=10
    )

    layer = ML.QuantumLayer(input_size=4, ansatz=ansatz)

    print("Testing large batch robustness...")
    large_batch_size = 128
    x = torch.rand(large_batch_size, 4)
    start = time.time()
    output = layer(x)
    batch_time = time.time() - start
    print(f"Large batch time: {batch_time:.4f}s")

    is_valid = benchmark_runner.validate_robustness_output_correctness(
        output, (large_batch_size, 10)
    )
    print(f"Output validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Output shape: {output.shape}")

    print("Testing extreme values robustness...")
    extreme_input = torch.full((16, 4), 10.0)
    start = time.time()
    extreme_output = layer(extreme_input)
    extreme_time = time.time() - start
    print(f"Extreme values time: {extreme_time:.4f}s")

    is_extreme_valid = benchmark_runner.validate_robustness_output_correctness(
        extreme_output, (16, 10)
    )
    print(f"Extreme values validation: {'PASS' if is_extreme_valid else 'FAIL'}")

    print("Robustness benchmark tests completed!")
