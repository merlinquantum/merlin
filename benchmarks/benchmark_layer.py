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
Comprehensive benchmark for quantum layer functionality.
Tests performance of quantum layer operations and transformations.
"""

import json
import os
import time
from typing import Any

import pytest
import torch

import merlin as ML


class LayerBenchmarkRunner:
    """Utility class for running and validating quantum layer benchmarks."""

    def __init__(self):
        self.results = []

    def validate_layer_output_correctness(
        self, output: torch.Tensor, expected_shape: tuple[int, ...]
    ) -> bool:
        """Validate that the quantum layer output is correct."""
        # Check output shape
        if output.shape != expected_shape:
            return False

        # Check that all values are finite
        if not torch.all(torch.isfinite(output)):
            return False

        # Check reasonable bounds for quantum outputs
        if torch.any(output < -1e6) or torch.any(output > 1e6):
            return False

        return True


# Test configurations for different complexity levels
BENCHMARK_CONFIGS = [
    {"n_modes": 4, "n_photons": 2, "input_size": 2, "output_size": 3, "name": "small"},
    {"n_modes": 6, "n_photons": 3, "input_size": 4, "output_size": 8, "name": "medium"},
    {"n_modes": 8, "n_photons": 3, "input_size": 6, "output_size": 10, "name": "large"},
    {
        "n_modes": 10,
        "n_photons": 4,
        "input_size": 8,
        "output_size": 15,
        "name": "xlarge",
    },
]

BATCH_CONFIGS = [8, 16, 32, 64]

DEVICE_CONFIGS = ["cpu"]

benchmark_runner = LayerBenchmarkRunner()


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_quantum_layer_forward_benchmark(benchmark, config: dict, device: str):
    """Benchmark quantum layer forward pass."""

    builder = ML.CircuitBuilder(n_modes=config["n_modes"])
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(
        modes=list(range(config["input_size"])), name="input", subset_combinations=True
    )
    builder.add_entangling_layer(trainable=True, name="U2")

    layer = ML.QuantumLayer(
        input_size=config["input_size"],
        n_photons=config["n_photons"],
        builder=builder,
    )

    model = torch.nn.Sequential(
        layer, torch.nn.Linear(layer.output_size, config["output_size"])
    )

    # Create larger batch for meaningful timing
    batch_size = 32
    x = torch.rand(batch_size, config["input_size"])

    def forward_pass():
        return model(x)

    # Run benchmark
    result = benchmark(forward_pass)

    # Validate correctness
    expected_shape = (batch_size, config["output_size"])
    assert benchmark_runner.validate_layer_output_correctness(result, expected_shape)


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_batched_computation_benchmark(
    benchmark, config: dict, batch_size: int, device: str
):
    """Benchmark batched quantum layer computation."""

    builder = ML.CircuitBuilder(n_modes=config["n_modes"])
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(config["input_size"])), name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    layer = ML.QuantumLayer(
        input_size=config["input_size"],
        n_photons=config["n_photons"],
        builder=builder,
    )

    model = torch.nn.Sequential(
        layer, torch.nn.Linear(layer.output_size, config["output_size"])
    )

    x = torch.rand(batch_size, config["input_size"])

    def batched_forward():
        return model(x)

    # Run benchmark
    result = benchmark(batched_forward)

    # Validate correctness
    expected_shape = (batch_size, config["output_size"])
    assert benchmark_runner.validate_layer_output_correctness(result, expected_shape)


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])  # Only test smaller configs
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_gradient_computation_benchmark(benchmark, config: dict, device: str):
    """Benchmark gradient computation through quantum layer."""

    builder = ML.CircuitBuilder(n_modes=config["n_modes"])
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(
        modes=list(range(config["input_size"])), name="input", subset_combinations=True
    )
    builder.add_entangling_layer(trainable=True, name="U2")

    layer = ML.QuantumLayer(
        input_size=config["input_size"],
        n_photons=config["n_photons"],
        builder=builder,
    )

    model = torch.nn.Sequential(
        layer, torch.nn.Linear(layer.output_size, config["output_size"])
    )

    x = torch.rand(16, config["input_size"], requires_grad=True)

    def compute_gradients():
        output = model(x)
        loss = output.sum()
        loss.backward()
        return loss

    # Run benchmark
    _loss = benchmark(compute_gradients)

    # Validate gradients exist
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    # Check that layer parameters have gradients
    has_trainable_params = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_trainable_params = True
            assert torch.all(torch.isfinite(param.grad))

    assert has_trainable_params, "Layer should have trainable parameters with gradients"


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_multiple_circuit_types_benchmark(benchmark, config: dict, device: str):
    """Benchmark different circuit types for quantum layers."""
    subset_combinations = [True, False]

    def test_all_circuit_types():
        results = []

        for subset in subset_combinations:
            builder = ML.CircuitBuilder(n_modes=config["n_modes"])
            builder.add_entangling_layer(trainable=True, name="U1")
            builder.add_angle_encoding(
                modes=list(range(config["input_size"])),
                name="input",
                subset_combinations=subset,
            )
            builder.add_entangling_layer(trainable=True, name="U2")

            layer = ML.QuantumLayer(
                input_size=config["input_size"],
                n_photons=config["n_photons"],
                builder=builder,
            )

            model = torch.nn.Sequential(
                layer, torch.nn.Linear(layer.output_size, config["output_size"])
            )

            x = torch.rand(16, config["input_size"])
            output = model(x)
            results.append(output)

        return results

    # Run benchmark
    results = benchmark(test_all_circuit_types)

    # Validate all results
    assert len(results) == len(subset_combinations)
    for result in results:
        expected_shape = (16, config["output_size"])
        assert benchmark_runner.validate_layer_output_correctness(
            result, expected_shape
        )


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_output_mapping_strategies_benchmark(benchmark, config: dict, device: str):
    """Benchmark different output mapping strategies."""
    strategies = [
        {
            "measurement_strategy": ML.MeasurementStrategy.probs(),
            "grouping_policy": None,
        },
        {
            "measurement_strategy": ML.MeasurementStrategy.probs(),
            "grouping_policy": ML.LexGrouping,
        },
        {
            "measurement_strategy": ML.MeasurementStrategy.probs(),
            "grouping_policy": ML.ModGrouping,
        },
    ]

    def test_all_strategies():
        results = []

        for strategy in strategies:
            builder = ML.CircuitBuilder(n_modes=config["n_modes"])
            builder.add_entangling_layer(trainable=True, name="U1")
            builder.add_angle_encoding(
                modes=list(range(config["input_size"])),
                name="input",
                subset_combinations=True,
            )
            builder.add_entangling_layer(trainable=True, name="U2")

            if strategy["grouping_policy"] is None:
                layer = ML.QuantumLayer(
                    input_size=config["input_size"],
                    n_photons=config["n_photons"],
                    builder=builder,
                    measurement_strategy=strategy["measurement_strategy"],
                )
                model = torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, config["output_size"])
                )
                x = torch.rand(16, config["input_size"])
                output = model(x)
                results.append(output)
            else:
                layer = ML.QuantumLayer(
                    input_size=config["input_size"],
                    n_photons=config["n_photons"],
                    builder=builder,
                    measurement_strategy=strategy["measurement_strategy"],
                )
                model = torch.nn.Sequential(
                    layer,
                    strategy["grouping_policy"](
                        layer.output_size, config["output_size"]
                    ),
                )
                x = torch.rand(16, config["input_size"])
                output = model(x)
                results.append(output)

        return results

    # Run benchmark
    results = benchmark(test_all_strategies)

    # Validate all results
    assert len(results) == len(strategies)
    for result in results:
        expected_shape = (16, config["output_size"])
        assert benchmark_runner.validate_layer_output_correctness(
            result, expected_shape
        )


# Performance regression tests
class TestLayerPerformanceRegression:
    """Test suite for detecting quantum layer performance regressions."""

    def test_forward_pass_performance_bounds(self):
        """Test that forward pass stays within reasonable time bounds."""

        builder = ML.CircuitBuilder(n_modes=8)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=list(range(6)), name="input", subset_combinations=True
        )
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=6,
            n_photons=3,
            builder=builder,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 10))

        x = torch.rand(32, 6)

        start_time = time.time()
        output = model(x)
        forward_time = time.time() - start_time

        # Assert reasonable performance bounds
        assert forward_time < 2.0, (
            f"Forward pass took {forward_time:.3f}s, expected < 2.0s"
        )
        assert benchmark_runner.validate_layer_output_correctness(output, (32, 10))

    def test_gradient_computation_performance_bounds(self):
        """Test that gradient computation stays within reasonable time bounds."""

        builder = ML.CircuitBuilder(n_modes=6)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=list(range(4)), name="input", subset_combinations=True
        )
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=4,
            n_photons=2,
            builder=builder,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 6))

        x = torch.rand(16, 4, requires_grad=True)

        start_time = time.time()
        output = model(x)
        loss = output.sum()
        loss.backward()
        gradient_time = time.time() - start_time

        # Assert reasonable performance bounds
        assert gradient_time < 5.0, (
            f"Gradient computation took {gradient_time:.3f}s, expected < 5.0s"
        )
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# Utility function to save benchmark results
def save_benchmark_results(
    results: list[dict[str, Any]], output_path: str = "layer-results.json"
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
    print("Running quantum layer benchmarks...")

    builder = ML.CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(
        modes=list(range(4)), name="input", subset_combinations=True
    )
    builder.add_entangling_layer(trainable=True, name="U2")

    layer = ML.QuantumLayer(
        input_size=4,
        n_photons=3,
        builder=builder,
    )

    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 8))

    print("Testing forward pass performance...")
    x = torch.rand(32, 4)
    start = time.time()
    output = model(x)
    forward_time = time.time() - start
    print(f"Forward pass time: {forward_time:.4f}s")

    is_valid = benchmark_runner.validate_layer_output_correctness(output, (32, 8))
    print(f"Output validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Output shape: {output.shape}")

    print("Quantum layer benchmark tests completed!")
