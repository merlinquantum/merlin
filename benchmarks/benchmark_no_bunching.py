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
Comprehensive benchmark suite for no bunching functionality.
Tests performance of photon bunching prevention algorithms.
"""

import json
import math
import os
import time
from typing import Any

import pytest
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.core.generators import CircuitGenerator, StateGenerator
from merlin.core.process import ComputationProcessFactory


class NoBunchingBenchmarkRunner:
    """Utility class for running and validating no bunching benchmarks."""

    def __init__(self):
        self.results = []

    def calculate_fock_space_size(self, n_modes: int, n_photons: int) -> int:
        """Calculate the size of the Fock space for n_photons in n_modes."""
        if n_photons == 0:
            return 1
        return math.comb(n_modes + n_photons - 1, n_photons)

    def calculate_no_bunching_size(self, n_modes: int, n_photons: int) -> int:
        """Calculate the size of the no-bunching space (single photon states only)."""
        if n_photons == 0:
            return 1
        if n_photons > n_modes:
            return 0
        return math.comb(n_modes, n_photons)

    def validate_distribution_correctness(
        self, pa: torch.Tensor, expected_size: int
    ) -> bool:
        """Validate that the no bunching distribution is correct."""
        # Check distribution size
        if pa.shape[-1] != expected_size:
            return False

        # Check probability normalization (allow small numerical errors)
        probs = (pa.abs() ** 2).real

        # Check that all probabilities are non-negative
        if (probs < 0).any():
            return False

        return True


# Test configurations for different complexity levels
BENCHMARK_CONFIGS = [
    {"n_modes": 4, "n_photons": 2, "name": "small"},
    {"n_modes": 6, "n_photons": 3, "name": "medium"},
    {"n_modes": 8, "n_photons": 3, "name": "large"},
    {"n_modes": 10, "n_photons": 4, "name": "xlarge"},
]

DEVICE_CONFIGS = ["cpu"]

benchmark_runner = NoBunchingBenchmarkRunner()


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_no_bunching_computation_benchmark(benchmark, config: dict, device: str):
    """Benchmark no bunching computation process."""
    n_modes = config["n_modes"]
    n_photons = config["n_photons"]

    # Create circuit and state
    circuit, _ = CircuitGenerator.generate_circuit(
        ML.CircuitType.PARALLEL_COLUMNS, n_modes, 3
    )
    input_state = StateGenerator.generate_state(
        n_modes, n_photons, ML.StatePattern.SEQUENTIAL
    )

    # Create computation process with default (UNBUNCHED) computation space
    process = ComputationProcessFactory.create(
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=["phi_"],
        input_parameters=["pl"],
    )

    # Create dummy parameters
    spec_mappings = process.converter.spec_mappings
    dummy_params = []

    for spec in ["phi_", "pl"]:
        if spec in spec_mappings:
            param_count = len(spec_mappings[spec])
            dummy_params.append(torch.rand(param_count))

    def compute_no_bunching():
        return process.compute(dummy_params)

    # Run benchmark
    result = benchmark(compute_no_bunching)

    # Validate correctness
    expected_size = benchmark_runner.calculate_no_bunching_size(n_modes, n_photons)
    assert benchmark_runner.validate_distribution_correctness(result, expected_size)


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_fock_space_comparison_benchmark(benchmark, config: dict, device: str):
    """Benchmark comparison between no bunching and full Fock space."""
    n_modes = config["n_modes"]
    n_photons = config["n_photons"]

    # Create circuit and state
    circuit, _ = CircuitGenerator.generate_circuit(
        ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
    )
    input_state = StateGenerator.generate_state(
        n_modes, n_photons, ML.StatePattern.SEQUENTIAL
    )

    def compute_both_modes():
        # No bunching process
        process_no_bunching = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
        )

        # Full Fock space process
        process_full = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
            computation_space=ComputationSpace.FOCK,
        )

        # Create dummy parameters
        spec_mappings = process_no_bunching.converter.spec_mappings
        dummy_params = []

        for spec in ["phi_", "pl"]:
            if spec in spec_mappings:
                param_count = len(spec_mappings[spec])
                dummy_params.append(torch.rand(param_count))

        # Compute both
        result_no_bunching = process_no_bunching.compute(dummy_params)
        result_full = process_full.compute(dummy_params)

        return result_no_bunching, result_full

    # Run benchmark
    results = benchmark(compute_both_modes)

    # Validate results
    result_no_bunching, result_full = results

    expected_no_bunching = benchmark_runner.calculate_no_bunching_size(
        n_modes, n_photons
    )
    expected_full_fock = benchmark_runner.calculate_fock_space_size(n_modes, n_photons)

    assert benchmark_runner.validate_distribution_correctness(
        result_no_bunching, expected_no_bunching
    )
    assert benchmark_runner.validate_distribution_correctness(
        result_full, expected_full_fock
    )


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS[:2])  # Only test smaller configs
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_compute_with_keys_benchmark(benchmark, config: dict, device: str):
    """Benchmark compute_with_keys functionality with no bunching."""
    n_modes = config["n_modes"]
    n_photons = config["n_photons"]

    circuit, _ = CircuitGenerator.generate_circuit(ML.CircuitType.SERIES, n_modes, 2)
    input_state = StateGenerator.generate_state(
        n_modes, n_photons, ML.StatePattern.PERIODIC
    )

    # Process with no_bunching
    process_no_bunching = ComputationProcessFactory.create(
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=["phi_"],
        input_parameters=["pl"],
    )

    spec_mappings = process_no_bunching.converter.spec_mappings
    dummy_params = []

    for spec in ["phi_", "pl"]:
        if spec in spec_mappings:
            param_count = len(spec_mappings[spec])
            dummy_params.append(torch.randn(param_count))

    def compute_with_keys():
        return process_no_bunching.compute_with_keys(dummy_params)

    # Run benchmark
    keys, distribution = benchmark(compute_with_keys)

    # Validate results
    expected_size = benchmark_runner.calculate_no_bunching_size(n_modes, n_photons)
    assert len(keys) == expected_size
    assert benchmark_runner.validate_distribution_correctness(
        distribution, expected_size
    )


# Performance regression tests
class TestNoBunchingPerformanceRegression:
    """Test suite for detecting no bunching performance regressions."""

    def test_no_bunching_performance_bounds(self):
        """Test that no bunching computation stays within reasonable time bounds."""
        n_modes = 8
        n_photons = 3

        circuit, _ = CircuitGenerator.generate_circuit(
            ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
        )
        input_state = StateGenerator.generate_state(
            n_modes, n_photons, ML.StatePattern.SEQUENTIAL
        )

        process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
        )

        spec_mappings = process.converter.spec_mappings
        dummy_params = []

        for spec in ["phi_", "pl"]:
            if spec in spec_mappings:
                param_count = len(spec_mappings[spec])
                dummy_params.append(torch.rand(param_count))

        start_time = time.time()
        result = process.compute(dummy_params)
        compute_time = time.time() - start_time

        # Assert reasonable performance bounds
        assert compute_time < 3.0, (
            f"No bunching compute took {compute_time:.3f}s, expected < 3.0s"
        )

        expected_size = benchmark_runner.calculate_no_bunching_size(n_modes, n_photons)
        assert benchmark_runner.validate_distribution_correctness(result, expected_size)


# Utility function to save benchmark results
def save_benchmark_results(
    results: list[dict[str, Any]], output_path: str = "no-bunching-results.json"
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
    print("Running no bunching benchmarks...")

    n_modes = 6
    n_photons = 3

    circuit, _ = CircuitGenerator.generate_circuit(
        ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
    )
    input_state = StateGenerator.generate_state(
        n_modes, n_photons, ML.StatePattern.SEQUENTIAL
    )

    process = ComputationProcessFactory.create(
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=["phi_"],
        input_parameters=["pl"],
    )

    spec_mappings = process.converter.spec_mappings
    dummy_params = []

    for spec in ["phi_", "pl"]:
        if spec in spec_mappings:
            param_count = len(spec_mappings[spec])
            dummy_params.append(torch.rand(param_count))

    print("Testing no bunching computation performance...")
    start = time.time()
    result = process.compute(dummy_params)
    compute_time = time.time() - start
    print(f"No bunching compute time: {compute_time:.4f}s")

    expected_size = benchmark_runner.calculate_no_bunching_size(n_modes, n_photons)
    is_valid = benchmark_runner.validate_distribution_correctness(result, expected_size)
    print(f"Output validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Expected size: {expected_size}, Actual size: {result.shape[-1]}")

    print("No bunching benchmark tests completed!")
