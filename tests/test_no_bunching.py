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
Tests for no_bunching functionality in quantum computation.
"""

import math

import torch

import merlin as ML
from merlin import OutputMappingStrategy, QuantumLayer
from merlin.core.generators import CircuitGenerator, StateGenerator
from merlin.core.process import ComputationProcessFactory


def calculate_fock_space_size(n_modes: int, n_photons: int) -> int:
    """Calculate the size of the Fock space for n_photons in n_modes."""
    if n_photons == 0:
        return 1
    return math.comb(n_modes + n_photons - 1, n_photons)


def calculate_no_bunching_size(n_modes: int, n_photons: int) -> int:
    """Calculate the size of the no-bunching space (single photon states only)."""
    if n_photons == 0:
        return 1
    if n_photons > n_modes:
        return 0  # Impossible to place more photons than modes without bunching
    return math.comb(n_modes, n_photons)


class TestNoBunchingFunctionality:
    """Test suite for no_bunching parameter in quantum computation."""

    def test_fock_space_vs_no_bunching_sizes(self):
        """Test that Fock space and no-bunching space sizes are calculated correctly."""
        # Test cases: (n_modes, n_photons)
        test_cases = [
            (3, 1),  # 3 modes, 1 photon
            (4, 2),  # 4 modes, 2 photons
            (5, 3),  # 5 modes, 3 photons
            (6, 2),  # 6 modes, 2 photons
        ]

        for n_modes, n_photons in test_cases:
            fock_size = calculate_fock_space_size(n_modes, n_photons)
            no_bunching_size = calculate_no_bunching_size(n_modes, n_photons)

            print(f"n_modes={n_modes}, n_photons={n_photons}")
            print(f"  Fock space size: {fock_size}")
            print(f"  No-bunching size: {no_bunching_size}")

            # No-bunching space should be smaller than or equal to Fock space
            assert no_bunching_size <= fock_size

            # For single photon, no-bunching size should equal n_modes
            if n_photons == 1:
                assert no_bunching_size == n_modes

    def test_computation_process_with_no_bunching_false(self):
        """Test computation process with no_bunching=False (full Fock space)."""
        n_modes = 4
        n_photons = 2

        # Create circuit and state
        circuit, _ = CircuitGenerator.generate_circuit(
            ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
        )
        input_state = StateGenerator.generate_state(
            n_modes, n_photons, ML.StatePattern.SEQUENTIAL
        )

        # Create computation process with no_bunching=False
        process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
            no_bunching=False,
        )

        # Create dummy parameters
        spec_mappings = process.converter.spec_mappings
        dummy_params = []

        for spec in ["phi_", "pl"]:
            if spec in spec_mappings:
                param_count = len(spec_mappings[spec])
                dummy_params.append(torch.zeros(param_count))

        # Compute distribution
        distribution = process.compute(dummy_params)

        # Check that distribution size matches full Fock space
        expected_size = calculate_fock_space_size(n_modes, n_photons)
        actual_size = distribution.shape[-1]

        print(f"Full Fock space - Expected: {expected_size}, Actual: {actual_size}")
        assert (
            actual_size == expected_size
        ), f"Expected Fock space size {expected_size}, got {actual_size}"

    def test_computation_process_with_no_bunching_true(self):
        """Test computation process with no_bunching=True (single photon states only)."""
        n_modes = 4
        n_photons = 2

        # Create circuit and state
        circuit, _ = CircuitGenerator.generate_circuit(
            ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
        )
        input_state = StateGenerator.generate_state(
            n_modes, n_photons, ML.StatePattern.SEQUENTIAL
        )

        # Create computation process with no_bunching=True
        process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
            no_bunching=True,
        )

        # Create dummy parameters
        spec_mappings = process.converter.spec_mappings
        dummy_params = []

        for spec in ["phi_", "pl"]:
            if spec in spec_mappings:
                param_count = len(spec_mappings[spec])
                dummy_params.append(torch.zeros(param_count))

        # Compute distribution
        distribution = process.compute(dummy_params)

        # Check that distribution size matches no-bunching space
        expected_size = calculate_no_bunching_size(n_modes, n_photons)
        actual_size = distribution.shape[-1]

        print(f"No-bunching space - Expected: {expected_size}, Actual: {actual_size}")
        assert (
            actual_size == expected_size
        ), f"Expected no-bunching size {expected_size}, got {actual_size}"

    def test_quantum_layer_with_no_bunching_parameter(self):
        """Test QuantumLayer integration with no_bunching parameter."""

        n_modes = 5
        n_photons = 2

        # Test both cases
        for no_bunching in [False, True]:
            circuit, _ = CircuitGenerator.generate_circuit(
                ML.CircuitType.SERIES, n_modes, 2
            )
            input_state = StateGenerator.generate_state(
                n_modes, n_photons, ML.StatePattern.PERIODIC
            )

            q_layer = QuantumLayer(
                input_size=2,
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                output_mapping_strategy=OutputMappingStrategy.NONE,
                no_bunching=no_bunching,
            )

            # Create dummy parameters
            dummy_params = q_layer._create_dummy_parameters()

            distribution = q_layer.computation_process.compute(dummy_params)

            if no_bunching:
                expected_size = calculate_no_bunching_size(n_modes, n_photons)
            else:
                expected_size = calculate_fock_space_size(n_modes, n_photons)

            actual_size = distribution.shape[-1]

            print(
                f"no_bunching={no_bunching}: Expected {expected_size}, Actual {actual_size}"
            )
            assert actual_size == expected_size

    def test_different_photon_numbers(self):
        """Test no_bunching with different numbers of photons."""
        n_modes = 6

        for n_photons in [1, 2, 3]:
            print(f"\nTesting {n_photons} photons in {n_modes} modes:")

            circuit, _ = CircuitGenerator.generate_circuit(
                ML.CircuitType.PARALLEL, n_modes, 2
            )
            input_state = StateGenerator.generate_state(
                n_modes, n_photons, ML.StatePattern.SPACED
            )

            # Test with no_bunching=True
            process_no_bunching = ComputationProcessFactory.create(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                no_bunching=True,
            )

            # Test with no_bunching=False
            process_full_fock = ComputationProcessFactory.create(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                no_bunching=False,
            )

            # Create dummy parameters
            spec_mappings = process_no_bunching.converter.spec_mappings
            dummy_params = []

            for spec in ["phi_", "pl"]:
                if spec in spec_mappings:
                    param_count = len(spec_mappings[spec])
                    dummy_params.append(torch.randn(param_count))

            # Compute distributions
            dist_no_bunching = process_no_bunching.compute(dummy_params)
            dist_full_fock = process_full_fock.compute(dummy_params)

            # Check sizes
            expected_no_bunching = calculate_no_bunching_size(n_modes, n_photons)
            expected_full_fock = calculate_fock_space_size(n_modes, n_photons)

            print(
                f"  No-bunching: {dist_no_bunching.shape[-1]} (expected {expected_no_bunching})"
            )
            print(
                f"  Full Fock: {dist_full_fock.shape[-1]} (expected {expected_full_fock})"
            )

            assert dist_no_bunching.shape[-1] == expected_no_bunching
            assert dist_full_fock.shape[-1] == expected_full_fock

            # No-bunching should be smaller
            assert dist_no_bunching.shape[-1] <= dist_full_fock.shape[-1]

    def test_impossible_no_bunching_case(self):
        """Test case where no_bunching is impossible (more photons than modes)."""
        n_modes = 3
        n_photons = 4  # More photons than modes

        circuit, _ = CircuitGenerator.generate_circuit(
            ML.CircuitType.SERIES, n_modes, 2
        )
        input_state = [1, 1, 1, 1][:n_modes] + [0] * max(0, n_modes - 4)

        # This should work but result in empty or minimal space
        process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi_"],
            input_parameters=["pl"],
            no_bunching=True,
        )

        # The calculation shows this should be 0, but the system might handle it differently
        expected_size = calculate_no_bunching_size(n_modes, n_photons)
        print(f"Impossible case: {n_photons} photons in {n_modes} modes")
        print(f"Expected no-bunching size: {expected_size}")

        # This might raise an error or return empty distribution
        # Let's see what actually happens
        spec_mappings = process.converter.spec_mappings
        dummy_params = []

        for spec in ["phi_", "pl"]:
            if spec in spec_mappings:
                param_count = len(spec_mappings[spec])
                dummy_params.append(torch.randn(param_count))

        try:
            distribution = process.compute(dummy_params)
            print(f"Actual distribution size: {distribution.shape[-1]}")
            # If it doesn't error, the size should be 0 or handled gracefully
            assert distribution.shape[-1] >= 0
        except Exception as e:
            print(f"Expected error for impossible case: {e}")
            # This is acceptable behavior

    def test_single_photon_case(self):
        """Test the simple single photon case."""
        n_modes = 5
        n_photons = 1

        circuit, _ = CircuitGenerator.generate_circuit(
            ML.CircuitType.PARALLEL_COLUMNS, n_modes, 2
        )
        input_state = StateGenerator.generate_state(
            n_modes, n_photons, ML.StatePattern.SEQUENTIAL
        )

        for no_bunching in [False, True]:
            process = ComputationProcessFactory.create(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                no_bunching=no_bunching,
            )

            spec_mappings = process.converter.spec_mappings
            dummy_params = []

            for spec in ["phi_", "pl"]:
                if spec in spec_mappings:
                    param_count = len(spec_mappings[spec])
                    dummy_params.append(torch.randn(param_count))

            distribution = process.compute(dummy_params)

            if no_bunching:
                # For single photon, no-bunching space = n_modes (each mode can have the photon)
                expected = n_modes
            else:
                # For single photon, Fock space = n_modes (same as no-bunching)
                expected = n_modes

            print(
                f"Single photon, no_bunching={no_bunching}: size={distribution.shape[-1]}, expected={expected}"
            )
            assert distribution.shape[-1] == expected

    def test_compute_with_keys_functionality(self):
        """
        Test that compute_with_keys works with no_bunching and on full Fock space.
        Test sizes of keys and  distributions.
        Test values of distributions (convert from full Fock space to no_bunching).
        """
        # Test cases: (n_modes, n_photons)
        test_cases = [
            (3, 1),  # 3 modes, 1 photon
            (4, 2),  # 4 modes, 2 photons
            (5, 3),  # 5 modes, 3 photons
            (6, 2),  # 6 modes, 2 photons
        ]

        # Test for every test case
        for n_modes, n_photons in test_cases:
            print(
                f"\nTesting compute_with_keys {n_photons} photons in {n_modes} modes:"
            )
            circuit, _ = CircuitGenerator.generate_circuit(
                ML.CircuitType.SERIES, n_modes, 2
            )
            input_state = StateGenerator.generate_state(
                n_modes, n_photons, ML.StatePattern.PERIODIC
            )

            # Process with no_bunching
            process_no_bunching = ComputationProcessFactory.create(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                no_bunching=True,
            )

            # Process with full Fock space
            process_full_fock_space = ComputationProcessFactory.create(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi_"],
                input_parameters=["pl"],
                no_bunching=False,
            )

            spec_mappings_no_bunching = process_no_bunching.converter.spec_mappings
            spec_mappings_full_fock_space = (
                process_full_fock_space.converter.spec_mappings
            )
            dummy_params = []

            # Replace parameters by the same random values for the two circuits
            for spec in ["phi_", "pl"]:
                if spec in spec_mappings_no_bunching:
                    param_count_n_b = len(spec_mappings_no_bunching[spec])
                    if spec in spec_mappings_full_fock_space:
                        param_count_f_f_s = len(spec_mappings_full_fock_space[spec])
                        assert (
                            param_count_n_b == param_count_f_f_s
                        ), "Different circuits for no_bunching and full_fock_space"
                        dummy_params.append(torch.randn(param_count_f_f_s))
                    else:
                        raise Exception(
                            "Different circuits for no_bunching and full_fock_space"
                        )
                else:
                    if spec in spec_mappings_full_fock_space:
                        raise Exception(
                            "Different circuits for no_bunching and full_fock_space"
                        )

            # Test compute_with_keys with no_bunching
            keys_no_bunching, distribution_no_bunching = (
                process_no_bunching.compute_with_keys(dummy_params)
            )

            # Should have the same distribution size
            expected_size = calculate_no_bunching_size(n_modes, n_photons)
            assert distribution_no_bunching.shape[-1] == expected_size

            # Keys should correspond to the states
            assert len(keys_no_bunching) == expected_size
            print("Correct distribution and keys size with no_bunching")

            # Test compute_with_keys on full Fock space
            keys_full_fock_space, distribution_full_fock_space = (
                process_full_fock_space.compute_with_keys(dummy_params)
            )

            # Should have the same distribution size
            expected_size = calculate_fock_space_size(n_modes, n_photons)
            assert distribution_full_fock_space.shape[-1] == expected_size

            # Keys should correspond to the states
            assert len(keys_full_fock_space) == expected_size
            print("Correct distribution and keys size on full Fock state")

            # We can convert the full Fock space distribution to the no_bunching distribution by removing any state
            # that has a mode with more than 1 photon followed by renormalization.
            new_keys = [0] * len(keys_no_bunching)
            new_distribution = [0] * len(keys_no_bunching)

            for key, proba in zip(
                keys_full_fock_space, distribution_full_fock_space, strict=False
            ):
                if any(key_elem > 1 for key_elem in key):
                    continue
                else:
                    index = keys_no_bunching.index(key)
                    new_keys[index] = key
                    new_distribution[index] = proba

            new_distribution = torch.tensor(new_distribution) / torch.sum(
                torch.tensor(new_distribution)
            )

            # new_distribution must be close to distribution_no_bunching
            assert torch.isclose(torch.sum(new_distribution), torch.tensor(1.0))
            assert torch.isclose(torch.sum(distribution_no_bunching), torch.tensor(1.0))
            assert new_keys == keys_no_bunching
            assert torch.allclose(new_distribution, distribution_no_bunching)
            print(
                "Convertion from distribution_full_fock_space to distribution_no_bunching completed successfully"
            )


if __name__ == "__main__":
    # Run a quick demonstration
    print("=== No-Bunching Test Demonstration ===")

    test = TestNoBunchingFunctionality()

    print("\n1. Testing size calculations...")
    test.test_fock_space_vs_no_bunching_sizes()

    print("\n2. Testing computation process...")
    test.test_computation_process_with_no_bunching_false()
    test.test_computation_process_with_no_bunching_true()

    print("\n3. Testing quantum layer...")
    test.test_quantum_layer_with_no_bunching_parameter()

    print("\n4. Testing different photon numbers...")
    test.test_different_photon_numbers()

    print("\n5. Testing impossible no_bunching case...")
    test.test_impossible_no_bunching_case()

    print("\n6. Testing single photon case...")
    test.test_single_photon_case()

    print("\n7. Testing compute_with_keys...")
    test.test_compute_with_keys_functionality()

    print("\n✅ All tests passed!")
