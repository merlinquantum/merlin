# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject of the following conditions:
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
Correctness tests for SLOS core functions.
Ensures that build_graph, compute, and compute_pa_inc produce correct outputs.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from merlin.pcvl_pytorch.slos_torchscript import (
    build_slos_distribution_computegraph,
    SLOSComputeGraph,
)


class TestSLOSCorrectness:
    """Test suite to validate correctness of SLOS core functions."""

    def create_test_unitary(self, m: int, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
        """Create a test unitary matrix."""
        # Use a simple beam splitter unitary for m=2 case for known results
        if m == 2:
            # 50:50 beam splitter
            bs = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / np.sqrt(2)
            return bs.to(dtype=dtype)
        else:
            # For larger systems, create random unitary
            real_part = torch.randn(m, m, dtype=torch.float32)
            imag_part = torch.randn(m, m, dtype=torch.float32)
            u = torch.complex(real_part, imag_part)
            q, _ = torch.linalg.qr(u)
            return q.to(dtype=dtype)

    def test_build_graph_correctness(self):
        """Test that build_slos_distribution_computegraph creates valid graph."""
        m = 4
        n_photons = 2
        input_state = [1, 1, 0, 0]

        graph = build_slos_distribution_computegraph(
            m=m,
            n_photons=n_photons,
            no_bunching=True,
            keep_keys=True,
            device="cpu",
            dtype=torch.float32
        )

        # Validate graph properties
        assert graph.m == m
        assert graph.n_photons == n_photons
        assert graph.no_bunching == True
        assert graph.keep_keys == True
        assert graph.dtype == torch.float32
        assert graph.complex_dtype == torch.cfloat

        # Validate that graph has expected structure
        assert len(graph.vectorized_operations) == n_photons
        assert graph.final_keys is not None
        assert len(graph.final_keys) > 0

    @pytest.mark.parametrize("m,n_photons", [(2, 1), (4, 2), (6, 3)])
    def test_compute_correctness(self, m: int, n_photons: int):
        """Test that compute function produces valid probability distributions."""
        # Create input state (distribute photons)
        input_state = [0] * m
        for i in range(n_photons):
            input_state[i] = 1

        # Build graph
        graph = build_slos_distribution_computegraph(
            m=m,
            n_photons=n_photons,
            no_bunching=True,
            keep_keys=True,
            device="cpu",
            dtype=torch.float32
        )

        # Create test unitary
        unitary = self.create_test_unitary(m, torch.cfloat)

        # Run compute
        keys, probs = graph.compute(unitary, input_state)

        # Validate output
        assert keys is not None, "Keys should not be None when keep_keys=True"
        assert len(keys) == len(probs), "Number of keys should match number of probabilities"
        
        # Check probability normalization
        prob_sum = probs.sum().item()
        assert 0.95 <= prob_sum <= 1.05, f"Probabilities should sum to ~1, got {prob_sum}"
        
        # Check non-negativity
        assert (probs >= 0).all(), "All probabilities should be non-negative"
        
        # Check photon number conservation
        n_input_photons = sum(input_state)
        for key in keys:
            assert sum(key) == n_input_photons, f"Output state {key} doesn't conserve photons"

    def test_compute_pa_inc_correctness(self):
        """Test that compute_pa_inc produces correct incremental results."""
        m = 4
        n_photons = 2
        
        # Create input states - both with 2 photons but different distributions
        input_state_prev = [1, 1, 0, 0]  # 2 photons in modes 0,1
        input_state = [2, 0, 0, 0]       # 2 photons in mode 0 (different distribution)

        # Build graph for 2 photons
        graph = build_slos_distribution_computegraph(
            m=m,
            n_photons=n_photons,
            no_bunching=False,  # Allow bunching for this test
            keep_keys=True,
            device="cpu",
            dtype=torch.float32
        )

        # Create test unitary
        unitary = self.create_test_unitary(m, torch.cfloat)

        # First run regular compute to set up prev_amplitudes
        keys_prev, probs_prev = graph.compute(unitary, input_state_prev)

        # Now run incremental compute
        keys_inc, probs_inc = graph.compute_pa_inc(unitary, input_state_prev, input_state)

        # Validate output
        assert keys_inc is not None
        assert len(keys_inc) == len(probs_inc)
        
        # Check probability normalization
        prob_sum = probs_inc.sum().item()
        assert 0.95 <= prob_sum <= 1.05, f"Probabilities should sum to ~1, got {prob_sum}"
        
        # Check non-negativity
        assert (probs_inc >= 0).all(), "All probabilities should be non-negative"
        
        # Check photon number conservation
        n_input_photons = sum(input_state)
        for key in keys_inc:
            assert sum(key) == n_input_photons, f"Output state {key} doesn't conserve photons"

        # Compare with direct compute for same state
        keys_direct, probs_direct = graph.compute(unitary, input_state)
        
        # Results should be very close (allowing for numerical precision)
        assert len(keys_inc) == len(keys_direct)
        prob_diff = torch.abs(probs_inc - probs_direct).max().item()
        assert prob_diff < 1e-6, f"Incremental and direct results differ by {prob_diff}"

    def test_beam_splitter_known_result(self):
        """Test known result for simple beam splitter case."""
        m = 2
        n_photons = 1
        input_state = [1, 0]  # One photon in first mode

        # Build graph
        graph = build_slos_distribution_computegraph(
            m=m,
            n_photons=n_photons,
            no_bunching=True,
            keep_keys=True,
            device="cpu",
            dtype=torch.float32
        )

        # 50:50 beam splitter
        bs_unitary = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / np.sqrt(2)

        # Run compute
        keys, probs = graph.compute(bs_unitary, input_state)

        # For a 50:50 beam splitter with one photon input in mode 0,
        # we should get equal probability (0.5) in both output modes
        expected_keys = [(0, 1), (1, 0)]
        expected_probs = [0.5, 0.5]

        # Sort results for comparison
        sorted_indices = sorted(range(len(keys)), key=lambda i: keys[i])
        sorted_keys = [keys[i] for i in sorted_indices]
        sorted_probs = [probs[i].item() for i in sorted_indices]

        # Check that we have the expected keys (order may vary)
        assert set(sorted_keys) == set(expected_keys), f"Expected keys {expected_keys}, got {sorted_keys}"
        
        for i, (actual, expected) in enumerate(zip(sorted_probs, expected_probs)):
            assert abs(actual - expected) < 1e-6, f"Probability {i}: expected {expected}, got {actual}"

    def test_dtype_consistency(self):
        """Test that different dtypes produce consistent results."""
        m = 4
        n_photons = 2
        input_state = [1, 1, 0, 0]

        # Create a fixed unitary for consistent comparison
        torch.manual_seed(42)  # Set seed for reproducibility
        real_part = torch.randn(m, m, dtype=torch.float32)
        imag_part = torch.randn(m, m, dtype=torch.float32)
        u = torch.complex(real_part, imag_part)
        base_unitary, _ = torch.linalg.qr(u)

        # Test with different precisions
        dtypes = [(torch.float32, torch.cfloat), (torch.float64, torch.cdouble)]
        results = []

        for float_dtype, complex_dtype in dtypes:
            graph = build_slos_distribution_computegraph(
                m=m,
                n_photons=n_photons,
                no_bunching=True,
                keep_keys=True,
                device="cpu",
                dtype=float_dtype
            )

            unitary = base_unitary.to(dtype=complex_dtype)
            keys, probs = graph.compute(unitary, input_state)
            results.append((keys, probs))

        # Compare results between different precisions
        keys1, probs1 = results[0]
        keys2, probs2 = results[1]

        assert keys1 == keys2, "Keys should be identical across dtypes"
        
        # Probabilities should be close (allowing for precision differences)
        prob_diff = torch.abs(probs1.double() - probs2.double()).max().item()
        assert prob_diff < 1e-5, f"Results differ significantly between dtypes: {prob_diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that CPU and GPU produce consistent results."""
        m = 4
        n_photons = 2
        input_state = [1, 1, 0, 0]

        devices = ["cpu", "cuda"]
        results = []

        for device in devices:
            graph = build_slos_distribution_computegraph(
                m=m,
                n_photons=n_photons,
                no_bunching=True,
                keep_keys=True,
                device=device,
                dtype=torch.float32
            )

            unitary = self.create_test_unitary(m, torch.cfloat).to(device)
            keys, probs = graph.compute(unitary, input_state)
            results.append((keys, probs.cpu()))

        # Compare CPU and GPU results
        keys_cpu, probs_cpu = results[0]
        keys_gpu, probs_gpu = results[1]

        assert keys_cpu == keys_gpu, "Keys should be identical across devices"
        
        prob_diff = torch.abs(probs_cpu - probs_gpu).max().item()
        assert prob_diff < 0.2, f"CPU and GPU results differ significantly: {prob_diff}"


if __name__ == "__main__":
    # Run a quick correctness check
    test = TestSLOSCorrectness()
    
    print("Running SLOS correctness tests...")
    
    test.test_build_graph_correctness()
    print("✓ Build graph correctness test passed")
    
    test.test_compute_correctness(4, 2)
    print("✓ Compute correctness test passed")
    
    test.test_compute_pa_inc_correctness()
    print("✓ Compute PA inc correctness test passed")
    
    test.test_beam_splitter_known_result()
    print("✓ Beam splitter known result test passed")
    
    test.test_dtype_consistency()
    print("✓ Dtype consistency test passed")
    
    print("All correctness tests completed successfully!")