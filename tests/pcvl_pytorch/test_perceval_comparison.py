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
Tests comparing MerLin probability distributions with direct Perceval QPU implementation.
"""

import numpy as np
import perceval as pcvl
import torch
from perceval.algorithm.sampler import Sampler

import merlin as ML


class TestPercevalComparison:
    """Test suite comparing MerLin and direct Perceval approaches."""

    TOLERANCE = 0.005
    N_MODES = 12
    N_PHOTONS = 3
    N_SAMPLES = 1000000

    def test_probability_distribution_comparison_simple(self):
        """Test that MerLin gives same probability distribution as direct Perceval QPU."""
        # Configuration

        # Create custom Perceval circuit following the provided pattern
        from perceval.components import BS, PS

        # Create random unitary for decomposition
        U = pcvl.Matrix.random_unitary(self.N_MODES)

        # Decomposition of the unitary for Pre-circuit and Reservoir
        pre_U = pcvl.Circuit.decomposition(
            U, BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")), phase_shifter_fn=PS
        )
        reservoir_U = pre_U.copy()

        # Add phases
        phases_U = pcvl.Circuit(self.N_MODES, name="phases")
        parameters = []
        for i in range(self.N_MODES):
            parameter = pcvl.P(f"φ{i}")
            phases_U.add(i, PS(phi=parameter))
            parameters.append(parameter)

        chip = (
            pcvl.Circuit(self.N_MODES, name="chip")
            .add(0, pre_U)
            .add(0, phases_U, merge=False)
            .add(0, reservoir_U, merge=False)
        )

        # Create input state
        input_state = [1] * self.N_PHOTONS + [0] * (
            self.N_MODES - self.N_PHOTONS
        )  # Place photons in first modes

        # Check what parameters the circuit actually has
        circuit_params = [p.name for p in chip.get_parameters()]

        # Select some parameters for training and some for input
        all_params = circuit_params
        input_size = len(all_params)

        # Create MerLin quantum layer with custom circuit
        merlin_layer = ML.QuantumLayer(
            input_size=len(all_params),
            circuit=chip,
            input_state=input_state,
            n_photons=self.N_PHOTONS,
            input_parameters=["φ"],
            trainable_parameters=[],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            shots=0,
        )

        # Create dummy input to get parameters (add batch dimension)
        dummy_input = torch.zeros(1, input_size, dtype=torch.float32)

        # Get MerLin probability distribution (no gradients to avoid sampling warnings)
        with torch.no_grad():
            merlin_params = merlin_layer.prepare_parameters([dummy_input])
            unitary = merlin_layer.computation_process.converter.to_tensor(
                *merlin_params
            )
            # perfect distribution (no sampling)
            keys, merlin_distribution = (
                merlin_layer.computation_process.simulation_graph.compute(
                    unitary, input_state
                )
            )
            probabilities = merlin_distribution.real**2 + merlin_distribution.imag**2
            sum_probs = probabilities.sum(dim=1, keepdim=True)
            # Only normalize when sum > 0 to avoid division by zero
            valid_entries = sum_probs > 0
            if valid_entries.any():
                probabilities = torch.where(
                    valid_entries,
                    probabilities
                    / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    probabilities,
                )
            merlin_probs = probabilities.numpy()

        # Set parameter values to match MerLin's computation
        for param_tensor in merlin_params:
            param_array = param_tensor.detach().numpy()
            for p, val in zip(parameters, param_array.flatten(), strict=True):
                p.set_value(val)

        # Create QPU processor
        qpu = pcvl.Processor("SLOS", chip)
        qpu.with_input(pcvl.BasicState(input_state))
        qpu.min_detected_photons_filter(self.N_PHOTONS)

        # Get probability distribution from Perceval
        sampler = Sampler(qpu)
        perceval_sample_count = sampler.sample_count(self.N_SAMPLES)["results"]

        # Convert to probability distribution
        perceval_probs = np.zeros(len(keys))

        # Create mapping from BasicState to index
        key_to_index = {}
        for i, key in enumerate(keys):
            key_to_index[pcvl.BasicState(key)] = i

        # Fill perceval probability array
        for state, count in perceval_sample_count.items():
            if state in key_to_index:
                perceval_probs[key_to_index[state]] = count / self.N_SAMPLES

        perceval_probs /= sum(perceval_probs)  # Normalize to ensure it sums to 1

        # Compare probability distributions
        # Allow for small numerical differences due to sampling
        tolerance = (
            self.TOLERANCE
        )  # 1% tolerance due to computation and sampling differences

        # Check that distributions are similar
        diff = np.abs(merlin_probs - perceval_probs)
        max_diff = np.max(diff)

        # print(f"MerLin probabilities: {merlin_probs}")
        # print(f"Perceval probabilities: {perceval_probs}")
        # print(f"Maximum difference: {max_diff}")
        # print(f"Mean absolute difference: {np.mean(diff)}")

        # The distributions should be reasonably close
        assert max_diff < tolerance, (
            f"Probability distributions differ by more than {tolerance}: max_diff={max_diff}"
        )
        assert np.mean(diff) < tolerance / 2, (
            f"Mean difference too large: {np.mean(diff)}"
        )
