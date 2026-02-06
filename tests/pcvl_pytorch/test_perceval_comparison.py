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
    N_SAMPLES = 1_000_000

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
            pcvl
            .Circuit(self.N_MODES, name="chip")
            .add(0, pre_U)
            .add(0, phases_U, merge=False)
            .add(0, reservoir_U, merge=False)
        )

        # Create input state (photons in the first modes)
        input_state = [1] * self.N_PHOTONS + [0] * (self.N_MODES - self.N_PHOTONS)

        # Parameter names present in the circuit
        circuit_params = [p.name for p in chip.get_parameters()]
        input_size = len(circuit_params)

        # --- FIX: do not pass `shots` to QuantumLayer.__init__ (modern API) ---
        merlin_layer = ML.QuantumLayer(
            input_size=input_size,
            circuit=chip,
            input_state=input_state,
            n_photons=self.N_PHOTONS,
            input_parameters=["φ"],
            trainable_parameters=[],
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        # Create dummy input to get parameters (add batch dimension)
        dummy_input = torch.zeros(1, input_size, dtype=torch.float32)

        # Get MerLin probability distribution (deterministic, no sampling)
        with torch.no_grad():
            merlin_params = merlin_layer.prepare_parameters([dummy_input])
            unitary = merlin_layer.computation_process.converter.to_tensor(
                *merlin_params
            )

            # exact amplitudes via simulation graph
            keys, merlin_distribution = (
                merlin_layer.computation_process.simulation_graph.compute(
                    unitary, input_state
                )
            )
            probabilities = merlin_distribution.real**2 + merlin_distribution.imag**2
            sum_probs = probabilities.sum(dim=1, keepdim=True)

            # Normalize only when sum > 0
            valid_entries = sum_probs > 0
            if valid_entries.any():
                probabilities = torch.where(
                    valid_entries,
                    probabilities
                    / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    probabilities,
                )

            merlin_probs = probabilities.detach().numpy()

        # Set parameter values in Perceval to match MerLin's input tensor
        for param_tensor in merlin_params:
            param_array = param_tensor.detach().numpy()
            for p, val in zip(parameters, param_array.flatten(), strict=True):
                p.set_value(val)

        # Build Perceval processor/QPU
        qpu = pcvl.Processor("SLOS", chip)
        qpu.with_input(pcvl.BasicState(input_state))
        qpu.min_detected_photons_filter(self.N_PHOTONS)

        # Empirical probability distribution from Perceval sampling
        sampler = Sampler(qpu)
        perceval_sample_count = sampler.sample_count(self.N_SAMPLES)["results"]

        # Map BasicState -> index into MerLin keys
        key_to_index = {pcvl.BasicState(k): i for i, k in enumerate(keys)}
        perceval_probs = np.zeros(len(keys), dtype=float)
        for state, count in perceval_sample_count.items():
            idx = key_to_index.get(state)
            if idx is not None:
                perceval_probs[idx] = count / self.N_SAMPLES

        # Normalize (defensive)
        total = perceval_probs.sum()
        if total > 0:
            perceval_probs /= total

        # Compare probability distributions (allow small sampling differences)
        diff = np.abs(merlin_probs - perceval_probs)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        assert max_diff < self.TOLERANCE, (
            f"Probability distributions differ by more than {self.TOLERANCE}: max_diff={max_diff}"
        )
        assert mean_diff < self.TOLERANCE / 2, f"Mean difference too large: {mean_diff}"
