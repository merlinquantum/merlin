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
Integration test for dtype propagation through QuantumLayer.

This test builds actual Perceval circuits and verifies that the dtype
parameter propagates correctly through the entire flow, including the
measurement mapping mask.

Run from project root:
    pytest tests/test_dtype_propagation.py -v
"""

from __future__ import annotations

import perceval as pcvl
import pytest
import torch

import merlin
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


def build_circuit(n_modes: int) -> pcvl.Circuit:
    """Build a simple interferometer circuit with input encoding."""
    circuit = pcvl.Circuit(n_modes)
    # Add beam splitters
    for k in range(0, n_modes, 2):
        if k + 1 < n_modes:
            circuit.add(k, pcvl.BS())
    # Add phase shifter for input encoding
    circuit.add(0, pcvl.PS(pcvl.P("px0")))
    return circuit


def build_circuit_no_params(n_modes: int) -> pcvl.Circuit:
    """
    Build a simple interferometer circuit with NO symbolic parameters.

    This is important for amplitude_encoding=True tests: amplitude encoding
    provides a complex statevector input, and we don't want any classical
    input parameter specs to be required by CircuitConverter.
    """
    circuit = pcvl.Circuit(n_modes)
    for k in range(0, n_modes, 2):
        if k + 1 < n_modes:
            circuit.add(k, pcvl.BS())
    # Fixed phase (numeric), not a Perceval Parameter
    circuit.add(0, pcvl.PS(0.0))
    return circuit


def _basis_size_from_layer(layer: QuantumLayer) -> int:
    """
    Basis size resolution for AMPLITUDES/amplitude-encoding path.

    Documented usage uses `len(layer.output_keys)`. Keep defensive.
    """
    if hasattr(layer, "output_keys") and layer.output_keys is not None:
        return len(layer.output_keys)
    mm = getattr(layer, "measurement_mapping", None)
    if mm is not None and hasattr(mm, "output_keys") and mm.output_keys is not None:
        return len(mm.output_keys)
    raise AttributeError(
        "Could not determine basis size for AMPLITUDES test: "
        "expected `layer.output_keys` (or `layer.measurement_mapping.output_keys`)."
    )


class TestQuantumLayerDtypePropagation:
    """Test dtype propagation through the full QuantumLayer flow."""

    @pytest.fixture
    def circuit_2mode(self) -> pcvl.Circuit:
        """2-mode circuit fixture."""
        return build_circuit(n_modes=2)

    @pytest.fixture
    def circuit_4mode(self) -> pcvl.Circuit:
        """4-mode circuit fixture."""
        return build_circuit(n_modes=4)

    @pytest.fixture
    def circuit_2mode_no_params(self) -> pcvl.Circuit:
        """2-mode circuit fixture with no symbolic parameters (for amplitude encoding)."""
        return build_circuit_no_params(n_modes=2)

    def test_mode_expectations_float64_mask_dtype(self, circuit_2mode):
        """Verify measurement mask is created with correct dtype."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        mask = qlayer.measurement_mapping.mask
        assert mask.dtype == torch.float64, (
            f"Expected mask dtype=torch.float64, got {mask.dtype}"
        )

    def test_mode_expectations_float32_mask_dtype(self, circuit_2mode):
        """Verify float32 still works (backward compatibility)."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float32,
        )

        mask = qlayer.measurement_mapping.mask
        assert mask.dtype == torch.float32

    def test_mode_expectations_float64_forward_pass(self, circuit_2mode):
        """The original bug: forward pass should not raise dtype mismatch."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        x = torch.zeros(1, 1, dtype=torch.float64)

        # This would previously raise:
        # RuntimeError: expected m1 and m2 to have the same dtype, but got: double != float
        output = qlayer(x)

        assert output.dtype == torch.float64
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == 2  # num modes

    def test_mode_expectations_float32_forward_pass(self, circuit_2mode):
        """Verify float32 forward pass still works."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float32,
        )

        x = torch.zeros(1, 1, dtype=torch.float32)
        output = qlayer(x)

        assert output.dtype == torch.float32

    def test_mode_expectations_batch_forward(self, circuit_2mode):
        """Test batched forward pass with float64."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        batch_size = 8
        x = torch.randn(batch_size, 1, dtype=torch.float64)
        output = qlayer(x)

        assert output.dtype == torch.float64
        assert output.shape == (batch_size, 2)

    def test_mode_expectations_4mode_circuit(self, circuit_4mode):
        """Test with larger circuit."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_4mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0, 1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        x = torch.zeros(1, 1, dtype=torch.float64)
        output = qlayer(x)

        assert output.dtype == torch.float64
        assert output.shape[1] == 4  # 4 modes

    def test_probabilities_strategy_float64(self, circuit_2mode):
        """Verify PROBABILITIES strategy works with float64."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            dtype=torch.float64,
        )

        x = torch.zeros(1, 1, dtype=torch.float64)
        output = qlayer(x)

        assert output.dtype == torch.float64

    def test_fock_computation_space_float64(self, circuit_2mode):
        """Test FOCK computation space with float64."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.FOCK,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        mask = qlayer.measurement_mapping.mask
        assert mask.dtype == torch.float64

        x = torch.zeros(1, 1, dtype=torch.float64)
        output = qlayer(x)
        assert output.dtype == torch.float64

    def test_gradient_flow_float64(self, circuit_2mode):
        """Verify gradients flow correctly with float64."""
        qlayer = QuantumLayer(
            input_size=1,
            circuit=circuit_2mode,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=ComputationSpace.UNBUNCHED,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch.float64,
        )

        x = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        output = qlayer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.dtype == torch.float64

    def test_amplitudes_amplitude_encoding_float32_outputs_cfloat(
        self, circuit_2mode_no_params
    ):
        """
        MeasurementStrategy.AMPLITUDES (amplitude_encoding=True):
        ensure dtype=torch.float32 leads to complex64 (torch.cfloat) amplitudes.

        IMPORTANT: use a circuit with *no symbolic parameters* to avoid requiring
        classical input specs (e.g. px0) when amplitude encoding is enabled.
        """
        layer = QuantumLayer(
            circuit=circuit_2mode_no_params,
            n_photons=1,
            amplitude_encoding=True,
            measurement_strategy=MeasurementStrategy.NONE,
            dtype=torch.float32,
        )

        num_states = _basis_size_from_layer(layer)
        psi_in = torch.randn(num_states, dtype=torch.cfloat)
        psi_in = psi_in / psi_in.norm()

        psi_out = layer(psi_in)

        assert psi_out.dtype == torch.cfloat, (
            f"Expected AMPLITUDES output dtype=torch.cfloat, got {psi_out.dtype}"
        )
        assert psi_out.shape in {(num_states,), (1, num_states)}

    def test_amplitudes_amplitude_encoding_float64_outputs_cdouble(
        self, circuit_2mode_no_params
    ):
        """
        MeasurementStrategy.AMPLITUDES (amplitude_encoding=True):
        ensure dtype=torch.float64 leads to complex128 (torch.cdouble) amplitudes.
        """
        layer = QuantumLayer(
            circuit=circuit_2mode_no_params,
            n_photons=1,
            amplitude_encoding=True,
            measurement_strategy=MeasurementStrategy.NONE,
            dtype=torch.float64,
        )

        num_states = _basis_size_from_layer(layer)
        psi_in = torch.randn(num_states, dtype=torch.cdouble)
        psi_in = psi_in / psi_in.norm()

        psi_out = layer(psi_in)

        assert psi_out.dtype == torch.cdouble, (
            f"Expected AMPLITUDES output dtype=torch.cdouble, got {psi_out.dtype}"
        )
        assert psi_out.shape in {(num_states,), (1, num_states)}


class TestOriginalBugReproduction:
    """
    Direct reproduction of the original bug report.

    This test class mirrors the exact reproduction script to ensure
    the fix addresses the reported issue.
    """

    def test_original_bug_is_fixed(self):
        """
        Reproduction of the original bug report.

        Before the fix, this would output:
            Building QuantumLayer with requested dtype=torch.float64
            Measurement mask dtype=torch.float32
            Forward call raised: RuntimeError(... double != float)

        After the fix, it should succeed with mask dtype=torch.float64.
        """
        # Build circuit (same as bug report)
        n_modes = 2
        circuit = pcvl.Circuit(n_modes)
        for k in range(0, n_modes, 2):
            if k + 1 < n_modes:
                circuit.add(k, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("px0")))

        # Build QuantumLayer with float64 (same as bug report)
        torch_dtype = torch.float64
        qlayer = merlin.QuantumLayer(
            input_size=1,
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["px"],
            input_state=[1, 0],
            computation_space=merlin.ComputationSpace.UNBUNCHED,
            measurement_strategy=merlin.MeasurementStrategy.MODE_EXPECTATIONS,
            dtype=torch_dtype,
        )

        # Verify mask dtype is now correct
        mask = qlayer.measurement_mapping.mask
        assert mask.dtype == torch.float64, (
            f"BUG NOT FIXED: mask dtype is {mask.dtype}, expected torch.float64"
        )

        # Verify forward pass succeeds (this was the crash point)
        x = torch.zeros(1, 1, dtype=torch_dtype)
        try:
            output = qlayer(x)
        except RuntimeError as exc:
            if "double != float" in str(exc) or "same dtype" in str(exc):
                pytest.fail(f"BUG NOT FIXED: dtype mismatch in forward pass: {exc}")
            raise

        assert output.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
