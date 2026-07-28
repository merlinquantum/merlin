# MIT License
#
# Copyright (c) 2026 Quandela
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

"""Tests for automatic Clements decomposition of Unitary blocks under phase noise.

The Clements fit reproduces each target unitary to ~1e-6 fidelity error, which
translates to matrix entries agreeing to ~1e-3, so unitaries are compared with
``_ELEMENTWISE_ATOL`` (the fit's global phase is compensated in production
code, making elementwise comparison valid) — never through raw phase values,
which differ between fits.
"""

from __future__ import annotations

import numpy as np
import perceval as pcvl
import pytest
import torch
from perceval.components import PS, Unitary

import merlin.pcvl_pytorch.locirc_to_tensor as locirc_to_tensor
from merlin.pcvl_pytorch.locirc_to_tensor import (
    CircuitConverter,
    _DECOMPOSITION_CACHE,
    _decompose_unitaries,
)

_ELEMENTWISE_ATOL = 1e-2
_FIDELITY_ATOL = 1e-5


@pytest.fixture(autouse=True)
def clear_decomposition_cache():
    """Keep process-global decomposition state isolated between tests."""
    locirc_to_tensor._DECOMPOSITION_CACHE.clear()
    yield
    locirc_to_tensor._DECOMPOSITION_CACHE.clear()


def _circuit_unitary(circuit: pcvl.Circuit) -> np.ndarray:
    return np.asarray(circuit.compute_unitary(), dtype=complex)


def _fidelity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.abs(np.trace(u.conj().T @ v)) / u.shape[-1])


def _has_black_box_unitary(circuit: pcvl.Circuit) -> bool:
    return any(
        isinstance(c, Unitary) and not isinstance(c, pcvl.PERM) for _, c in circuit
    )


def _single_unitary_circuit(m: int = 4) -> pcvl.Circuit:
    return pcvl.Circuit(m) // pcvl.Unitary(pcvl.Matrix.random_unitary(m))


def _reservoir_circuit(m: int = 4, n_features: int = 2) -> pcvl.Circuit:
    circuit = pcvl.Circuit(m)
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)))
    for i in range(n_features):
        circuit.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)))
    return circuit


def test_decompose_replaces_unitary_with_equivalent_mesh():
    np.random.seed(0)
    circuit = _single_unitary_circuit(4)

    decomposed = _decompose_unitaries(circuit)

    assert decomposed is not circuit
    assert not _has_black_box_unitary(decomposed)
    assert decomposed.get_parameters() == []
    original = _circuit_unitary(circuit)
    rebuilt = _circuit_unitary(decomposed)
    assert _fidelity(original, rebuilt) == pytest.approx(1.0, abs=_FIDELITY_ATOL)
    assert np.allclose(rebuilt, original, atol=_ELEMENTWISE_ATOL)


def test_decompose_returns_same_object_without_black_box_unitary():
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.PERM([1, 0]))

    assert _decompose_unitaries(circuit) is circuit


def test_decompose_compensates_global_phase_of_sub_mode_blocks():
    # A block on a subset of modes must not pick up the fit's arbitrary
    # global phase: relative to the untouched modes it would be physical.
    # Elementwise comparison of the full-circuit unitary catches it.
    np.random.seed(1)
    circuit = pcvl.Circuit(4)
    circuit.add(1, pcvl.Unitary(pcvl.Matrix.random_unitary(2)))

    decomposed = _decompose_unitaries(circuit)

    assert np.allclose(
        _circuit_unitary(decomposed), _circuit_unitary(circuit), atol=_ELEMENTWISE_ATOL
    )


def test_decomposition_cache_reuses_fixed_mesh(monkeypatch):
    """Reuse a cached mesh after all of its parameters are fixed."""
    np.random.seed(101)
    circuit = _single_unitary_circuit(4)
    _DECOMPOSITION_CACHE.clear()

    first = _decompose_unitaries(circuit)
    first_mesh = next(component for _, component in first)
    first_parameters = [
        float(parameter)
        for parameter in first_mesh.get_parameters(all_params=True)
        if parameter.name.startswith("phL")
    ]

    optimizer_call_count = 0
    original_optimizer = locirc_to_tensor.CircuitOptimizer.optimize_rectangle

    def count_optimizer_calls(self, target):
        nonlocal optimizer_call_count
        optimizer_call_count += 1
        return original_optimizer(self, target)

    monkeypatch.setattr(
        locirc_to_tensor.CircuitOptimizer,
        "optimize_rectangle",
        count_optimizer_calls,
    )
    monkeypatch.setattr(locirc_to_tensor, "_DEFAULT_OPTIMIZE_RECTANGLE", count_optimizer_calls)
    second = _decompose_unitaries(circuit)
    second_mesh = next(component for _, component in second)
    second_parameters = [
        float(parameter)
        for parameter in second_mesh.get_parameters(all_params=True)
        if parameter.name.startswith("phL")
    ]

    assert optimizer_call_count == 0
    assert second_parameters == first_parameters


def test_no_noise_converter_keeps_unitary_fast_path():
    np.random.seed(2)
    circuit = _single_unitary_circuit(4)

    converter = CircuitConverter(circuit, dtype=torch.float64)

    assert converter.circuit is circuit
    assert len(converter.list_rct) == 1
    assert isinstance(converter.list_rct[0][1], torch.Tensor)


def test_phase_error_triggers_decomposition_and_noiseless_eval_matches():
    np.random.seed(3)
    circuit = _single_unitary_circuit(4)

    converter = CircuitConverter(circuit, dtype=torch.float64, phase_error=0.1)

    assert converter.circuit is not circuit
    assert any(isinstance(c, PS) for _, c in converter.list_rct)
    unitary = converter.to_tensor().numpy()
    original = _circuit_unitary(circuit)
    assert _fidelity(original, unitary) == pytest.approx(1.0, abs=_FIDELITY_ATOL)
    assert np.allclose(unitary, original, atol=_ELEMENTWISE_ATOL)


def test_decomposed_mesh_keeps_all_phase_shifters_noise_sensitive():
    """Test that every phase shifter in a Clements mesh receives phase noise."""
    np.random.seed(31)
    mode_count = 4
    circuit = _single_unitary_circuit(mode_count)

    converter = CircuitConverter(
        circuit,
        dtype=torch.float64,
        phase_error=0.1,
    )

    expected_phase_shifter_count = (
        2 * (mode_count * (mode_count - 1) // 2) + mode_count
    )
    noisy_phase_shifters = [
        component for _, component in converter.list_rct if isinstance(component, PS)
    ]

    assert len(noisy_phase_shifters) == expected_phase_shifter_count


def test_phase_error_sampling_varies_and_is_reproducible():
    np.random.seed(4)
    circuit = _single_unitary_circuit(4)
    converter = CircuitConverter(circuit, dtype=torch.float64, phase_error=0.1)

    torch.manual_seed(123)
    first = converter.to_tensor(apply_phase_error=True)
    second = converter.to_tensor(apply_phase_error=True)
    torch.manual_seed(123)
    replayed = converter.to_tensor(apply_phase_error=True)

    assert not torch.allclose(first, second)
    assert torch.allclose(first, replayed)


def test_imprecision_only_triggers_decomposition_and_quantizes():
    np.random.seed(5)
    circuit = _single_unitary_circuit(4)

    converter = CircuitConverter(circuit, dtype=torch.float64, phase_imprecision=0.5)

    assert converter.circuit is not circuit
    # Mesh phases are quantized with a coarse step, so the effective unitary
    # must deviate from the target well beyond the decomposition error.
    unitary = converter.to_tensor().numpy()
    assert not np.allclose(unitary, _circuit_unitary(circuit), atol=_ELEMENTWISE_ATOL)


def test_reservoir_circuit_keeps_input_parameters_and_gradients():
    np.random.seed(6)
    circuit = _reservoir_circuit(4, 2)

    converter = CircuitConverter(
        circuit, input_specs=["px"], dtype=torch.float64, phase_error=0.05
    )

    assert converter.spec_mappings["px"] == ["px1", "px2"]
    params = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    unitary = converter.to_tensor(params)
    assert unitary.shape == (2, 4, 4)
    identity = torch.eye(4, dtype=unitary.dtype).expand(2, 4, 4)
    assert torch.allclose(unitary @ unitary.mH, identity, atol=1e-6)
    unitary.real.sum().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_perm_is_not_decomposed():
    np.random.seed(7)
    circuit = pcvl.Circuit(4)
    circuit.add(0, pcvl.PERM([1, 0]))
    circuit.add(2, pcvl.Unitary(pcvl.Matrix.random_unitary(2)))
    perm = next(c for _, c in circuit if isinstance(c, pcvl.PERM))

    decomposed = _decompose_unitaries(circuit)

    assert any(c is perm for _, c in decomposed)
    assert not _has_black_box_unitary(decomposed)


def test_polarized_unitary_raises_value_error():
    circuit = pcvl.Circuit(1)
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(2), use_polarization=True))

    with pytest.raises(ValueError, match="polarized unitary"):
        _decompose_unitaries(circuit)


def test_non_convergence_error_names_component(monkeypatch):
    def _fail(self, target):
        raise RuntimeError("optimization above threshold")

    monkeypatch.setattr(locirc_to_tensor.CircuitOptimizer, "optimize_rectangle", _fail)
    circuit = _single_unitary_circuit(4)

    with pytest.raises(RuntimeError, match=r"did not converge .* modes \(0, 1, 2, 3\)"):
        _decompose_unitaries(circuit)


def test_one_mode_unitary_creates_single_ps():
    """Test that a 1-mode unitary is replaced by a single PS without optimizer."""
    np.random.seed(2)
    # Create a 1-mode circuit with a bare global phase
    circuit = pcvl.Circuit(1)
    phase_value = np.pi / 4
    u = pcvl.Matrix.random_unitary(1)
    circuit.add(0, pcvl.Unitary(u))

    # Decompose with phase noise enabled
    decomposed = _decompose_unitaries(circuit, phase_imprecision=0.1, phase_error=0.05)

    # Should replace Unitary with PS, not an MZI mesh
    assert decomposed is not circuit
    components = [c for _, c in decomposed]
    assert len(components) == 1
    assert isinstance(components[0], PS)


def test_slow_decomposition_warning_for_large_unitary():
    """Test that a warning is emitted for m >= 16 mode unitaries."""
    np.random.seed(3)
    circuit = pcvl.Circuit(16)
    circuit.add((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                pcvl.Unitary(pcvl.Matrix.random_unitary(16)))

    # Should emit warning about slow decomposition
    with pytest.warns(UserWarning, match=r"16-mode.*Clements mesh.*O\(m\^3\)"):
        _decompose_unitaries(circuit, phase_imprecision=0.1)


def test_fit_error_vs_noise_scale_warning(monkeypatch):
    """Test that a warning is emitted when fit error exceeds 0.1 * noise_scale."""
    np.random.seed(4)
    circuit = _single_unitary_circuit(4)

    # Mock CircuitOptimizer.optimize_rectangle to return a mesh with controlled fit error
    class MockMesh(pcvl.Circuit):
        def __init__(self, target):
            super().__init__(4)
            self.target = target
            for index in range(4):
                self.add(index, PS(pcvl.P(f"phL{index}")))

        def get_parameters(self, all_params=False):
            return [
                type(
                    "MockParameter",
                    (),
                    {
                        "name": f"phL{index}",
                        "__float__": lambda self: 0.1,
                        "fix_value": lambda self, value: None,
                    },
                )()
                for index in range(4)
            ]

        def compute_unitary(self):
            # Apply a small unitary perturbation that is larger than the
            # configured noise threshold after global-phase alignment.
            phase_rotation = np.diag(np.exp(1j * np.array([0.002, 0, 0, 0])))
            return self.target @ phase_rotation

    def mock_optimize(self, target):
        return MockMesh(target)

    monkeypatch.setattr(locirc_to_tensor.CircuitOptimizer, "optimize_rectangle", mock_optimize)

    # Use phase_error small enough that 0.1 * noise_scale < fit_error (≈0.001)
    # With phase_error = 0.005, noise_scale = 0.005, threshold = 0.0005 < 0.001 ✓
    with pytest.warns(UserWarning, match=r"fit residual.*comparable.*phase noise scale"):
        _decompose_unitaries(circuit, phase_error=0.005)


def test_fit_quality_error_if_overlap_too_small(monkeypatch):
    """Test that RuntimeError is raised if fit overlap magnitude is too small."""
    np.random.seed(5)
    circuit = _single_unitary_circuit(4)

    # Mock CircuitOptimizer.optimize_rectangle to return a mesh with terrible fit
    class BadMesh:
        def __init__(self, target):
            self.target = target

        def get_parameters(self):
            return []

        def compute_unitary(self):
            # Return an orthogonal matrix unrelated to target
            # This gives overlap ≈ 0, triggering the fit quality error
            return np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, -1, 0]], dtype=complex)

    def mock_optimize(self, target):
        return BadMesh(target)

    monkeypatch.setattr(locirc_to_tensor.CircuitOptimizer, "optimize_rectangle", mock_optimize)

    with pytest.raises(RuntimeError, match=r"fit quality check failed.*overlap magnitude"):
        _decompose_unitaries(circuit)


def test_phL_count_mismatch_error(monkeypatch):
    """Test that RuntimeError is raised if mesh has wrong number of output phases."""
    np.random.seed(6)
    circuit = _single_unitary_circuit(4)

    # Mock CircuitOptimizer.optimize_rectangle to return a mesh with wrong structure
    class BadStructureMesh:
        def __init__(self, target):
            self.target = target

        def get_parameters(self, all_params=False):
            # Return parameters with no phL entries (missing output layer)
            # This simulates a malformed mesh structure
            return [type('param', (), {'name': f'other{i}', '__float__': lambda self: 0.1})()
                    for i in range(5)]

        def compute_unitary(self):
            # Return the target itself so overlap check passes (overlap = 1)
            # Then the phL count check fails
            return self.target

    def mock_optimize(self, target):
        return BadStructureMesh(target)

    monkeypatch.setattr(locirc_to_tensor.CircuitOptimizer, "optimize_rectangle", mock_optimize)

    with pytest.raises(RuntimeError, match=r"Unexpected mesh structure.*expected 4 output phases.*found 0"):
        _decompose_unitaries(circuit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_circuit_converter_with_phase_noise_on_gpu():
    """Test CircuitConverter unitary decomposition and phase noise on GPU.

    Verifies that:
    - Unitary decomposition works on GPU device
    - Phase imprecision and phase_error produce correct outputs on GPU
    - Tensors are created on the specified GPU device
    - Stochastic sampling works correctly with fresh samples per call
    - Results match CPU baseline (fidelity check)
    """
    np.random.seed(42)
    circuit = _single_unitary_circuit(4)

    # Create converter with phase noise on GPU
    converter_gpu = CircuitConverter(
        circuit,
        dtype=torch.float64,
        device="cuda",
        phase_imprecision=0.1,
        phase_error=0.05,
    )

    assert str(converter_gpu.device) == "cuda:0" or str(converter_gpu.device) == "cuda"

    # Get unitary on GPU
    unitary_gpu = converter_gpu.to_tensor()
    assert unitary_gpu.device.type == "cuda"
    assert unitary_gpu.shape == (4, 4)

    # Verify decomposition occurred
    assert converter_gpu.circuit is not circuit
    assert any(isinstance(c, PS) for _, c in converter_gpu.list_rct)

    # Decomposed mesh PS phases receive noise; BS parameters do not.
    # The converter's output should still be a valid unitary (U†U ≈ I).
    # Verify this rather than comparing against the unquantized original.
    unitary_gpu_numpy = unitary_gpu.cpu().numpy()
    identity = np.eye(unitary_gpu_numpy.shape[0], dtype=complex)
    assert np.allclose(
        unitary_gpu_numpy.conj().T @ unitary_gpu_numpy, identity, atol=1e-6
    )

    # Phase error sampling should vary on GPU
    torch.manual_seed(123)
    first_sample_gpu = converter_gpu.to_tensor(apply_phase_error=True)
    second_sample_gpu = converter_gpu.to_tensor(apply_phase_error=True)

    assert first_sample_gpu.device.type == "cuda"
    assert second_sample_gpu.device.type == "cuda"
    assert not torch.allclose(first_sample_gpu, second_sample_gpu)

    # Verify reproducibility with same seed
    torch.manual_seed(456)
    third_sample_gpu = converter_gpu.to_tensor(apply_phase_error=True)
    torch.manual_seed(456)
    replayed_sample_gpu = converter_gpu.to_tensor(apply_phase_error=True)

    assert torch.allclose(third_sample_gpu, replayed_sample_gpu)

    # Compare CPU and GPU decomposition without quantization. Quantization is
    # discontinuous at grid boundaries, and CPU/CUDA can round a boundary
    # value differently even when the source mesh parameters are identical.
    torch.manual_seed(789)
    converter_cpu = CircuitConverter(
        circuit,
        dtype=torch.float64,
        device="cpu",
        phase_imprecision=0.0,
        phase_error=0.05,
    )
    unitary_cpu = converter_cpu.to_tensor()

    # The GPU result above includes quantization; compare fresh unquantized
    # GPU output with the CPU baseline.
    converter_gpu_no_quantization = CircuitConverter(
        circuit,
        dtype=torch.float64,
        device="cuda",
        phase_error=0.05,
    )
    unitary_gpu_no_quantization = converter_gpu_no_quantization.to_tensor()
    assert torch.allclose(
        unitary_gpu_no_quantization.cpu(), unitary_cpu, atol=1e-3
    )
