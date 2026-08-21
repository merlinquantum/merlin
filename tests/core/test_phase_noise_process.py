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

"""Tests for phase-noise handling in ComputationProcess."""

from __future__ import annotations

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer_utils import NoiseGroups
from merlin.core.computation_space import ComputationSpace
from merlin.core.process import ComputationProcess, ComputationProcessFactory
from merlin.core.sectored_distribution import SectoredDistribution, SectorResult
from merlin.utils.normalization import (
    normalize_probabilities,
    probabilities_from_amplitudes,
)


def _mzi_circuit() -> pcvl.Circuit:
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.BS.H())
    return circuit


def _mzi_circuit_with_component_phase_error() -> pcvl.Circuit:
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi"), max_error=0.2))
    circuit.add((0, 1), pcvl.BS.H())
    return circuit


def _process(
    noise_groups: NoiseGroups | None = None,
    *,
    circuit: pcvl.Circuit | None = None,
    input_state: list[int] | torch.Tensor | None = None,
    n_phase_error_samples: int = 3,
) -> ComputationProcess:
    if input_state is None:
        input_state = [1, 0]
    return ComputationProcess(
        circuit=_mzi_circuit() if circuit is None else circuit,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=noise_groups,
        n_phase_error_samples=n_phase_error_samples,
    )


def _phase_parameter(value: float = 0.37) -> list[torch.Tensor]:
    return [torch.tensor([value], dtype=torch.float64, requires_grad=True)]


def test_phase_error_accepts_fock_input_with_amplitude_encoding_flag():
    """Deprecated amplitude encoding does not force tensor support preparation."""
    process = _process(
        noise_groups=NoiseGroups(
            source=None,
            circuit={"phase_error": 0.1},
            post_measurement=None,
        )
    )

    probabilities = process.compute(
        _phase_parameter(),
        amplitude_encoding=True,
    )

    assert isinstance(probabilities, torch.Tensor)


def _manual_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
    *,
    amplitude_encoding: bool = False,
) -> torch.Tensor | SectoredDistribution:
    accumulated: torch.Tensor | SectoredDistribution | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        probabilities = process._compute_probabilities_for_unitary(
            unitary, amplitude_encoding=amplitude_encoding
        )
        if accumulated is None:
            accumulated = probabilities
            continue
        if isinstance(accumulated, SectoredDistribution):
            assert isinstance(probabilities, SectoredDistribution)
            accumulated = process._add_sectored_distributions(
                accumulated, probabilities
            )
        else:
            assert isinstance(probabilities, torch.Tensor)
            accumulated = accumulated + probabilities

    assert accumulated is not None
    if isinstance(accumulated, SectoredDistribution):
        return process._divide_sectored_distribution(
            accumulated, process._n_phase_error_samples
        )
    return accumulated / process._n_phase_error_samples


def _manual_coherent_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    accumulated: torch.Tensor | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        amplitudes = process._compute_superposition_amplitudes_for_unitary(unitary)
        probabilities = probabilities_from_amplitudes(amplitudes)
        probabilities = normalize_probabilities(
            probabilities, process.computation_space
        )
        accumulated = (
            probabilities if accumulated is None else accumulated + probabilities
        )

    assert accumulated is not None
    return accumulated / process._n_phase_error_samples


def _sector_key_groups(
    distribution: SectoredDistribution,
) -> list[list[tuple[int, ...]]]:
    return [list(sector.keys) for sector in distribution.sectors]


def _manual_incoherent_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    accumulated: torch.Tensor | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        probabilities = process._compute_source_probabilities_for_unitary(
            unitary, amplitude_encoding=True
        )
        assert isinstance(probabilities, torch.Tensor)
        accumulated = (
            probabilities if accumulated is None else accumulated + probabilities
        )

    assert accumulated is not None
    return accumulated / process._n_phase_error_samples


def test_add_sectored_distributions_aligns_basis_keys():
    process = _process()
    left_keys = ((1, 0), (0, 1))
    right_keys = ((0, 1), (1, 0))
    left = SectoredDistribution((
        SectorResult(
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            n_modes=2,
            n_photons=1,
            keys=left_keys,
        ),
    ))
    right = SectoredDistribution((
        SectorResult(
            torch.tensor([[30.0, 10.0]], dtype=torch.float64),
            n_modes=2,
            n_photons=1,
            keys=right_keys,
        ),
    ))

    result = process._add_sectored_distributions(left, right)

    assert result.sectors[0].keys == left_keys
    assert torch.allclose(
        result.sectors[0].tensor,
        torch.tensor([[11.0, 32.0]], dtype=torch.float64),
    )


def test_add_sectored_distributions_rejects_duplicate_basis_keys():
    process = _process()
    left = SectoredDistribution((
        SectorResult(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        ),
    ))
    right = SectoredDistribution((
        SectorResult(
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (1, 0)),
        ),
    ))

    with pytest.raises(ValueError, match="unique"):
        process._add_sectored_distributions(left, right)


def test_g2_source_superposition_accumulation_aligns_sector_keys(monkeypatch):
    input_state = torch.tensor([1.0, 1.0], dtype=torch.complex128)
    process = _process(
        NoiseGroups(
            source={
                "g2": 0.1,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        ),
        input_state=input_state,
    )
    left_keys = ((1, 0), (0, 1))
    right_keys = ((0, 1), (1, 0))

    def compute_probs(_unitary, fock_state):
        if tuple(fock_state) == (1, 0):
            tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
            keys = left_keys
        else:
            tensor = torch.tensor([30.0, 10.0], dtype=torch.float64)
            keys = right_keys
        return SectoredDistribution((
            SectorResult(tensor, n_modes=2, n_photons=1, keys=keys),
        ))

    monkeypatch.setattr(process.simulation_graph, "compute_probs", compute_probs)

    output = process._compute_source_probabilities_for_unitary(
        torch.eye(2, dtype=torch.complex128),
        amplitude_encoding=True,
    )

    assert isinstance(output, SectoredDistribution)
    assert output.sectors[0].keys == left_keys
    assert torch.allclose(
        output.sectors[0].tensor,
        torch.tensor([[5.5, 16.0]], dtype=torch.float64),
    )


def test_no_noise_compute_returns_amplitudes():
    process = _process()

    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert output.is_complex()


def test_phase_imprecision_only_compute_still_returns_amplitudes():
    process = _process(
        NoiseGroups(
            source=None,
            circuit={"phase_imprecision": 0.5},
            post_measurement=None,
        )
    )

    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert output.is_complex()
    assert process.converter._phase_imprecision == pytest.approx(0.5)


def test_phase_error_compute_returns_probabilities():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )

    torch.manual_seed(12)
    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert not output.is_complex()
    assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0, dtype=output.dtype))


def test_component_phase_error_compute_returns_probabilities_without_noise_model():
    process = _process(
        circuit=_mzi_circuit_with_component_phase_error(),
        n_phase_error_samples=4,
    )

    torch.manual_seed(12)
    output = process.compute(_phase_parameter())

    assert process.noise_groups is not None
    assert process.noise_groups.circuit == {"phase_error": None}
    assert isinstance(output, torch.Tensor)
    assert not output.is_complex()
    assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0, dtype=output.dtype))


def test_phase_error_matches_manual_probability_average():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert torch.allclose(output, expected)


def test_phase_error_rejects_probability_key_order_mismatch(monkeypatch):
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=1,
    )
    reversed_keys = list(reversed(process.simulation_graph.mapped_keys))

    def compute_probs(_unitary, _input_state):
        return reversed_keys, torch.ones(
            len(reversed_keys),
            dtype=torch.float64,
        )

    monkeypatch.setattr(process.simulation_graph, "compute_probs", compute_probs)

    with pytest.raises(ValueError, match="Probability keys"):
        process.compute(_phase_parameter())


def test_phase_error_with_tensor_superposition_averages_probabilities():
    input_state = torch.tensor(
        [1.0, 1.0j],
        dtype=torch.complex128,
    )
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters, amplitude_encoding=True)
    torch.manual_seed(123)
    expected = _manual_coherent_phase_error_average(process, parameters)
    torch.manual_seed(123)
    incoherent_mixture = _manual_incoherent_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert not output.is_complex()
    assert torch.allclose(output, expected)
    assert not torch.allclose(expected, incoherent_mixture)
    assert torch.allclose(
        output.sum(dim=-1),
        torch.ones(output.shape[:-1], dtype=output.dtype, device=output.device),
    )


def test_compute_with_keys_phase_error_matches_compute_default_for_tensor_input_state():
    input_state = torch.tensor(
        [1.0, 1.0j],
        dtype=torch.complex128,
    )
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    unkeyed_output = process.compute(parameters)
    torch.manual_seed(123)
    keys, keyed_output = process.compute_with_keys(parameters)

    assert keys == process.simulation_graph.mapped_keys
    assert isinstance(unkeyed_output, torch.Tensor)
    assert isinstance(keyed_output, torch.Tensor)
    assert torch.allclose(keyed_output, unkeyed_output)


def test_compute_with_keys_phase_error_can_use_tensor_superposition():
    input_state = torch.tensor(
        [1.0, 1.0j],
        dtype=torch.complex128,
    )
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    keys, keyed_output = process.compute_with_keys(
        parameters,
        use_input_state_superposition=True,
    )
    torch.manual_seed(123)
    expected = _manual_coherent_phase_error_average(process, parameters)
    torch.manual_seed(123)
    default_output = process.compute(parameters)

    assert keys == process.simulation_graph.mapped_keys
    assert isinstance(keyed_output, torch.Tensor)
    assert torch.allclose(keyed_output, expected)
    assert not torch.allclose(keyed_output, default_output)


def test_phase_error_with_source_noise_averages_noisy_probabilities():
    process = _process(
        NoiseGroups(
            source={"indistinguishability": 0.8},
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert torch.allclose(output, expected)


def test_phase_error_with_g2_averages_sectored_distributions():
    process = _process(
        NoiseGroups(
            source={"g2": 0.05},
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=3,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, SectoredDistribution)
    assert isinstance(expected, SectoredDistribution)
    assert {sector.n_photons for sector in output.sectors} == {
        sector.n_photons for sector in expected.sectors
    }
    for sector in output.sectors:
        expected_sector = expected.get_sector(sector.n_photons)
        assert torch.allclose(sector.tensor, expected_sector.tensor)


def test_compute_with_keys_returns_source_g2_sector_keys():
    process = _process(
        NoiseGroups(
            source={"g2": 0.05},
            circuit=None,
            post_measurement=None,
        )
    )

    keys, output = process.compute_with_keys(_phase_parameter())
    unkeyed_output = process.compute(_phase_parameter())

    assert isinstance(output, SectoredDistribution)
    assert isinstance(unkeyed_output, SectoredDistribution)
    assert keys == _sector_key_groups(output)
    assert keys == process.simulation_graph.mapped_keys


def test_compute_with_keys_returns_phase_error_g2_sector_keys():
    process = _process(
        NoiseGroups(
            source={"g2": 0.05},
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=3,
    )

    torch.manual_seed(123)
    keys, output = process.compute_with_keys(_phase_parameter())

    assert isinstance(output, SectoredDistribution)
    assert keys == _sector_key_groups(output)
    assert keys == process.simulation_graph.mapped_keys


def test_n_phase_error_samples_must_be_integer():
    with pytest.raises(TypeError, match="n_phase_error_samples must be an integer."):
        _process(n_phase_error_samples=1.5)  # type: ignore[arg-type]


def test_n_phase_error_samples_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_phase_error_samples must be at least 1."):
        _process(n_phase_error_samples=0)


def test_computation_process_defaults_to_one_phase_error_sample():
    process = ComputationProcess(
        circuit=_mzi_circuit(),
        input_state=[1, 0],
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=NoiseGroups(
            source=None,
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
    )

    assert process._n_phase_error_samples == 1


def test_compute_superposition_state_raises_with_phase_error():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None)
    )

    with pytest.raises(RuntimeError, match="phase_error"):
        process.compute_superposition_state(_phase_parameter())


def test_compute_ebs_simultaneously_raises_with_phase_error():
    input_state = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
    )

    with pytest.raises(RuntimeError, match="phase_error"):
        process.compute_ebs_simultaneously(_phase_parameter())


def test_compute_with_keys_returns_phase_error_probabilities():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )

    torch.manual_seed(123)
    keys, output = process.compute_with_keys(_phase_parameter())

    assert keys == process.simulation_graph.mapped_keys
    assert isinstance(output, torch.Tensor)
    assert not output.is_complex()


def test_factory_forwards_n_phase_error_samples():
    process = ComputationProcessFactory.create(
        circuit=_mzi_circuit(),
        input_state=[1, 0],
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=NoiseGroups(
            source=None,
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=7,
    )

    assert process._n_phase_error_samples == 7


def test_factory_defaults_to_one_phase_error_sample():
    process = ComputationProcessFactory.create(
        circuit=_mzi_circuit(),
        input_state=[1, 0],
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=NoiseGroups(
            source=None,
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
    )

    assert process._n_phase_error_samples == 1
