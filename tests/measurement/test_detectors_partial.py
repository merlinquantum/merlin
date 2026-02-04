from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import perceval as pcvl
import pytest
import torch

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.measurement import DetectorTransform
from merlin.utils.combinadics import Combinadics


def test_partial_detector_transform_outputs_probability_and_amplitudes():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
    result = transform(amplitudes)

    assert len(result) >= 2
    assert set(result[0].keys()) == {(1, None)}
    assert set(result[1].keys()) == {(0, None)}
    assert transform.output_keys == [(0,), (1,)]

    remaining_modes = sum(detector is None for detector in detectors)

    entries_one = result[0][(1, None)]
    entries_zero = result[1][(0, None)]

    assert len(entries_one) == 1
    assert len(entries_zero) == 1

    prob_one, rem_one = entries_one[0]
    prob_zero, rem_zero = entries_zero[0]

    assert torch.allclose(prob_one, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(prob_zero, torch.tensor(0.0), atol=1e-6)
    assert (
        rem_one.shape[-1]
        == Combinadics("fock", n=0, m=remaining_modes).compute_space_size()
    )
    assert (
        rem_zero.shape[-1]
        == Combinadics("fock", n=1, m=remaining_modes).compute_space_size()
    )


def test_partial_detector_transform_requires_complex_tensor():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    with pytest.raises(TypeError, match="complex-valued"):
        transform(torch.ones(2))


def test_partial_detector_transform_preserves_batch_dimension():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([
        [1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j],
    ])
    result = transform(amplitudes)

    assert len(result) >= 2

    remaining_modes = sum(detector is None for detector in detectors)

    for remaining_n, level in enumerate(result):
        for full_key, entries in level.items():
            for probability, collapsed in entries:
                assert probability.shape == (amplitudes.shape[0],)
                expected_dim = Combinadics(
                    "fock", n=remaining_n, m=remaining_modes
                ).compute_space_size()
                assert collapsed.shape == (amplitudes.shape[0], expected_dim)
            measured_values = tuple(
                full_key[mode]
                for mode, detector in enumerate(detectors)
                if detector is not None
            )
            assert measured_values in {(1,), (0,)}
            assert remaining_n in {0, 1}

    total = torch.zeros(amplitudes.shape[0], device=amplitudes.device)
    for level in result:
        for entries in level.values():
            for probability, _ in entries:
                total = total + probability
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_partial_detector_transform_handles_no_detectors():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [None, None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
    result = transform(amplitudes)

    assert len(result) >= 2
    entries = result[1][(None, None)]
    assert len(entries) == 1
    probability, collapsed = entries[0]
    assert torch.allclose(probability, torch.tensor(1.0), atol=1e-6)
    assert collapsed.shape[-1] == len(simulation_keys)


@pytest.mark.parametrize(
    "n,m,input_state,detector_factories",
    [
        (
            3,
            4,
            [1, 1, 0, 1],
            [(0, lambda: pcvl.Detector.ppnr(2)), (3, pcvl.Detector.threshold)],
        ),
        (
            2,
            3,
            [1, 1, 0],
            [(0, pcvl.Detector.threshold), (1, pcvl.Detector.ppnr(2))],
        ),
        (
            4,
            5,
            [1, 1, 1, 1, 0],
            [(0, lambda: pcvl.Detector.ppnr(3)), (2, pcvl.Detector.threshold)],
        ),
    ],
)
def test_partial_detector_transform_mixed_pnr_case(
    n: int,
    m: int,
    input_state: list[int],
    detector_factories: list[tuple[int, Callable[[], pcvl.Detector]]],
):
    c = pcvl.Unitary(pcvl.Matrix.random_unitary(m))
    combinator = Combinadics("fock", n=n, m=m)

    # calculate the baseline
    exp = pcvl.Experiment(c)
    for mode, factory in detector_factories:
        detector = factory() if callable(factory) else factory
        exp.add(mode, detector)
    input_basic_state = pcvl.BasicState(input_state)
    exp.with_input(input_basic_state)

    # first with Perceval
    processor = pcvl.Processor("SLOS", exp)
    processor.min_detected_photons_filter(0)
    sampler = pcvl.algorithm.Sampler(processor)
    sample_counts = sampler.probs()["results"]

    ql = QuantumLayer(
        experiment=exp,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.FOCK
        ),
    )
    probabilities = ql().squeeze(0)

    for idx, keys in enumerate(ql.output_keys):
        assert torch.isclose(
            torch.tensor(
                sample_counts.get(pcvl.BasicState(keys), 0), dtype=probabilities.dtype
            ),
            probabilities[idx],
            atol=1e-6,
        ), f"Mismatch for outcome {keys}"

    # let us do the same without detectors to check the partial measurement
    exp_nodetect = pcvl.Experiment(c)
    exp_nodetect.with_input(input_basic_state)

    ql_nodetect = QuantumLayer(
        experiment=exp_nodetect,
        measurement_strategy=MeasurementStrategy.amplitudes(
            computation_space=ComputationSpace.FOCK
        ),
    )
    amplitudes = ql_nodetect().squeeze(0)

    transform = DetectorTransform(
        combinator.iter_states(),
        exp.detectors,
        partial_measurement=True,
    )

    result = transform(amplitudes)

    unmeasured_modes = tuple(
        idx for idx, detector in enumerate(exp.detectors) if detector is None
    )
    remaining_modes = len(unmeasured_modes)

    for remaining_n, level in enumerate(result):
        expected_dim = (
            1
            if remaining_modes == 0
            else Combinadics(
                "fock", n=remaining_n, m=remaining_modes
            ).compute_space_size()
        )
        for _measured_key, entries in level.items():
            for _, collapsed in entries:
                assert collapsed.shape[-1] == expected_dim

    remap_probs: defaultdict[tuple[int, ...], float] = defaultdict(float)
    state_cache: dict[int, list[tuple[int, ...]]] = {}

    def remaining_states(count: int) -> list[tuple[int, ...]]:
        if remaining_modes == 0:
            return [()]
        if count not in state_cache:
            state_cache[count] = list(
                Combinadics("fock", n=count, m=remaining_modes).iter_states()
            )
        return state_cache[count]

    for remaining_n, level in enumerate(result):
        states = remaining_states(remaining_n)
        for full_key, entries in level.items():
            for probability, collapsed in entries:
                collapsed_flat = collapsed.reshape(-1)
                if not states:
                    remap_probs[tuple(full_key)] += probability.item()
                    continue
                for idx_state, remaining_key in enumerate(states):
                    completed = list(full_key)
                    for pos, mode in enumerate(unmeasured_modes):
                        completed[mode] = remaining_key[pos]
                    contrib = (
                        probability.item()
                        * collapsed_flat[idx_state].abs().pow(2).item()
                    )
                    remap_probs[tuple(completed)] += contrib

    baseline = {
        tuple(key): probabilities[idx] for idx, key in enumerate(ql.output_keys)
    }

    assert set(remap_probs.keys()) == set(baseline.keys())

    for key, target in baseline.items():
        remapped = torch.tensor(remap_probs[key], dtype=target.dtype)
        assert torch.isclose(remapped, target, atol=1e-6), f"Mismatch for outcome {key}"
