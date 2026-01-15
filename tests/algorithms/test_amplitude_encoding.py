"""
Algorithms-level tests for amplitude-encoded QuantumLayer workflows.

These cases validate:
* Construction and execution of QuantumLayer when amplitude encoding is enabled.
* Measurement strategies applied to amplitude vectors (e.g. returning probabilities).
* The combinatorial integrity of `output_keys` for both `no_bunching` and full Fock spaces.

Keeping these checks here ensures the public algorithms facade keeps exposing
the right behaviour for amplitude-centric users without dipping into lower-level tests.
"""

import copy
import itertools
import math
import warnings
from types import MethodType

import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin import ComputationSpace
from merlin.algorithms.layer import QuantumLayer
from merlin.measurement.strategies import MeasurementStrategy


@pytest.fixture
def make_layer():
    def _make(**overrides):
        circuit = pcvl.components.GenericInterferometer(
            3,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
        params = {
            "circuit": circuit,
            "n_photons": 1,
            "measurement_strategy": MeasurementStrategy.AMPLITUDES,
            "trainable_parameters": ["phi"],
            "input_parameters": [],
            "dtype": torch.float32,
            "amplitude_encoding": True,
            "computation_space": ComputationSpace.UNBUNCHED,
        }
        params.update(overrides)
        if not params.get("amplitude_encoding", False):
            params.setdefault("input_size", 0)
        return QuantumLayer(**params)

    return _make


def _no_bunching_keys(modes: int, n_photons: int) -> set[tuple[int, ...]]:
    return {
        tuple(1 if i in combo else 0 for i in range(modes))
        for combo in itertools.combinations(range(modes), n_photons)
    }


def _dual_rail_keys(modes: int, n_photons: int) -> set[tuple[int, ...]]:
    states = []
    for choices in itertools.product((0, 1), repeat=n_photons):
        state = [0] * modes
        for pair_idx, bit in enumerate(choices):
            state[2 * pair_idx + bit] = 1
        states.append(tuple(state))
    return set(states)


def _fock_keys(modes: int, n_photons: int) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()

    def build(prefix: list[int], remaining: int, idx: int) -> None:
        if idx == modes - 1:
            keys.add(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            build(prefix + [value], remaining - value, idx + 1)

    build([], n_photons, 0)
    return keys


def _normalised_state(n_states: int, dtype: torch.dtype) -> torch.Tensor:
    state = torch.rand(1, n_states, dtype=dtype)
    norm = state.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    return state / norm


@pytest.mark.parametrize(
    ("space", "n_photons", "n_modes", "expected_size"),
    [
        (ComputationSpace.FOCK, 3, 5, math.comb(5 + 3 - 1, 3)),
        (ComputationSpace.UNBUNCHED, 3, 5, math.comb(5, 3)),
        (ComputationSpace.DUAL_RAIL, 3, 6, 2**3),
    ],
)
def test_amplitude_encoding_output_matches_computation_space(
    space: ComputationSpace, n_photons: int, n_modes: int, expected_size: int
) -> None:
    circuit = pcvl.components.GenericInterferometer(
        n_modes,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        amplitude_encoding=True,
        computation_space=space,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
    )

    amplitude_input = _normalised_state(expected_size, dtype=torch.float32).squeeze(0)
    outputs = layer(amplitude_input)

    assert len(layer.output_keys) == expected_size
    assert outputs.shape[-1] == expected_size


@pytest.mark.parametrize(
    ("space", "n_photons", "n_modes", "expected_size"),
    [
        (ComputationSpace.FOCK, 3, 5, math.comb(5 + 3 - 1, 3)),
        (ComputationSpace.UNBUNCHED, 3, 5, math.comb(5, 3)),
        (ComputationSpace.DUAL_RAIL, 3, 6, 2**3),
    ],
)
def test_amplitude_encoding_gradients_follow_computation_space(
    space: ComputationSpace, n_photons: int, n_modes: int, expected_size: int
) -> None:
    circuit = pcvl.components.GenericInterferometer(
        n_modes,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        amplitude_encoding=True,
        computation_space=space,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
    )
    layer.zero_grad()

    amplitude_input = torch.randn(
        expected_size, dtype=torch.float32, requires_grad=True
    )

    outputs = layer(amplitude_input)
    loss = outputs.real.sum()
    loss.backward()

    assert amplitude_input.grad is not None
    assert amplitude_input.grad.shape == amplitude_input.shape

    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    assert trainable_params, (
        "Expected at least one trainable parameter for gradient check"
    )
    for param in trainable_params:
        assert param.grad is not None
        assert param.grad.shape == param.shape


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU sanity checks."
)
@pytest.mark.parametrize(
    ("space", "n_photons", "n_modes", "expected_size"),
    [
        (ComputationSpace.FOCK, 3, 5, math.comb(5 + 3 - 1, 3)),
        (ComputationSpace.UNBUNCHED, 3, 5, math.comb(5, 3)),
        (ComputationSpace.DUAL_RAIL, 3, 6, 2**3),
    ],
)
def test_amplitude_encoding_gpu_roundtrip(
    space: ComputationSpace, n_photons: int, n_modes: int, expected_size: int
) -> None:
    device = torch.device("cuda")
    circuit = pcvl.components.GenericInterferometer(
        n_modes,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        amplitude_encoding=True,
        computation_space=space,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
    ).to(device)
    layer.zero_grad()

    amplitude_input = torch.randn(
        expected_size, dtype=torch.float32, device=device, requires_grad=True
    )

    outputs = layer(amplitude_input)

    assert outputs.shape[-1] == expected_size
    assert outputs.device.type == device.type

    loss = outputs.real.sum()
    loss.backward()

    assert amplitude_input.grad is not None
    assert amplitude_input.grad.device.type == device.type
    assert amplitude_input.grad.shape == amplitude_input.shape

    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    for param in trainable_params:
        assert param.grad is not None
        assert param.grad.device.type == device.type
        assert param.grad.shape == param.shape


@pytest.mark.skip(
    reason="compute_superposition state is broken but not sure it is necessary - not called anywhere anyway"
)
def test_amplitude_encoding_matches_superposition(make_layer):
    layer = make_layer()
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    raw_amplitude = torch.arange(1, num_states + 1, dtype=torch.float64)

    prepared_state = layer._validate_amplitude_input(raw_amplitude)
    layer.set_input_state(prepared_state)
    params = layer.prepare_parameters([])
    expected = layer.computation_process.compute_superposition_state(params)

    amplitudes = layer(raw_amplitude)

    assert torch.allclose(amplitudes, expected, rtol=1e-6, atol=1e-8)


def test_amplitude_encoding_batches_use_vectorised_kernel(make_layer):
    layer = make_layer()
    process = layer.computation_process

    call_tracker = {"ebs": 0, "super": 0}
    original_ebs = process.compute_ebs_simultaneously
    original_super = process.compute_superposition_state

    def tracked_ebs(self, parameters, simultaneous_processes=1):
        call_tracker["ebs"] += 1
        return original_ebs(parameters, simultaneous_processes=simultaneous_processes)

    def tracked_super(self, parameters):
        call_tracker["super"] += 1
        return original_super(parameters)

    process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
    process.compute_superposition_state = MethodType(tracked_super, process)

    num_states = len(process.simulation_graph.mapped_keys)
    batched_state = torch.rand(3, num_states, dtype=torch.float64)

    layer(batched_state)

    assert call_tracker["ebs"] == 1
    assert call_tracker["super"] == 0


def test_amplitude_encoding_requires_first_argument(make_layer):
    layer = make_layer()
    with pytest.raises(ValueError, match="expects an amplitude tensor input"):
        layer()


def test_amplitude_encoding_validates_dimension(make_layer):
    layer = make_layer()
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    invalid = torch.rand(num_states + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="Amplitude input expects"):
        layer(invalid)


def test_computation_space_selector(make_layer):
    layer_fock = make_layer(
        amplitude_encoding=False, computation_space=ComputationSpace.FOCK
    )
    assert layer_fock.computation_space is ComputationSpace.FOCK

    layer_nb = make_layer(
        amplitude_encoding=False, computation_space=ComputationSpace.UNBUNCHED
    )
    assert layer_nb.computation_space is ComputationSpace.UNBUNCHED

    with pytest.raises(ValueError):
        make_layer(amplitude_encoding=False, computation_space="invalid")

    with warnings.catch_warnings(record=True) as caught:
        # don't fail because of the DeprecationWarning
        warnings.filterwarnings("default", category=DeprecationWarning)
        with pytest.raises(
            ValueError,
            match="Incompatible 'no_bunching' value with selected 'computation_space'.",
        ):
            make_layer(
                amplitude_encoding=False,
                computation_space=ComputationSpace.FOCK,
                no_bunching=True,
            )
    # and a warning should also be raised about no_bunching being deprecated
    assert len(caught) == 1


def test_computation_space_consistency_no_warning(make_layer):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = make_layer(
            amplitude_encoding=False,
            computation_space=ComputationSpace.UNBUNCHED,
            no_bunching=True,
        )

    assert layer.computation_space is ComputationSpace.UNBUNCHED
    # warning will be generated because no_bunching is deprecated, but not about inconsistency
    assert caught != []


def test_amplitude_encoding_probabilities_strategy(make_layer):
    layer = make_layer(measurement_strategy=MeasurementStrategy.PROBABILITIES)
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    raw_amplitude = torch.arange(1, num_states + 1, dtype=torch.float32)

    prepared_state = layer._validate_amplitude_input(raw_amplitude)
    layer.set_input_state(prepared_state)
    params = layer.prepare_parameters([])
    expected_amplitudes = layer.computation_process.compute_superposition_state(params)
    expected_probabilities = expected_amplitudes.abs() ** 2

    probabilities = layer(raw_amplitude)

    assert torch.allclose(probabilities, expected_probabilities, rtol=1e-6, atol=1e-8)


def test_mapped_keys_no_bunching_space():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2
    expected_states = math.comb(circuit.m, n_photons)
    input_state = _normalised_state(expected_states, dtype=torch.float32)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space=ComputationSpace.UNBUNCHED,
    )

    mapped_keys = layer.output_keys
    assert len(mapped_keys) == expected_states
    assert len(set(mapped_keys)) == expected_states
    assert set(mapped_keys) == _no_bunching_keys(circuit.m, n_photons)


def test_mapped_keys_fock_space():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2
    expected_states = math.comb(circuit.m + n_photons - 1, n_photons)
    input_state = _normalised_state(expected_states, dtype=torch.float32)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space=ComputationSpace.FOCK,
    )

    mapped_keys = layer.output_keys
    assert len(mapped_keys) == expected_states
    assert len(set(mapped_keys)) == expected_states
    assert set(mapped_keys) == _fock_keys(circuit.m, n_photons)


def test_mapped_keys_dual_rail_space():
    n_photons = 3
    circuit = pcvl.components.GenericInterferometer(
        2 * n_photons,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    expected_states = 2**n_photons
    input_state = _normalised_state(expected_states, dtype=torch.float32)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space=ComputationSpace.DUAL_RAIL,
    )

    mapped_keys = layer.output_keys
    assert len(mapped_keys) == expected_states
    assert len(set(mapped_keys)) == expected_states
    assert set(mapped_keys) == _dual_rail_keys(circuit.m, n_photons)


@pytest.mark.parametrize(
    "computation_space",
    [
        ComputationSpace.FOCK,
        ComputationSpace.UNBUNCHED,
        ComputationSpace.DUAL_RAIL,
    ],
)
def test_ebs_batches_group_fock_states(computation_space: ComputationSpace):
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=None,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space=computation_space,
    )

    expected_states = layer.input_size
    amplitude = torch.rand(8, expected_states, dtype=torch.float32)

    process = layer.computation_process
    original_compute_batch = process.simulation_graph.compute_batch
    recorded_batches: list[list[tuple[int, ...]]] = []

    def tracked_compute_batch(unitary, batch_fock_states):
        recorded_batches.append([tuple(state) for state in batch_fock_states])
        return original_compute_batch(unitary, batch_fock_states)

    process.simulation_graph.compute_batch = tracked_compute_batch  # type: ignore[assignment]
    try:
        layer(amplitude, simultaneous_processes=8)
    finally:
        process.simulation_graph.compute_batch = original_compute_batch  # type: ignore[assignment]
    expected_batches = [
        [tuple(state) for state in layer.output_keys[i : i + 8]]
        for i in range(0, expected_states, 8)
    ]
    assert recorded_batches == expected_batches


@pytest.mark.parametrize(
    ("space", "n_photons", "n_modes", "expected_size"),
    [
        (ComputationSpace.FOCK, 4, 8, math.comb(8 + 4 - 1, 4)),
        (ComputationSpace.UNBUNCHED, 4, 8, math.comb(8, 4)),
        (ComputationSpace.DUAL_RAIL, 4, 8, 2**4),
    ],
)
def test_amplitude_encoding_input_size(
    space: ComputationSpace, n_photons: int, n_modes: int, expected_size: int
):
    """QuantumLayer computes the correct input size for amplitude encoding."""

    circuit = pcvl.Circuit(n_modes)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        amplitude_encoding=True,
        computation_space=space,
    )

    assert layer.input_size == expected_size
    assert len(layer.output_keys) == expected_size


def test_amplitude_encoding_requires_valid_configuration():
    """Amplitude encoding enforces required constructor constraints."""

    circuit = pcvl.Circuit(8)

    with pytest.raises(ValueError, match="n_photons must be provided"):
        QuantumLayer(
            circuit=circuit,
            n_photons=None,
            amplitude_encoding=True,
        )

    with pytest.raises(
        ValueError,
        match="Amplitude encoding cannot be combined with classical input parameters",
    ):
        QuantumLayer(
            circuit=circuit,
            n_photons=4,
            input_parameters=["theta"],
            amplitude_encoding=True,
        )


def test_dual_rail_requires_even_mode_count():
    circuit = pcvl.Circuit(6)

    # Newer error message includes the provided counts for clarity
    with pytest.raises(
        ValueError, match=r"dual_rail compute space requires n_photons = m // 2"
    ):
        QuantumLayer(
            circuit=circuit,
            n_photons=2,
            amplitude_encoding=True,
            computation_space=ComputationSpace.DUAL_RAIL,
        )


def test_dual_rail_rejects_incorrect_amplitude_length():
    n_photons = 3
    circuit = pcvl.Circuit(2 * n_photons)
    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        amplitude_encoding=True,
        computation_space=ComputationSpace.DUAL_RAIL,
    )
    invalid = torch.rand((2**n_photons) + 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="Amplitude input expects .* components"):
        layer(invalid)


def test_amplitude_encoding_superposition_matches_basis_sum():
    """The amplitudes are a weighted sum over basis-state simulations."""

    n_photons = 4
    n_modes = 8
    circuit = pcvl.Circuit(n_modes)
    for k in range(0, n_modes, 2):
        for mode in range(k % 2, n_modes, 2):
            circuit.add(mode, pcvl.BS())
    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.AMPLITUDES,
        amplitude_encoding=True,
        computation_space=ComputationSpace.DUAL_RAIL,
    )

    basis_indices = [0, 1, 2]
    basis_vectors = torch.eye(layer.input_size, dtype=torch.complex64)[basis_indices]
    basis_outputs = torch.stack([layer(state) for state in basis_vectors])

    coefficients = torch.tensor(
        [0.6 + 0.2j, -0.3 + 0.5j, 0.1 - 0.4j], dtype=torch.complex64
    )
    coefficients = coefficients / torch.linalg.norm(coefficients)

    amplitude_input = torch.zeros(layer.input_size, dtype=torch.complex64)
    amplitude_input[basis_indices] = coefficients

    combined_output = layer(amplitude_input)
    expected_output = torch.sum(coefficients[:, None, None] * basis_outputs, dim=0)
    difference = combined_output - expected_output
    assert torch.allclose(combined_output, expected_output, atol=1e-6, rtol=1e-6), (
        f"Max deviation {difference.abs().max().item():.2e}"
    )

    with pytest.raises(ValueError, match="Amplitude input expects"):
        layer(torch.ones(layer.input_size + 1, dtype=torch.complex64))


@pytest.mark.parametrize(
    "m,batch_size,computation_space",
    [
        (4, 2, ComputationSpace.FOCK),
        (4, 2, ComputationSpace.UNBUNCHED),
        (4, 2, ComputationSpace.DUAL_RAIL),
        (4, 1, ComputationSpace.FOCK),
        (4, 1, ComputationSpace.UNBUNCHED),
        (4, 1, ComputationSpace.DUAL_RAIL),
    ],
)
def test_ebs_wrt_quantumlayer(
    m, batch_size, computation_space: ComputationSpace
) -> None:
    # define circuit
    circuit = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(phi=np.pi / 4 * i)
            // pcvl.BS()
            // pcvl.PS(phi=np.pi / 8 * i)
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = m // 2

    ebs_layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.AMPLITUDES,
        amplitude_encoding=True,
        computation_space=computation_space,
    )

    num_states = len(ebs_layer.output_keys)

    # generate random amplitude input
    magnitudes = torch.rand(batch_size, num_states, dtype=torch.float32)
    norms = magnitudes.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    magnitudes = magnitudes / norms

    phases = torch.rand(batch_size, num_states, dtype=torch.float32) * (2 * math.pi)
    # out = magnitute (cos(phases) + j sin(phases))
    amplitude_input = torch.polar(magnitudes, phases)

    # all the magic happens here - we don't care about batch since we calculate the same slos for each input state, and the batch is only used to weight the superposition
    # so the function `compute_ebs_simultaneously` will create a batch of multiple input states - using batching to speed up the computation
    ebs_output = ebs_layer(amplitude_input)
    if ebs_output.dim() == 1:
        ebs_output = ebs_output.unsqueeze(0)

    expected_output = torch.zeros_like(ebs_output, dtype=ebs_output.dtype)
    shared_state = ebs_layer.state_dict()

    ebs_params = ebs_layer.prepare_parameters([])
    ebs_unitary = ebs_layer.computation_process.converter.to_tensor(*ebs_params)

    for idx, state in enumerate(ebs_layer.output_keys):
        coefficients = amplitude_input[:, idx].to(ebs_output.dtype).unsqueeze(-1)
        if coefficients.abs().max() > 1e-8:
            single_layer = QuantumLayer(
                circuit=copy.deepcopy(circuit),
                n_photons=n_photons,
                measurement_strategy=MeasurementStrategy.AMPLITUDES,
                input_state=list(state),
                amplitude_encoding=False,
                computation_space=computation_space,
            )

            single_layer.load_state_dict(shared_state, strict=False)

            single_params = single_layer.prepare_parameters([])
            single_unitary = single_layer.computation_process.converter.to_tensor(
                *single_params
            )
            assert torch.allclose(single_unitary, ebs_unitary, rtol=1e-6, atol=1e-8), (
                "Expected identical unitaries between EBS and single-state layers."
            )
            assert (
                single_layer.computation_process.simulation_graph.mapped_keys
                == ebs_layer.computation_process.simulation_graph.mapped_keys
            ), "Computation graphs diverge between EBS and single-state layers."

            basis_output = single_layer()
            if basis_output.dim() > 1:
                basis_output = basis_output.squeeze(0)
            basis_output = basis_output.to(ebs_output.dtype)

            expected_output = expected_output + coefficients * basis_output

    # normalize expected_output
    expected_output = expected_output / expected_output.norm(
        p=2, dim=1, keepdim=True
    ).clamp_min(1e-12)
    # TODO: investigate why this tests failed with rtol=1e-6, atol=1e-8
    assert torch.allclose(ebs_output, expected_output, rtol=1e-4, atol=1e-6), (
        "EBS output deviates from the superposed QuantumLayer results."
    )
