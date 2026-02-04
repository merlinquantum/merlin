"""
Tests for superposition handling in QuantumLayer.
"""

import copy
import math
from types import MethodType

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.core import ComputationSpace
from merlin.measurement.strategies import MeasurementStrategy
from merlin.utils.combinadics import Combinadics


def classical_method(layer, input_state, *x):
    output_classical = torch.zeros(1, layer.output_size)
    dtype = layer.computation_process.simulation_graph.prev_amplitudes.dtype
    output_classical = output_classical.to(dtype)

    for key, value in input_state.items():
        layer.computation_process.input_state = key
        _ = layer(*x)

        # retrieve amplitudes from the computation graph
        amplitudes = layer.computation_process.simulation_graph.prev_amplitudes
        amplitudes /= torch.norm(amplitudes, p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        output_classical += value * amplitudes

    output_classical /= torch.norm(output_classical, p=2, dim=-1, keepdim=True)

    output_probs = (
        layer.computation_process.simulation_graph.compute_probs_from_amplitudes(
            output_classical
        )
    )
    return output_probs[1]


class TestOutputSuperposedState:
    """Test cases for measurement-driven outputs in QuantumLayer.simple()."""

    def test_superposed_state(self, benchmark):
        """Test default measurement behaviour when output_size is not constrained."""
        # With the default measurement distribution the output size matches the underlying Fock distribution
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        n_photons = 3
        expected_states = math.comb(circuit.m, n_photons)
        input_state = torch.rand(3, expected_states, dtype=torch.float64)

        sum_values = (input_state**2).sum(dim=-1, keepdim=True)

        input_state = input_state / sum_values

        layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
        )

        input_state_superposed = {
            layer.computation_process.simulation_graph.mapped_keys[k]: input_state[1, k]
            for k in range(len(input_state[0]))
        }

        output_superposed = benchmark(layer)

        output_classical = classical_method(layer, input_state_superposed)

        assert torch.allclose(
            output_superposed[1], output_classical, rtol=3e-4, atol=1e-7
        )

    def test_classical_method(self, benchmark):
        """Test probability distribution behaviour when output_size is not constrained."""
        # With the default measurement distribution the output size matches the underlying Fock distribution
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        n_photons = 3
        expected_states = math.comb(circuit.m, n_photons)
        input_state = torch.rand(3, expected_states, dtype=torch.float64)

        sum_values = (input_state**2).sum(dim=-1, keepdim=True)

        input_state = input_state / torch.sqrt(sum_values)

        layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
        )

        input_state_superposed = {
            layer.computation_process.simulation_graph.mapped_keys[k]: input_state[0, k]
            for k in range(len(input_state[0]))
        }

        output_superposed = layer()

        output_classical = benchmark(
            lambda: classical_method(layer, input_state_superposed)
        )

        assert torch.allclose(
            output_superposed[0], output_classical, rtol=3e-4, atol=1e-7
        )

    def test_forward_infers_batch_for_superposed_state(self):
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        n_photons = 3
        expected_states = math.comb(circuit.m, n_photons)
        input_state = torch.rand(2, expected_states, dtype=torch.float64)
        sum_values = (input_state**2).sum(dim=-1, keepdim=True)
        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
        )

        process = layer.computation_process
        call_tracker = {"ebs": 0, "super": 0}

        original_ebs = process.compute_ebs_simultaneously
        original_super = process.compute_superposition_state

        def tracked_ebs(self, parameters, simultaneous_processes=1):
            call_tracker["ebs"] += 1
            return original_ebs(
                parameters, simultaneous_processes=simultaneous_processes
            )

        def tracked_super(self, parameters):
            call_tracker["super"] += 1
            return original_super(parameters)

        process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
        process.compute_superposition_state = MethodType(tracked_super, process)

        layer()

        assert call_tracker["ebs"] == 0
        assert call_tracker["super"] == 1

    def test_forward_infers_single_state_without_batch(self):
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        n_photons = 3
        expected_states = math.comb(circuit.m, n_photons)
        input_state = torch.rand(1, expected_states, dtype=torch.float64)
        sum_values = (input_state**2).sum(dim=-1, keepdim=True)
        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
        )

        process = layer.computation_process
        call_tracker = {"ebs": 0, "super": 0}

        original_ebs = process.compute_ebs_simultaneously
        original_super = process.compute_superposition_state

        def tracked_ebs(self, parameters, simultaneous_processes=1):
            call_tracker["ebs"] += 1
            return original_ebs(
                parameters, simultaneous_processes=simultaneous_processes
            )

        def tracked_super(self, parameters):
            call_tracker["super"] += 1
            return original_super(parameters)

        process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
        process.compute_superposition_state = MethodType(tracked_super, process)

        layer()

        assert call_tracker["ebs"] == 0
        assert call_tracker["super"] == 1

    def test_superposition_state_classical_batch(self):
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.PS(pcvl.P("px_0")))
        circuit.add(1, pcvl.PS(pcvl.P("px_1")))
        circuit.add(2, pcvl.PS(pcvl.P("px_2")))

        n_photons = 1
        expected_states = math.comb(circuit.m, n_photons)
        input_state = torch.rand(2, expected_states, dtype=torch.float64)
        input_state = input_state / input_state.norm(p=2, dim=1, keepdim=True)

        layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            dtype=torch.float64,
        )

        process = layer.computation_process
        call_tracker = {"ebs": 0, "super": 0}
        original_ebs = process.compute_ebs_simultaneously
        original_super = process.compute_superposition_state

        def tracked_ebs(self, parameters, simultaneous_processes=1):
            call_tracker["ebs"] += 1
            call_tracker["simultaneous_processes"] = simultaneous_processes
            # Surface the tensor shape returned by the batching kernel for regression checks.
            result = original_ebs(
                parameters, simultaneous_processes=simultaneous_processes
            )
            call_tracker["result_shape"] = result.shape
            return result

        def tracked_super(self, parameters, return_keys=False):
            call_tracker["super"] += 1
            return original_super(parameters, return_keys=return_keys)

        process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
        process.compute_superposition_state = MethodType(tracked_super, process)

        batch_size = expected_states
        classical_input = torch.rand(batch_size, layer.input_size, dtype=torch.float64)

        output = layer(classical_input)

        assert call_tracker["ebs"] == 1
        assert call_tracker["super"] == 0
        assert call_tracker["simultaneous_processes"] == expected_states
        assert call_tracker["result_shape"] == (
            batch_size,
            input_state.shape[0],
            layer.output_size,
        )
        assert output.shape == (
            batch_size,
            input_state.shape[0],
            layer.output_size,
        )

    @pytest.mark.parametrize(
        "computation_space,n_modes,n_photons",
        [
            (ComputationSpace.FOCK, 3, 2),
            (ComputationSpace.UNBUNCHED, 5, 3),
            (ComputationSpace.DUAL_RAIL, 6, 3),
        ],
    )
    def test_superposition_state_input(
        self,
        computation_space: ComputationSpace,
        n_modes: int,
        n_photons: int,
    ):
        circuit = pcvl.Circuit(n_modes)
        for mode in range(n_modes):
            circuit.add(mode, pcvl.PS(pcvl.P(f"theta_{mode}")))

        circuit.add(
            0,
            pcvl.components.GenericInterferometer(
                n_modes,
                pcvl.components.catalog["mzi phase last"].generate,
                shape=pcvl.InterferometerShape.RECTANGLE,
            ),
        )

        if computation_space is ComputationSpace.DUAL_RAIL:
            assert n_modes == 2 * n_photons
        elif computation_space is ComputationSpace.UNBUNCHED:
            assert n_photons <= n_modes

        combinadics = Combinadics(computation_space.value, n_photons, n_modes)
        expected_states = combinadics.compute_space_size()

        magnitudes = torch.rand(1, expected_states, dtype=torch.float64)
        magnitudes = magnitudes / magnitudes.norm(p=2, dim=1, keepdim=True).clamp_min(
            1e-12
        )
        phases = torch.rand(1, expected_states, dtype=torch.float64) * (2 * math.pi)
        input_state = torch.polar(magnitudes, phases)

        amplitude_layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.amplitudes(
                computation_space=computation_space
            ),
            input_state=input_state,
            input_parameters=["theta"],
            trainable_parameters=["phi"],
            dtype=torch.float64,
        )

        classical_dim = amplitude_layer.input_size or n_modes
        dummy_input = torch.rand(4, classical_dim, dtype=torch.float64)
        dummy_input = dummy_input / dummy_input.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        output = amplitude_layer(dummy_input)
        if output.dim() == 2:
            output = output.unsqueeze(1)

        assert output.shape == (
            dummy_input.shape[0],
            input_state.shape[0],
            amplitude_layer.output_size,
        )

        coefficients = amplitude_layer.computation_process.input_state.to(output.dtype)
        if coefficients.dim() == 1:
            coefficients = coefficients.unsqueeze(0)

        shared_state = amplitude_layer.state_dict()
        amplitude_params = amplitude_layer.prepare_parameters([dummy_input])
        reference_unitary = amplitude_layer.computation_process.converter.to_tensor(
            *amplitude_params
        )
        reference_keys = tuple(
            amplitude_layer.computation_process.simulation_graph.mapped_keys
        )
        amplitude_params_dict = dict(amplitude_layer.named_parameters())

        expected_amplitudes = torch.zeros_like(output, dtype=output.dtype)
        for idx, state in enumerate(amplitude_layer.output_keys):
            basis_layer = QuantumLayer(
                circuit=copy.deepcopy(circuit),
                n_photons=n_photons,
                measurement_strategy=MeasurementStrategy.amplitudes(
                    computation_space=computation_space
                ),
                input_state=list(state),
                input_parameters=["theta"],
                trainable_parameters=["phi"],
                dtype=torch.float64,
            )

            load_result = basis_layer.load_state_dict(shared_state, strict=False)
            assert not load_result.missing_keys
            assert not load_result.unexpected_keys
            for name, param in basis_layer.named_parameters():
                assert torch.allclose(param, amplitude_params_dict[name]), (
                    f"Parameter mismatch for {name}"
                )

            layer_params = basis_layer.prepare_parameters([dummy_input])
            layer_unitary = basis_layer.computation_process.converter.to_tensor(
                *layer_params
            )
            assert torch.allclose(
                layer_unitary, reference_unitary, rtol=1e-6, atol=1e-8
            ), "Unitary mismatch between layers"
            assert (
                tuple(basis_layer.computation_process.simulation_graph.mapped_keys)
                == reference_keys
            ), "Simulation graph keys mismatch"

            basis_output = basis_layer(dummy_input)
            if basis_output.dim() == 2:
                basis_output = basis_output.unsqueeze(1)
            elif basis_output.dim() == 1:
                basis_output = basis_output.unsqueeze(0).unsqueeze(1)
            basis_output = basis_output.to(expected_amplitudes.dtype)

            coefficient = coefficients[:, idx].to(expected_amplitudes.dtype)
            weight = coefficient.unsqueeze(0).unsqueeze(-1)
            if weight.abs().max() < 1e-10:
                continue

            expected_amplitudes += weight * basis_output

        expected_amplitudes /= expected_amplitudes.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        assert torch.allclose(output, expected_amplitudes, rtol=3e-4, atol=1e-7), (
            "Superposed output deviates from the superposed QuantumLayer results."
        )

    def test_superposition_state_statevector(self):
        n_modes = 10
        n_photons = 5

        circuit = pcvl.Circuit(n_modes)
        for mode in range(n_modes):
            circuit.add(mode, pcvl.PS(pcvl.P(f"theta_{mode}")))

        circuit.add(
            0,
            pcvl.components.GenericInterferometer(
                n_modes,
                pcvl.components.catalog["mzi phase last"].generate,
                shape=pcvl.InterferometerShape.RECTANGLE,
            ),
        )

        combinadics = Combinadics(ComputationSpace.DUAL_RAIL, n_photons, n_modes)

        # build a superposition of 5 basic states
        sv = pcvl.StateVector()
        for idx in torch.randint(0, combinadics.compute_space_size(), (5,)):
            sv += ((idx % 2) and -1 or 1) * pcvl.BasicState(
                combinadics.index_to_fock(idx)
            )

        layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.DUAL_RAIL
            ),
            input_state=sv,
            trainable_parameters=["phi"],
            input_parameters=["theta"],
            dtype=torch.float64,
        )

        # input is a batch of 10 tensor
        x = torch.rand(10, n_modes)
        out = layer(x)

        assert out.shape == (10, combinadics.compute_space_size())

        # now check that we could have obtained this probability by weighted adding of probability amplitude of each component
        dict_sv = {tuple(k): v for k, v in sv.unnormalized_iterator()}

        for k in range(x.shape[0]):
            x_k = x[k]
            out_k = out[k]
            expected_out_k = classical_method(layer, dict_sv, x_k)

            assert torch.allclose(out_k, expected_out_k, rtol=3e-4, atol=1e-7), (
                "output between classical and superposition method differ"
            )
