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


from typing import Any

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import (
    AmplitudesStrategy,
    MeasurementStrategy,
    ModeExpectationsStrategy,
    ProbabilitiesStrategy,
    resolve_measurement_strategy,
)


class TestQuantumLayerMeasurementStrategy:
    def test_measurement_distribution(self):
        # Probabilities is equivalent to OutputMappingStrategy.NONE
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, 3)

        # Ensure positive values (that represent probabilities)
        assert torch.all(output >= -1e6)

        # Ensure values sum to 1 (within numerical precision)
        assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0), atol=1e-6)

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # Ensure that QuantumLayer with Probabilities strategy can be initialized without specifying output_size and that its output_size is accessible
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size
        output.sum().backward()
        assert x.grad is not None

    def test_circuit_infers_input_size_from_input_parameters(self):
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]

        layer = ML.QuantumLayer(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        assert layer.input_size == 2

        x = torch.rand(4, 2)
        output = layer(x)
        assert output.shape == (4, layer.output_size)

    def test_experiment_infers_input_size_from_input_parameters(self):
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        experiment = pcvl.Experiment(circuit)
        input_state = [1, 1, 0]

        layer = ML.QuantumLayer(
            experiment=experiment,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        assert layer.input_size == 2

        x = torch.rand(3, 2)
        output = layer(x)
        assert output.shape == (3, layer.output_size)

        # Error: cannot specify output_size
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        with pytest.raises(TypeError):
            layer = ML.QuantumLayer(
                input_size=2,
                output_size=5,  # cannot specify output_size
                n_photons=1,
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

    def test_linear_equivalent(self):
        # OutputMappingStrategy.LINEAR is equivalent to Probabilities + torch.nn.Linear
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        linear = torch.nn.Linear(layer.output_size, 2)
        x = torch.rand(5, 2, requires_grad=True)
        output = linear(layer(x))

        # Good output shape
        assert output.shape == (5, 2)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 2))
        x = torch.rand(2, 2, requires_grad=True)
        output = model(x)
        assert output.shape[-1] == 2
        output.sum().backward()
        assert x.grad is not None

    def test_mode_expectations(self):
        # ModeExpectations is a new strategy unlike any previous OutputMappingStrategy
        n_modes = 3
        n_photons = 2

        builder = ML.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=n_photons,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.MODE_EXPECTATIONS,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, n_modes)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # By default, no_bunching=True so output values cannot surpass 1
        assert torch.all(output <= 1.0 + 1e-6)
        assert torch.all(output >= -1e-6)  # No negative values

        # ModeExpectations with explicit no_bunching=True
        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=n_photons,
            builder=builder,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
        )
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None
        assert torch.all(output <= 1.0 + 1e-6)
        assert torch.all(output >= -1e-6)  # No negative values

        # ModeExpectations with no_bunching=False (has some output values > 1)
        builder = ML.CircuitBuilder(n_modes=n_modes)
        builder.add_superpositions((0, 1), name="BS")
        builder.add_angle_encoding(modes=[0, 1], name="input")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[2, 1, 0],
            builder=builder,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            computation_space=ComputationSpace.FOCK,
        )
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None
        assert torch.all(output >= -1e-6)  # No negative values
        assert torch.all(
            output <= n_photons + 1e-6
        )  # Values cannot surpass the number of photons
        # output[:, 0] and output[:, 1] should have values superior to 1 because their expected number of photons is higher than 1 with no_bunching=False
        assert torch.all(output[:, 0] > torch.ones_like(output[:, 0]))
        assert torch.all(output[:, 1] > torch.ones_like(output[:, 1]))

        # QuantumLayer's output_size is accessible and equal to n_modes
        builder = ML.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=n_photons,
            builder=builder,
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
        )
        assert layer.output_size == n_modes
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == 3
        output.sum().backward()
        assert x.grad is not None

    def test_amplitude_vector(self):
        # Amplitudes is equivalent to return_amplitudes=True
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, layer.output_size)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        probs = output.abs().pow(2)
        targets = torch.ones_like(probs)
        loss = torch.sum(targets - probs)
        loss.backward()
        assert x.grad is not None

        # Ensure that it is normalized
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )

        # QuantumLayer's output_size is accessible and equal to the number of possible Fock states
        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.AMPLITUDES,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size

        # Backprop compatibility
        probs = output.abs().pow(2)
        targets = torch.ones_like(probs)
        loss = torch.sum(targets - probs)
        loss.backward()
        assert x.grad is not None

        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )


def test_resolve_measurement_strategy():
    assert isinstance(
        resolve_measurement_strategy(MeasurementStrategy.PROBABILITIES),
        ProbabilitiesStrategy,
    )
    assert isinstance(
        resolve_measurement_strategy(MeasurementStrategy.MODE_EXPECTATIONS),
        ModeExpectationsStrategy,
    )
    assert isinstance(
        resolve_measurement_strategy(MeasurementStrategy.AMPLITUDES),
        AmplitudesStrategy,
    )


def test_probabilities_strategy_applies_transforms_and_sampling():
    strategy = ProbabilitiesStrategy()
    distribution = torch.tensor([1.0])
    amplitudes = torch.tensor([0.5])

    def apply_photon_loss(dist: torch.Tensor) -> torch.Tensor:
        return dist * 2.0

    def apply_detectors(dist: torch.Tensor) -> torch.Tensor:
        return dist + 1.0

    def sample_fn(dist: torch.Tensor, shots: int) -> torch.Tensor:
        assert torch.allclose(dist, torch.tensor([3.0]))
        assert shots == 5
        return dist * 0 + shots

    result = strategy.process(
        distribution=distribution,
        amplitudes=amplitudes,
        apply_sampling=True,
        effective_shots=5,
        sample_fn=sample_fn,
        apply_photon_loss=apply_photon_loss,
        apply_detectors=apply_detectors,
    )

    assert torch.allclose(result, torch.tensor([5.0]))


def test_mode_expectations_strategy_skips_sampling_when_disabled():
    strategy = ModeExpectationsStrategy()
    distribution = torch.tensor([2.0])
    amplitudes = torch.tensor([0.25])
    sample_called = False

    def apply_photon_loss(dist: torch.Tensor) -> torch.Tensor:
        return dist * 3.0

    def apply_detectors(dist: torch.Tensor) -> torch.Tensor:
        return dist - 1.0

    def sample_fn(dist: torch.Tensor, shots: int) -> torch.Tensor:
        nonlocal sample_called
        sample_called = True
        return dist

    result = strategy.process(
        distribution=distribution,
        amplitudes=amplitudes,
        apply_sampling=False,
        effective_shots=10,
        sample_fn=sample_fn,
        apply_photon_loss=apply_photon_loss,
        apply_detectors=apply_detectors,
    )

    assert sample_called is False
    assert torch.allclose(result, torch.tensor([5.0]))


def test_amplitudes_strategy_returns_amplitudes_and_blocks_sampling():
    strategy = AmplitudesStrategy()
    distribution = torch.tensor([1.0])
    amplitudes = torch.tensor([0.75])

    def apply_photon_loss(dist: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Photon loss transform should not be called.")

    def apply_detectors(dist: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Detector transform should not be called.")

    def sample_fn(dist: torch.Tensor, shots: int) -> torch.Tensor:
        raise AssertionError("Sampling should not be called.")

    result = strategy.process(
        distribution=distribution,
        amplitudes=amplitudes,
        apply_sampling=False,
        effective_shots=0,
        sample_fn=sample_fn,
        apply_photon_loss=apply_photon_loss,
        apply_detectors=apply_detectors,
    )
    assert torch.allclose(result, amplitudes)

    with pytest.raises(RuntimeError):
        strategy.process(
            distribution=distribution,
            amplitudes=amplitudes,
            apply_sampling=True,
            effective_shots=1,
            sample_fn=sample_fn,
            apply_photon_loss=apply_photon_loss,
            apply_detectors=apply_detectors,
        )


class _DummyComputationProcess:
    def __init__(self):
        self.input_state = None
        self.called = None
        self.last_simultaneous_processes = None

    def compute_ebs_simultaneously(self, params, simultaneous_processes):
        self.called = "ebs"
        self.last_simultaneous_processes = simultaneous_processes
        return torch.tensor([float(simultaneous_processes)])

    def compute_superposition_state(self, params):
        self.called = "superposition"
        return torch.tensor([1.0])

    def compute(self, params):
        self.called = "compute"
        return torch.tensor([2.0])


def _build_test_layer(amplitude_encoding: bool = False) -> ML.QuantumLayer:
    builder = ML.CircuitBuilder(n_modes=2)
    builder.add_entangling_layer(trainable=True, name="U1")
    kwargs: dict[str, Any] = {"builder": builder}
    if amplitude_encoding:
        kwargs["amplitude_encoding"] = True
        kwargs["n_photons"] = 1
    else:
        builder.add_angle_encoding(modes=[0, 1], name="input")
        kwargs["input_size"] = 2
        kwargs["n_photons"] = 1
    return ML.QuantumLayer(**kwargs)


def test_compute_amplitudes_helper_prefers_amplitude_ebs_path():
    layer = _build_test_layer(amplitude_encoding=True)
    stub = _DummyComputationProcess()
    layer.computation_process = stub

    inferred_state = torch.ones(4)
    result = layer._compute_amplitudes(
        params=[torch.tensor([0.0])],
        inferred_state=inferred_state,
        parameter_batch_dim=0,
        simultaneous_processes=None,
    )

    assert stub.called == "ebs"
    assert stub.last_simultaneous_processes == 1
    assert torch.allclose(result, torch.tensor([1.0]))


def test_compute_amplitudes_helper_uses_across_batch_when_requested():
    layer = _build_test_layer(amplitude_encoding=True)
    stub = _DummyComputationProcess()
    layer.computation_process = stub

    inferred_state = torch.ones(3, 5)
    result = layer._compute_amplitudes(
        params=[torch.tensor([0.0])],
        inferred_state=inferred_state,
        parameter_batch_dim=0,
        simultaneous_processes=4,
    )

    assert stub.called == "ebs"
    assert stub.last_simultaneous_processes == 4
    assert torch.allclose(result, torch.tensor([4.0]))


def test_compute_amplitudes_helper_handles_missing_state_in_amplitude_mode():
    layer = _build_test_layer(amplitude_encoding=True)
    layer.computation_process = _DummyComputationProcess()

    with pytest.raises(TypeError):
        layer._compute_amplitudes(
            params=[torch.tensor([0.0])],
            inferred_state=None,
            parameter_batch_dim=0,
            simultaneous_processes=None,
        )


def test_compute_amplitudes_helper_batches_classical_inputs():
    layer = _build_test_layer()
    stub = _DummyComputationProcess()
    layer.computation_process = stub

    inferred_state = torch.ones(2, 6)
    result = layer._compute_amplitudes(
        params=[torch.tensor([0.0])],
        inferred_state=inferred_state,
        parameter_batch_dim=2,
        simultaneous_processes=None,
    )

    assert stub.called == "ebs"
    assert stub.last_simultaneous_processes == inferred_state.shape[-1]
    assert torch.allclose(result, torch.tensor([6.0]))


def test_compute_amplitudes_helper_uses_superposition_when_unbatched():
    layer = _build_test_layer()
    stub = _DummyComputationProcess()
    layer.computation_process = stub

    inferred_state = torch.ones(3, 2)
    result = layer._compute_amplitudes(
        params=[torch.tensor([0.0])],
        inferred_state=inferred_state,
        parameter_batch_dim=0,
        simultaneous_processes=None,
    )

    assert stub.called == "superposition"
    assert torch.allclose(result, torch.tensor([1.0]))


def test_compute_amplitudes_helper_delegates_to_compute_when_state_missing():
    layer = _build_test_layer()
    stub = _DummyComputationProcess()
    layer.computation_process = stub

    result = layer._compute_amplitudes(
        params=[torch.tensor([0.0])],
        inferred_state=None,
        parameter_batch_dim=0,
        simultaneous_processes=None,
    )

    assert stub.called == "compute"
    assert torch.allclose(result, torch.tensor([2.0]))
