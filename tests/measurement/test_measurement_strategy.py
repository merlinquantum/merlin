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


import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import MeasurementStrategy


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
                output_size=5, # cannot specify output_size
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
