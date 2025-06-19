import torch
import pytest
import math
from merlin import QuantumLayer, OutputMappingStrategy  # Replace with actual import path
import perceval as pcvl

def classical_method(layer, input_state):
    output_classical = torch.zeros(layer.output_size)
    for key, value in input_state.items():
        layer.computation_process.input_state = key
        output_classical += value * layer()
    return output_classical


class TestOutputSuperposedState:
    """Test cases for output mapping strategies in QuantumLayer.simple()."""

    def test_superposed_state(self, benchmark):
        """Test NONE strategy when output_size is not specified."""
        print("\n=== Testing Superposed input state method ===")

        # When using NONE strategy without specifying output_size,
        # the output size should equal the distribution size
        circuit = pcvl.components.GenericInterferometer(
            6,
            pcvl.components.catalog['mzi phase last'].generate,
            shape=pcvl.InterferometerShape.RECTANGLE
        )
        input_state_superposed = {(1, 1, 1, 0, 0, 0): 0.6, (0, 1, 1, 1, 0, 0):0.3,
            (0, 0, 1, 0, 1, 1):0.4, (0, 1, 1, 0, 1, 0):0.25,
            (0, 0, 1, 1, 0, 1):0.45, (1, 1, 0, 1, 0, 0):0.4,
            (1, 1, 0, 0, 0, 1):0.25}
        sum_values = sum([k**2 for k in list(input_state_superposed.values())])
        for key in input_state_superposed.keys():
            input_state_superposed[key] = input_state_superposed[key] / (sum_values)**0.5
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state_superposed,
            trainable_parameters=["phi"],
        )

        output_superposed = benchmark(layer)

        output_classical = classical_method(layer, input_state_superposed)
        assert torch.allclose(output_superposed, output_classical, rtol=1e-3, atol=1e-6)


    def test_classical_method(self, benchmark):
        """Test NONE strategy when output_size is not specified."""
        print("\n=== Testing Superposed input state method ===")

        # When using NONE strategy without specifying output_size,
        # the output size should equal the distribution size
        circuit = pcvl.components.GenericInterferometer(
            6,
            pcvl.components.catalog['mzi phase last'].generate,
            shape=pcvl.InterferometerShape.RECTANGLE
        )
        input_state_superposed = {(1, 1, 1, 0, 0, 0): 0.6, (0, 1, 1, 1, 0, 0):0.3,
            (0, 0, 1, 0, 1, 1):0.4, (0, 1, 1, 0, 1, 0):0.25,
            (0, 0, 1, 1, 0, 1):0.45, (1, 1, 0, 1, 0, 0):0.4,
            (1, 1, 0, 0, 0, 1):0.25}
        sum_values = sum([k**2 for k in list(input_state_superposed.values())])
        for key in input_state_superposed.keys():
            input_state_superposed[key] = input_state_superposed[key] / (sum_values)**0.5
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state_superposed,
            trainable_parameters=["phi"],
        )

        output_superposed = layer()

        output_classical = benchmark(lambda: classical_method(layer, input_state_superposed))
        assert torch.allclose(output_superposed, output_classical, rtol=1e-3, atol=1e-7)
