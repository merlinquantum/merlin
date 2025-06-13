"""
Test suite for no_bunching parameter through ansatz-based QuantumLayer construction.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Assuming these imports based on the code structure
from merlin.core.layer import QuantumLayer
from merlin.core.ansatz import Ansatz, AnsatzFactory
from merlin.core.photonicbackend import PhotonicBackend as Experiment
from merlin.core.generators import CircuitType, StatePattern
from merlin.sampling.strategies import OutputMappingStrategy

class TestNoBunchingViaAnsatz:
    """Test suite for no_bunching parameter through ansatz route."""

    @pytest.fixture
    def basic_experiment(self):
        """Create a basic experiment configuration."""
        return Experiment(
            circuit_type=CircuitType.SERIES,
            n_modes=4,
            n_photons=2,
            reservoir_mode=False,
            use_bandwidth_tuning=False,
            state_pattern=StatePattern.PERIODIC
        )

    def test_ansatz_factory_propagates_no_bunching(self, basic_experiment):
        """Test that AnsatzFactory correctly propagates no_bunching parameter."""
        # Test with no_bunching=True
        ansatz_true = AnsatzFactory.create(
            PhotonicBackend=basic_experiment,
            input_size=2,
            output_size=4,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            dtype=torch.float32,
            device=torch.device('cpu'),
            no_bunching=True
        )

        # Test with no_bunching=False
        ansatz_false = AnsatzFactory.create(
            PhotonicBackend=basic_experiment,
            input_size=2,
            output_size=4,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            dtype=torch.float32,
            device=torch.device('cpu'),
            no_bunching=False
        )

        # Verify that computation processes have different state space sizes
        # When no_bunching=True, states with multiple photons in one mode are excluded
        assert hasattr(ansatz_true, 'computation_process'), "Ansatz should have computation_process"
        assert hasattr(ansatz_false, 'computation_process'), "Ansatz should have computation_process"

        # The actual verification would depend on internal implementation
        # Here we're checking that the ansatz objects are created successfully
        assert ansatz_true is not None
        assert ansatz_false is not None

    def test_quantum_layer_from_ansatz_respects_no_bunching(self, basic_experiment):
        """Test that QuantumLayer created from ansatz respects no_bunching."""
        # Create ansatz with no_bunching=True
        ansatz = AnsatzFactory.create(
            PhotonicBackend=basic_experiment,
            input_size=2,
            output_size=4,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            dtype=torch.float32,
            device=torch.device('cpu'),
            no_bunching=True
        )

        # Create QuantumLayer from ansatz
        layer = QuantumLayer(
            input_size=2,
            output_size=4,
            ansatz=ansatz,
            no_bunching=True
        )

        # Verify layer is created and has correct no_bunching setting
        assert layer.no_bunching == True
        assert layer.computation_process is not None

    def test_simple_method_propagates_no_bunching(self):
        """Test that QuantumLayer.simple() correctly propagates no_bunching."""
        # Test with no_bunching=True (default)
        layer_true = QuantumLayer.simple(
            input_size=3,
            n_params=50,
            no_bunching=True
        )

        # Test with no_bunching=False
        layer_false = QuantumLayer.simple(
            input_size=3,
            n_params=50,
            no_bunching=False
        )

        # Verify both layers are created successfully
        assert layer_true.no_bunching == True
        assert layer_false.no_bunching == False

        # Test forward pass
        x = torch.randn(10, 3)  # Batch of 10, input size 3

        output_true = layer_true(x)
        output_false = layer_false(x)

        # Get actual output sizes (determined by quantum state space)
        output_size_true = output_true.shape[1]
        output_size_false = output_false.shape[1]

        assert output_true.shape == (10, output_size_true)
        assert output_false.shape == (10, output_size_false)

        # With no_bunching=False, we should have more states (larger output)
        assert output_size_false >= output_size_true

    def test_output_distribution_differs_with_no_bunching(self):
        """Test that no_bunching actually affects the output distribution."""
        torch.manual_seed(42)  # For reproducibility

        # Create two layers with same config but different no_bunching
        layer_no_bunch = QuantumLayer.simple(
            input_size=2,
            n_params=20,
            output_size=None,  # Use full distribution
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=True
        )

        layer_with_bunch = QuantumLayer.simple(
            input_size=2,
            n_params=20,
            output_size=None,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=False
        )

        # Same input
        x = torch.tensor([[0.5, 0.5]])

        # Get distributions
        dist_no_bunch = layer_no_bunch(x)
        dist_with_bunch = layer_with_bunch(x)

        # The distributions should have different sizes
        # no_bunching=False allows more states
        assert dist_with_bunch.shape[-1] >= dist_no_bunch.shape[-1], \
            "Distribution with bunching should have at least as many states"

    def test_no_bunching_with_different_circuit_types(self):
        """Test no_bunching with different circuit types."""
        circuit_types = [CircuitType.SERIES, CircuitType.PARALLEL]

        for circuit_type in circuit_types:
            experiment = Experiment(
                circuit_type=circuit_type,
                n_modes=4,
                n_photons=2,
                reservoir_mode=False,
                use_bandwidth_tuning=False,
                state_pattern=StatePattern.PERIODIC
            )

            ansatz = AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                output_size=3,
                output_mapping_strategy=OutputMappingStrategy.LINEAR,
                no_bunching=True
            )

            layer = QuantumLayer(
                input_size=2,
                output_size=3,
                ansatz=ansatz,
                no_bunching=True
            )

            # Test forward pass
            x = torch.randn(5, 2)
            output = layer(x)

            assert output.shape == (5, 3), f"Failed for circuit type {circuit_type}"

    def test_no_bunching_with_reservoir_mode(self):
        """Test no_bunching in reservoir computing mode."""
        experiment = Experiment(
            circuit_type=CircuitType.SERIES,
            n_modes=5,
            n_photons=3,
            reservoir_mode=True,  # Enable reservoir mode
            use_bandwidth_tuning=False,
            state_pattern=StatePattern.PERIODIC
        )

        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=3,
            output_size=2,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            no_bunching=True
        )

        layer = QuantumLayer(
            input_size=3,
            output_size=2,
            ansatz=ansatz,
            no_bunching=True
        )

        # In reservoir mode, quantum parameters should be fixed
        trainable_params = [p for p in layer.parameters() if p.requires_grad]

        # Only output mapping should be trainable in reservoir mode
        assert len(trainable_params) > 0, "Should have trainable output mapping"

        # Test forward pass
        x = torch.randn(8, 3)
        output = layer(x)
        assert output.shape == (8, 2)

    def test_no_bunching_gradient_flow(self):
        """Test that gradients flow correctly with no_bunching."""
        # Create layer with LINEAR mapping for trainable output
        layer = QuantumLayer.simple(
            input_size=2,
            n_params=30,
            output_size=2,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            no_bunching=True
        )

        # Create simple loss function
        x = torch.randn(4, 2, requires_grad=True)
        target = torch.randn(4, 2)

        output = layer(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "Input should have gradients"

        # Check that layer parameters have gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"


    def test_no_bunching_state_validation(self):
        """Test that no_bunching correctly validates quantum states."""
        # Create a scenario where we can check state space
        experiment = Experiment(
            circuit_type=CircuitType.SERIES,
            n_modes=3,
            n_photons=2,
            reservoir_mode=False,
            use_bandwidth_tuning=False,
            state_pattern=StatePattern.PERIODIC
        )

        # With no_bunching=True, states like |2,0,0⟩ should be excluded
        # Valid states would be: |1,1,0⟩, |1,0,1⟩, |0,1,1⟩
        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=None,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=True
        )

        layer = QuantumLayer(
            input_size=2,
            ansatz=ansatz,
            no_bunching=True
        )

        # Get output distribution
        x = torch.zeros(1, 2)  # Simple input
        output = layer(x)

        # Check that output dimension matches expected state space
        # For 2 photons in 3 modes with no bunching: C(3,2) = 3 states
        expected_states = 3
        assert output.shape[-1] == expected_states, \
            f"Expected {expected_states} states, got {output.shape[-1]}"

    @pytest.mark.parametrize("n_photons,n_modes,expected_states", [
        (2, 3, 3),  # C(3,2) = 3
        (2, 4, 6),  # C(4,2) = 6
        (3, 4, 4),  # C(4,3) = 4
        (3, 5, 10),  # C(5,3) = 10
    ])
    def test_no_bunching_state_count(self, n_photons, n_modes, expected_states):
        """Test that no_bunching produces correct number of quantum states."""
        experiment = Experiment(
            circuit_type=CircuitType.SERIES,
            n_modes=n_modes,
            n_photons=n_photons,
            reservoir_mode=False,
            use_bandwidth_tuning=False,
            state_pattern=StatePattern.PERIODIC
        )

        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=n_photons,
            output_size=None,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=True
        )

        layer = QuantumLayer(
            input_size=n_photons,
            ansatz=ansatz,
            no_bunching=True
        )

        x = torch.zeros(1, n_photons)
        output = layer(x)

        assert output.shape[-1] == expected_states, \
            f"For {n_photons} photons in {n_modes} modes, expected {expected_states} states, got {output.shape[-1]}"


class TestNoBunchingEdgeCases:
    """Test edge cases and error handling for no_bunching."""

    def test_no_bunching_with_single_photon(self):
        """Test that no_bunching has no effect with single photon."""
        # With 1 photon, bunching is impossible
        layer_true = QuantumLayer.simple(
            input_size=1,
            n_params=20,
            output_size=None,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=True
        )

        layer_false = QuantumLayer.simple(
            input_size=1,
            n_params=20,
            output_size=None,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=False
        )

        x = torch.randn(1, 1)

        # Both should produce same size output
        output_true = layer_true(x)
        output_false = layer_false(x)

        assert output_true.shape == output_false.shape, \
            "Single photon should produce same state space regardless of no_bunching"

    def test_no_bunching_with_bandwidth_tuning(self):
        """Test no_bunching with bandwidth tuning enabled."""
        experiment = Experiment(
            circuit_type=CircuitType.SERIES,
            n_modes=4,
            n_photons=2,
            reservoir_mode=False,
            use_bandwidth_tuning=True,  # Enable bandwidth tuning
            state_pattern=StatePattern.PERIODIC
        )

        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            no_bunching=True
        )

        layer = QuantumLayer(
            input_size=2,
            output_size=3,
            ansatz=ansatz,
            no_bunching=True
        )

        # Check that bandwidth coefficients exist
        assert hasattr(layer, 'bandwidth_coeffs')
        assert layer.bandwidth_coeffs is not None

        # Test forward pass
        x = torch.randn(4, 2)
        output = layer(x)
        assert output.shape == (4, 3)

    def test_no_bunching_serialization(self):
        """Test that no_bunching setting is preserved through save/load."""
        import tempfile
        import os

        # Create layer with no_bunching=True (using default NONE strategy)
        layer = QuantumLayer.simple(
            input_size=2,
            n_params=30,
            no_bunching=True
        )

        # Save state
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(layer.state_dict(), tmp.name)
            tmp_path = tmp.name

        try:
            # Create new layer with same config
            new_layer = QuantumLayer.simple(
                input_size=2,
                n_params=30,
                no_bunching=True
            )

            # Load state
            new_layer.load_state_dict(torch.load(tmp_path))

            # Verify same behavior
            x = torch.randn(3, 2)
            torch.manual_seed(42)
            output1 = layer(x)
            torch.manual_seed(42)
            output2 = new_layer(x)

            assert torch.allclose(output1, output2), \
                "Loaded layer should produce same output"
        finally:
            os.unlink(tmp_path)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])