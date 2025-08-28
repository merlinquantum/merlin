"""
Tests for quantum kernel functionality using section referencing.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

try:
    import perceval as pcvl

    PERCEVAL_AVAILABLE = True
except ImportError:
    PERCEVAL_AVAILABLE = False

from merlin.builder import CircuitBuilder
from merlin.models import QuantumModel, QuantumConfig


@pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not available")
class TestKernelWithSections:
    """Test quantum kernels using section referencing."""

    def test_kernel_with_different_inputs(self):
        """Test kernel computation with shared trainable but different input params."""
        n_modes = 3
        n_photons = 2

        # Build kernel circuit with section referencing
        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # First feature map for x_i
        builder.begin_section("feature_map_i")
        builder.add_entangling_layer(trainable=True, depth=1)
        builder.add_input_encoding(name="xi")  # xi1, xi2, xi3
        builder.add_entangling_layer(trainable=True, depth=1)
        builder.end_section()

        # Adjoint feature map for x_j with same trainable params
        builder.add_adjoint_section(
            "feature_map_j",
            reference="feature_map_i",
            share_trainable=True,  # Same feature map weights
            share_input=False  # Different input parameters (xi4, xi5, xi6)
        )

        circuit = builder.build()

        # Verify we have different input parameters
        assert 'sections' in circuit.metadata
        assert len(circuit.metadata['sections']) == 2

        # Create model
        config = QuantumConfig(
            n_modes=n_modes,
            n_photons=n_photons,
            backend_type="photonic",
            backend_options={
                'input_state': [1, 1, 0],
                'no_bunching': True,
                'input_parameters': ['xi']  # Will match xi1-xi6
            }
        )

        model = QuantumModel(circuit, config=config)

        # For kernel K(x_i, x_j), we need to pass both x_i and x_j
        # The first 3 features go to xi1,xi2,xi3 and next 3 to xi4,xi5,xi6
        x_i = torch.tensor([0.1, 0.2, 0.3])
        x_j = torch.tensor([0.4, 0.5, 0.6])
        x_combined = torch.cat([x_i, x_j]).unsqueeze(0)  # Shape [1, 6]

        # Get probability distribution
        probs = model(x_combined)

        # Should return to initial state with some probability
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)

        # For identical inputs, should have high return probability
        x_same = torch.cat([x_i, x_i]).unsqueeze(0)
        probs_same = model(x_same)
        assert probs_same.max().item() > 0.5  # High probability at initial state

    def test_autoencoder_pattern(self):
        """Test autoencoder with shared everything."""
        n_modes = 3
        n_photons = 2

        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # Encoder
        builder.begin_section("encoder")
        builder.add_entangling_layer(trainable=True)
        builder.add_input_encoding()
        builder.add_entangling_layer(trainable=True)
        builder.end_section()

        # Decoder (adjoint with all parameters shared)
        builder.add_adjoint_section(
            "decoder",
            reference="encoder",
            share_trainable=True,
            share_input=True  # Same input for autoencoder
        )

        circuit = builder.build()

        config = QuantumConfig(
            n_modes=n_modes,
            n_photons=n_photons,
            backend_type="photonic",
            backend_options={
                'input_state': [1, 1, 0],
                'no_bunching': True,
                'input_parameters': ['px']
            }
        )

        model = QuantumModel(circuit, config=config)

        # Should reconstruct for any input
        x = torch.zeros(1, 3)
        probs = model(x)
        assert probs.max().item() > 0.9  # Returns to initial state

    def test_inverse_all_pattern(self):
        """Test taking inverse of everything before sections."""
        n_modes = 3
        n_photons = 2

        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # Build some circuit
        builder.add_rotation_layer(trainable=True)
        builder.add_entangling_layer()
        builder.add_input_encoding()

        # Add inverse of everything
        builder.begin_section(
            "inverse",
            compute_adjoint=True,
            reference="_all_",
            share_trainable=True,
            share_input=True
        )

        circuit = builder.build()

        config = QuantumConfig(
            n_modes=n_modes,
            n_photons=n_photons,
            backend_type="photonic",
            backend_options={
                'input_state': [1, 1, 0],
                'no_bunching': True,
                'input_parameters': ['px']
            }
        )

        model = QuantumModel(circuit, config=config)

        x = torch.zeros(1, 3)
        probs = model(x)
        assert probs.max().item() > 0.9  # Identity transformation


@pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not available")
class TestParameterSharing:
    """Test parameter sharing mechanisms."""

    def test_trainable_parameter_sharing(self):
        """Verify trainable parameters are shared when requested."""
        builder = CircuitBuilder(n_modes=3)

        builder.begin_section("original")
        builder.add_rotation_layer(trainable=True, name="theta")
        builder.end_section()

        # Share trainable
        builder.begin_section(
            "copy_shared",
            reference="original",
            share_trainable=True
        )

        # Don't share trainable
        builder.begin_section(
            "copy_not_shared",
            reference="original",
            share_trainable=False
        )

        circuit = builder.build()

        # Check parameter names in components
        comp_names = []
        for comp in circuit.components:
            if hasattr(comp, 'custom_name') and comp.custom_name:
                comp_names.append(comp.custom_name)

        # Should have original names and copy names
        assert any('theta' in name and 'copy' not in name for name in comp_names)
        assert any('copy' in name for name in comp_names)

    def test_input_parameter_sharing(self):
        """Verify input parameters are NOT shared by default."""
        builder = CircuitBuilder(n_modes=3)

        builder.begin_section("original")
        builder.add_input_encoding()  # px1, px2, px3
        builder.end_section()

        # Don't share input (default)
        builder.begin_section(
            "copy",
            reference="original",
            share_input=False
        )

        circuit = builder.build()

        # Count px parameters
        px_params = []
        for comp in circuit.components:
            if hasattr(comp, 'custom_name') and comp.custom_name and 'px' in comp.custom_name:
                px_params.append(comp.custom_name)

        # Should have 6 unique px parameters
        assert len(set(px_params)) == 6
        assert 'px1' in px_params and 'px4' in px_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])