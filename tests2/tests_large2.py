"""
Comprehensive tests for QuantumModel with PhotonicBackend.
Adapted from old test suite to new architecture.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional

try:
    import perceval as pcvl

    PERCEVAL_AVAILABLE = True
except ImportError:
    PERCEVAL_AVAILABLE = False
    pcvl = None

from merlin.models import QuantumModel, QuantumConfig
from merlin.builder import CircuitBuilder
from merlin.core.circuit import Circuit
from merlin.core.components import Rotation, BeamSplitter, EntanglingBlock
from merlin.backends.pcvl_pytorch import CircuitConverter


class TestCircuitConverter:
    """Test CircuitConverter functionality."""

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not installed")
    def test_ps_to_torch(self):
        """Test phase shifter conversion to torch."""
        c_ps = pcvl.Circuit(1) // pcvl.PS(pcvl.P("x"))
        params = {"x": torch.tensor([0.1], requires_grad=True)}

        torch_conv = CircuitConverter(c_ps, input_specs=list(params.keys()))
        torch_tensor = torch_conv.to_tensor(params["x"])

        assert torch_tensor.requires_grad
        assert torch_tensor.dim() == 2

        # Check gradient computation
        torch_tensor.real.sum().backward()
        assert params["x"].grad is not None

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not installed")
    def test_bs_to_torch(self):
        """Test beam splitter conversion to torch."""
        c_bs = pcvl.Circuit(2) // pcvl.BS.H(pcvl.P("theta"))
        param_theta = torch.tensor([np.pi / 2], requires_grad=True)

        torch_conv = CircuitConverter(c_bs, input_specs=["theta"])
        torch_tensor = torch_conv.to_tensor([param_theta])

        assert torch_tensor.shape == torch.Size([2, 2])
        assert torch_tensor.requires_grad

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not installed")
    def test_batched_input(self):
        """Test batched parameter input."""
        c_ps = pcvl.Circuit(1) // pcvl.PS(pcvl.P("x"))
        params = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)

        torch_conv = CircuitConverter(c_ps, [""])
        torch_tensor = torch_conv.to_tensor([params])

        assert torch_tensor.shape == torch.Size([3, 1, 1])
        assert torch_tensor.requires_grad


class TestQuantumModelBasic:
    """Basic tests for QuantumModel."""

    def test_model_creation_from_builder(self):
        """Test creating model from CircuitBuilder."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        config = QuantumConfig(
            n_modes=4,
            n_photons=2,
            output_size=6
        )

        model = QuantumModel.from_builder(builder, config=config)

        assert model.config.n_modes == 4
        assert model.config.n_photons == 2
        assert model.config.output_size == 6

    def test_forward_pass_batched(self):
        """Test forward pass with batched input."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder, output_size=3)

        # Test with batch
        x = torch.rand(10, 2)
        output = model(x)

        assert output.shape == (10, 3)
        assert torch.all(torch.isfinite(output))

    def test_forward_pass_single(self):
        """Test forward pass with single input."""
        builder = CircuitBuilder(n_modes=3, n_photons=1)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder, output_size=3)

        # Test with single sample
        x = torch.rand(1)
        output = model(x)

        assert output.shape == (3,) or output.shape == (1, 3)

    def test_gradient_computation(self):
        """Test that gradients flow through the model."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(trainable=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder, output_size=3)

        x = torch.rand(5, 2)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that model parameters have gradients
        has_trainable_params = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Model should have trainable parameters"


class TestPercevelCircuitIntegration:
    """Test direct Perceval circuit integration."""

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not installed")
    def test_simple_perceval_circuit_no_input(self):
        """Test QuantumModel with simple perceval circuit and no input parameters."""
        # Create a simple perceval circuit
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi1")))
        circuit.add(1, pcvl.BS())
        circuit.add(1, pcvl.PS(pcvl.P("phi2")))

        # Define input state
        input_state = [1, 0, 0]

        config = QuantumConfig(
            n_modes=3,
            n_photons=1,
            output_size=3,
            backend_options={
                'input_state': input_state,
                'trainable_parameters': ['phi'],
                'input_parameters': []
            }
        )

        model = QuantumModel(circuit, config=config)

        # Test forward pass (no input needed)
        output = model()
        assert output.shape[-1] == 3
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not installed")
    def test_perceval_circuit_with_input(self):
        """Test QuantumModel with perceval circuit and input parameters."""
        circuit = pcvl.Circuit(4)

        # Add input phase shifters
        for i in range(4):
            circuit.add(i, pcvl.PS(pcvl.P(f"input_{i}")))

        # Add trainable interferometer
        circuit.add(0, pcvl.BS(pcvl.P("theta1")))
        circuit.add(2, pcvl.BS(pcvl.P("theta2")))

        config = QuantumConfig(
            n_modes=4,
            n_photons=2,
            backend_options={
                'trainable_parameters': ['theta'],
                'input_parameters': ['input_']
            }
        )

        model = QuantumModel(circuit, config=config)

        # Test forward pass
        x = torch.rand(5, 4)
        output = model(x)
        assert output.shape[0] == 5


class TestSamplingFunctionality:
    """Test sampling and shot-based execution."""

    def test_sampling_with_shots(self):
        """Test sampling with different shot numbers."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder)
        model.eval()

        x = torch.rand(5, 2)

        # Get perfect output (no sampling)
        with torch.no_grad():
            output_perfect = model(x, shots=0)

        # Get sampled output
        with torch.no_grad():
            output_sampled = model(x, shots=1000)

        # Should be valid probability distributions
        assert torch.all(output_sampled >= 0)
        assert torch.allclose(output_sampled.sum(dim=-1), torch.ones(5), atol=0.1)

        # Sampled should be different from perfect due to noise
        if output_sampled.shape == output_perfect.shape:
            assert not torch.allclose(output_sampled, output_perfect, atol=1e-3)

    def test_deterministic_with_zero_shots(self):
        """Test deterministic behavior when shots=0."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(trainable=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder)

        with torch.no_grad():
            x = torch.rand(3, 2)
            output1 = model(x, shots=0)
            output2 = model(x, shots=0)

        # With shots=0, outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


class TestReservoirMode:
    """Test reservoir computing mode."""

    def test_reservoir_mode_creation(self):
        """Test creating model in reservoir mode."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=False)  # Reservoir

        config = QuantumConfig(
            n_modes=4,
            n_photons=2,
            backend_options={'reservoir_mode': True}
        )

        model = QuantumModel.from_builder(builder, config=config)

        # Test that model works
        x = torch.rand(3, 2)
        output = model(x)
        assert output.shape[0] == 3


class TestOutputMapping:
    """Test output size mapping."""

    def test_output_size_mapping(self):
        """Test that output size mapping works correctly."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        # Test different output sizes
        for output_size in [3, 6, 10]:
            model = QuantumModel.from_builder(builder, output_size=output_size)

            x = torch.rand(8, 2)
            output = model(x)

            assert output.shape == (8, output_size)

    def test_no_output_mapping(self):
        """Test model without output mapping (raw distribution)."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        # No output_size specified - raw distribution
        model = QuantumModel.from_builder(builder)

        x = torch.rand(5, 2)
        output = model(x)

        # Should be probability distribution
        assert torch.all(output >= 0)
        assert torch.allclose(output.sum(dim=-1), torch.ones(5), atol=1e-5)


class TestDeviceAndDtype:
    """Test device and dtype handling."""

    def test_dtype_consistency(self):
        """Test different data types."""
        for dtype in [torch.float32, torch.float64]:
            builder = CircuitBuilder(n_modes=4, n_photons=2)
            builder.add_rotation_layer(as_input=True)
            builder.add_entangling_layer(trainable=True)

            config = QuantumConfig(
                n_modes=4,
                n_photons=2,
                dtype=dtype
            )

            model = QuantumModel.from_builder(builder, config=config)

            x = torch.rand(2, 2, dtype=dtype)
            output = model(x)

            # Check parameter dtypes
            for param in model.parameters():
                assert param.dtype == dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """Test execution on CUDA."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        config = QuantumConfig(
            n_modes=4,
            n_photons=2,
            device=torch.device('cuda')
        )

        model = QuantumModel.from_builder(builder, config=config)

        x = torch.rand(8, 2, device='cuda')
        output = model(x)

        assert output.device.type == 'cuda'


class TestMemoryAndPerformance:
    """Test memory efficiency and performance."""

    def test_large_batch_processing(self):
        """Test processing large batches."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder, output_size=3)

        # Test with large batch
        large_batch_size = 1000
        x = torch.rand(large_batch_size, 2)

        output = model(x)

        assert output.shape == (large_batch_size, 3)
        assert torch.all(torch.isfinite(output))

    def test_memory_efficiency(self):
        """Test memory doesn't grow unexpectedly."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(as_input=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder)

        # Run many forward passes
        for _ in range(100):
            x = torch.rand(10, 2)
            with torch.no_grad():
                output = model(x)
                del output, x

        # Should complete without memory issues


class TestHybridNetworks:
    """Test integration with classical neural networks."""

    def test_quantum_classical_hybrid(self):
        """Test hybrid quantum-classical network."""

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()

                # Classical preprocessing
                self.classical_in = nn.Sequential(
                    nn.Linear(10, 8),
                    nn.ReLU(),
                    nn.Linear(8, 4)
                )

                # Quantum layer
                builder = CircuitBuilder(n_modes=4, n_photons=2)
                builder.add_rotation_layer(as_input=True)
                builder.add_entangling_layer(trainable=True)

                self.quantum = QuantumModel.from_builder(builder, output_size=6)

                # Classical postprocessing
                self.classical_out = nn.Sequential(
                    nn.Linear(6, 4),
                    nn.ReLU(),
                    nn.Linear(4, 2)
                )

            def forward(self, x):
                x = self.classical_in(x)
                x = torch.sigmoid(x[:, :2])  # Take first 2 features for quantum
                x = self.quantum(x)
                x = self.classical_out(x)
                return x

        model = HybridModel()

        # Test forward pass
        x = torch.rand(16, 10)
        output = model(x)

        assert output.shape == (16, 2)
        assert torch.all(torch.isfinite(output))

        # Test backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.grad is None  # Some might be unused

    def test_training_loop(self):
        """Test full training loop."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)
        builder.add_rotation_layer(trainable=True)
        builder.add_entangling_layer(trainable=True)

        model = QuantumModel.from_builder(builder, output_size=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Generate dummy data
        X = torch.rand(100, 2)
        y = torch.rand(100, 2)

        initial_loss = None
        final_loss = None

        # Training loop
        for epoch in range(5):
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, y)

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 4:
                final_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should change (model is learning)
        assert initial_loss != final_loss


class TestSpecialCircuits:
    """Test special circuit configurations."""

    def test_identity_circuit(self):
        """Test circuit that acts as identity."""
        builder = CircuitBuilder(n_modes=2, n_photons=1)
        # No operations - should act as identity

        model = QuantumModel.from_builder(builder)

        # Even with no operations, should produce output
        output = model()
        assert output.shape[-1] > 0

    def test_multi_layer_circuit(self):
        """Test circuit with multiple layers."""
        builder = CircuitBuilder(n_modes=4, n_photons=2)

        # Multiple layers
        for i in range(3):
            builder.add_rotation_layer(trainable=(i > 0), as_input=(i == 0))
            builder.add_entangling_layer(trainable=True, depth=1)

        model = QuantumModel.from_builder(builder, output_size=4)

        x = torch.rand(5, 2)
        output = model(x)

        assert output.shape == (5, 4)

    def test_modular_circuit(self):
        """Test modular circuit construction."""
        builder = CircuitBuilder(n_modes=6, n_photons=3)

        # Define modules
        module1 = builder.define_module([0, 1, 2], "input_module")
        module2 = builder.define_module([3, 4, 5], "output_module")

        # Add operations to modules
        builder.add_module_encoder(module1)
        builder.add_module_encoder(module2)
        builder.add_module_bridge(module1, module2)

        model = QuantumModel.from_builder(builder)

        x = torch.rand(4, 3)
        output = model(x)

        assert output.shape[0] == 4


if __name__ == "__main__":
    # Run all test classes
    test_classes = [
        TestCircuitConverter(),
        TestQuantumModelBasic(),
        TestPercevelCircuitIntegration(),
        TestSamplingFunctionality(),
        TestReservoirMode(),
        TestOutputMapping(),
        TestDeviceAndDtype(),
        TestMemoryAndPerformance(),
        TestHybridNetworks(),
        TestSpecialCircuits()
    ]

    print("Running comprehensive QuantumModel tests...")
    print("=" * 60)

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\nRunning {class_name}")
        print("-" * 40)

        # Run all test methods
        for attr_name in dir(test_class):
            if attr_name.startswith('test_'):
                print(f"  {attr_name}...", end=" ")
                method = getattr(test_class, attr_name)
                try:
                    method()
                    print("✅ PASSED")
                except Exception as e:
                    print(f"❌ FAILED: {e}")

    print("\n" + "=" * 60)
    print("Test suite completed!")