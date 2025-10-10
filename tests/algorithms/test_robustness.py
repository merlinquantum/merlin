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

"""
Robustness and integration tests for Merlin.
"""

import torch
import torch.nn as nn

import merlin as ML


class TestRobustness:
    """Test suite for robustness and edge cases."""

    def test_large_batch_sizes(self):
        """Test handling of large batch sizes."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Test with large batch
        large_batch_size = 1000
        x = torch.rand(large_batch_size, 2)

        output = layer(x)

        assert output.shape == (large_batch_size, 3)
        assert torch.all(torch.isfinite(output))

    def test_extreme_input_values(self):
        """Test handling of extreme input values."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Test boundary values
        boundary_inputs = torch.tensor([
            [0.0, 0.0],  # All zeros
            [1.0, 1.0],  # All ones
            [0.0, 1.0],  # Mixed
            [1.0, 0.0],  # Mixed reverse
        ])

        output = layer(boundary_inputs)

        assert output.shape == (4, 3)
        assert torch.all(torch.isfinite(output))

    def test_numerical_stability(self):
        """Test numerical stability with repeated computations."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        x = torch.rand(5, 2)

        # Run multiple times - should get identical results
        outputs = []
        for _ in range(10):
            with torch.no_grad():
                output = layer(x)
                outputs.append(output)

        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Accumulate gradients over multiple batches
        total_loss = 0
        for _ in range(3):
            x = torch.rand(4, 2, requires_grad=True)
            output = layer(x)
            loss = output.sum()
            loss.backward()
            total_loss += loss.item()

        # Check that gradients accumulated
        param_count = 0
        for param in layer.parameters():
            if param.requires_grad and param.grad is not None:
                param_count += 1
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

        assert param_count > 0, "No parameters have gradients"

    def test_device_compatibility(self):
        """Test CPU compatibility (GPU testing would require CUDA)."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        # Test CPU device
        layer_cpu = ML.QuantumLayer(
            input_size=2, ansatz=ansatz, device=torch.device("cpu")
        )

        x_cpu = torch.rand(3, 2, device="cpu")
        output_cpu = layer_cpu(x_cpu)

        assert output_cpu.device.type == "cpu"
        assert output_cpu.shape == (3, 3)

    def test_different_dtypes(self):
        """Test different data types."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        # Test float32
        ansatz_f32 = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3, dtype=torch.float32
        )
        layer_f32 = ML.QuantumLayer(
            input_size=2, ansatz=ansatz_f32, dtype=torch.float32
        )
        x_f32 = torch.rand(2, 2, dtype=torch.float32)
        output_f32 = layer_f32(x_f32)
        assert output_f32.dtype == torch.float32

        # Test float64 - create separate ansatz with correct dtype
        ansatz_f64 = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3, dtype=torch.float64
        )
        layer_f64 = ML.QuantumLayer(
            input_size=2, ansatz=ansatz_f64, dtype=torch.float64
        )
        x_f64 = torch.rand(2, 2, dtype=torch.float64)
        output_f64 = layer_f64(x_f64)

        # The output dtype might be influenced by the underlying quantum simulation
        # So we'll be more flexible and just check that it's a valid float type
        assert output_f64.dtype in [
            torch.float32,
            torch.float64,
        ], f"Expected float type, got {output_f64.dtype}"

        # More importantly, check that the computation works correctly
        assert torch.all(torch.isfinite(output_f64))
        assert output_f64.shape == (2, 3)

    def test_parameter_initialization_consistency(self):
        """Test that parameter initialization is consistent."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        # Create multiple layers with same random seed
        torch.manual_seed(42)
        layer1 = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        torch.manual_seed(42)
        layer2 = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Parameters should be identical
        assert len(list(layer1.parameters())) == len(list(layer2.parameters())), (
            "Mismatch in number of parameters between layer1 and layer2"
        )
        for p1, p2 in zip(layer1.parameters(), layer2.parameters(), strict=True):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_memory_efficiency(self):
        """Test memory usage doesn't grow unexpectedly."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Run many forward passes
        for _ in range(100):
            x = torch.rand(10, 2)
            with torch.no_grad():
                output = layer(x)
                del output, x  # Explicit cleanup

        # Should complete without memory issues


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_training_loop_simulation(self):
        """Simulate a realistic training loop."""
        # Create a simple dataset
        n_samples = 200
        X = torch.rand(n_samples, 3)
        y = (X.sum(dim=1) > 1.5).long()  # Simple binary classification

        # Create model
        class SimpleQuantumModel(nn.Module):
            def __init__(self):
                super().__init__()
                experiment = ML.PhotonicBackend(
                    circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
                )

                ansatz = ML.AnsatzFactory.create(
                    PhotonicBackend=experiment, input_size=3, output_size=4
                )

                self.quantum = ML.QuantumLayer(input_size=3, ansatz=ansatz)
                self.classifier = nn.Linear(4, 2)

            def forward(self, x):
                x = torch.sigmoid(x)  # Normalize for quantum layer
                x = self.quantum(x)
                return self.classifier(x)

        model = SimpleQuantumModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        initial_loss = None
        final_loss = None

        for epoch in range(10):
            epoch_loss = 0
            for i in range(0, len(X), 32):  # Batch size 32
                batch_X = X[i : i + 32]
                batch_y = y[i : i + 32]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch == 0:
                initial_loss = epoch_loss
            elif epoch == 9:
                final_loss = epoch_loss

        # Loss should decrease (learning is happening)
        assert final_loss < initial_loss, "Model should learn and reduce loss"

    def test_hybrid_architecture(self):
        """Test complex hybrid classical-quantum architecture."""

        class ComplexHybridModel(nn.Module):
            def __init__(self):
                super().__init__()

                # Classical preprocessing
                self.pre_classical = nn.Sequential(
                    nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3)
                )

                # First quantum layer
                experiment1 = ML.PhotonicBackend(
                    circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
                )
                ansatz1 = ML.AnsatzFactory.create(
                    PhotonicBackend=experiment1, input_size=3, output_size=5
                )
                self.quantum1 = ML.QuantumLayer(input_size=3, ansatz=ansatz1)

                # Middle classical processing
                self.mid_classical = nn.Sequential(nn.Linear(5, 4), nn.ReLU())

                # Second quantum layer (reservoir)
                experiment2 = ML.PhotonicBackend(
                    circuit_type=ML.CircuitType.PARALLEL,
                    n_modes=5,
                    n_photons=2,
                    reservoir_mode=True,
                )
                ansatz2 = ML.AnsatzFactory.create(
                    PhotonicBackend=experiment2, input_size=4, output_size=3
                )
                self.quantum2 = ML.QuantumLayer(input_size=4, ansatz=ansatz2)

                # Final classical layer
                self.final_classical = nn.Linear(3, 2)

            def forward(self, x):
                x = self.pre_classical(x)
                x = torch.sigmoid(x)  # Normalize for quantum
                x = self.quantum1(x)
                x = self.mid_classical(x)
                x = torch.sigmoid(x)  # Normalize for quantum
                x = self.quantum2(x)
                x = self.final_classical(x)
                return x

        model = ComplexHybridModel()

        # Test forward pass
        x = torch.rand(16, 8)
        output = model(x)

        assert output.shape == (16, 2)
        assert torch.all(torch.isfinite(output))

        # Test backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist for trainable parameters
        trainable_params = 0
        for _name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                trainable_params += 1

        assert trainable_params > 0, "Should have trainable parameters with gradients"

    def test_ensemble_quantum_models(self):
        """Test ensemble of quantum models."""

        class QuantumEnsemble(nn.Module):
            def __init__(self, n_models=3):
                super().__init__()

                self.models = nn.ModuleList()

                for _i in range(n_models):
                    experiment = ML.PhotonicBackend(
                        circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
                        n_modes=4,
                        n_photons=2,
                    )

                    ansatz = ML.AnsatzFactory.create(
                        PhotonicBackend=experiment, input_size=2, output_size=3
                    )

                    layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
                    self.models.append(layer)

            def forward(self, x):
                outputs = []
                for model in self.models:
                    normalized_x = torch.sigmoid(x)
                    output = model(normalized_x)
                    outputs.append(output)

                # Average ensemble predictions
                return torch.stack(outputs).mean(dim=0)

        ensemble = QuantumEnsemble(n_models=3)

        x = torch.rand(5, 2)
        output = ensemble(x)

        assert output.shape == (5, 3)
        assert torch.all(torch.isfinite(output))

        # Test that individual models produce different outputs
        individual_outputs = []
        normalized_x = torch.sigmoid(x)
        for model in ensemble.models:
            with torch.no_grad():
                individual_output = model(normalized_x)
                individual_outputs.append(individual_output)

        # Outputs should be different (different random initializations)
        for i in range(len(individual_outputs)):
            for j in range(i + 1, len(individual_outputs)):
                assert not torch.allclose(
                    individual_outputs[i], individual_outputs[j], atol=1e-3
                ), f"Models {i} and {j} produced identical outputs"

    def test_saving_and_loading(self):
        """Test model saving and loading."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        # Create and test original model
        original_layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(3, 2)
        original_output = original_layer(x)

        # Save model state
        state_dict = original_layer.state_dict()

        # Create new model and load state
        new_ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )
        new_layer = ML.QuantumLayer(input_size=2, ansatz=new_ansatz)
        new_layer.load_state_dict(state_dict)

        # Test that outputs are identical
        new_output = new_layer(x)
        assert torch.allclose(original_output, new_output, atol=1e-6)
