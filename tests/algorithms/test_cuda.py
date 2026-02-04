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
import torch.nn as nn

import merlin as ml

ANSATZ_SKIP = pytest.mark.skip(
    reason="Legacy ansatz-based QuantumLayer API removed; test pending migration."
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_model_on_cuda():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 1, 0, 0],
        trainable_parameters=["phi"],
        device=torch.device("cuda"),
        measurement_strategy=ml.MeasurementStrategy.probs(),
    )
    model = nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1)).to(
        torch.device("cuda")
    )
    assert layer.device == torch.device("cuda")
    assert next(model.parameters()).device.type == "cuda"
    if len(layer.thetas) > 0:
        assert layer.thetas[0].device == torch.device("cuda", index=0)
    assert layer.computation_process.converter.device == torch.device("cuda")
    assert layer.computation_process.simulation_graph.device == torch.device("cuda")
    assert layer.computation_process.converter.list_rct[0][1].device == torch.device(
        "cuda", index=0
    )
    assert layer.computation_process.simulation_graph.vectorized_operations[-1][
        0
    ].device == torch.device("cuda", index=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_switch_model_to_cuda():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 1, 0, 0],
        trainable_parameters=["phi"],
        device=torch.device("cpu"),
        measurement_strategy=ml.MeasurementStrategy.probs(),
    )
    model = nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1)).to(
        torch.device("cpu")
    )
    assert layer.device == torch.device("cpu")
    assert next(model.parameters()).device.type == "cpu"
    layer = layer.to(torch.device("cuda"))
    model = model.to(torch.device("cuda"))
    _ = layer()
    layer.computation_process.input_state = torch.rand(
        (3, 6), device=torch.device("cuda")
    )
    _ = layer()
    assert layer.device == torch.device("cuda")
    if len(layer.thetas) > 0:
        assert layer.thetas[0].device == torch.device("cuda", index=0)
    assert layer.computation_process.converter.device == torch.device("cuda")
    assert layer.computation_process.simulation_graph.device == torch.device("cuda")
    assert layer.computation_process.converter.list_rct[0][1].device == torch.device(
        "cuda", index=0
    )
    assert layer.computation_process.simulation_graph.vectorized_operations[-1][
        0
    ].device == torch.device("cuda", index=0)


class QuantumClassifier_withBuilder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=10,
        modes=10,
        num_classes=2,
        device=torch.device("cpu"),
    ):
        super().__init__()

        # This layer downscales the inputs to fit in the QLayer
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)

        builder = ml.CircuitBuilder(n_modes=modes)
        builder.add_entangling_layer(trainable=True, name="U1")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        available_modes = list(range(modes))
        if not available_modes:
            raise ValueError("modes must be at least 1")
        # we need to add as many input layers as needed to encode all features
        full_chunks, remainder = divmod(hidden_dim, len(available_modes))
        if hidden_dim < len(available_modes):
            builder.add_angle_encoding(modes=available_modes[:hidden_dim], name="input")
        else:
            for _ in range(full_chunks):
                builder.add_angle_encoding(modes=available_modes, name="input")
                builder.add_superpositions(depth=2)
            if remainder:
                builder.add_angle_encoding(
                    modes=available_modes[:remainder], name="input"
                )

        builder.add_rotations(trainable=True, name="theta")
        builder.add_superpositions(depth=1)

        self.q_circuit = ml.QuantumLayer(
            input_size=hidden_dim,
            builder=builder,
            n_photons=modes // 2,
            measurement_strategy=ml.MeasurementStrategy.probs(),
            device=device,
        )

        # Linear output layer as in the original paper
        self.output_layer = nn.Linear(
            self.q_circuit.output_size, num_classes, device=device
        )

    def forward(self, x):
        # Forward pass through the quantum-classical hybrid
        x = self.downscaling_layer(x)
        x = torch.sigmoid(x)  # Normalize for quantum layer
        x = self.q_circuit(x)
        out = self.output_layer(x)
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_QuantumClassifier_withBuilder():
    """Test QuantumClassifier_withBuilder functionality on GPU"""

    device = torch.device("cuda")

    # Test parameters
    batch_size = 4
    input_dim = 768
    hidden_dim = 10
    modes = 10
    num_classes = 2

    # Create the quantum classifier
    model = QuantumClassifier_withBuilder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        modes=modes,
        num_classes=num_classes,
        device=device,
    )

    # Move model to GPU
    model = model.to(device)

    # Create sample input data
    sample_input = torch.randn(batch_size, input_dim, device=device)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input)

    # Test with different batch sizes
    for test_batch_size in [1, 2, 8]:
        test_input = torch.randn(test_batch_size, input_dim, device=device)
        with torch.no_grad():
            model(test_input)

    # Test training mode
    model.train()

    # Create dummy labels
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Test with gradient computation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Single training step
    optimizer.zero_grad()
    output = model(sample_input)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n✅ All tests passed successfully!")

    # Assertions for pytest
    assert output.shape == torch.Size([batch_size, num_classes])
    assert output.device.type == device.type
    assert total_params > 0
    assert trainable_params > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_different_configurations():
    """Test different model configurations"""

    device = torch.device("cuda")

    # Test configurations
    configs = [
        {"modes": 4, "hidden_dim": 50},
        {"modes": 6, "hidden_dim": 100},
    ]

    for _i, config in enumerate(configs):
        model = QuantumClassifier_withBuilder(
            input_dim=768,
            hidden_dim=config["hidden_dim"],
            modes=config["modes"],
            num_classes=2,
            device=device,
        ).to(device)

        # Test with sample input
        sample_input = torch.randn(2, 768, device=device)
        with torch.no_grad():
            output = model(sample_input)

        # Assertions for pytest
        assert output.shape == torch.Size([2, 2])
        assert output.device.type == device.type
