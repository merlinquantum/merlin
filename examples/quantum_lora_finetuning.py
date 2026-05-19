import torch
import torch.nn as nn
from merlin import QuantumLoRALayer, convert_to_quantum_lora
from merlin.measurement.strategies import MeasurementStrategy

def main():
    # 1. Define a simple dummy model with nested modules
    class SubModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        def forward(self, x):
            return self.linear(x)

    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 64)
            self.sub = SubModule(64)
            self.linear2 = nn.Linear(64, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.sub(x)
            return self.linear2(x)

    input_dim = 10
    output_dim = 2
    model = SimpleModel(input_dim, output_dim)

    print("Original Model:")
    print(model)

    # 2. Automatically apply Quantum LoRA to specific layers
    # We target 'linear1' and the nested 'sub.linear'
    print("\nApplying Quantum LoRA auto-injection...")
    convert_to_quantum_lora(
        model, 
        r=4, 
        n_photons=2, 
        target_modules=["linear1", "linear"]
    )

    print("\nModel with Quantum LoRA (Auto-Injected):")
    print(model)

    # 3. Visualize the quantum circuit of one of the layers
    print("\nVisualizing Quantum Circuit for 'linear1':")
    # This will print the circuit description or try to display it
    model.linear1.visualize_circuit()

    # 4. Simulate a training step
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nNumber of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # Dummy data
    x = torch.randn(5, input_dim)
    y = torch.randint(0, output_dim, (5,))

    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)

    # Backward pass
    loss.backward()

    print(f"\nForward pass output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    print("Backward pass successful.")

if __name__ == "__main__":
    main()
