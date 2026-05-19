import torch
import torch.nn as nn
import pytest
from merlin import QuantumLoRALayer

def test_quantum_lora_initialization():
    linear = nn.Linear(10, 20)
    r = 4
    n_photons = 2
    lora = QuantumLoRALayer(linear, r, n_photons)
    
    # Check structure
    assert isinstance(lora.lora_down, nn.Linear)
    assert isinstance(lora.lora_up, nn.Linear)
    assert lora.lora_down.in_features == 10
    assert lora.lora_down.out_features == r
    assert lora.lora_up.out_features == 20
    
    # Check freezing
    for param in lora.original_layer.parameters():
        assert not param.requires_grad
        
    # Check trainable
    assert lora.lora_down.weight.requires_grad
    assert lora.lora_up.weight.requires_grad

def test_quantum_lora_forward_pass():
    input_dim = 6
    output_dim = 8
    linear = nn.Linear(input_dim, output_dim)
    r = 4
    n_photons = 2
    
    lora = QuantumLoRALayer(linear, r, n_photons)
    
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    
    output = lora(x)
    assert output.shape == (batch_size, output_dim)

def test_quantum_lora_gradient_flow():
    input_dim = 6
    output_dim = 8
    linear = nn.Linear(input_dim, output_dim)
    r = 4
    n_photons = 2
    
    lora = QuantumLoRALayer(linear, r, n_photons)
    
    x = torch.randn(1, input_dim)
    y_target = torch.randn(1, output_dim)
    
    output = lora(x)
    loss = nn.MSELoss()(output, y_target)
    loss.backward()
    
    # Check gradients exist for LoRA paths
    assert lora.lora_down.weight.grad is not None
    assert lora.lora_up.weight.grad is not None
    
    # Check no gradients for original layer
    assert linear.weight.grad is None

def test_quantum_lora_output_differs():
    # Verify that the quantum branch actually contributes
    input_dim = 6
    output_dim = 8
    linear = nn.Linear(input_dim, output_dim)
    r = 4
    n_photons = 2
    
    lora = QuantumLoRALayer(linear, r, n_photons)
    
    # Initialize up projection to non-zero to ensure contribution
    # (Default init for up projection is zero)
    nn.init.constant_(lora.lora_up.weight, 0.1)
    
    x = torch.randn(1, input_dim)
    
    with torch.no_grad():
        original_out = linear(x)
        lora_out = lora(x)
        
    assert not torch.allclose(original_out, lora_out)


def test_quantum_lora_ansatz_types():
    linear = nn.Linear(10, 20)
    for ansatz_type in ["simple", "universal", "hardware_efficient"]:
        lora_layer = QuantumLoRALayer(linear, r=4, n_photons=2, ansatz=ansatz_type)
        assert lora_layer.ansatz == ansatz_type
        # Verify forward pass doesn't crash
        x = torch.randn(2, 10)
        out = lora_layer(x)
        assert out.shape == (2, 20)


def test_convert_to_quantum_lora():
    class Sub(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(5, 5)
    
        def forward(self, x):
            return self.l(x)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 5)
            self.sub = Sub()
            self.l2 = nn.Linear(5, 2)
            
        def forward(self, x):
            return self.l2(self.sub(self.l1(x)))
            
    model = Model()
    # Replace only l1 and the nested sub.l
    from merlin import convert_to_quantum_lora
    convert_to_quantum_lora(model, r=2, n_photons=1, target_modules=[r"^l1$", r"^l$"])
    
    assert isinstance(model.l1, QuantumLoRALayer)
    assert isinstance(model.sub.l, QuantumLoRALayer)
    assert not isinstance(model.l2, QuantumLoRALayer) # Should remain nn.Linear


def test_visualize_circuit():
    linear = nn.Linear(4, 4)
    lora_layer = QuantumLoRALayer(linear, r=2, n_photons=1)
    # This should not raise an exception
    lora_layer.visualize_circuit()
