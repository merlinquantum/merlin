import torch
import torch.nn as nn
import time
from merlin import QuantumLoRALayer, convert_to_quantum_lora, QuantumAnsatz

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_efficiency():
    print("=== Quantum LoRA Efficiency Benchmark ===\n")
    
    input_dim = 128
    hidden_dim = 512
    output_dim = 128
    rank = 8
    
    # Base model
    model_orig = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    print(f"Original trainable params: {count_parameters(model_orig):,}")
    
    # 1. Standard LoRA approximation (Calculated)
    # Down: input*rank, Up: rank*output
    std_lora_params = (input_dim * rank) + (rank * hidden_dim) + (hidden_dim * rank) + (rank * output_dim)
    print(f"Standard LoRA added params (rank {rank}): ~{std_lora_params:,}")
    
    # 2. Quantum LoRA
    model_q = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    # Freeze base
    for p in model_q.parameters():
        p.requires_grad = False
        
    convert_to_quantum_lora(model_q, r=rank, n_photons=rank//2, ansatz=QuantumAnsatz.SIMPLE)
    
    q_lora_params = count_parameters(model_q)
    print(f"Quantum LoRA added params (rank {rank}): {q_lora_params:,}")
    
    reduction = (1 - q_lora_params/std_lora_params) * 100
    print(f"\nParameter reduction vs Standard LoRA: {reduction:.2f}%")
    
    # Memory/Speed dummy test
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    print("\nRunning dummy forward pass...")
    start = time.time()
    for _ in range(10):
        _ = model_q(x)
    end = time.time()
    print(f"Avg forward pass time: {(end-start)/10*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_efficiency()
