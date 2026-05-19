# Quantum-Enhanced Low-Rank Adaptation (Quantum LoRA)

Quantum LoRA brings the expressive power of photonic quantum neural networks to the Low-Rank Adaptation fine-tuning paradigm. By replacing classical low-rank matrices with parameterized quantum circuits, we can potentially capture complex features with extreme parameter efficiency.

## How it Works

The standard LoRA update $\Delta W = A \times B$ is replaced by a quantum-enhanced path:

$$\Delta W = A_{down} \rightarrow \text{QuantumCircuit}(\theta) \rightarrow B_{up}$$

- **Down-Projection**: Projects the high-dimensional input into a low-dimensional rank $r$.
- **Quantum Circuit (Ansatz)**: Operates on $r$ modes using $n$ photons. The circuit performs rotations and entangling operations designed to execute high-dimensional transformations in Hilbert space.
- **Up-Projection**: Projects the quantum measurement results back to the original output dimension.

## Advanced Usage

### Auto-Injection

You don't need to manually modify your model architecture. Use `convert_to_quantum_lora` to adapt any PyTorch model:

```python
import merlin
import torch.nn as nn

model = nn.Transformer(...)

# Convert all Attention projection layers
merlin.convert_to_quantum_lora(
    model,
    r=4,
    n_photons=2,
    target_modules=[r".*attn\..*_proj"], # Regex supported!
    exclude_modules=["output_proj"],    # Exclusions too!
    ansatz="universal"
)
```

### Predefined Ansätze

Choose the architecture that fits your task:

1. **`simple`**: Fast, lightweight, suitable for small adaptations.
2. **`universal`**: A deeper circuit designed for maximum expressivity.
3. **`hardware_efficient`**: Tailored for execution on physical photonic processors.

## Visualization

Inspect your quantum circuit using the built-in visualization tool:

```python
model.attention.q_proj.visualize_circuit()
```

## Tips for Efficiency

- **Rank vs Photons**: For most LLM tasks, a small rank ($r=4$ or $r=8$) with 1-2 photons provides a good balance of speed and power.
- **Initialization**: Quantum LoRA initializes the up-projection to zero by default, ensuring the model starts identical to the pre-trained version.
