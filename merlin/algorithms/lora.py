from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from .layer import QuantumLayer
from ..measurement.strategies import MeasurementStrategy


from dataclasses import dataclass


@dataclass
class QuantumAnsatz:
    """Predefined quantum ansätze for LoRA."""
    SIMPLE = "simple"  # Entangling -> Encoding -> Entangling
    UNIVERSAL = "universal"  # More expressive universal-style circuit
    HARDWARE_EFFICIENT = "hardware_efficient" # Repeated blocks of entangling and rotations


class QuantumLoRALayer(nn.Module):
    """
    Quantum-Enhanced Low-Rank Adaptation (LoRA) layer.

    This layer wraps a standard torch.nn.Linear layer and adds a parallel
    branch consisting of:
      Input -> [Down-projection (Linear)] -> [QuantumLayer] -> [Up-projection (Linear)] -> Output

    The output is calculated as:
      h = W_0 x + (alpha / r) * (W_up * Q(W_down * x))
    where Q is the quantum layer.

    Parameters
    ----------
    original_layer : nn.Linear
        The pre-trained linear layer to be adapted.
    r : int
        The rank of the adaptation (dimension of the quantum layer input/output).
    n_photons : int
        Number of photons to use in the quantum layer.
    alpha : float, default: 1.0
        Scaling factor for the LoRA update.
    ansatz : str | QuantumAnsatz, default: "simple"
        The type of quantum circuit architecture to use.
    quantum_layer_params : dict[str, Any] | None, optional
        Additional parameters to pass to the QuantumLayer constructor.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int,
        n_photons: int,
        alpha: float = 1.0,
        ansatz: str | QuantumAnsatz = "simple",
        quantum_layer_params: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.ansatz = ansatz

        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Down-projection: input_dim -> r
        self.lora_down = nn.Linear(original_layer.in_features, r, bias=False)

        # Quantum Layer: r -> r
        q_params = quantum_layer_params or {}
        if "input_size" not in q_params:
            q_params["input_size"] = r
        if "n_photons" not in q_params:
            q_params["n_photons"] = n_photons
        if "measurement_strategy" not in q_params:
            q_params["measurement_strategy"] = MeasurementStrategy.probs()

        # Check if user provided a builder or circuit
        if "builder" in q_params or "circuit" in q_params:
            self.quantum_layer = QuantumLayer(**q_params)
        else:
            # Default construction using CircuitBuilder
            from ..builder import CircuitBuilder
            
            input_size = q_params.get("input_size", r)
            q_params.pop("input_size", None)
            
            builder = CircuitBuilder(n_modes=input_size)
            
            if ansatz == "simple":
                # Simple ansatz: Entangling -> Encoding -> Entangling
                builder.add_entangling_layer(trainable=True, name="LI_lora")
                builder.add_angle_encoding(name="input", subset_combinations=False)
                builder.add_entangling_layer(trainable=True, name="RI_lora")
            elif ansatz == "universal":
                # Deeper expressive ansatz
                builder.add_entangling_layer(trainable=True, name="L1_lora")
                builder.add_angle_encoding(name="input", subset_combinations=False)
                builder.add_entangling_layer(trainable=True, name="L2_lora")
                builder.add_entangling_layer(trainable=True, name="L3_lora")
                builder.add_entangling_layer(trainable=True, name="L4_lora")
            elif ansatz == "hardware_efficient":
                # Hardware efficient: One encoding, multiple entangling blocks
                builder.add_angle_encoding(name="input", subset_combinations=False)
                for i in range(3):
                    builder.add_entangling_layer(trainable=True, name=f"H{i}_ent")

            # Setup input state if not provided
            if "input_state" not in q_params:
                current_n_photons = q_params.get("n_photons", n_photons) or (input_size // 2)
                q_params["n_photons"] = current_n_photons

            self.quantum_layer = QuantumLayer(
                input_size=input_size,
                builder=builder,
                **q_params
            )

        # Up-projection: r -> output_dim
        q_out_size = self.quantum_layer.output_size
        self.lora_up = nn.Linear(q_out_size, original_layer.out_features, bias=False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def visualize_circuit(self):
        """
        Visualize the quantum circuit inside the LoRA layer.
        """
        try:
            import perceval as pcvl
            return pcvl.pdisplay(self.quantum_layer.circuit)
        except ImportError:
            print("Perceval not installed, cannot visualize circuit.")
        except Exception as e:
            print(f"Visualization failed: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original branch
        original_out = self.original_layer(x)

        # Quantum LoRA branch
        # 1. Down-project
        h_down = self.lora_down(x)

        # 2. Quantum Layer
        h_q = self.quantum_layer(h_down)

        # 3. Up-project
        h_up = self.lora_up(h_q)

        # Combine
        return original_out + h_up * self.scaling


import re

def convert_to_quantum_lora(
    model: nn.Module,
    r: int,
    n_photons: int,
    alpha: float = 1.0,
    target_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    ansatz: str | QuantumAnsatz = "simple",
    **kwargs: Any,
) -> nn.Module:
    """
    Utility function to automatically replace nn.Linear layers with QuantumLoRALayer.

    Parameters
    ----------
    model : nn.Module
        The model to modify.
    r : int
        Rank of LoRA.
    n_photons : int
        Number of photons for quantum layers.
    alpha : float, default: 1.0
        LoRA scaling factor.
    target_modules : list[str] | None, optional
        List of module names (or regex patterns) to replace. If None, replaces all nn.Linear layers.
    exclude_modules : list[str] | None, optional
        List of module names (or regex patterns) to explicitly exclude from replacement.
    ansatz : str | QuantumAnsatz, default: "simple"
        Quantum ansatz to use.
    kwargs : Any
        Additional parameters for QuantumLoRALayer.

    Returns
    -------
    nn.Module
        The modified model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Check if this linear layer should be replaced
            should_replace = False
            
            # Check for exclusion first
            if exclude_modules is not None:
                is_excluded = False
                for pattern in exclude_modules:
                    if re.search(pattern, name):
                        is_excluded = True
                        break
                if is_excluded:
                    should_replace = False
                elif target_modules is None:
                    should_replace = True
            
            if not should_replace:
                if target_modules is None:
                    should_replace = True
                else:
                    # Use regex matching for target modules
                    for pattern in target_modules:
                        if re.search(pattern, name):
                            should_replace = True
                            break
            
            if should_replace:
                new_layer = QuantumLoRALayer(
                    original_layer=module,
                    r=r,
                    n_photons=n_photons,
                    alpha=alpha,
                    ansatz=ansatz,
                    quantum_layer_params=kwargs
                )
                setattr(model, name, new_layer)
        else:
            # Recursive call for nested modules
            convert_to_quantum_lora(
                module,
                r=r,
                n_photons=n_photons,
                alpha=alpha,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
                ansatz=ansatz,
                **kwargs
            )
    return model
