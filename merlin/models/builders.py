"""Convenient builders for common quantum models."""

from typing import Optional, List, Dict, Any, Union
import torch
import torch
import torch.nn as nn  # FIXED: Added this import
from .quantum_model import QuantumModel, QuantumConfig
from ..builder import CircuitBuilder
from ..encoding import EncodingStrategy


class ModelBuilder:
    """Utility class for building quantum models."""

    @staticmethod
    def create_variational_classifier(n_features: int,
                                     n_classes: int,
                                     n_qubits: Optional[int] = None,
                                     n_layers: int = 3,
                                     encoding: str = "fourier",
                                     backend: str = "auto") -> QuantumModel:
        """
        Create a variational quantum classifier.

        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            n_qubits: Number of qubits (default: n_features)
            n_layers: Number of variational layers
            encoding: Encoding type
            backend: Backend type

        Returns:
            Configured QuantumModel for classification
        """
        if n_qubits is None:
            n_qubits = max(n_features, 4)

        config = QuantumConfig(
            n_modes=n_qubits,
            n_features=n_features,
            output_size=n_classes,
            encoding_type=encoding,
            backend_type=backend
        )

        return QuantumModel.from_template("variational", config=config)

    @staticmethod
    def create_quantum_kernel(n_features: int,
                            n_qubits: Optional[int] = None,
                            feature_map: str = "iqp",
                            backend: str = "auto") -> QuantumModel:
        """
        Create a quantum kernel model.

        Args:
            n_features: Number of input features
            n_qubits: Number of qubits
            feature_map: Type of feature map
            backend: Backend type

        Returns:
            Quantum kernel model
        """
        if n_qubits is None:
            n_qubits = n_features

        config = QuantumConfig(
            n_modes=n_qubits,
            n_features=n_features,
            backend_type=backend
        )

        if feature_map == "iqp":
            return QuantumModel.from_template("iqp", config=config)
        else:
            from ..encoding import QuantumFeatureMap
            fm = QuantumFeatureMap(n_qubits, n_features, feature_map)
            circuit = fm.build_feature_map()
            return QuantumModel(circuit, config)

    @staticmethod
    def create_photonic_neural_network(n_modes: int,
                                      n_photons: int,
                                      n_features: int,
                                      n_layers: int = 3,
                                      reservoir: bool = False) -> QuantumModel:
        """
        Create a photonic neural network.

        Args:
            n_modes: Number of optical modes
            n_photons: Number of photons
            n_features: Number of input features
            n_layers: Number of network layers
            reservoir: Use reservoir computing

        Returns:
            Photonic neural network model
        """
        config = QuantumConfig(
            n_modes=n_modes,
            n_photons=n_photons,
            n_features=n_features,
            backend_type="photonic",
            reservoir_mode=reservoir,
            no_bunching=True
        )

        # Build circuit
        def build_pnn(builder: CircuitBuilder) -> CircuitBuilder:
            # Input encoding
            for i in range(n_features):
                mode = i % n_modes
                builder.circuit.rotation(mode, f'input_{i}', 'z')

            # Network layers
            for layer in range(n_layers):
                # Entangling
                builder.add_entangling_layer(pattern='all_to_all')

                # Processing
                if not reservoir:
                    for mode in range(n_modes):
                        builder.circuit.rotation(
                            mode,
                            f'layer{layer}_mode{mode}',
                            'z'
                        )

            return builder

        return QuantumModel.from_builder(build_pnn, n_modes=n_modes, n_photons=n_photons, **{k:v for k,v in config.__dict__.items() if k not in ['n_modes', 'n_photons', 'n_elements'] and not k.startswith('_')})

    @staticmethod
    def create_qcnn(n_features: int,
                   n_classes: int,
                   n_conv_layers: int = 2,
                   n_pool_layers: int = 1,
                   backend: str = "auto") -> QuantumModel:
        """
        Create a Quantum Convolutional Neural Network.

        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            n_conv_layers: Number of convolutional layers
            n_pool_layers: Number of pooling layers
            backend: Backend type

        Returns:
            QCNN model
        """
        n_qubits = 2 ** ((n_features - 1).bit_length())  # Next power of 2

        config = QuantumConfig(
            n_modes=n_qubits,
            n_features=n_features,
            output_size=n_classes,
            backend_type=backend
        )

        return QuantumModel.from_template(
            "qcnn",
            n_modes=n_qubits,
            n_features=n_features,
            config=config
        )

    @staticmethod
    def create_hybrid_model(classical_layers: List[int],
                          quantum_layers: int = 2,
                          n_qubits: int = 4,
                          encoding: str = "amplitude") -> nn.Module:
        """
        Create a hybrid classical-quantum model.

        Args:
            classical_layers: Sizes of classical layers
            quantum_layers: Number of quantum layers
            n_qubits: Number of qubits
            encoding: Encoding type

        Returns:
            Hybrid model
        """
        import torch.nn as nn

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()

                # Classical preprocessing
                layers = []
                for i in range(len(classical_layers) - 1):
                    layers.append(nn.Linear(classical_layers[i], classical_layers[i+1]))
                    layers.append(nn.ReLU())
                self.classical = nn.Sequential(*layers)

                # Quantum processing
                config = QuantumConfig(
                    n_modes=n_qubits,
                    n_features=classical_layers[-1],
                    encoding_type=encoding
                )
                self.quantum = QuantumModel.from_template("variational", config)

                # Classical postprocessing
                self.output = nn.Linear(2**n_qubits, classical_layers[0])

            def forward(self, x):
                x = self.classical(x)
                x = self.quantum(x)
                x = self.output(x)
                return x

        return HybridModel()


class PretrainedModels:
    """Access to pre-configured model architectures."""

    @staticmethod
    def get_model(name: str, **kwargs) -> QuantumModel:
        """
        Get a pre-configured model.

        Available models:
        - 'mnist_classifier': 4-qubit MNIST classifier
        - 'iris_classifier': 4-qubit Iris dataset classifier
        - 'quantum_gan': Quantum GAN generator
        - 'vqe_ansatz': VQE ansatz for quantum chemistry
        """

        configs = {
            'mnist_classifier': {
                'template': 'variational',
                'n_modes': 4,
                'n_features': 16,  # Reduced MNIST
                'output_size': 10,
                'encoding_type': 'amplitude'
            },
            'iris_classifier': {
                'template': 'variational',
                'n_modes': 4,
                'n_features': 4,
                'output_size': 3,
                'encoding_type': 'fourier'
            },
            'quantum_gan': {
                'template': 'variational',
                'n_modes': 6,
                'n_features': 4,
                'encoding_type': 'reupload'
            },
            'vqe_ansatz': {
                'template': 'variational',
                'n_modes': 4,
                'n_features': 0,  # No encoding
                'ansatz_layers': 6
            }
        }

        if name not in configs:
            raise ValueError(f"Unknown model: {name}")

        model_config = configs[name]
        model_config.update(kwargs)

        template = model_config.pop('template')
        config = QuantumConfig(**model_config)

        return QuantumModel(template, config)
