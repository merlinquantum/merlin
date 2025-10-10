:github_url: https://github.com/merlinquantum/merlin

=========================
Your First Quantum Layers
=========================
Merlin provides three approaches to create quantum neural networks, from simple to advanced:

1. `Simple API <#method-1-simple-api-easiest>`_ : Quick start with minimal configuration
2. `Factory Method <#method-2-factory-method-intermediate>`_ : More control with pre-built architectures
3. `Direct Circuit Definition <#method-3-direct-circuit-definition-advanced>`_ : Full control over quantum circuit design

We'll start with the easiest approach and progress to more advanced methods.

Method 1: Simple API (Easiest)
------------------------------

The simplest way to create a quantum layer is using the ``.simple()`` method. This is perfect for beginners who want to quickly integrate quantum processing into their neural networks.

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.nn as nn
    import merlin as ML

    class HybridIrisClassifier(nn.Module):
        """
        Hybrid model for Iris classification:
        - Classical layer reduces 4 features to 3
        - Quantum layer processes the 3 features
        - Classical output layer for 3-class classification
        """
        def __init__(self):
            super(HybridIrisClassifier, self).__init__()

            # Classical preprocessing layer
            self.classical_in = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 3),
                nn.Tanh()  # Normalize to [-1, 1] for quantum layer
            )

            # Quantum layer
            # Just specify input size and approximate parameter count
            self.quantum = ML.QuantumLayer.simple(
                input_size=3,      # 3 input features
                n_params=100,      # ~100 quantum parameters
                shots=10000        # Number of measurements
            )

            # Classical output layer
            self.classical_out = nn.Sequential(
                nn.Linear(self.quantum.output_size, 8),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(8, 3)  # 3 classes
            )

            print(f"\nModel Architecture:")
            print(f"  Input: 4 features")
            print(f"  Classical preprocessing: 4 → 3")
            print(f"  Quantum layer: 3 → {self.quantum.output_size}")
            print(f"  Classical output: {self.quantum.output_size} → 3 classes")

        def forward(self, x):
            # Preprocess with classical layer
            x = self.classical_in(x)

            # Process through quantum layer
            # Note: quantum layer expects inputs in [0, 1], so we shift tanh output
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
            x = self.quantum(x)

            # Final classification
            x = self.classical_out(x)
            return x

Training the Simple Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create model
    model = HybridIrisClassifier()

    # Standard PyTorch training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop (standard PyTorch)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()



Method 2: Factory Method (Intermediate)
---------------------------------------

For more control over the quantum architecture while still using pre-built circuits, use the factory method approach.

Creating Quantum Layers with Factories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   ``Ansatz`` helpers now live in :mod:`merlin.builder.ansatz` (formerly
   ``merlin.core.ansatz``). The top-level ``merlin`` namespace continues to
   re-export :class:`~merlin.builder.ansatz.Ansatz` and
   :class:`~merlin.builder.ansatz.AnsatzFactory` for convenience.

.. code-block:: python

    from merlin import PhotonicBackend, CircuitType, StatePattern, AnsatzFactory, QuantumLayer
    from merlin import OutputMappingStrategy

    # Define the quantum experiment configuration
    n_modes = 6
    n_photons = 3
    input_size = 4

    # Create experiment with Series circuit
    photonicbackend = PhotonicBackend(
        circuit_type=CircuitType.SERIES,
        n_modes=n_modes,
        n_photons=n_photons,
        reservoir_mode=True,         # Non-trainable quantum layer
        use_bandwidth_tuning=False,  # No bandwidth tuning
        state_pattern=StatePattern.PERIODIC
    )

    # Create ansatz with automatic output size
    ansatz = AnsatzFactory.create(
        PhotonicBackend=photonicbackend,
        input_size=input_size,
        # output_size not specified - will be calculated automatically unless specified
        output_mapping_strategy=OutputMappingStrategy.NONE
    )

    # Create quantum layer
    quantum_layer = QuantumLayer(
        input_size=input_size,
        ansatz=ansatz,
        shots=10000,           # Number of measurement shots
        no_bunching=False
    )

Available Circuit Types
^^^^^^^^^^^^^^^^^^^^^^^

merlin provides several pre-built circuit architectures:

- ``CircuitType.SERIES``: Gan et al paper circuit design implementation
- Other circuit types available in the library

State Patterns
^^^^^^^^^^^^^^

Control how photons are distributed:

- ``StatePattern.PERIODIC``: 1 photon every 2 modes i.e 101010
- ``StatePattern.SEQUENTIAL``: photons injected into the first modes seen i.e 111000

Using Factory-Created Layers in Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class FactoryHybridModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Classical preprocessing
            self.preprocess = nn.Linear(10, 4)

            # Quantum layer from factory
            photonicbackend = PhotonicBackend(
                circuit_type=CircuitType.SERIES,
                n_modes=8,
                n_photons=4,
                state_pattern=StatePattern.PERIODIC
            )

            ansatz = AnsatzFactory.create(
                PhotonicBackend=photonicbackend,
                input_size=4,
                output_size = 10,
                output_mapping_strategy=OutputMappingStrategy.LINEAR
            )

            self.quantum = QuantumLayer(
                input_size=4,
                ansatz=ansatz,
                shots=5000
            )

            # Classical output
            self.output = nn.Linear(self.quantum.output_size, 3)

        def forward(self, x):
            x = torch.relu(self.preprocess(x))
            x = torch.sigmoid(x)  # Normalize to [0, 1]
            x = self.quantum(x)
            return self.output(x)



Method 3: Direct Circuit Definition (Advanced)
----------------------------------------------

For complete control over the quantum circuit, define it directly using Perceval.

Circuit Definition
^^^^^^^^^^^^^^^^^^

The quantum circuit consists of three main parts:

.. code-block:: python

    import perceval as pcvl

    def create_quantum_circuit(m):
        # 1. Left interferometer - trainable transformation
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                     pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

        # 2. Input encoding - maps classical data to quantum parameters
        c_var = pcvl.Circuit(m)
        for i in range(4):  # 4 input features
            px = pcvl.P(f"px{i + 1}")
            c_var.add(i + (m - 4) // 2, pcvl.PS(px))

        # 3. Right interferometer - trainable transformation
        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
                     pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

        # Combine all components
        return wl // c_var // wr

**Key Components**:

- ``pcvl.BS()``: Beam splitter for quantum interference
- ``pcvl.PS(pcvl.P("name"))``: Phase shifter with trainable parameter
- ``pcvl.GenericInterferometer``: Creates complex interference patterns
- ``pcvl.Circuit``: Container for quantum components

Create Quantum Layer from Circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create quantum layer
    m = 6  # 6 optical modes
    circuit = create_quantum_circuit(m)

    quantum_layer = ML.QuantumLayer(
        input_size=4,                                              # 4 input features
        output_size=3,                                             # 3 output classes
        circuit=circuit,                                           # Quantum circuit
        trainable_parameters=["theta"],                            # Parameters to train
        input_parameters=["px"],                                   # Input encoding parameters
        input_state=[1, 0, 1, 0, 1, 0],                           # Initial photon state
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR    # Output mapping
    )

    # Test the layer
    x = torch.rand(10, 4)  # Batch of 10 samples, 4 features each
    output = quantum_layer(x)
    print(f"Input shape: {x.shape}")      # [10, 4]
    print(f"Output shape: {output.shape}")  # [10, 3]

Understanding Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    quantum_layer = ML.QuantumLayer(
        input_size=4,                                              # Classical input features
        output_size=3,                                             # Desired output size
        circuit=circuit,                                           # Quantum circuit
        trainable_parameters=["theta"],                            # Which parameters to train
        input_parameters=["px"],                                   # Input encoding parameters
        input_state=[1, 0] * (m // 2) + [0] * (m % 2),           # Initial photon distribution
        no_bunching=False,                                         # Allow photon bunching
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR    # How to map quantum output
    )

**Parameter Explanation**:

- ``trainable_parameters``: Parameters updated during backpropagation
- ``input_parameters``: Parameters that encode classical input data
- ``input_state``: Initial photon configuration (e.g., [1,0,1,0,0,0] = photons in modes 0,2)
- ``no_bunching``: Whether multiple photons can occupy the same mode
- ``output_mapping_strategy``: How quantum probabilities become classical outputs

Complete Hybrid Network Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class AdvancedHybridClassifier(nn.Module):
        def __init__(self, input_dim=8, n_classes=3, n_modes=6):
            super().__init__()

            # Classical preprocessing
            self.classical_input = nn.Sequential(
                nn.Linear(input_dim, 6),
                nn.ReLU(),
                nn.Linear(6, 4)  # Reduce to quantum layer input size
            )

            # Create quantum circuit
            circuit = create_quantum_circuit(n_modes)

            # Quantum processing layer
            self.quantum_layer = ML.QuantumLayer(
                input_size=4,
                output_size=6,
                circuit=circuit,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 0] * (n_modes // 2) + [0] * (n_modes % 2),
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
            )

            # Classical output layer
            self.classifier = nn.Sequential(
                nn.Linear(6, n_classes),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            # Classical preprocessing
            x = self.classical_input(x)

            # Normalize for quantum layer (required: inputs must be in [0,1])
            x = torch.sigmoid(x)

            # Quantum transformation
            x = self.quantum_layer(x)

            # Classical output
            return self.classifier(x)

    # Create and test model
    model = AdvancedHybridClassifier(input_dim=8, n_classes=3, n_modes=6)
    x = torch.rand(16, 8)  # Batch of 16 samples
    output = model(x)
    print(f"Model output shape: {output.shape}")  # [16, 3]

Training the Advanced Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train your hybrid model with standard PyTorch workflows:

.. code-block:: python

    import torch.optim as optim
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=8, n_classes=3,
        n_informative=6, random_state=42
    )

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Setup training
    model = AdvancedHybridClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(50):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                test_acc = (test_outputs.argmax(1) == y_test).float().mean()

            print(f"Epoch {epoch}: Loss={loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
            model.train()
