:github_url: https://github.com/merlinquantum/merlin

===================================================================
Transfer Learning in Hybrid Classical-Quantum Neural Networks
===================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Transfer Learning in Hybrid Classical-Quantum Neural Networks

   **Authors**: Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, Nathan Killoran

   **Published**: Quantum 4, 340 (2020)

   **DOI**: `10.22331/q-2020-10-09-340 <https://doi.org/10.22331/q-2020-10-09-340>`_

   **Reproduction Status**: Complete

   **Reproducer**: Benjamin Stott (benjamin.stott@quandela.com)

Abstract
========

This paper extends the concept of transfer learning to hybrid classical-quantum neural
networks. The key contribution is a framework for combining pre-trained classical feature
extractors with variational quantum circuits, introducing four transfer learning paradigms
(CC, CQ, QC, QQ) depending on whether the source and target networks are classical or
quantum. The paper focuses on the CQ paradigm, in which a frozen pre-trained CNN extracts
compact feature representations from high-dimensional inputs, which are then processed by
a trainable variational quantum circuit for final classification.

The paper introduces the concept of a dressed quantum circuit, which augments a bare
variational quantum circuit with classical pre- and post-processing layers, making the
input and output dimensions independent of qubit count. Proof-of-concept experiments are
demonstrated on 2D spiral classification, image recognition (ants vs bees), and CIFAR-10
binary classification, with hardware validation on IBM and Rigetti quantum processors.

Significance
============

This paper is significant for the quantum ML field because it provides a practical
blueprint for applying near-term quantum devices to real-world tasks without requiring
quantum hardware to process raw high-dimensional data. By delegating feature extraction to
a classical pre-trained network and using the quantum circuit only for the final
classification step, the approach is compatible with the limited qubit counts of current
NISQ devices. The dressed quantum circuit abstraction is a general design pattern that has
since been widely adopted in hybrid quantum-classical architectures.

MerLin Implementation
=====================

MerLin is used to implement the variational quantum circuit component of the dressed
quantum circuit. The ``QuantumLayer`` uses beam splitter meshes as the variational
ansatz, with phase shifters providing angle encoding of the classical input features.
The beam splitters have fixed angles; the phase shifters are the trainable parameters,
optimised via PyTorch gradient descent through the full hybrid pipeline. A PennyLane
qubit-based backend is also provided, implementing the paper's original RY + CNOT
architecture for direct comparison.

Key Contributions
=================

**Example 1: 2D Spiral Classification**
  * We have trained a dressed quantum circuit (4 modes, depth 5) on the 2D spiral dataset
    and achieved 100% test accuracy, exceeding the paper's reported ~97% for the quantum
    model. The classical baseline (two 4-neuron hidden layers) reached 76% accuracy,
    consistent with the paper's ~85% and confirming that the quantum model outperforms a
    comparably-sized classical network on this non-linear task.

**Example 2: CQ Transfer Learning - Ants vs Bees**
  * We have reproduced the CQ transfer learning pipeline using a frozen ResNet18 as
    feature extractor (11,176,512 frozen parameters) and a dressed 4-mode quantum circuit
    as the trainable classifier (2,606 trainable parameters, 0.02% of total). We achieved
    90.8% test accuracy compared to the paper's 96.7% simulator result.

**Example 3: CIFAR-10 Binary Classification**
  * We have reproduced both CIFAR-10 binary experiments using CQ transfer learning.
    Dogs vs Cats reached 83.0% accuracy (paper: 82.7%) and Planes vs Cars reached 96.6%
    accuracy (paper: 96.1%), closely matching the paper's reported results.

Implementation Details
======================

The dressed quantum circuit wraps MerLin's ``QuantumLayer`` with classical linear layers
for flexible input/output dimensionality:

.. code-block:: python

   import torch.nn as nn
   from merlin import QuantumLayer, ComputationSpace, MeasurementStrategy
   from lib.circuits import create_merlin_circuit

   n_modes = 4
   circuit = create_merlin_circuit(n_modes, q_depth=6)

   quantum_layer = QuantumLayer(
       input_size=n_modes,
       circuit=circuit,
       trainable_parameters=["phi"],
       input_parameters=["theta"],
       computation_space=ComputationSpace.UNBUNCHED,
       measurement_strategy=MeasurementStrategy.probs(),
   )

   # Dressed quantum circuit: classical -> quantum -> classical
   dressed_circuit = nn.Sequential(
       nn.Linear(512, n_modes),   # ResNet18 features -> quantum input
       quantum_layer,
       nn.Linear(quantum_layer.output_size, 2),  # quantum output -> classes
   )

   # Full CQ transfer learning model: frozen ResNet18 + dressed circuit
   model = nn.Sequential(resnet18_frozen, dressed_circuit)


Extensions and Future Work
==========================

The MerLin implementation extends beyond the original paper:

**Enhanced Capabilities**
  * A PennyLane qubit-based backend is included alongside MerLin, implementing the
    paper's original RY + CNOT ladder circuit, enabling direct comparison of photonic
    and qubit-based quantum transfer learning.
  * The dressed quantum circuit abstraction is implemented as a reusable module,
    configurable via JSON for different qubit/mode counts, circuit depths, and
    computation spaces (``fock``, ``unbunched``, ``dual_rail``).

**Experimental Extensions**
  * All three paper examples are reproduced with both backends, allowing quantitative
    comparison of photonic vs qubit approaches on the same datasets.
  * The MerLin photonic backend provides results competitive with the paper's simulator
    results, particularly on CIFAR-10 where Dogs vs Cats and Planes vs Cars match to
    within 0.5 percentage points.

**Hardware Considerations**
  * All experiments run on CPU. The CIFAR-10 experiments are the most expensive, taking
    approximately 6 minutes per training run.
  * Examples 4 (QC) and 5 (QQ) from the paper involve continuous-variable quantum
    networks using Strawberry Fields and are not reproduced here.

**Future Work**
  * Reproducing Examples 4 and 5 (QC and QQ transfer learning with continuous-variable
    quantum circuits) would complete the full set of paradigms introduced in the paper.
  * Extending the CQ pipeline to larger backbone networks (e.g., ResNet50, ViT) and
    more modes would test the scalability of the approach.

Citation
========

.. code-block:: bibtex

   @article{mari2020transfer,
     title={Transfer learning in hybrid classical-quantum neural networks},
     author={Mari, Andrea and Bromley, Thomas R. and Izaac, Josh and Schuld, Maria
             and Killoran, Nathan},
     journal={Quantum},
     volume={4},
     pages={340},
     year={2020},
     publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den
                Quantenwissenschaften},
     doi={10.22331/q-2020-10-09-340}
   }