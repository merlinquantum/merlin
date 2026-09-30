:github_url: https://github.com/merlinquantum/merlin

=======================================================================
QTRL: Toward Practical Quantum Reinforcement Learning via Quantum-Train
=======================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: QTRL: Toward Practical Quantum Reinforcement Learning via Quantum-Train

   **Authors**: Chen-Yu Liu, Chu-Hsuan Abraham Lin, Chao-Han Huck Yang, Kuan-Cheng Chen, Min-Hsiu Hsieh

   **Published**: IEEE International Conference on Quantum Computing and Engineering (QCE) (2024)

   **Paper URL**: `arXiv:2407.06103 <https://arxiv.org/abs/2407.06103>`_

   **Reproduction Status**: ✅ Complete

Abstract
========

This reproduction studies the Quantum-Train Reinforcement Learning (QTRL) framework proposed by Liu et al. The QTRL method uses a quantum neural network combined with a classical mapping model to generate the parameters for a classical policy network. This approach overcomes traditional quantum reinforcement learning (QRL) challenges, such as complex data encoding and the requirement of quantum hardware during the inference stage, leading to a highly practical framework that enjoys polylogarithmic parameter reduction while running entirely on classical hardware during inference.

The MerLin reproduction implements the QTRL setup using various backends for the parameter-generation model: a photonic quantum circuit using MerLin, a Matrix Product State (MPS) mapping, a gate-based quantum model using TorchQuantum, and a classical MLP baseline. These models are evaluated on standard Gym environments like CartPole and MiniGrid.

Significance
============

The QTRL approach represents a significant step toward practical quantum reinforcement learning. By using a quantum model strictly as a weight-generator for a classical linear policy, it completely bypasses the bottleneck of encoding classical environment states into quantum states at every step. This decoupling means that inference requires zero quantum operations—a critical requirement for RL tasks needing low-latency, real-time feedback.

MerLin Implementation
=====================

The implementation is structured around two main files: ``runner.py`` for the training loop and ``util.py`` for the hybrid architectures.

Model Architecture (``util.py``)
--------------------------------

The core of the architecture relies on multiple configurable backends that produce the weights for the target classical policy:

* **QLayer**: A photonic quantum layer built with the MerLin framework. It uses a ``CircuitBuilder`` to add a trainable U3 entangling layer on a specified number of modes, followed by lexical grouping (``LexGrouping``) to adjust the output dimension.
* **MappingModel**: A classical Multi-Layer Perceptron (MLP) mapping network that projects the quantum layer's output (or a classical baseline's output) to the exact number of weights required by the RL agent (``state_dim * action_dim``).
* **Hybrid Models**:
   * **HybridMLPModel**: Combines the MerLin ``QLayer`` with the ``MappingModel``.
   * **HybridMPSModel**: Combines the MerLin ``QLayer`` with a Matrix Product State (``MPS``) tensor network instead of a classical MLP.
   * **TorchQuantumModel**: A baseline replacing MerLin's continuous-variable photonic circuit with a gate-based parameterized quantum circuit (using U3 and CU3 gates from TorchQuantum).
   * **classic_model**: A pure classical MLP replacing the quantum component entirely, serving as a classical baseline.

Training Loop (``runner.py``)
-----------------------------

The RL training loop utilizes a REINFORCE-style policy gradient algorithm to optimize the hybrid parameter-generation network:

1. **Environment Setup**: Supports standard Gym environments (e.g., ``CartPole-v1``) and image-flattened gridworlds (``MiniGrid-Empty-5x5-v0`` via a custom ``MinigridImageOnlyWrapper``).
2. **Weight Generation**: For each episode, a forward pass of the selected hybrid model (e.g., ``HybridMLPModel``) generates a flattened 1D tensor of weights.
3. **Policy Execution**: The ``rl_agent_forward`` function reshapes these generated weights into a matrix (``output_dim x input_dim``) and applies a linear transformation to the current environment state to produce action logits.
4. **Optimization**: The agent samples actions from a categorical distribution, collects rewards, and computes discounted returns. The REINFORCE policy loss is then backpropagated entirely through the weight-generation network (hybrid or classical) to update its parameters.

Key Contributions Reproduced
============================

**Photonic QTRL implementation**
  * Extended the original QTRL concept (which typically relies on gate-model qubits) to continuous-variable photonic circuits using the MerLin framework.
  * Demonstrated how a photonic network (``QLayer``) can effectively generate parameters for a discrete action-space linear policy.

**Unified backend comparison**
  * Built a flexible training pipeline supporting multiple representation backends: MerLin (Photonic + MLP), MerLin (Photonic + MPS), TorchQuantum (Gate-model), and Classical MLP.
  * Enabled direct comparison of hybrid architectures against classical and alternative quantum baselines under identical RL training conditions.

Citation
========

.. code-block:: bibtex

   @inproceedings{liu2024qtrl,
     title={QTRL: Toward Practical Quantum Reinforcement Learning via Quantum-Train},
     author={Liu, Chen-Yu and Lin, Chu-Hsuan Abraham and Yang, Chao-Han Huck and Chen, Kuan-Cheng and Hsieh, Min-Hsiu},
     booktitle={2024 IEEE International Conference on Quantum Computing and Engineering (QCE)},
     year={2024},
     organization={IEEE}
   }