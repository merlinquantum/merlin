Quantum Adversarial Machine Learning
=====================================

.. admonition:: Paper Information

   **Title:** Quantum Adversarial Machine Learning

   **Authors:** Sirui Lu, Lu-Ming Duan, Dong-Ling Deng

   **Published:** Physical Review Research, 2, 033212 (2020)

   **DOI:** `10.1103/PhysRevResearch.2.033212 <https://doi.org/10.1103/PhysRevResearch.2.033212>`_

   **Reproduction Status:** Complete

   **Reproducer:** [Benjamin Stott] ([benjamin.stott@quandela.com])

Abstract
--------

This paper investigates whether quantum machine learning models are vulnerable to adversarial
attacks - imperceptible perturbations that cause misclassification - in a way analogous to
classical neural networks. Lu et al. demonstrate that hybrid classical-quantum classifiers
are susceptible to standard gradient-based attacks (FGSM, BIM, PGD, MIM) regardless of
whether the input is classical image data or quantum ground-state data.

A key finding is that adversarial perturbations are qualitatively different from random noise:
at the same perturbation magnitude, adversarial attacks can reduce accuracy to near zero while
random noise leaves accuracy almost unchanged. The paper also shows that adversarial examples
crafted against classical surrogates transfer to quantum classifiers, and that adversarial
training can substantially recover robustness.

Significance
------------

This paper is one of the first systematic studies of adversarial robustness in quantum machine
learning. It establishes that adversarial vulnerability is not a property of the classical
computing paradigm - it carries over to quantum classifiers. This has direct implications for
any near-term application of quantum ML in security-sensitive domains. The work also introduces
functional attacks, where perturbations are applied as local unitary operations, which is the
natural attack model for quantum-native inputs such as ground states of physical systems.

MerLin Implementation
---------------------

MerLin is used to construct photonic quantum classifiers based on beam splitter meshes and
phase shifters. The ``QuantumLayer`` performs amplitude encoding: classical input vectors are
mapped to Fock basis state amplitudes, so a 256-pixel MNIST image is encoded into a
superposition over the 286 Fock states of a 13-mode, 3-photon circuit. The beam splitters
have fixed angles; the trainable parameters are the phase shifters, which are optimised via
standard PyTorch gradient descent. Adversarial gradients are back-propagated through the
quantum layer in the same way.

Key Contributions
-----------------

Demonstration that quantum classifiers are vulnerable to adversarial attacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have trained a photonic quantum classifier on binary MNIST (digits 1 vs 9) achieving 99.6%
clean accuracy. We then showed that a BIM attack at perturbation magnitude ε = 0.1 reduces
accuracy from 99.6% to 0.0%, faithfully reproducing the paper's central finding.

We have also reproduced the key contrast between adversarial and random noise: at ε = 0.1,
adversarial accuracy collapses to 0.0% while uniform random noise of the same magnitude
leaves accuracy at 99.2%, and photon loss noise reduces it by less than 0.3 percentage
points.

Adversarial attack suite
~~~~~~~~~~~~~~~~~~~~~~~~

We have implemented FGSM, BIM, PGD, and MIM attacks as differentiable PyTorch operations
compatible with the MerLin photonic backend. We have also implemented functional attacks using
local phase shifter perturbations, the photonic-native analogue of the unitary perturbations
described in the paper for quantum-native inputs.

Adversarial training defense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have reproduced the adversarial training defense using a smaller model (8 modes, 2
photons). After adversarial training with BIM at epsilon = 0.1, clean accuracy is maintained
at 100% and adversarial accuracy recovers to 42%, compared to 0% for the undefended model.
Adversarial training with the larger model (13 modes, 3 photons) is more sensitive to
hyperparameter choices and requires more epochs to converge under adversarial training
pressure.

Implementation Details
----------------------

The photonic quantum classifier uses amplitude encoding where classical data is normalised
to unit L2 norm and mapped directly to Fock basis state amplitudes. For MNIST, 13 modes and
3 photons give C(13,3) = 286 Fock states, sufficient to encode 256-pixel images without any
classical compression.

.. code-block:: python

   from lib.circuits import MerLinAmplitudeClassifier
   from lib.models import create_model

   model = create_model({
       "type": "amplitude_quantum",
       "input_dim": 256,           # 16×16 pixels
       "n_outputs": 2,             # Binary classification
       "n_modes": 13,              # C(13,3) = 286 ≥ 256
       "n_photons": 3,
       "n_layers": 2,
       "computation_space": "unbunched",
   })

   # Adversarial examples via back-propagation through the quantum layer
   from lib.attacks import bim_attack

   adv_images = bim_attack(
       model, images, labels,
       epsilon=0.1, alpha=0.01, num_iter=10,
   )

   # Adversarial training
   from lib.defense import adversarial_training

   robust_model = adversarial_training(
       model, train_loader, attack_method="bim", epsilon=0.1, epochs=100,
   )

Interactive Exploration
-----------------------

Jupyter Notebook: *notebooks/quantum_adversarial_ml.ipynb*

The notebook provides:

- A self-contained introduction to adversarial ML and the paper's key findings, with the
  amplitude encoding circuit construction explained step by step.
- Live training of a photonic quantum classifier on binary MNIST with loss curves.
- Visualisation of adversarial examples alongside clean images and perturbations.
- The Figure 11 noise comparison experiment with a printed summary table.
- An extended amplitude vs angle encoding robustness comparison.
- Shell script reference for reproducing all paper figures from the command line.

Extensions and Future Work
--------------------------

Enhanced Capabilities
~~~~~~~~~~~~~~~~~~~~~~

- A PennyLane gate-based backend is included alongside MerLin, implementing the original
  paper's architecture (Z-X-Z Euler rotations and a CNOT ladder), allowing direct comparison
  of adversarial vulnerability across photonic and qubit-based quantum classifiers.
- Photon loss is added as a physically motivated noise model using Perceval's ``NoiseModel``.

Experimental Extensions
~~~~~~~~~~~~~~~~~~~~~~~~

- An amplitude vs angle encoding robustness comparison reveals that the classical compression
  bottleneck in angle encoding provides a degree of adversarial robustness as a by-product,
  with angle encoding reaching 41.8% accuracy at ε = 0.05 compared to 2.2% for direct
  amplitude encoding.
- The ``encoding_comparison.sh`` script benchmarks three encoding strategies (direct
  amplitude, compressed amplitude, and angle) systematically across multiple ε values.

Hardware Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

- All experiments can be run on CPU. Smoke-test configs run in under 30 seconds.
- Full adversarial training approximately doubles wall-clock training time and is best run
  via the provided shell scripts.

Future Work
~~~~~~~~~~~

- A systematic evaluation across black-box, gray-box, and white-box threat models would
  give a more complete picture of quantum classifier vulnerability than the white-box attacks
  reproduced here.
- Studying adversarial robustness across multiple circuit depths and the two encoding schemes
  together would clarify the capacity-robustness trade-off suggested by our amplitude vs
  angle encoding comparison.


Citation
--------

.. code-block:: bibtex

   @article{lu2020quantum,
     title={Quantum adversarial machine learning},
     author={Lu, Sirui and Duan, Lu-Ming and Deng, Dong-Ling},
     journal={Physical Review Research},
     volume={2},
     number={3},
     pages={033212},
     year={2020},
     publisher={APS},
     doi={10.1103/PhysRevResearch.2.033212}
   }
