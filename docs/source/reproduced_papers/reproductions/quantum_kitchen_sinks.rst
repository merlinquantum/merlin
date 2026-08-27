:github_url: https://github.com/merlinquantum/merlin

====================================================================================
Quantum Kitchen Sinks
====================================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum Kitchen Sinks: An algorithm for machine learning on near-term quantum computers

   **Authors**: C. M. Wilson, J. S. Otterbach, N. Tezak, R. S. Smith, A. M. Polloreno, P. J. Karalekas, S. Heidel, M. S. Alam, G. E. Crooks, M. P. da Silva

   **Published**: arXiv (2018; v2 2019)

   **DOI**: `arXiv:1806.08321 <https://arxiv.org/abs/1806.08321>`_

   .. merlin-citations-badge:: quantum_kitchen_sinks

   **Reproduction Status**: ✅ Complete (simulation only — the paper's QPU runs are not reproduced)

   **Reproducer**: Jean Senellart (jean.senellart@quandela.com)

Project Repository
====================================================================================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/quantum_kitchen_sinks_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
====================================================================================

Quantum Kitchen Sinks (QKS) is an *open-loop* hybrid algorithm: the quantum
processor is never trained, it is used as a random non-linear feature
extractor. For each of ``E`` independent episodes, the input vector ``u`` is
mapped to circuit angles by a fresh random linear encoding
``theta_e = Omega_e u + beta_e`` (``Omega ~ N(0, sigma^2)`` sparse,
``beta ~ U(0, 2 pi)``), a small fixed-depth circuit of ``RX`` rotations and an
entangling layer is executed, and **a single bitstring is sampled**.
Concatenating the bitstrings of all episodes gives an ``E*q``-dimensional binary
feature vector fed to a *linear* classifier — under the paper's Linear Baseline
rule, every non-linearity in the model comes from the circuit.

The reproduction covers the gate-model results (picture frames, (3,5)-MNIST,
1/2/4 qubits) and the classical baselines, and adds a **photonic adaptation** of
the same open-loop recipe built with MerLin.

Significance
====================================================================================

QKS sidesteps the training difficulties of variational QML — no parameter
optimisation on the quantum device, no barren plateaus, one shot per circuit —
while still producing features a linear model cannot produce on its own. That
makes it a natural fit for photonics: a QKS episode is a fixed random
interferometer with a data-dependent phase, which is what a photonic chip runs
natively. The paper's implicit kernel is also known in closed form, which turns
the reproduction into an exact test of the feature map rather than an accuracy
comparison.

MerLin Implementation
====================================================================================

MerLin builds the per-episode photonic circuits, with all phases frozen (QKS is
open-loop), data driving an angle encoding, and one shot sampled per episode.
Four architectures are implemented, from a generic random interferometer to a
deterministic dual-rail chip read out with threshold detectors.

Key Contributions Reproduced
====================================================================================

**Picture frames (Fig. 3)**
  * Logistic regression alone is at chance (49.25%); QKS with the two-qubit CNOT
    ansatz reaches **100.0 ± 0.0%** test accuracy at its best ``(sigma, E)``.
  * The Fig. 2(b) CZ ansatz is chance-level everywhere on a ``sigma`` × ``E``
    sweep, as the paper reports. This is exact rather than empirical: the
    entangler acts on ``|++>`` before the rotations, so each qubit is maximally
    mixed and every feature bit is a fair coin — a useful negative control.

**(3,5)-MNIST (Fig. 5)**
  * Linear baseline 4.40% test error; QKS reaches **1.77 ± 0.24%** (two qubits)
    at a fixed ``E = 5000``, against the paper's best point of 1.4% obtained
    after optimising ``sigma`` and ``E`` up to 20 000.

**Closed-form implicit kernel**
  * The simulator is checked against the analytic QKS kernel derived in the
    paper — ``1/2 + (1/8) exp(-sigma^2 ||u1 - v1||^2 / 2) + (1/16) exp(-sigma^2 ||u - v||^2 / 2)``
    for Fig. 2(a), the constant ``1/2`` for Fig. 2(b) — a sharper test than any
    accuracy figure.

**Photonic adaptation (new)**
  * A balanced 50:50 splitter, a data-dependent phase, and a second balanced
    splitter give an even-rail click probability of exactly ``sin^2(theta/2)``:
    the photonic featurizer *is* the paper's ``RX(theta)`` ansatz, agreeing with
    the gate code to 1e-5. Two-qubit equivalence follows from a post-selected
    KLM CNOT (success 1/9), exact to 1.3e-07.
  * Reproducing that gate is not necessary. A deterministic 4-mode circuit — two
    MZI encoders, one balanced splitter joining the logical-|1> rails (HOM
    interference, no ancillas), threshold detectors, **no post-selection** —
    reaches **1.60 ± 0.00%** on (3,5)-MNIST, at parity with the gate ansatz.
  * Circuit design, not photonics, is what matters here: a generic random mesh
    is unbalanced, has low fringe visibility, and never beats the linear
    baseline (4.43 ± 0.34% against 4.40%).

Implementation Details
====================================================================================

Each episode is a ``QuantumLayer`` with frozen parameters. The deterministic
threshold-readout ansatz is built through the explicit-circuit interface:

.. code-block:: python

   import merlin as ml
   import numpy as np
   import perceval as pcvl

   # MerLin parameterises a splitter by R = cos^2(theta / 2): balanced is pi / 2.
   BALANCED = np.pi / 2
   circuit = pcvl.Circuit(4)
   for lo in (0, 2):                                 # MZI first arm
       circuit.add(lo, pcvl.BS.H(theta=BALANCED))
   for i, mode in enumerate((1, 2)):                 # data-dependent phases
       circuit.add(mode, pcvl.PS(pcvl.P(f"px_{i}")))
   for lo in (0, 2):                                 # MZI second arm
       circuit.add(lo, pcvl.BS.H(theta=BALANCED))
   circuit.add(1, pcvl.BS.H(theta=BALANCED))         # entangling HOM splitter

   layer = ml.QuantumLayer(
       circuit=circuit,
       input_parameters=["px"],
       input_state=(1, 0, 0, 1),
       n_photons=2,
       measurement_strategy=ml.MeasurementStrategy.probs(
           computation_space=ml.ComputationSpace.FOCK
       ),
   )

Experimental Results
====================================================================================

(3,5)-MNIST, 4 000 train / 1 000 test, test error over 3 seeds on one shared
subsample:

.. list-table::
   :header-rows: 1
   :widths: 45 25 25

   * - Model
     - Paper
     - Reproduced
   * - Logistic regression (linear baseline)
     - 4.1%
     - 4.40%
   * - QKS, 1 qubit (``E = 5000``)
     - —
     - 1.87 ± 0.09%
   * - QKS, 2 qubits (``E = 5000``)
     - 1.4% (best point, ``E`` up to 20 000)
     - 1.77 ± 0.24%
   * - Photonic QKS, dual-rail MZI (``m = 4``)
     - n/a
     - 1.43 ± 0.24%
   * - Photonic QKS, threshold readout + HOM splitter
     - n/a
     - 1.60 ± 0.00%
   * - Photonic QKS, generic random mesh (``m = 6``)
     - n/a
     - 4.43 ± 0.34%

On picture frames, logistic regression is at 49.25%, the gate-model CNOT ansatz
at 100.0 ± 0.0% and the photonic dual-rail model at 99.50 ± 0.41%.

Full tables, the ``sigma`` × ``E`` heatmaps and the hardware-aware reporting are
in the project
`README <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/quantum_kitchen_sinks/README.md>`_.

Code Access and Documentation
====================================================================================

**GitHub Repository**: `reproduced_papers/papers/quantum_kitchen_sinks <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/quantum_kitchen_sinks/>`_

The complete implementation includes:

* A batched-NumPy statevector simulator for the paper's 1/2/4-qubit ansätze
* Four MerLin photonic architectures (random mesh, dual-rail MZI, post-selected
  KLM CNOT, deterministic threshold readout)
* 29 experiment configs, committed run artifacts under ``results/``, and a
  notebook that reads them without needing a prior run
* 30 tests, including the closed-form kernel checks and photonic-versus-gate
  equivalence assertions

Related Reproductions
====================================================================================

* :doc:`fock_state_expressivity` covers Gan et al. (2022), whose Algorithm 3
  pushes the kitchen-sinks idea into a Fock-state photonic regime. The two
  reproductions are complementary: this one covers the original gate-model
  formulation of Wilson et al.

Citation
====================================================================================

.. code-block:: bibtex

   @article{wilson2019qks,
       title   = {Quantum {Kitchen} {Sinks}: An algorithm for machine learning on near-term quantum computers},
       author  = {Wilson, C. M. and Otterbach, J. S. and Tezak, N. and Smith, R. S. and Polloreno, A. M. and Karalekas, P. J. and Heidel, S. and Alam, M. S. and Crooks, G. E. and da Silva, M. P.},
       journal = {arXiv:1806.08321},
       url     = {https://arxiv.org/abs/1806.08321},
       year    = {2019},
   }
