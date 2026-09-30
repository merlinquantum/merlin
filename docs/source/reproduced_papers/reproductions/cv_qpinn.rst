:github_url: https://github.com/merlinquantum/merlin

===========================================================
CV-QPINN: Quantum Physics-Informed Neural Networks for PDEs
===========================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum physics informed neural networks for multi-variable partial differential equations

   **Authors**: Giorgio Panichi, Sebastiano Corli, Enrico Prati

   **Published**: arXiv preprint (2025)

   **DOI**: `10.48550/arXiv.2503.12244 <https://doi.org/10.48550/arXiv.2503.12244>`_

   .. merlin-citations-badge:: cv_qpinn

   **Paper URL**: `arXiv:2503.12244 <https://arxiv.org/abs/2503.12244>`_

   **Reproduction Status**: ⚠️ Partial

   **Reproducer**: Benjamin Stott (benjamin.stott@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/cv_qpinn_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

The paper combines the continuous-variable (CV) quantum neural network ansatz of
Killoran et al. with the Physics-Informed Neural Network loss design of Raissi et
al. Its contribution is a multi-output architecture with a consistency loss. The
network exposes one homodyne read-out per derivative order, and a loss term pins
each output to the auto-differentiated derivative of the preceding one. This
removes the nested automatic differentiation that otherwise dominates memory in
CV simulators. The method is demonstrated on the 1D Poisson equation and the 1D
heat equation, with a photon-loss noise model derived from Xanadu's X8 processor.

Significance
============

Nested automatic differentiation through a Fock-truncated CV simulator limits the
derivative order such a network can reach. The consistency loss addresses that
limit at the level of the architecture rather than the simulator, so it applies
beyond photonic hardware. The reproduction tests two separate questions: whether
the architecture reproduces the reported errors, and whether the consistency loss
is preferable to nested autograd at a matched epoch budget.

MerLin Implementation
=====================

The CV simulator is written directly in PyTorch and does not use Strawberry
Fields. ``lib/cv_simulator.py`` implements rotation, beam-splitter, squeezing,
displacement and Kerr gates on one- and two-qumode systems, together with the
photon-loss channel. ``lib/qpinn_model.py`` provides the Killoran-style
multi-qumode and single-qumode layers, and ``lib/losses.py`` the consistency-loss
objectives for both PDEs.

Two comparators are implemented alongside the QPINN. ``lib/pinn_baseline.py`` is a
classical fully-connected network whose width is set to match the QPINN parameter
count, returning the same ``(u, u_x)`` pair and trained with the same loss.
``lib/merlin_pinn.py`` applies the consistency-loss scheme to a MerLin
interferometer with angle encoding, mapping ``UNBUNCHED`` detection probabilities
to ``(u, u_x)`` through two trainable linear heads.

Key Contributions Reproduced
============================

**Consistency-loss architecture on two PDEs**
  * Implemented the multi-output QPINN and the consistency loss for the 1D Poisson
    and 1D heat equations.
  * The 5-seed heat-equation sweep reaches RMSE 1.23e-2 against the paper's
    reported 1.24e-2.

**Matched-parameter classical baseline**
  * Applied the consistency-loss scheme to the classical network as well, so the
    two differ only in the function family.
  * Under matched-effort training the classical PINN attains lower RMSE than the
    QPINN on the heat equation.

**Nested-versus-consistency ablation**
  * Ran both loss formulations head to head on the same architecture at 200 epochs
    and two Fock cutoffs.
  * Measured the memory saving the consistency loss provides and the accuracy cost
    that accompanies it.

**Photonic adaptation**
  * Trained a MerLin linear-optics network with the same consistency loss on the
    Poisson task, reaching RMSE 2.37e-4.

Implementation Details
======================

Runs are launched from the reproduced-papers repository root.

.. code-block:: bash

   # Poisson QPINN, 2+2 layers, cutoff 8
   python implementation.py --paper CV_QPINN_PDE --config configs/poisson_smoke.json

   # Paper-accurate Poisson QPINN (Table V hyperparameters)
   python implementation.py --paper CV_QPINN_PDE --config configs/poisson_original.json

   # Classical PINN baseline, matched on parameter count
   python implementation.py --paper CV_QPINN_PDE --config configs/poisson_pinn.json

   # MerLin photonic adaptation
   python implementation.py --paper CV_QPINN_PDE --config configs/poisson_merlin.json

The heat-equation counterparts are ``heat_smoke.json``, ``heat_original.json`` and
``heat_pinn.json``. The ``*_original.json`` configurations use the paper's Table
V and VI hyperparameters and require multi-hour CPU runs. The smoke
configurations complete in single-digit minutes.

Experimental Results
====================

1D Poisson equation
-------------------

Reproduction rows are single-seed.

.. list-table:: Poisson: error against the analytic solution
   :header-rows: 1
   :widths: 40 12 16 16 16

   * - Setting
     - Params
     - RMSE
     - NMSE
     - Wall time
   * - Paper QPINN (8 layers, cutoff 10, 5000 ep)
     - 88
     - 1.09e-4
     - 6.08e-6
     - not stated
   * - Reproduction QPINN (2+2 layers, cutoff 8, 200 ep)
     - 48
     - 4.64e-3
     - 1.11e-2
     - 168 s
   * - Reproduction classical PINN (3000 ep)
     - 90
     - 8.28e-4
     - 3.53e-4
     - 102 s
   * - Reproduction MerLin PINN (6 modes, 3 photons, 600 ep)
     - 162
     - 2.37e-4
     - 2.90e-5
     - 25 s

The MerLin adaptation reaches the same order of magnitude as the paper's reported
QPINN RMSE in 600 epochs. The CV-QPINN smoke configuration runs 25 times fewer
epochs at a lower Fock cutoff and lands correspondingly further from the paper.

.. figure:: ../../_static/reproduced_papers/CV_QPINN_PDE/poisson_compare.png
   :alt: Poisson 1D predictions from the CV-QPINN, classical PINN and MerLin PINN against the analytic solution
   :align: center
   :width: 85%

   Poisson predictions against the analytic solution.

1D heat equation
----------------

Reproduction rows are means over 5 seeds (42, 7, 123, 256, 1024).

.. list-table:: Heat equation: error against an RK45 reference
   :header-rows: 1
   :widths: 38 10 16 12 12 12

   * - Setting
     - Params
     - RMSE (mean)
     - RMSE std
     - MAE
     - L∞
   * - Paper QPINN (4 layers, cutoff 20, 1000 ep)
     - 44
     - 1.24e-2
     - not stated
     - 9.63e-3
     - 3.93e-2
   * - Paper classical PINN
     - 44
     - 2.09e-2
     - not stated
     - 1.48e-2
     - 9.04e-2
   * - Reproduction QPINN (2+2, cutoff 10, 60+200 ep)
     - 48
     - 1.23e-2
     - 4.8e-3
     - 8.6e-3
     - 5.6e-2
   * - Reproduction classical PINN (300+1000 ep)
     - 42
     - 8.74e-3
     - 1.2e-3
     - 7.0e-3
     - 3.0e-2

The reproduced QPINN matches the paper's reported QPINN RMSE to within 1%. The
reproduced classical baseline attains 8.74e-3, a factor of 2.4 below the paper's
reported classical PINN and a factor of 1.40 below the reproduced QPINN. The
QPINN mean sits 2.9 classical-PINN standard deviations above the classical mean,
and its seed spread is about four times wider. The paper's quantum-advantage
reading of Table IV is therefore not supported under a matched-effort comparison.

.. figure:: ../../_static/reproduced_papers/CV_QPINN_PDE/heat_qpinn.png
   :alt: Heat-equation QPINN prediction, RK45 reference and absolute error over the space-time domain
   :align: center
   :width: 100%

   A single heat QPINN run (60 + 250 epochs, RMSE 8.95e-3 for that seed) against
   the RK45 reference. Error concentrates at early times near the peak of the
   Gaussian initial condition.

Nested autograd against the consistency loss
--------------------------------------------

Both loss formulations were run on the same architecture at 200 epochs.

.. list-table:: Poisson: nested autograd against the consistency loss
   :header-rows: 1
   :widths: 14 26 22 18 20

   * - Cutoff
     - Loss
     - RMSE
     - Wall time
     - Derivative
   * - 8
     - nested
     - 4.24e-5
     - 85 s
     - autograd of autograd
   * - 8
     - consistency
     - 4.64e-3
     - 168 s
     - paper's scheme
   * - 12
     - nested
     - 1.51e-4
     - 141 s
     - autograd of autograd
   * - 12
     - consistency
     - 1.85e-3
     - 115 s
     - paper's scheme

The consistency loss delivers the memory saving it is designed for: peak-RSS
delta for the nested path rises from 2 MB at cutoff 10 to 30 MB at cutoff 12. At
equal epoch budget in this regime it costs a factor of 12 to 100 in accuracy,
which the paper does not report. At the paper's cutoff of 15 to 20 the nested
memory cost is expected to reverse the ordering, but the range in which the
consistency loss is strictly preferable is narrower than the paper indicates.

Hardware-Aware Settings
=======================

.. list-table:: MerLin variant
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Value
   * - Computation space
     - ``UNBUNCHED``
   * - Detector model
     - threshold
   * - Photon number
     - 3
   * - Number of modes
     - 6
   * - Input state
     - ``[1, 0, 1, 0, 1, 0]``
   * - Encoding
     - angle, modes ``[0, 1, 2]``, scale ``π/2``
   * - Measurement strategy
     - ``MeasurementStrategy.probs(computation_space=UNBUNCHED)``
   * - Postselection
     - none
   * - Simulator
     - MerLin CPU simulator (analytic, shots = 0)
   * - Seeds
     - 42

Limitations
===========

* The CV simulator is re-implemented in PyTorch with ``matrix_exp`` inside a Fock
  truncation. Each gate is mathematically equivalent to its Strawberry Fields
  counterpart, but the gradient path differs, and results that depend on the
  TensorFlow gradient path could behave differently.
* MerLin targets linear-optical photonic computing, in which squeezing,
  displacement and Kerr gates are not native. The MerLin variant applies the
  consistency-loss scheme to a different architecture and is not a CV port of the
  paper's circuit.
* The X8 photon-loss study of Section V is implemented at the gate level but not
  run as part of the headline results.
* Poisson results are single-seed. Only the heat-equation comparison uses five
  seeds.
* The simulator defaults to ``complex128`` to preserve trace under deep circuits.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (CV_QPINN_PDE) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/CV_QPINN_PDE>`_

The reproduction folder includes a notebook that builds the CV simulator, the
QPINN and the consistency loss step by step, trains the smoke configurations, and
re-plots the committed runs.

Citation
========

.. code-block:: bibtex

   @misc{panichi_quantum_2025,
     title={Quantum physics informed neural networks for multi-variable partial differential equations},
     author={Panichi, Giorgio and Corli, Sebastiano and Prati, Enrico},
     year={2025},
     eprint={2503.12244},
     archivePrefix={arXiv},
     primaryClass={quant-ph},
     doi={10.48550/arXiv.2503.12244}
   }

Related Reproductions
=====================

* :doc:`hqpinn` applies a hybrid quantum PINN to the same problem family and
  differs in how the quantum layer enters the PINN loss.
