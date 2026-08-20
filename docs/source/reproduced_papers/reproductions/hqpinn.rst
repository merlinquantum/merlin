:github_url: https://github.com/merlinquantum/merlin

================================================================================
Hybrid Quantum Physics-Informed Neural Network for High-Speed Flows
================================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Hybrid Quantum Physics-informed Neural Network: Towards Efficient Learning of High-speed Flows

   **Authors**: Fong Yew Leong, Wei-Bin Ewe, Si Bui Quang Tran, Zhongyuan Zhang, Jun Yong Khoo

   **Published**: Computers & Fluids, Volume 301, 106782 (2025)

   **DOI**: `10.1016/j.compfluid.2025.106782 <https://doi.org/10.1016/j.compfluid.2025.106782>`_

   .. merlin-citations-badge:: hqpinn

   **Paper URL**: `arXiv:2503.02202 <https://arxiv.org/abs/2503.02202>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Jérôme Ricciardi (jerome.ricciardi@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/hqpinn_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This reproduction targets the hybrid quantum physics-informed neural network
(HQPINN) benchmark proposed by Leong et al. The paper studies whether quantum
branches can improve physics-informed learning on high-speed-flow problems by
comparing classical-classical, hybrid quantum-classical, and quantum-quantum
architectures.

The benchmark covers four problems:

* ``DHO``: damped harmonic oscillator.
* ``SEE``: smooth one-dimensional Euler equation.
* ``DEE``: discontinuous one-dimensional Euler equation.
* ``TAF``: steady two-dimensional transonic flow around a NACA0012 airfoil.

Each model is trained with a PINN objective combining data or boundary-condition
terms with physics-residual terms computed through automatic differentiation.

Significance
============

The paper is relevant for near-term quantum machine learning because it tests
quantum models on scientific machine-learning tasks where the target is not only
data fitting but also equation consistency. It separates easier low-dimensional
physics problems from harder high-speed-flow cases and compares quantum branches
against direct classical PINN baselines.

For MerLin, this reproduction is a useful stress test for photonic
``QuantumLayer`` models inside differentiable PDE-constrained training loops.
It exercises coordinate encoding, Fock-space probability readout, branch fusion,
and repeated automatic-differentiation residual evaluations.

MerLin Implementation
=====================

The reproduced implementation uses the repository-level ``implementation.py``
runtime. The shared entrypoint dispatches into ``papers/HQPINN`` and then calls:

.. code-block:: text

   lib.runner.train_and_evaluate(cfg, run_dir)

The implementation supports the paper's main architecture families and adds
MerLin photonic variants:

.. list-table:: Architecture Variants
   :header-rows: 1
   :widths: 18 82

   * - Variant
     - Meaning
   * - ``cc``
     - Classical-classical PINN with two classical branches.
   * - ``hy-pl``
     - Hybrid model with one PennyLane quantum branch and one classical branch.
   * - ``hy-m``
     - Hybrid model with one MerLin photonic branch and one classical branch.
   * - ``hy-mp``
     - DHO-only hybrid model using a manual Perceval/MerLin photonic branch.
   * - ``qq-pl``
     - Quantum-quantum model with two PennyLane branches.
   * - ``qq-m``
     - Quantum-quantum model with two MerLin photonic branches.
   * - ``qq-mp``
     - DHO-only quantum-quantum model with manual Perceval/MerLin branches.


How ``QuantumLayer`` Fits This Reproduction
===========================================

During local training, MerLin branches are differentiable PyTorch modules. The
branch maps physical coordinates to angle features, evaluates a trainable
photonic circuit, groups Fock-space probabilities with MerLin's measurement
strategy, and applies a small linear readout before fusing branch outputs.

.. list-table:: MerLin Feature Maps
   :header-rows: 1
   :widths: 18 82

   * - Benchmark
     - Feature map input
   * - ``DHO``
     - Time ``t`` encoded as harmonic angle features.
   * - ``SEE``
     - ``(x, t)``, including the traveling-wave coordinate ``x - t``.
   * - ``DEE``
     - ``(x, t)``, including the shock-relative coordinate ``x - (x0 + u*t)``.
   * - ``TAF``
     - ``(x, y)``, including a compact coordinate interaction ``x - y``.

Remote mode is inference-only. It loads a local checkpoint, rebuilds the MerLin
branch with a ``MerlinProcessor``, and evaluates the saved model on the selected
backend:

.. code-block:: bash

   python implementation.py --paper HQPINN --config configs/dho_hy_m_run.json --mode remote --backend sim:ascella

Remote mode does not train with remote gradients. Training configs run locally;
remote mode is only used after a matching checkpoint exists in ``models/``.

Key Contributions Reproduced
============================

Our reproduction focuses on the following contributions from the paper:

**Benchmark implementation**

* Implemented the four benchmark families: ``DHO``, ``SEE``, ``DEE``, and
  ``TAF``.
* Preserved the paper's ``cc``, ``hy``, and ``qq`` architecture split.
* Added config-driven execution through the repository shared runtime.

**Photonic branch implementation**

* Added MerLin photonic branches for hybrid and quantum-quantum PINNs.
* Added DHO-only manual Perceval/MerLin circuits for closer circuit-level
  control.
* Kept PennyLane variants for comparison where local runtime is practical.

Experimental Results
====================

The committed results are small classical-classical baseline runs used as a
documentation snapshot. They are not the full paper matrix.

.. list-table:: Committed Baseline Results
   :header-rows: 1
   :widths: 16 30 30

   * - Benchmark
     - Config
     - Main metric
   * - ``DHO``
     - ``configs/dho_cc_train.json``
     - Relative L2 error ``4.161749e-01``.
   * - ``SEE``
     - ``configs/see_cc_train_10-4.json``
     - Density error ``4.158964e-03``; pressure error ``2.587267e-04``.
   * - ``DEE``
     - ``configs/dee_cc_train_10-4.json``
     - Density error ``3.934973e-02``; pressure error ``5.138812e-05``.

Current qualitative status:

* ``DHO``, ``SEE``, and ``DEE`` have runnable train and inference paths.
* ``TAF`` is implemented as a geometry-aware PINN baseline because the original
  internal CFD target fields are unavailable.
* The default config is a lightweight DHO inference smoke check.
* PennyLane variants outside ``DHO`` are expensive on CPU and are not part of
  the standard batch launcher.

Comparison With the Paper
=========================

The implementation follows the paper's benchmark split and architecture naming.
The main mismatch is ``TAF``: without supervised CFD targets for internal
points, this reproduction cannot fully match the paper's transonic-airfoil
results or Figure 7 flow structure.

Other known deviations:

* Full-batch training from the paper is not reproduced for all cases; minibatch
  training is used where needed for local runtime.
* PennyLane runs outside ``DHO`` are limited by CPU latency.
* The paper describes hybrid output fusion as a fixed branch combination, while
  the reproduction uses explicit local branch fusion code.
* The MerLin interferometer is a photonic formulation of the branch, not an
  exact circuit-for-circuit reproduction of the original PennyLane ansatz.

Interactive Exploration
=======================

The paper folder includes a `notebook <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/HQPINN/notebook.ipynb>`_
for interactive DHO exploration and helper utilities.

Extensions and Future Work
==========================

Current next steps are:

* Test additional MerLin interferometer shapes and photonic encodings.
* Build more manual Perceval circuits that mimic the original PQC structure,
  following the DHO manual-circuit path.
* Contact the paper authors for the missing TAF internal CFD target fields.
* Clarify original quantum implementation details that are underspecified in
  the paper.
* Complete the full result matrix for MerLin and PennyLane variants where
  runtime permits.

Code Access and Documentation
=============================

**Reproduction Repository**: `merlinquantum/reproduced_papers (HQPINN) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/HQPINN>`_

For command-line usage, config naming, output layout, and limitations, see the
project README:
`HQPINN README <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/HQPINN/README.md>`_.

Citation
========

.. code-block:: bibtex

   @article{leong2025hybrid,
        title={Hybrid quantum physics-informed neural network: Towards efficient learning of high-speed flows},
        author={Leong, Fong Yew and Ewe, Wei-Bin and Tran, Si Bui Quang and Zhang, Zhongyuan and Khoo, Jun Yong},
        journal={Computers \& Fluids},
        pages={106782},
        year={2025},
        publisher={Elsevier}
        }
