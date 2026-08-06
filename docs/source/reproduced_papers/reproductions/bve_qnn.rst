:github_url: https://github.com/merlinquantum/merlin

================================================================================
Potential of Quantum Scientific Machine Learning Applied to Weather Modelling
================================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Potential of quantum scientific machine learning applied to weather modelling

   **Authors**: Ben Jaderberg, Antonio A. Gentile, Atiyo Ghosh, Vincent E. Elfving, Caitlin Jones, Davide Vodola, John Manobianco, Horst Weiss

   **Published**: Phys. Rev. A 110, 052423 (2024)

   **DOI**: `10.1103/PhysRevA.110.052423 <https://doi.org/10.1103/PhysRevA.110.052423>`_

   **Paper URL**: `arXiv:2404.08737 <https://arxiv.org/abs/2404.08737>`_

   **Reproduction Status**: Partial (Experiment 1 photonic MerLin complete; Experiment 2 not targeted)

   **Reproducer**: Cyril Deloince (cyrilde9@gmail.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/bve_qnn_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This paper explores quantum scientific machine learning (QSciML) for weather
modelling with parameterised quantum circuits. Experiment 1 trains a supervised
quantum neural network to regress the global atmospheric stream function
:math:`\psi(t, x, y, z)` against a Spectral Element Method (SEM) reference at
:math:`4^\circ` resolution. Experiment 2 studies physics-informed solving of the
barotropic vorticity equation (BVE).

This MerLin reproduction focuses on **Experiment 1**. The original architecture
is a neutral-atom / qubit Hardware-Efficient Ansatz (HEA) QNN with a serial
trainable-frequency feature map, total-magnetisation observable
:math:`\sum_m Z_m`, and a learnable affine output map. We translate that model
into a **photonic dual-rail** MerLin implementation and report the resulting
figures of merit.

Significance
============

The paper is a rare QSciML benchmark outside toy ML tasks: it targets real
geophysical stream-function fields and reports concrete MRE / PPMCC metrics.
For MerLin, it is a useful stress test of ``QuantumLayer`` on a long
differentiable regression pipeline (dual-rail expectations, deep HEA, float64
training).

It also makes the photonic vs neutral-atom comparison explicit: linear optics
cannot implement a native CNOT (KLM), so the photonic HEA must replace fixed
two-qubit gates by beamsplitter mixing. Measuring that gap is itself a valuable
platform result.

MerLin Implementation
=====================

The reproduction lives in the shared
`reproduced_papers <https://github.com/merlinquantum/reproduced_papers>`_
repository under ``papers/bve_qnn`` and is launched with the common runtime:

.. code-block:: bash

   python implementation.py --paper bve_qnn --config configs/example.json

The model is a MerLin ``QuantumLayer`` wrapping a Perceval circuit built from
sparse dual-rail primitives (``pcvl.BS``, ``pcvl.PS``), not from a dense
generic interferometer / ``CircuitBuilder`` mesh. Measurement uses dual-rail
mode expectations, mapped to magnetisation via
:math:`\sum_m(\langle n_{\mathrm{left},m}\rangle-\langle n_{\mathrm{right},m}\rangle)`,
then an affine classical head.

Key Contributions Reproduced
============================

**Experiment 1 supervised stream-function QNN (photonic)**
  * Dual-rail encoding of :math:`N=6` logical qubits (12 modes, 6 photons) with
    HEA depth :math:`l=32`.
  * Serial trainable-frequency feature map and learnable output scaling matching
    the paper protocol.
  * Trainable nearest-neighbour photonic mixing as a CNOT substitute.

**Metrics and packaging**
  * Reported median MRE **14.85%** and median PPMCC **0.754** for the trainable-
    mixing photonic model (1006 parameters).
  * Compared against the paper neutral-atom baseline (MRE 7.1–10.9%, PPMCC 0.870)
    and an external faithful Qadence reproduction (MRE 9.15%, PPMCC 0.873).
  * Full CLI / tests / checkpoint / dataset package under repository conventions.

Implementation Details
======================

.. code-block:: python

   from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
   import torch

   # Dual-rail HEA is assembled with Perceval primitives on fixed mode pairs,
   # then trained through MerLin QuantumLayer.
   layer = QuantumLayer(
       input_size=4 * 6,  # (t,x,y,z) x N qubits
       circuit=circuit,   # Perceval Circuit(n_modes=12)
       input_state=[1, 0] * 6,
       trainable_parameters=["fm", "hea"],
       input_parameters=["input"],
       measurement_strategy=MeasurementStrategy.mode_expectations(
           computation_space=ComputationSpace.DUAL_RAIL
       ),
       dtype=torch.float64,
   )

Experimental Results
====================

**Experiment 1 (supervised stream-function regression)**

.. list-table:: Median figures of merit
   :header-rows: 1
   :widths: 40 20 20 20

   * - Model
     - Params
     - Median MRE
     - Median PPMCC
   * - Paper (neutral-atom HEA)
     - 654
     - 7.1–10.9%
     - 0.870
   * - Neutral-atom reproduction (external)
     - 654
     - 9.15%
     - 0.873
   * - MerLin photonic (trainable mixing)
     - 1006
     - 14.85%
     - 0.754

The photonic model learns real stream-function structure (PPMCC 0.754) but does
not match the neutral-atom baseline. We interpret this as a physical limitation
of linear-optical entanglement relative to native CNOT gates, not as a failed
engineering reproduction.

Technical Implementation Details
================================

**Dual-rail observable**
  * Logical :math:`Z_m` is realised as
    :math:`n_{\mathrm{left},m}-n_{\mathrm{right},m}`.

**Why Perceval primitives inside MerLin**
  * The paper HEA requires sparse mode pairing
    ``(2q, 2q+1)`` and nearest-neighbour mixers
    ``(2q+1, 2(q+1))``.
  * A dense all-to-all MZI mesh is the wrong primitive for that topology.

**Training**
  * Adam, ``lr=1e-2``, batch size 1602, 5000 steps, ``float64``.
  * A trained checkpoint is shipped for evaluation without full retrain.

Interactive Exploration
=======================

**Jupyter Notebook**: :doc:`../../notebooks/reproduced_papers/bve_qnn`

The notebook provides:

* Dataset loading from ``data/bve_qnn/``
* Dual-rail circuit / ``QuantumLayer`` construction
* Checkpoint loading and evaluation (MRE / PPMCC)
* Mollweide SEM vs Quantum comparison at :math:`t=22\mathrm{h}`

The runnable package notebook is also available in
`reproduced_papers <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/bve_qnn/notebook.ipynb>`_.

Extensions and Future Work
==========================

**Possible next steps**
  * Stronger photonic entanglement (measurement feed-forward, alternative encodings)
  * Photonic reproduction of Experiment 2 (physics-informed BVE)
  * Ablations on mixing depth / parameter budget

**Out of scope here**
  * Claiming photonic parity with the paper neutral-atom metrics under the current
    dual-rail + beamsplitter design

Code Access and Documentation
=============================

**Reproduction package**:
`reproduced_papers/papers/bve_qnn <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/bve_qnn>`_

The package includes:

* ``lib/runner.py`` shared-runtime entrypoint
* ``lib/model.py`` dual-rail MerLin QNN
* ``configs/example.json`` paper-faithful evaluation config
* committed checkpoint and SEM dataset
* tests and Mollweide plotting utility

Citation
========

.. code-block:: bibtex

   @article{PhysRevA.110.052423,
     title = {Potential of quantum scientific machine learning applied to weather modeling},
     author = {Jaderberg, Ben and Gentile, Antonio A. and Ghosh, Atiyo and Elfving, Vincent E. and Jones, Caitlin and Vodola, Davide and Manobianco, John and Weiss, Horst},
     journal = {Phys. Rev. A},
     volume = {110},
     issue = {5},
     pages = {052423},
     numpages = {14},
     year = {2024},
     month = {Nov},
     publisher = {American Physical Society},
     doi = {10.1103/PhysRevA.110.052423}
   }

Related Reproductions
=====================

* **HQPINN**: another MerLin SciML / physics-informed benchmark.
* **QRNN / QLSTM**: sequential / temporal modelling reproductions in MerLin.

Impact and Applications
=======================

* **QSciML benchmarking**: non-toy weather regression for photonic QML stacks.
* **Platform comparison**: quantifies dual-rail photonic HEA vs CNOT-based HEA.
* **Tooling feedback**: motivates sparse HEA construction patterns on top of MerLin.
