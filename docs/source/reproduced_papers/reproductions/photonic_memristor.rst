:github_url: https://github.com/merlinquantum/merlin

==============================================================
Experimental Neuromorphic Computing Based on Quantum Memristor
==============================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Experimental neuromorphic computing based on quantum memristor

   **Authors**: Mirela Selimovic, Iris Agresti, Michal Siemaszko, Joshua Morris, Borivoje Dakic, Riccardo Albiero, Andrea Crespi, Francesco Ceccarelli, Roberto Osellame, Magdalena Stobinska, Philip Walther

   **Published**: arXiv preprint (2025)

   **DOI**: `10.48550/arXiv.2504.18694 <https://doi.org/10.48550/arXiv.2504.18694>`_

   **Reproduction Status**: Partial

   **Reproducer**: Vassilis Apostolou (vassilis.apostolou@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/memristor_gallery.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This work studies neuromorphic computing with a photonic quantum memristor and evaluates the approach on temporal and nonlinear prediction tasks. The core idea is to introduce memory effects directly in a quantum reservoir so the model can capture richer time-dependent dynamics than memoryless alternatives.

The MerLin reproduction includes both quantum and classical baselines and follows the paper workflow for configurable experiments. At the current stage, the NARMA10 and nonlinear function tasks are validated, while the Mackey-Glass and Santa Fe experiments are still under stabilization.

Significance
============

The paper connects quantum photonics and neuromorphic computing through a memristive element that acts as a physically motivated memory mechanism. This is relevant for near-term quantum machine learning because many practical forecasting and signal-processing tasks depend on temporal memory.

MerLin Implementation
=====================

The reproduction code and runnable examples are maintained in the dedicated README:

`qrc_memristor README <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/qrc_memristor/README.md>`_.

Memristor ``QuantumLayer`` usage (from the implementation):

.. code-block:: python

   self.quantum_layer = ml.QuantumLayer(
       input_size=input_dim + 1,  # input + feedback
       circuit=circuit,
       trainable_parameters=["theta"],
       input_parameters=["px"],
       input_state=[0, 1, 0],
       measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
       no_bunching=True,
   )

   phi_enc = encode_phase(x)
   theta_t = R_to_theta(R_t)  # feedback-derived phase (memristive state)
   quantum_input = torch.cat([phi_enc, theta_t], dim=1)  # [px_0, px_1]
   out = self.quantum_layer(quantum_input)

The full command-line interface, task coverage, and results workflow are documented in that README.

Key Contributions Reproduced
============================

**Quantum reservoir variants**
  * Implemented quantum reservoir computing with memristor (``memristor``) and without memristor (``nomem``).

**Classical baselines for comparison**
  * Implemented linear and quadratic models (``L``, ``Q``) and their memory-augmented variants (``L+M``, ``Q+M``).
  * Enabled direct mode switching between quantum and classical experiments.

Experimental Results
====================

For the nonlinear transformation benchmark, the reproduced results show a clear error reduction when the quantum memristor is enabled.

.. list-table:: Nonlinear Task (reported in reproduction)
   :header-rows: 1
   :widths: 60 40

   * - Method
     - MSE
   * - Linear regression baseline
     - 1.7e-1
   * - Quantum reservoir without memristor
     - 2.8e-2
   * - Quantum reservoir with memristor
     - 1.5e-3

.. figure:: ../../_static/reproduced_papers/memristor/results.png
   :align: center
   :width: 50%
   :alt: Nonlinear task plot comparing target x^4 with linear baseline, quantum reservoir without memristor, and quantum reservoir with memristor.

   Nonlinear task reproduction plot. The memristor-enhanced quantum reservoir follows the target curve more accurately than the baselines.

Performance Analysis
====================

**Advantages of the memristor-enhanced model**
  * Lower nonlinear-task error than both classical linear regression and no-memristor quantum reservoir.
  * Better fit near the high-curvature region of the target function.

**Current limitations**
  * Full parity across all target datasets is not yet reached.
  * Performance claims are currently strongest on NARMA10 and nonlinear benchmarks.

Citation
========

.. code-block:: bibtex

   @misc{selimovic2025experimentalneuromorphiccomputingquantum,
      title={Experimental neuromorphic computing based on quantum memristor},
      author={Mirela Selimovic and Iris Agresti and Michal Siemaszko and Joshua Morris and Borivoje Dakic and Riccardo Albiero and Andrea Crespi and Francesco Ceccarelli and Roberto Osellame and Magdalena Stobinska and Philip Walther},
      year={2025},
      eprint={2504.18694},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      doi={10.48550/arXiv.2504.18694},
      url={https://arxiv.org/abs/2504.18694}
   }
