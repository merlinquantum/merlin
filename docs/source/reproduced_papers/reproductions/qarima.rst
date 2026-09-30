:github_url: https://github.com/merlinquantum/merlin

====================================================
QARIMA: A Quantum Approach to Time-Series Analysis
====================================================

.. admonition:: Paper Information
   :class: note

   **Title**: QARIMA: A Quantum Approach To Classical Time Series Analysis

   **Authors**: N. Mohanty, B. K. Behera, B. Mukherjee, P. Dash

   **Published**: arXiv preprint (2026)

   **DOI**: `10.48550/arXiv.2604.08277 <https://doi.org/10.48550/arXiv.2604.08277>`_

   .. merlin-citations-badge:: qarima

   **Paper URL**: `arXiv:2604.08277 <https://arxiv.org/abs/2604.08277>`_

   **Reproduction Status**: ⚠️ Partial — headline claim not supported under fair baselines

   **Reproducer**: Cassandre Notton

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/qarima_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

QARIMA augments a Box--Jenkins ARIMA pipeline with quantum-informed order
selection, compact-swap-test ACF and PACF estimates, and variational quantum
circuits for AR and MA coefficients. This reproduction implements the pipeline
on five univariate datasets and compares classical, gate-based, and MerLin
photonic coefficient refiners under the same forecasting protocol.

Significance
============

The reproduction isolates which parts of QARIMA can change forecast quality.
In the analytic swap-test limit, the phase-corrected prediction reduces exactly
to a linear dot product. This makes matched-order classical, gate, and photonic
refiners directly comparable and exposes whether reported gains come from the
quantum parameterization or from model-order and baseline choices.

MerLin Implementation
=====================

All three refiners share one ARIMA implementation:

* ``classical`` refines an OLS warm start against the paper's loss.
* ``gate`` uses an RY/CNOT-ladder statevector VQC and COBYLA.
* ``merlin`` uses a trainable photonic ``QuantumLayer`` followed by a linear
  coefficient readout, with the same warm-started form.

The implementation includes dynamic out-of-sample forecasting, finite-shot
swap-test noise, MSE and MAPE metrics, Diebold--Mariano tests, order sweeps, and
fair seasonal baselines.

Experimental Results
====================

The matched-order refiners agree to at least three significant figures on all
five datasets. This supports the analytic conclusion that the quantum
coefficient refinement adds no measurable predictive advantage.

.. list-table:: Out-of-Sample Multi-Step MSE
   :header-rows: 1
   :widths: 19 20 20 24 17

   * - Dataset
     - Paper classical
     - Paper best-Q
     - Reproduced best-Q (classical/gate/MerLin)
     - Fair seasonal
   * - Sunspots
     - 2181.6
     - 2146.9
     - 2108 / 2108 / 2108
     - —
   * - CO2
     - 78.4
     - 10.03
     - 83.7 / 83.5 / 83.7
     - 0.40
   * - Australian Beer
     - 1491.8
     - 59.8
     - 216.7 / 216.7 / 216.7
     - 181.8
   * - Woollen Yarn
     - 528230
     - 530506
     - 470305 / 470305 / 470308
     - 3.50e6
   * - Sydney
     - 11.44
     - 11.36
     - 27.9 / 27.9 / 27.9
     - 25.4

The CO2 order sweep shows that a classical AR(14,1,0) reaches approximately the
paper's best-Q error, while a seasonal ARIMA obtains MSE 0.40. The reproduced
evidence therefore attributes the headline gap to AR-order and baseline choices,
not to the gate or photonic refiner.

Implementation Details
======================

Run a dataset from the reproduced-papers repository root:

.. code-block:: bash

   python implementation.py --paper QARIMA --config configs/co2.json --seed 42

The available configs cover Sunspots, Mauna Loa CO2, Australian Beer,
Australian Woollen Yarn, and Sydney temperature. Each run writes metrics,
forecasts, plots, and the resolved configuration to a timestamped directory.

Deviations and Limitations
==========================

* The swap test is analytic by default, with optional finite-shot noise.
* The forecast is multi-step because the paper's reported error scale is
  inconsistent with one-step rolling evaluation.
* The paper's Sydney station contains no temperature data. The reproduction
  uses Sydney Observatory Hill, so its absolute Sydney values are not directly
  comparable.
* The VQC coefficient readout is underspecified in the paper and is implemented
  as a documented warm-started refinement.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (QARIMA) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/QARIMA>`_

Citation
========

.. code-block:: bibtex

   @article{mohanty2026qarima,
     title={QARIMA: A Quantum Approach To Classical Time Series Analysis},
     author={Mohanty, N. and Behera, B. K. and Mukherjee, B. and Dash, P.},
     journal={arXiv preprint arXiv:2604.08277},
     year={2026},
     doi={10.48550/arXiv.2604.08277}
   }
