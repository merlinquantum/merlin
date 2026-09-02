:github_url: https://github.com/merlinquantum/merlin

===========================================
Quantum vs. Classical Time-Series Benchmark
===========================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum vs. Classical: A Comprehensive Benchmark Study for Predicting Time Series with Variational Quantum Machine Learning

   **Authors**: Andreas Fellner, Christian Kreplin, Daniel Tovey, Christian Holm

   **Published**: Machine Learning: Science and Technology, Volume 7, 010501 (2026)

   **DOI**: `10.1088/2632-2153/ae365f <https://doi.org/10.1088/2632-2153/ae365f>`_

   .. merlin-citations-badge:: variational_qml_ts_benchmark

   **Paper URL**: `arXiv:2504.12416 <https://arxiv.org/abs/2504.12416>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: MerLin reproduced-papers contributors

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/variational_qml_ts_benchmark_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This reproduction evaluates variational quantum and classical models for
forecasting the Hénon, Mackey--Glass, and Lorenz chaotic systems. It reproduces
the paper's central 27-task ranking from the authors' released results, checks a
reduced model set with independent training, and adds MerLin photonic dressed-QNN
and reservoir experiments.

Significance
============

The benchmark controls datasets, forecast horizons, sequence lengths, training,
and model selection across quantum and classical model families. It provides a
more reliable comparison than isolated best-case results and shows why the
stopping rule and training budget must be matched before claiming an advantage.

Key Contributions Reproduced
============================

* Aggregated the released grid-search results across all 27 forecasting tasks.
* Reproduced the exact model ordering reported in the paper's Figure 5.
* Independently implemented five quantum and three classical model families.
* Added a photonic dressed QNN and static, sequential, and memristive frozen
  photonic reservoirs.
* Compared fixed-epoch training with validation-plateau stopping.

Experimental Results
====================

Full paper ranking
------------------

.. list-table:: Mean Rank Across 27 Tasks
   :header-rows: 1
   :widths: 15 35 25 25

   * - Rank
     - Model
     - Mean rank
     - Type
   * - 1
     - LSTM
     - 1.78
     - Classical
   * - 2
     - RNN
     - 2.70
     - Classical
   * - 3
     - le-QLSTM
     - 2.85
     - Quantum
   * - 4
     - MLP
     - 4.22
     - Classical
   * - 5
     - d-QNN
     - 4.70
     - Quantum
   * - 6
     - ru-QNN
     - 5.33
     - Quantum
   * - 7
     - QLSTM
     - 7.04
     - Quantum
   * - 8
     - QRNN
     - 7.37
     - Quantum

Classical models obtain mean rank 2.90 and quantum models mean rank 5.46. The
ordering matches the paper exactly.

Training-budget check
---------------------

In reduced six-task experiments, the photonic dressed QNN ranks first after a
fixed 400 epochs, but LSTM ranks first when training follows a validation
plateau with a 3000-epoch cap. The apparent photonic lead is therefore a
training-budget effect.

Photonic reservoir extension
----------------------------

The selected 8-mode, 4-photon reservoir comparison gives LSTM and the sequential
photonic reservoir the best mean rank, both at 2.50. The sequential and
memristive reservoirs are close, with a geometric-mean test-MSE ratio of 0.89.
This result does not isolate a memristor benefit and is not evidence of quantum
advantage.

Implementation Details
======================

Run the default workflow from the reproduced-papers repository root:

.. code-block:: bash

   python implementation.py --paper variational_qml_ts_benchmark --config configs/defaults.json

The exact paper ranking can be regenerated without training by running
``utils/plot_paper_figures.py`` from the paper directory. Photonic CLI options
reuse ``num-qubits`` for optical modes and ``hidden-size`` for photons.

Limitations
===========

* The exact 27-task comparison uses the authors' released result files.
* Independent gate-model runs cover two tasks, and photonic runs cover six.
* Live runs use three seeds and representative configurations rather than the
  paper's ten seeds and complete grid.
* The photonic reservoir study covers one selected optical configuration and
  does not establish general superiority.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (time-series benchmark) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/variational_qml_ts_benchmark>`_

Citation
========

.. code-block:: bibtex

   @article{fellner2026quantum,
     title={Quantum vs. Classical: A Comprehensive Benchmark Study for Predicting Time Series with Variational Quantum Machine Learning},
     author={Fellner, Andreas and Kreplin, Christian and Tovey, Daniel and Holm, Christian},
     journal={Machine Learning: Science and Technology},
     volume={7},
     pages={010501},
     year={2026},
     doi={10.1088/2632-2153/ae365f}
   }
