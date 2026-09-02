:github_url: https://github.com/merlinquantum/merlin

=============================================
Photonic Quantum-Accelerated Machine Learning
=============================================

.. admonition:: Paper Information
   :class: note

   **Title**: Photonic Quantum-Accelerated Machine Learning

   **Authors**: Markus Rambach, Abhishek Roy, Alexei Gilchrist, Akitada Sakurai, William J. Munro, Kae Nemoto, Andrew G. White

   **Published**: arXiv preprint (2025)

   **DOI**: `10.48550/arXiv.2512.08318 <https://doi.org/10.48550/arXiv.2512.08318>`_

   .. merlin-citations-badge:: photonic_quantum_accelerated_ml

   **Paper URL**: `arXiv:2512.08318 <https://arxiv.org/abs/2512.08318>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: MerLin reproduced-papers contributors

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/photonic_quantum_accelerated_ml_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This reproduction implements the paper's frozen boson-sampling reservoir as a
quantum feature accelerator for classical machine-learning readouts. It extends
the original QORC reproduction to controlled MNIST imbalance, sparse-data
training, imperfect photon sources, biomedical MedMNIST tasks, and optional
photonic-QPU execution.

Significance
============

The work tests photonic quantum features under constraints that matter for
near-term use: limited training data, source noise, and real hardware. Since the
reservoir remains fixed and only the classical readout is trained, the workflow
separates quantum feature generation from inexpensive classical optimization.

MerLin Implementation
=====================

The implementation shares the ``papers/QORC`` package with the earlier quantum
optical reservoir reproduction. Images are reduced and encoded into phase
shifters between fixed random interferometers. Photon-count probabilities form
the reservoir features used by linear or neural-network readouts.

The reproduced experiment suite includes:

* MNIST QORC versus a raw-pixel linear-softmax baseline.
* Accuracy versus source indistinguishability for 12- and 20-mode reservoirs.
* QORC versus multinomial logistic regression on four MedMNIST datasets.
* Accuracy versus balanced MNIST training-set size for ideal, noisy, and
  optional QPU reservoirs.
* A noisy output-distribution comparison between simulation and Quandela
  Ascella measurements.
* Pixel-only versus QORC-augmented readout architectures on full MNIST.

Experimental Results
====================

MedMNIST reproduction
---------------------

The Fig. 3 experiment uses 20 modes, 3 photons, 200 training epochs, and three
seeds. The values are test macro-F1 means with one standard deviation.

.. list-table:: MedMNIST Test Macro-F1
   :header-rows: 1
   :widths: 24 38 38

   * - Dataset
     - MLR
     - QORC
   * - OCT
     - 0.247 ± 0.001
     - 0.321 ± 0.010
   * - OrganS
     - 0.405 ± 0.004
     - 0.537 ± 0.003
   * - OrganA
     - 0.604 ± 0.001
     - 0.722 ± 0.000
   * - Derma
     - 0.218 ± 0.039
     - 0.400 ± 0.019

Source-noise and sparse-data studies
------------------------------------

The source-indistinguishability sweep evaluates three-photon QORC on the full
MNIST train and test sets. The sparse-data experiment evaluates balanced subsets
from 100 to 60,000 images and compares ideal and noisy local simulations with an
optional QPU branch. These sweeps make the dependence on optical quality and
training-set size explicit instead of reporting only an ideal-simulator result.

Implementation Details
======================

Run the MedMNIST experiment from the reproduced-papers repository root:

.. code-block:: bash

   python implementation.py --paper QORC --config configs/QORC_medmnist.json

The source-noise and sparse-data experiments use
``configs/noisy_QORC_indistinguishability.json`` and
``configs/fig4_dataset_size_comparison.json``. The full sparse-data sweep is
computationally intensive because it evaluates 50 independently sampled subsets
at each configured training size.

Limitations
===========

* QPU execution is optional and requires Perceval credentials.
* The Fig. 5 image and output ordering are not specified by the paper; the
  reproduction uses a seeded image selection and lexicographic no-bunching order.
* Noisy simulations and QPU runs use finite shots and should be interpreted with
  their run configuration and random seed.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (QORC) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/QORC>`_

The repository README documents every experiment, configuration, output file,
and the relationship between this paper and the earlier QORC reproduction.

Citation
========

.. code-block:: bibtex

   @article{rambach2025photonic,
     title={Photonic Quantum-Accelerated Machine Learning},
     author={Rambach, Markus and Roy, Abhishek and Gilchrist, Alexei and Sakurai, Akitada and Munro, William J. and Nemoto, Kae and White, Andrew G.},
     journal={arXiv preprint arXiv:2512.08318},
     year={2025},
     doi={10.48550/arXiv.2512.08318}
   }
