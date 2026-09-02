:github_url: https://github.com/merlinquantum/merlin

===========================================================
RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks
===========================================================

.. admonition:: Paper Information
   :class: note

   **Title**: RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks: Architecture, Representation, and Hardware Validation

   **Authors**: Abdallah Aaraba, Alexis Vieloszynski, Remon Polus, Soumaya Cherkaoui, Ola Ahmad

   **Published**: IEEE Quantum Week (2026)

   **DOI**: `10.48550/arXiv.2607.13897 <https://doi.org/10.48550/arXiv.2607.13897>`_

   .. merlin-citations-badge:: rf_rqks

   **Paper URL**: `arXiv:2607.13897 <https://arxiv.org/abs/2607.13897>`_

   **Reproduction Status**: ⚠️ Partial — evaluated on one LTE band

   **Reproducer**: MerLin reproduced-papers contributors

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/rf_rqks_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This reproduction implements a binary radio-frequency anomaly detector based
on random quantum kitchen sinks. Measured LTE IQ captures provide the normal
signal, while chirp, barrage-jamming, and frequency-hopping anomalies are
injected at controlled jammer-to-signal ratios. Low-frequency DCT coefficients
from the resulting spectrograms are processed by photonic MerLin or gate-based
Qiskit feature maps and classical readouts.

Significance
============

The work connects quantum random features to a hardware-relevant signal
processing task. It also provides an end-to-end reproducibility boundary: raw
capture handling, anomaly synthesis, spectrogram construction, representation
caching, leakage-safe splitting, ablation studies, and optional QPU execution
are all explicit.

MerLin Implementation
=====================

The five-stage ablation studies:

1. Mode count, episode count, and entanglement.
2. Circuit depth for the Stage 1 shortlist.
3. Matched depth and episode count.
4. Photon count.
5. Held-out direct-readout versus quantum-kitchen-sink comparison.

The photonic implementation emits probability features from MerLin circuits.
A simulator-only Qiskit implementation follows the paper's ``RX``/``RY`` upload
layers and optional CZ ring, allowing the same protocol to be compared across
photonic and gate-model samplers.

Dataset and Representation
==========================

Each measured LTE segment contains 1,300,000 complex IQ samples at 61.44 MHz.
The reproduction pairs the unchanged segment with an anomalous version and
uses disjoint source segments for training and testing. A Hann-window STFT with
a 3,250-point window, 8,192-point FFT, and 25% overlap is cropped to 48 MHz and
resized to 400 by 400 pixels.

The model input is the flattened 64 by 64 low-frequency block of an orthonormal
two-dimensional DCT. Standardization is fitted on training rows only. Grouped
splits keep translated copies of a source pair in the same partition.

Experimental Results
====================

The complete WASD IQ collection is larger than 200 GB and was not downloaded
for this reproduction. Reported results use the independently downloaded
``36_LTE_1`` subset.

.. list-table:: Dataset Coverage
   :header-rows: 1
   :widths: 34 22 22 22

   * - Dataset
     - Source pairs
     - Labelled rows
     - Independent captures
   * - Reproduced ``36_LTE_1`` subset
     - 505
     - 1,010
     - 505
   * - Paper-sized target
     - 14,862
     - 29,724
     - Not available locally

Direct readouts remain competitive with the quantum features on the held-out
single-band test split. The best Qiskit readout is a linear SVM, while the
approximate-kernel readout is close to chance level. These values are not
directly comparable with the full multi-band paper experiment.

Implementation Details
======================

After generating the measured-data representation, run the reduced photonic
ablation from the reproduced-papers repository root:

.. code-block:: bash

   python implementation.py --paper RF-RQKS --config papers/RF-RQKS/configs/ablation_36_lte_1.json

A no-config invocation runs a deterministic synthetic-data integration check.
It verifies the pipeline only and is not used for reported results.

Limitations
===========

* The available dataset covers one LTE band and has about 29.4 times fewer
  labelled rows than the paper-sized target.
* Optional augmentation matches row counts but cannot create new independent
  measurements or recover multi-band diversity.
* Several anomaly-duration and resize distributions are unspecified by the
  paper; the reproduction exposes and records those choices.
* Photonic sweeps and QPU execution can be computationally expensive.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (RF-RQKS) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/RF-RQKS>`_

Citation
========

.. code-block:: bibtex

   @article{aaraba2026rf,
     title={RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks: Architecture, Representation, and Hardware Validation},
     author={Aaraba, Abdallah and Vieloszynski, Alexis and Polus, Remon and Cherkaoui, Soumaya and Ahmad, Ola},
     journal={arXiv preprint arXiv:2607.13897},
     year={2026},
     doi={10.48550/arXiv.2607.13897}
   }
