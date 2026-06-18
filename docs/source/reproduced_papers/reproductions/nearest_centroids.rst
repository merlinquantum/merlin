:github_url: https://github.com/merlinquantum/merlin

===========================================================================
Nearest Centroid Classification on a Trapped Ion Quantum Computer
===========================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Nearest Centroid Classification on a Trapped Ion Quantum Computer

   **Authors**: Sonika Johri, Shantanu Debnath, Avinash Mocherla, Alexandros Singh, Anupam Prakash, Jungsang Kim, Iordanis Kerenidis

   **Published**: arXiv preprint (2020)

   **DOI**: `arXiv:2012.04145 <https://arxiv.org/abs/2012.04145>`_

   **Reproduction Status**: Complete

   **Reproducer**: Benjamin Stott (benjamin.stott@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/nearest_centroids_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This paper proposes a quantum algorithm for nearest centroid classification, a distance-based
machine learning method, and demonstrates it on an IonQ trapped-ion quantum processor. The
core contribution is a quantum inner product estimation circuit that encodes classical
vectors into quantum states using amplitude encoding, achieving an asymptotic speedup of
:math:`O(k \log d)` over the classical :math:`O(kd)` cost for :math:`d`-dimensional data
with :math:`k` classes.

The algorithm uses unary amplitude encoding with Reconfigurable Beam Splitter (RBS) gates
arranged in a logarithmic-depth binary tree structure. By measuring the probability of a
photon returning to its initial mode after encoding both a test point and a class centroid,
the circuit estimates their inner product, from which Euclidean distance is derived. The
paper validates this approach on synthetic, Iris, and MNIST datasets, demonstrating that the
quantum classifier matches classical accuracy under ideal conditions and characterises the
effect of hardware noise and post-selection error mitigation.

Significance
============

This paper is significant for the photonic quantum ML field because it provides one of the
first end-to-end experimental demonstrations of a quantum classifier on real quantum
hardware. It establishes a concrete data loading primitive - the RBS binary tree - that
encodes a :math:`d`-dimensional vector in :math:`O(\log d)` circuit depth using a single
photon, a technique that has since become foundational in linear optical quantum ML. The
post-selection error mitigation strategy described also serves as a template for handling
hardware noise in near-term photonic devices.

MerLin Implementation
=====================

MerLin's ``QuantumLayer`` is used as a drop-in quantum distance oracle inside scikit-learn's
``NearestCentroid`` classifier. The ``QuantumLayer`` operates in the unbunched single-photon
regime (``ComputationSpace.UNBUNCHED``) and returns measurement probabilities over basis
states (``MeasurementStrategy.probs()``). The RBS binary-tree circuit is constructed
programmatically from the input dimension, and the angle encoding for both the test vector
and the reference centroid is concatenated and passed as the ``input_parameters``. No
trainable parameters are used - the circuit is entirely data-driven.

A parallel Cirq-based reference implementation is included, enabling direct comparison
between MerLin and the paper's original ideal simulation.

Key Contributions
=================

**Quantum inner product estimation and nearest centroid classification**
  * We have implemented unary amplitude encoding of :math:`d`-dimensional vectors into
    :math:`d` photonic modes using a single photon and an RBS binary-tree circuit of depth
    :math:`O(\log d)`. We have shown that the inner product between two vectors can be
    estimated by measuring the probability of a photon returning to mode 0 after applying
    the loader for :math:`\mathbf{x}` followed by the inverse loader for
    :math:`\mathbf{y}`, and that Euclidean distance can be recovered from this estimate.
  * We have validated multiclass nearest centroid classification on synthetic Gaussian
    clusters, the Iris dataset (3 classes, 4 features), and MNIST (binary and multiclass
    configurations up to 4 classes), with PCA preprocessing to reduce high-dimensional
    inputs to a power-of-2 number of components as required by the RBS circuit structure.

**MerLin and Cirq parity**
  * We have demonstrated that MerLin's ``QuantumLayer`` produces results statistically
    identical to the Cirq reference simulator across all datasets and configurations.
    Differences between the two backends are within one standard deviation and attributable
    solely to shot noise, confirming correctness of the MerLin implementation.

Implementation Details
======================

The ``QuantumNearestCentroid`` classifier (Cirq backend) and ``MLQuantumNearestCentroid``
classifier (MerLin backend) both subclass scikit-learn's ``NearestCentroid``, overriding
only the distance metric:

.. code-block:: python

   from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
   from sklearn.neighbors import NearestCentroid
   from lib.utils import create_circuit, get_angles
   import torch, numpy as np

   n = 8  # number of modes (must be power of 2)
   circuit = create_circuit(n)  # RBS binary-tree circuit

   layer = QuantumLayer(
       input_size=2 * (n - 1),
       circuit=circuit,
       trainable_parameters=[],
       input_parameters=["theta"],
       input_state=[1] + [0] * (n - 1),
       computation_space=ComputationSpace.UNBUNCHED,
       measurement_strategy=MeasurementStrategy.probs(),
   )

   class MLQuantumNearestCentroid(NearestCentroid):
       def get_metric(self, x, y):
           norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
           angles = torch.cat((2 * get_angles(x), -2 * get_angles(y).flip(0)))
           inner = layer(angles, shots=1000, sampling_method="multinomial").sqrt()[0]
           return np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner.item())

       def __init__(self, n=8, repetitions=500):
           self.layer = layer
           super().__init__(metric=self.get_metric)

Before each experiment, data is preprocessed through a fixed pipeline: optional stratified
subsampling, a 50/50 stratified train/test split, PCA reduction to ``n_components``, and
MinMax scaling to [0, 1]. The classical ``NearestCentroid`` baseline uses PCA features
without scaling, consistent with the paper.

Experimental Results
====================

**Synthetic Data (Paper Figure 8)**

Gaussian clusters generated on the unit sphere with controlled centroid separation
(``min_centroid_distance = 0.3``, ``gaussian_variance = 0.05``), 10 points per cluster,
averaged over 10 repeated splits.

.. list-table::
   :header-rows: 1
   :widths: 20 12 20 20 20

   * - Config
     - Shots
     - Classical
     - Cirq
     - MerLin
   * - Nq=4, Nc=2
     - 100
     - 100.0 %
     - 100.0 %
     - 99.0 %
   * - Nq=4, Nc=4
     - 500
     - 94.5 %
     - 94.5 %
     - 93.5 %
   * - Nq=8, Nc=2
     - 1000
     - 85.0 %
     - 79.0 %
     - 79.0 %
   * - Nq=8, Nc=4
     - 1000
     - 94.0 %
     - 93.0 %
     - 92.5 %

**IRIS Dataset (Paper Figure 9)**

Fisher Iris dataset (150 samples, 4 features, 3 classes). No PCA required since the data
is already 4-dimensional (Nq=4). Shot count is swept to illustrate the effect of
measurement statistics on accuracy.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 25

   * - Shots
     - Classical
     - Cirq
     - MerLin
   * - 100
     - 93.5 %
     - 91.7 %
     - 91.9 %
   * - 500
     - 93.5 %
     - 91.5 %
     - 91.9 %
   * - 1000
     - 93.5 %
     - 92.4 %
     - 92.0 %

**MNIST Dataset (Paper Figure 11)**

MNIST images reduced from 784 pixels to 8 PCA components (Nq=8), 1000 shots.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Task
     - Classical
     - Cirq
     - MerLin
   * - Digits 0 vs 1
     - 99.0 %
     - 98.8 %
     - 99.5 %
   * - Digits 2 vs 7
     - 94.2 %
     - 93.7 %
     - 95.5 %
   * - 4-class (digits 0-3)
     - 87.9 %
     - 88.8 %
     - 88.2 %

Code Access and Documentation
=============================

**GitHub Repository**: `merlin/reproductions/nearest_centroids <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/nearest_centroids_merlin>`_



Extensions and Future Work
==========================

The MerLin implementation extends beyond the original paper:

**Enhanced Capabilities**
  * Both a Cirq reference backend and a MerLin ``QuantumLayer`` backend are provided,
    enabling direct numerical comparison of the two frameworks on identical inputs.
  * Per-experiment prediction arrays are saved to JSON, supporting fine-grained analysis
    of agreement between backends at the individual sample level.
  * Hyperparameters (``n_shots``, ``n_repeats``, ``n_components``, ``max_samples``) are
    fully configurable via JSON config files without code changes.

**Experimental Extensions**
  * Shot-count sweep on the Iris dataset (100, 500, 1000 shots) extends the paper's single
    reported configuration.
  * 4-class MNIST extends the binary tasks in the original paper.
  * Synthetic data generation with controllable cluster separation and Gaussian variance
    allows systematic study of how data geometry affects quantum vs. classical accuracy.

**Hardware Considerations**
  * All experiments run on CPU with no GPU requirement.
  * The 10-class MNIST experiment (400 samples x 1000 shots x 10 repeats) is
    computationally expensive and is disabled by default (``RUN_10CLASS = False``).
  * The ``error_rate`` and ``error_mitigation`` parameters in both classifiers are ready
    for hardware noise experiments without code changes.

**Future Work**
  * Noise-aware experiments building on the error model in ``lib/noise.py`` would
    complement the hardware results in the original paper and validate the post-selection
    error mitigation strategy against a realistic noise model.

Citation
========

.. code-block:: bibtex

   @article{johri2020nearest,
     title={Nearest Centroid Classification on a Trapped Ion Quantum Computer},
     author={Johri, Sonika and Debnath, Shantanu and Mocherla, Avinash and Singh, Alexandros
             and Prakash, Anupam and Kim, Jungsang and Kerenidis, Iordanis},
     journal={arXiv preprint arXiv:2012.04145},
     year={2020}
   }
