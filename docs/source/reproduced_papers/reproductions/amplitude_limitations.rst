:github_url: https://github.com/merlinquantum/merlin

============================================================
Limitations of Amplitude Encoding on Quantum Classification
============================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Limitations of Amplitude Encoding on Quantum Classification

   **Authors**: Xin Wang, Yabo Wang, Bo Qi, Rebing Wu

   **Published**: ([2025])

   **DOI**: `https://arxiv.org/abs/2503.01545 <https://arxiv.org/abs/2503.01545>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Louis-Félix Vigneux (louis-felix.vigneux@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/amplitude_limitations_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

This paper prooves mathematically the main limitations of the amplitude encoding paradigm is QML. Indeed, the authors introduce the concentraion phenomena of encoded states that tend towards a loss plateau. The authors then proove on simple synthtic datasets that the limitations are also showing experimentally. To show that those cases are not just outliers, the paper shows that for known datasets such as CIFAR-10, EURO-SAT and MedMNIST, the same limitations occur.

Significance
============

This paper is significant as the QML community mostly sees amplitude encoding as the most efficient and useful encoding strategy. Indeed, it is a strategy that can encode classical data in very few qubits. Although, there is not much discussion on the impact normalization factor in amplitude encoding that greatly limits its capabilities. It is important for the community to question the choice of embedding if a quantum model does not perform as well as they thought.

MerLin Implementation
=====================

MerLin is used to create small and easy to train model for the experimental part of the paper. 

Key Contributions Reproduced
============================

**Limitations of amplitude encoding**
  * Three simple datasets are shown to have the same amplitude encoded average state
  * These datasets are then classified with a simple amplitude encoding model.

**Classify known datasets**
  * Four known datasets ((MNIST, EuroSAT, CIFAR-10 and PathMNIST)) are shown to have the same amplitude encoded average state
  * These datasets are then classified with QCNN amplitude encoding bases-models to show that the limitations are also visible in practical cases.


Implementation Details
======================

We use the ``CircuitBuilder`` to create small quantum models easily with amplitude encoding.

.. code-block:: python

   import merlin as ml

    circuit = ml.CircuitBuilder(n_modes=n_modes)
    for _ in range(num_layers):
        if n_modes == 1:
            circuit.add_rotations(trainable=True)
        else:
            circuit.add_entangling_layer()
    qlayer = ml.QuantumLayer(
        builder=circuit,
        amplitude_encoding=True,
        n_photons=n_photons,
        measurement_strategy=measurement_strategy,
    )

Experimental Results
====================

To see the result plots, consult the  `ReadMe <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/AA_study/README.md>`_ of the project.

Interactive Exploration
=======================

**Jupyter Notebook**: :doc:`../../notebooks/reproduced_papers/amplitude_limitations_tutorial`

The notebook provides shows that even for simple datasets, amplitude encoding is not always the best suited encoding strategy.

Extensions and Future Work
==========================

This paper sparked the interest to explore more encoding strategies and find their strengths and limitations so that MerLin users can choose the encoding that is the most appropriate for them.

Code Access and Documentation
=============================

**GitHub Repository**: `merlin/reproductions/AA_study <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/AA_study>`_

The complete implementation includes:

* The simple synthetic datasets proving the limitations of amplitude encoding.
* For those datasets, classification tasks on a amplitude and angle encoding based models.
* A gate based and photonic based QCNN used to classify known datasets (MNIST, EuroSAT, CIFAR-10 and PathMNIST). They both use amplitude encoding.

Citation
========

.. code-block:: bibtex

  @misc{wang_limitations_2025,
    title = {Limitations of {Amplitude} {Encoding} on {Quantum} {Classification}},
    url = {http://arxiv.org/abs/2503.01545},
    doi = {10.48550/arXiv.2503.01545},
    abstract = {It remains unclear whether quantum machine learning (QML) has real advantages when dealing with practical and meaningful tasks. Encoding classical data into quantum states is one of the key steps in QML. Amplitude encoding has been widely used owing to its remarkable efficiency in encoding a number of \$2{\textasciicircum}\{n\}\$ classical data into \$n\$ qubits simultaneously. However, the theoretical impact of amplitude encoding on QML has not been thoroughly investigated. In this work we prove that under some broad and typical data assumptions, the average of encoded quantum states via amplitude encoding tends to concentrate towards a specific state. This concentration phenomenon severely constrains the capability of quantum classifiers as it leads to a loss barrier phenomenon, namely, the loss function has a lower bound that cannot be improved by any optimization algorithm. In addition, via numerical simulations, we reveal a counterintuitive phenomenon of amplitude encoding: as the amount of training data increases, the training error may increase rather than decrease, leading to reduced decrease in prediction accuracy on new data. Our results highlight the limitations of amplitude encoding in QML and indicate that more efforts should be devoted to finding more efficient encoding strategies to unlock the full potential of QML.},
    publisher = {arXiv},
    author = {Wang, Xin and Wang, Yabo and Qi, Bo and Wu, Rebing},
    month = mar,
    year = {2025},
    note = {arXiv:2503.01545 [quant-ph]},
    keywords = {Quantum Physics},
    annote = {Comment: 18 pages, 11 figures},
  }

Related Reproductions
=====================

This work uses another reproduction:

* **Photonic QCNN with Adaptive State Injection**: For the usual datasets, a QCNN is used. In the paper it is a gate based version. So, we used a photonic QCNN for our experiments. We used the one reproduced in this paper.

.. merlin-gallery::
   :data: _data/galleries/AA_study_gallery.json
