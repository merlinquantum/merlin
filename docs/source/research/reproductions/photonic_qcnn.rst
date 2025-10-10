:github_url: https://github.com/merlinquantum/merlin

====================================================
Photonic Quantum Convolutional Neural Networks with Adaptive State Injection
====================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Photonic Quantum Convolutional Neural Networks with Adaptive State Injection

   **Authors**: Léo Monbroussou, Beatrice Polacchi, Verena Yacoub, Eugenio Caruccio, Giovanni Rodari, Francesco Hoch, Gonzalo Carvacho, Nicolò Spagnolo, Taira Giordani, Mattia Bossi, Abhiram Rajan, Niki Di Giano, Riccardo Albiero, Francesco Ceccarelli, Roberto Osellame, Elham Kashefi, and Fabio Sciarrino

   **Published**: EPJ Quantum Technol. 9, 16 (2025)

   **DOI**: `https://doi.org/10.48550/arXiv.2504.20989`_

   **Reproduction Status**: ⏳ On hold

   **Reproducer**: Philippe Schoeb (philippe.schoeb@quandela.com) and Anthony Walsh (anthony.walsh@quandela.com)

Abstract
========

This paper presents a photonic vision architecture for quantum machine learning. This work proposes a novel approach to apply pooling within a photonic quantum convolutional neural network: state injection. This method allows to preserve the same number of photons throughout the whole circuit.

After presenting this model, this paper also presents its obtained results by running numerical simulations of this architecture on datasets of different sizes. Finally, it presents a comparison between the expected results versus the ones obtained by implementing the proposed architecture on hardware.

Significance
============

This paper introduces an innovative photonic vision model that is compatible with near-term quantum devices. In addition, it shows promise in terms of scalability, running time complexity and of the number of parameters needed with respect to other quantum neural networks for vision.

MerLin Implementation
=====================

Through this reproduction, MerLin was used to define each layer of the PQCNN as well as the complete model. This allows the user to use the torch optimization framework in order to optimize the parameters within the convolutional and dense layers of the quantum model.

Key Contributions
=================

**Reproduce the architecture proposed**
  * We have reproduced the described model very precisely, especially for specific architecture used during simulation on the three datasets.

**Simulation of the PQCNN on 3 different datasets**
  * We have run and optimized the model on the BAS, Custom BAS and binary MNIST datasets.
  * Using the provided code for simulation and with further hyperparameter exploration, we have been able to attain significantly better test accuracies than the ones reported on Custom BAS and MNIST.
  * Finally, the MerLin version of the PQCNN reaches equivalent accuracies through simulation.

**Testing of two measurement layer strategies**
  * Two fold readout: Our results with this strategy were much better than the ones reported, attaining near-perfect accuracy most of the time. To understand the reason behind this, more time would need to be inested in this.
  * Mode pair readout: With this approach, our results were equivalent to the ones obtained in the paper.

Implementation Details
======================

The key role of MerLin in our implementation, although it might be subtle, is to calculate the output density matrix or the output amplitudes of a Perceval circuit while updating the computational graph for backpropagation.

Experimental Results
====================

**PQCNN simulation results**

 *Using MerLin
+------------+----------------------+----------------+---------------+
| Dataset    | Number of Parameters | Train Accuracy | Test Accuracy |
+============+======================+================+===============+
| BAS        |           8          | 94.7 ± 1.0%    | 93.0 ± 1.1%   |
+------------+----------------------+----------------+---------------+
| Custom BAS |           8          | 98.4 ± 1.9%    | 98.2 ± 2.2%   |
+------------+----------------------+----------------+---------------+
| MNIST      |          30          | 99.7 ± 0.5%    | 98.8 ± 1.0%   |
+------------+----------------------+----------------+---------------+

 *Using their provided code
+------------+----------------------+----------------+---------------+
| Dataset    | Number of Parameters | Train Accuracy | Test Accuracy |
+============+======================+================+===============+
| BAS        |           8          | 92.5 ± 1.1%    | 93.1 ± 2.1%   |
+------------+----------------------+----------------+---------------+
| Custom BAS |           8          | 97.3 ± 1.6%    | 98.2 ± 2.0%   |
+------------+----------------------+----------------+---------------+
| MNIST      |          30          | 100.0 ± 0.0%   | 97.2 ± 1.3%   |
+------------+----------------------+----------------+---------------+

Interactive Exploration
=======================

**Jupyter Notebooks**:

:doc:`../../notebooks/photonic_QCNN`

This notebook not only defines all components of the PQCNN, it allows the user to explore with certain hyperparameters and optimize the model for binary classification on MNIST (0 vs 1). Furthermore, it then compares the performance with the one of a classical CNN.

Extensions and Future Work
==========================

The MerLin implementation extends beyond the original paper:

**Experimental Extensions**
  * Further hyperparameter optimization was conducted which led to a gain in test accuracy on Custom BAS and MNIST.
  * Note that the PQCNN implementation within the notebook and in the reproduced_papers repository cannot take multi-channel images but we have implemented a PQCNN which can handle these types of images.

**Hardware Considerations**
  * Every experiment from this section can and has been designed to be run on a CPU.

**Future work**
  * Benchmarking of the PQCNN model.
  * Hardware implementation of the PQCNN on Quandela's quantum device to compare its performance with expected results.


Citation
========

.. code-block:: bibtex

   @misc{monbroussou2025photonicquantumconvolutionalneural,
      title={Photonic Quantum Convolutional Neural Networks with Adaptive State Injection},
      author={Léo Monbroussou and Beatrice Polacchi and Verena Yacoub and Eugenio Caruccio and Giovanni Rodari and Francesco Hoch and Gonzalo Carvacho and Nicolò Spagnolo and Taira Giordani and Mattia Bossi and Abhiram Rajan and Niki Di Giano and Riccardo Albiero and Francesco Ceccarelli and Roberto Osellame and Elham Kashefi and Fabio Sciarrino},
      year={2025},
      eprint={2504.20989},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2504.20989},
}

----