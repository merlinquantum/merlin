:github_url: https://github.com/merlinquantum/merlin

====================================================
Fock State-enhanced Expressivity of Quantum Machine Learning Models
====================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Fock State-enhanced Expressivity of Quantum Machine Learning Models

   **Authors**: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis

   **Published**: EPJ Quantum Technol. 9, 16 (2022)

   **DOI**: `10.1140/epjqt/s40507-022-00135-0 <10.1140/epjqt/s40507-022-00135-0>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Philippe Schoeb (philippe.schoeb@quandela.com)

Abstract
========

This paper proposes a data-encoding scheme that projects the classical data points into the high-dimensional Fock space using a photonic circuit. Their efficient methodology serves as foundational work for using photonic quantum systems for machine learning. In addition, it delves into the theory behind the expressivity of photonic quantum circuits and demonstrates their findings with experimental results.

Furthermore, it presents three different noisy photonic intermediate-scale binary classification methods.

Significance
============

This paper introduces a data-embedding process that is now commonly used in photonic quantum machine learning because it is more effective than the previously used methods. It also showcases the expressivity of photonic quantum circuits that is dependent on the number of photons used.

MerLin Implementation
=====================

In brief, MerLin is used in our implementations to define quantum layers that are part of a hybrid network model for which PyTorch is used to define the classical layers.

Key Contributions
=================

**Display the expressive power of the variational linear quantum photonic circuit**
  * We have shown that the expressive power of a VQC depends on the number of photons used. More specifically, we have obtained that a VQC using 1 photon (initial state : [1, 0, 0]) cannot fit a degree 3 Fourier series. Then, we have that a VQC with 2 photons (intial state : [1, 1, 0]) can reach a better fit but does not have expressive power needed to do so perfectly. Finally, using 3 photons (intial state : [1, 1, 1]), the VQC can fit the target function without problem.

**Usage of linear quantum photonic circuits as a variational quantum classifiers**
  * We have used a VQC for binary classification on three different datasets and we have reached decision boundaries very similar to the ones found in the paper.
  * Our results also showcase the increase in model expressivity when increasing the number of input photons. This led to a better perfomance on the moon dataset but it also led to a decrease in performance on the ciruclar dataset because of overfitting.
  * Finally, our implementation allows the comparison of hybrid models with classical models of comparable size.

**Usage of linear quantum photonic circuits as Gaussian kernel samplers**
  * We have used 2 modes circuits with varying number of photons to fit Gaussian kernel functions with various standard deviations.
  * Our results showcase once again the larger expressivity of photonic circuits with more photons. Although, our results are not as precise as the ones presented by the original paper.
  * Finally, we have used our trained Gaussian kernel samplers for a binary classification task on three different datasets.

**Usage of quantum-enhanced random kitchen sinks**
  * We have used two modes circuits to approximate the random kitchen sinks algorithm with many tunable hyperparameters.
  * Namely, the number of photons used is changeable, the hybrid model can be used with or without training on a fitting task using data from the moon dataset or using generated data.
  * The classical random kitchen sinks algorithm was also implemented to allow direct comparison.

Implementation Details
======================

The key role of MerLin in our implementations is to give us access to trainable quantum layers. Then, we can include this quantum layer in our model using torch.nn.Sequential.

.. code-block:: python

   import merlin as ml

   vqc = ml.QuantumLayer(
                input_size=1,
                output_size=1,
                circuit=create_vqc_general(3, 1),
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state= initial_state,
                no_bunching=False,
                output_mapping_strategy=OutputMappingStrategy.LINEAR,
            )
   # Assemble with a previously defined scale_layer
   model = nn.Sequential(scale_layer, vqc)

Interactive Exploration
=======================

**Jupyter Notebooks**:

:doc:`../../notebooks/GanEtAl_VQC_fourier_series`

This notebook provides a tutorial on how to train a variational quantum circuit to fit a 1D Fourier series.

:doc:`../../notebooks/GanEtAl_VQC_classification`

This notebook provides a tutorial on how to use a photonic quantum circuit for binary classification tasks. It is also a good environment for you to experiment by varying several hyperparameters.

:doc:`../../notebooks/GanEtAl_q_gaussian_kernel`

This notebook provides a tutorial on how to use a photonic quantum circuit as a Gaussian kernel sampler. It also provides the possibility of putting to the test your trained Gaussian kernel sampler on three basic classifying tasks and observe the results.

:doc:`../../notebooks/GanEtAl_q_random_kitchen_sinks`


This notebook provides a tutorial on how to use a photonic quantum circuit to approximate the random kitchen sinks algorithm with or without training. It also allows you to experiment by varying several hyperparameters.

Extensions and Future Work
==========================

The MerLin implementation extends beyond the original paper:

**Enhanced Capabilities**
  * Usage of PyTorch for model optimization
  * Easier quantum model definition

**Experimental Extensions**
  * Comparison of the VQC results on the linear; circular and moon datasets with classical models such as an MLP and a SVM.
  * Usage of the trained quantum Gaussian kernel sampler on three binary classifying tasks to see just how accurate it is.
  * Comparison of the quantum-enhanced random kitchen sinks with its classical counter part.
  * Hyperparameter exploration was conducted and is possible with the code provided which is essential for a thorough analysis.

**Hardware Considerations**
  * Every experiment from this section can and has been designed to be run on a CPU.

**Future work**
  * For the experiement on Algorithm 2: **Linear quantum photonic circuits as Gaussian kernel samplers**, further investigation is needed to fully understand why the obtained fits are significantly less accurate than the ones presented in the reference paper and to solve this issue.

Citation
========

.. code-block:: bibtex

   @article{Gan_2022,
   title={Fock state-enhanced expressivity of quantum machine learning models},
   volume={9},
   ISSN={2196-0763},
   url={http://dx.doi.org/10.1140/epjqt/s40507-022-00135-0},
   DOI={10.1140/epjqt/s40507-022-00135-0},
   number={1},
   journal={EPJ Quantum Technology},
   publisher={Springer Science and Business Media LLC},
   author={Gan, Beng Yee and Leykam, Daniel and Angelakis, Dimitris G.},
   year={2022},
   month=jun }

----
