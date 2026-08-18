:github_url: https://github.com/merlinquantum/merlin

==================================================================================
Experimental data re-uploading with provable enhanced learning capabilities
==================================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Experimental data re-uploading with provable enhanced learning capabilities

   **Authors**: Martin F. X. Mauser, Solène Four, Lena Marie Predl, Riccardo Albiero, Francesco Ceccarelli, Roberto Osellame, Philipp Petersen, Borivoje Dakić, Iris Agresti, and Philip Walther

   **Published**: arXiv preprint, 14 (2025)

   **DOI**: `https://doi.org/10.48550/arXiv.2507.05120 <https://doi.org/10.48550/arXiv.2507.05120>`_

   .. merlin-citations-badge:: data_reuploading

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Hugo Izadi (hugoizadi@gmail.com) and Philippe Schoeb (philippe.schoeb@quandela.com)

Project Repository
==================================================================================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/data_reuploading_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
==================================================================================

The reference paper's main contribution is to present a well performing and resource-efficient data re-uploading scheme on a photonic quantum processor. It showcases the model's performance on four datasets of increasing complexity. It also provides an analytical proof that the proposed model in a universal classifier.  

Significance
==================================================================================

This research is significant since it proposes a resource-efficient model that could reduce ernegy consumption for classifying tasks. In addition, energy efficiency is becoming an increasingly important argument for using quantum circuits in machine learning.

MerLin Implementation
==================================================================================

MerLin's QuantumLayer is used in our reproduction to define the optimizable photonic circuit that comprises almost all the proposed model.

Key Contributions Reproduced
==================================================================================

**Effect of increasing number of re-uploading layers on the circular dataset**
  * Three re-uploading layers are necessary to fully capture the expressivity of the circular dataset

**Experimental accuracies or the photonic data re-uploading model on four different datasets based on its number of layers**
  * Increasing number of layers increases the obtained train and test accuracies
  * Faster improvement of accuracy when increasing number of layers on the Tetromino dataset 
  * Slower improvement of accuracy when increasing number of layers on the circular dataset

Implementation Details
==================================================================================

The trainable photonic circuit used to encode data is defined with MerLin: 

.. code-block:: python

   import merlin as ml

   quantum_layer = ml.QuantumLayer(
       input_size=self.dimension,
       circuit=circuit_model.circuit,
       trainable_parameters=["var"],
       input_parameters=["x"],
       input_state=circuit_model.input_state,
       measurement_strategy=ml.MeasurementStrategy.probs(),
   )

Extensions and Future Work
==================================================================================

The MerLin implementation extends beyond the original paper:

**Circuit architecture benchmark**
  * Three different circuit schemes were explored for data encoding and for training
  * Design C is clearly defective

**Hyperparameter grid search**
  * Exploration for different values of tau (Fisher loss temperature) and alpha (phase scaling)
  * Trade off between too simple and too complex decision boundaries

**Hardware Considerations**
  * Every experiment done for this reproduction has been designed for simulation on CPU

**Future work**
  * Extend to more complex datasets to see the limit of this architecture (if there is any)
  * Compare with classical baseline models
  * Deploy on quantum hardware

Code Access and Documentation
==================================================================================

**GitHub Repository**: `merlin/reproductions/data_reuploading <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/data_reuploading>`_

The complete implementation includes:

* Model implementation
* Testing on the four dataset used
* Two additional benchmarking
* A notebook to explore with the proposed architecture in different settings

Citation
==================================================================================

.. code-block:: bibtex

   @misc{mauser2025experimentaldatareuploadingprovable,
      title={Experimental data re-uploading with provable enhanced learning capabilities}, 
      author={Martin F. X. Mauser and Solène Four and Lena Marie Predl and Riccardo Albiero and Francesco Ceccarelli and Roberto Osellame and Philipp Petersen and Borivoje Dakić and Iris Agresti and Philip Walther},
      year={2025},
      eprint={2507.05120},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2507.05120}, 
   }
