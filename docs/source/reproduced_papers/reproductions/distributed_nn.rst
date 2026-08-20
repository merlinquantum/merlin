:github_url: https://github.com/merlinquantum/merlin

====================================================================================
Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing
====================================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing

   **Authors**: Kuan-Cheng Chen, Chen-Yu Liu, Yu Shang, Felix Bur, Kin K. Leung 

   **Published**: IEEE Future Networks Webinar (2025)

   **DOI**: `[https://arxiv.org/abs/2505.08474] <[https://arxiv.org/abs/2505.08474]>`_

   .. merlin-citations-badge:: distributed_nn

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Louis-Félix Vigneux (louis-felix.vigneux@quandela.com)

Project Repository
====================================================================================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/distributed_nn_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
====================================================================================

This paper presents a novel way to do use a QPU to do machine learning. Indeed, with the use of distributed learning, boson samplers are used as advanced compression techniques for the parameters of a CNN. Indeed with the use of a simple MPS layer to map the quantum output to parameters, the boson samplers are trained to generate all of the parameters of the classical CNN. The model is used to classify the full 10 MNIST digits.

The full pipeline is the following:

#. Boson samplers
#. A MPS taking the probability distribution from the boson samplers and map it to the correct number of classical parameters
#. The classical model updates the parameters from the values given by the MPS

The data to classify only goes through the classical model that is not trained directly. Only the boson samplers and the MPS is trained to generate the correct parameters.

The paper also compares the performance of the quantum model with classical parameter compression techniques to show the utility of quantum.

Significance
====================================================================================

Most papers in QML use the QPU as a main part in the model classifying the data. Here, the boson samplers do not treat the data directly as it only trains the parameters of the CNN that actually classifies the data. This paper is significant since it is a way to observe the advantages of quantum (few quantum parameters to train to generate exponential classical parameters) while using all of the benefits of big classical models. Also, only small boson samplers are needed, which is useful for the current limited access to quantum resources.

MerLin Implementation
====================================================================================

MerLin is used to instantiate the boson samplers that generate the classical data. It also allows us to use faster gradient-based optimization instead of the COBYLA optimizer used in the paper.

Key Contributions Reproduced
====================================================================================

**Investigate the relation between the bond dimension of the MPS and the CNN's accuracy**
  * We classify a subset of the MNIST dataset and vary the MPS' bond dimension (1 to 10) to increase the representability of the model and the number of parameters.
  * Our results show that the bigger the bond dimension is, better accuracy is obtained. Although, a small bond dimension of 4 gives very good results while only needing 688 parameters to train instead of the full 6690 of the classical model.
  * Our implementation, using gradient based methods is much faster than the original implementation.

**Comparing the quantum algorithm to classical parameter compression techniques**
  * We compare the full boson sampler model with classical parameter compression techniques such as weight sharing and pruning.
  * From the plotted results, we see that the quantum method uses less parameters than the classical methods while showing better accuracy overall.


Implementation Details
====================================================================================

The key role of MerLin is to train efficiently the boson samplers. We use a QuantumLayer to do so.

.. code-block:: python

   import merlin as ml

   model = ml.QuantumLayer(
      input_size=0,
      n_photons= n_photons,
      circuit= circuit,
   )

Where the `circuit` is a `Perceval` circuit implementing the interferometer described in the paper.

Experimental Results
====================================================================================

To see the result plots, consult the  `ReadMe <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/DQNN/README.md>`_ of the project.


Code Access and Documentation
====================================================================================

**GitHub Repository**: `merlin/reproductions/DQNN <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/DQNN/>`_

The complete implementation includes:

* The bond dimension comparison experiment
* The parameter compression comparison experiment
* An ablation study
* A tutorial on how to use the code

Citation
====================================================================================
.. code-block:: bibtex

  @misc{chen_distributed_2025,
      title = {Distributed {Quantum} {Neural} {Networks} on {Distributed} {Photonic} {Quantum} {Computing}},
      url = {http://arxiv.org/abs/2505.08474},
      doi = {10.48550/arXiv.2505.08474},
      abstract = {We introduce a distributed quantum-classical framework that synergizes photonic quantum neural networks (QNNs) with matrix-product-state (MPS) mapping to achieve parameter-efficient training of classical neural networks. By leveraging universal linear-optical decompositions of \$M\$-mode interferometers and photon-counting measurement statistics, our architecture generates neural parameters through a hybrid quantum-classical workflow: photonic QNNs with \$M(M+1)/2\$ trainable parameters produce high-dimensional probability distributions that are mapped to classical network weights via an MPS model with bond dimension \$χ\$. Empirical validation on MNIST classification demonstrates that photonic QT achieves an accuracy of \$95.50{\textbackslash}\% {\textbackslash}pm 0.84{\textbackslash}\%\$ using 3,292 parameters (\$χ= 10\$), compared to \$96.89{\textbackslash}\% {\textbackslash}pm 0.31{\textbackslash}\%\$ for classical baselines with 6,690 parameters. Moreover, a ten-fold compression ratio is achieved at \$χ= 4\$, with a relative accuracy loss of less than \$3{\textbackslash}\%\$. The framework outperforms classical compression techniques (weight sharing/pruning) by 6--12{\textbackslash}\% absolute accuracy while eliminating quantum hardware requirements during inference through classical deployment of compressed parameters. Simulations incorporating realistic photonic noise demonstrate the framework's robustness to near-term hardware imperfections. Ablation studies confirm quantum necessity: replacing photonic QNNs with random inputs collapses accuracy to chance level (\$10.0{\textbackslash}\% {\textbackslash}pm 0.5{\textbackslash}\%\$). Photonic quantum computing's room-temperature operation, inherent scalability through spatial-mode multiplexing, and HPC-integrated architecture establish a practical pathway for distributed quantum machine learning, combining the expressivity of photonic Hilbert spaces with the deployability of classical neural networks.},
      publisher = {arXiv},
      author = {Chen, Kuan-Cheng and Liu, Chen-Yu and Shang, Yu and Burt, Felix and Leung, Kin K.},
      month = may,
      year = {2025},
      note = {arXiv:2505.08474 [quant-ph]},
      keywords = {Computer Science - Artificial Intelligence, Computer Science - Distributed, Parallel, and Cluster Computing, Quantum Physics},
  }
