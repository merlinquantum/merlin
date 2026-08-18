:github_url: https://github.com/merlinquantum/merlin

=================
Reproduced Papers
=================

MerLin provides reproducible implementations of published quantum machine learning papers.
Each card links to a dedicated reproduction page with paper metadata, implementation details, code access, and results.

Research Impact
---------------

MerLin's reproductions span published quantum machine learning research; the
figures below summarize how many papers have been reproduced and how widely
those papers have been cited.

.. merlin-citations-summary::

Available Reproductions
-----------------------

The reproductions are organized by topic. Each card opens the corresponding paper-reproduction page.

Kernel Methods
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/photonic_kernel
   reproductions/nearest_centroids

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_kernel_methods.json
   :columns: 3
   :contour-color: #5648ED

For a Better Understanding of Photonic QML Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/fock_state_expressivity
   reproductions/hqnn-myth
   reproductions/data_reuploading
   reproductions/amplitude_limitations
   reproductions/nqe
   reproductions/BP_QNN

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_variational_methods.json
   :columns: 3
   :contour-color: #5648ED

Computer Vision
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/quantum_reservoir_computing
   reproductions/photonic_qcnn
   reproductions/QCNN_data_classification

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_computer_vision.json
   :columns: 3
   :contour-color: #5648ED


Sequential Tasks
~~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_sequential.json
   :columns: 3
   :contour-color: #5648ED

Generative Algorithms
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/photonic_QGAN
   reproductions/LatentQGAN

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_generative_algorithms.json
   :columns: 2
   :contour-color: #5648ED

Advanced Training Paradigms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/qllm_finetuning
   reproductions/qssl
   reproductions/quantum_adversarial_ml
   reproductions/hqpinn
   reproductions/quantum_transfer_learning

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_advanced_training.json
   :columns: 3
   :contour-color: #5648ED

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/distributed_nn

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_distributed_training.json
   :columns: 2
   :contour-color: #5648ED

Future-proofing
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/photonic_memristor

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_future_proofing.json
   :columns: 2
   :contour-color: #5648ED

Reproductions by Citations
--------------------------

The papers reproduced in MerLin, ordered by citation count. Each row links to
the reproduction page; the arrow opens the original paper.

.. merlin-citations-table::

Contributing Reproductions
--------------------------

We welcome contributions of additional paper reproductions.

**Requirements**:

* High-impact quantum ML papers (>50 citations preferred)
* Photonic/optical quantum computing focus
* Implementable with current MerLin features
* Clear experimental validation

**Submission Process**:

1. **Propose** the paper in our `GitHub Discussions <https://github.com/merlinquantum/merlin/discussions>`_
2. **Implement** using MerLin following our guidelines
3. **Validate** results against original paper
4. **Document** in Jupyter notebook format
5. **Submit** via pull request a complete reproduction folder and a summary page in :code:`docs/source/reproduced_papers/reproductions/` directory

**Mandatory Structure for a Reproduction**:

.. code-block:: text

   papers/NAME/            # Non-ambiguous acronym or fullname of the reproduced paper
   ├── .gitignore          # specific .gitignore rules for clean repository
   ├── notebook.ipynb      # Interactive exploration of key concepts
   ├── README.md           # Paper overview and results overview
   ├── requirements.txt    # additional requirements for the scripts
   ├── configs/            # defaults + CLI/runtime descriptors consumed by the repo root runner
   ├── lib/                # code used by the shared runner and notebooks - as an integrated library (import shared data helpers from papers/shared/<paper>/)
   ├── models/             # Trained models
   ├── results/            # Selected generated figures, tables, or outputs from trained models
   ├── tests/              # Validation tests
   └── utils/              # additional commandline utilities for visualization, launch of multiple trainings, etc...

**Template Summary Page**: :doc:`this document <reproductions/template>`

.. toctree::
   :maxdepth: 1
   :hidden:

   reproductions/template

Recognition
-----------

Contributors to reproductions are recognized in:

* Paper reproduction documentation
* MerLin project contributors list
* Academic citations in MerLin publications

*Have a paper you'd like to see reproduced?* `Start a discussion <https://github.com/merlinquantum/merlin/discussions/new>`_.
