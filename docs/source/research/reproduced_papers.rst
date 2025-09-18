:github_url: https://github.com/merlinquantum/merlin

=================
Reproduced Papers
=================

MerLin enables researchers to reproduce and build upon published quantum machine learning research.
This section provides implementations of key papers in the quantum ML field, complete with working code, analysis, and extensions.

Overview
========

Each reproduction may include:

* **Original paper implementation** - Faithful recreation of the paper's methodology
* **Reproduction status** - Indicating whether the reproduction is partial or complete
* **Jupyter notebooks** - Interactive exploration of results and concepts
* **Full code** - Available on GitHub for easy access and modification
* **Performance analysis** - Comparison with paper results
* **Extension opportunities** - Ideas for building upon the work

.. note::
   All reproductions are implemented using MerLin's high-level API, making them accessible to ML practitioners without deep quantum expertise.

Available Reproductions
=======================

.. list-table::
   :header-rows: 1
   :widths: 40 25 20 15 60

   * - Paper Title
     - Authors
     - Year
     - Status
     - Description/Category
   * - :doc:`reproductions/fock_state_expressivity`
     - Gan et al.
     - 2021
     - Complete
     - Foundational work on photonic circuit architectures
   * - :doc:`reproductions/quantum_reservoir_computing`
     - Sakurai et al.
     - 2025
     - In Progress
     - Boson sampling for quantum reservoir computing
   * - :doc:`reproductions/qllm_finetuning`
     - Jern et al.
     - 2025
     - In Progress
     - Fine Tuning a LLM with a photonic neural network
   * - :doc:`reproductions/photonic_qcnn`
     - Monbroussou et al.
     - 2025
     - Complete
     - Photonic vision model

Contributing Reproductions
==========================

We welcome contributions of additional paper reproductions!

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
5. **Submit** via pull request a complete reproduction folder and a summary page in :code:`docs/source/reproductions/` directory

**Template Structure**:

.. code-block:: text

   paper_reproduction/
   ├── README.md             # Paper overview and results
   ├── implementation.py     # Core implementation
   ├── notebook.ipynb        # Interactive exploration showing the key concepts, not necessarily the full implementation
   ├── data/                 # Datasets and preprocessing
   ├── results/              # Figures and analysis
   └── tests/                # Validation tests

**Template Summary Page**: :doc:`this document <reproductions/template>`

Recognition
-----------

Contributors to reproductions are recognized in:

* Paper reproduction documentation
* MerLin project contributors list
* Academic citations in MerLin publications

Upcoming Reproductions
======================

**Near-term (Q2 2025)**:
  *Currently accepting proposals*

**Medium-term (Q3-Q4 2025)**:
  *Community voting in progress*

**Community Requested**:
  Vote on upcoming reproductions in our `paper requests discussions <https://github.com/merlinquantum/merlin/discussions/categories/paper-requests>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reproduced Papers

   reproductions/fock_state_expressivity
   reproductions/quantum_reservoir_computing
   reproductions/template
   reproductions/qllm_finetuning

----

*Have a paper you'd like to see reproduced?* `Start a discussion <https://github.com/merlinquantum/merlin/discussions/new>`_ *and let us know!*
