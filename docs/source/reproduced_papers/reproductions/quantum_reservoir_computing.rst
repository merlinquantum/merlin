:github_url: https://github.com/merlinquantum/merlin

=============================================================
Quantum Optical Reservoir Computing Powered by Boson Sampling
=============================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum optical reservoir computing powered by boson sampling

   **Authors**: Akitada Sakurai, Aoi Hayashi, William John Munro, and Kae Nemoto

   **Published**: Optica Quantum 3, 238-245 (2025)

   **DOI**: `https://doi.org/10.1364/OPTICAQ.541432 <https://doi.org/10.1364/OPTICAQ.541432>`_

   .. merlin-citations-badge:: quantum_reservoir_computing

   **Reproduction Status**: 🚧In Progress

   **Reproducer**: Jean Senellart (jean.senellart@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/quantum_reservoir_computing_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

The authors demonstrate that the random interferometer used in boson sampling can generate complex dynamics suitable for quantum reservoir computing, providing a practical application for this previously limited quantum computation model. They successfully apply this approach to image recognition tasks, showing its utility even with small-scale quantum systems.

Significance
============

This work represents a breakthrough in finding practical applications for boson sampling beyond computational supremacy demonstrations. By repurposing the random interferometer structure for reservoir computing, the authors provide a pathway for near-term quantum advantage in machine learning tasks.

MerLin Implementation
=====================

The MerLin reproduction implements the quantum reservoir computing framework using the library's photonic simulation capabilities. The implementation leverages MerLin's boson sampling modules and reservoir computing extensions.

Key Contributions Reproduced
============================

**Boson Sampling Reservoir**
  * Implementation of random interferometer as quantum reservoir
  * Characterization of computational expressivity

**Image Recognition Tasks**
  * MNIST digit classification experiments
  * Comparison with classical reservoir computing methods

**Scalability Analysis**
  * In progress: exploring performance with increasing reservoir size


Interactive Exploration
=======================

**Jupyter Notebook**: :doc:`quantum_reservoir </notebooks/reproduced_papers/quantum_reservoir>`

The notebook provides:

* First implementation of the quantum reservoir computing model


Citation
========

.. code-block:: bibtex

   @article{sakurai2025quantum,
     title={Quantum optical reservoir computing powered by boson sampling},
     author={Sakurai, Akitada and Hayashi, Aoi and Munro, William John and Nemoto, Kae},
     journal={Optica Quantum},
     volume={3},
     pages={238--245},
     year={2025},
     publisher={Optica Publishing Group},
     doi={10.1364/OPTICAQ.541432}
   }
