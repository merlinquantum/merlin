:github_url: https://github.com/merlinquantum/merlin

====================================================
Quantum Large Language Model Fine-Tuning
====================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum Large Language Model Fine-Tuning

   **Authors**: Sang Hyub Kim, Jonathan Mei, Claudio Girotto, Masako Yamada, Marin Roetteler

   **Published**: eprint arXiv,(2025)

   **DOI**: `arXiv:2504.08732v1 <https://arxiv.org/abs/2504.08732v1>`_

   **Reproduction Status**: 🚧 In Progress

   **Reproducer**: Cassandre Notton (cassandre.notton@quandela.com)

Abstract
========

This paper explores the use of a quantum parametrized circuit for LLM fine-tuning. The authors observe up to 3.14% improvements in accuracy over classical architectures of comparable model size.

Significance
============

This paper introduces a novel hybrid approach to LLM fine-tuning that addresses current limitations in the field. Given the critical role of fine-tuning in optimizing large language models for specific tasks, investigating how hybrid methodologies can enhance this process represents a significant research opportunity




Key Contributions Reproduced
============================

**We observe up to 3.14% improvements in accuracy over classical architectures**

We present below the final results

Performance Comparison
-----------------------




Implementation Details
======================

For a first analysis, we use a Generic Interferometer:

.. code-block:: python

   import merlin as ML # Package: merlinquantum, import: merlin
   import torch

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Create a simple quantum layer
   experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.SERIES,
            n_modes=modes,
            n_photons=sum(input_state) if input_state else modes // 2,
            state_pattern=ML.StatePattern.PERIODIC
        )

Experimental Results
====================

.. table:: Model Performance Results
   :widths: auto

   ========================== ========== ==========
   Method                     Validation Test
   ========================== ========== ==========
   **Classical Methods**
   LogisticRegression         0.8000     0.7669
   SVC                        0.8080     0.7846
   MLP                        0.8040     0.7701
   SVC (paper)                0.8508     --
   MLP (paper)                91.44      --
   **Quantum Methods**
   2 modes                    0.5800     0.5691
   4 modes                    0.8120     0.7990
   6 modes                    0.8360     0.8280
   8 modes                    0.8480     0.8376
   Single sQE 10Q (paper)     90.21      --
   Multi sQE 14Q (paper)      92.7       --
   ========================== ========== ==========


We use smaller models as in the paper and are able to reach +4% in accuracy for the 8-mode interferometer.
Next results will include a comparison of the number of parameters !


Interactive Exploration
=======================

**Jupyter Notebook**: :doc:`../../notebooks/QLLM_fine_tuning`


Citation
========

.. code-block:: bibtex

   @article{hyub2025quantum,
  title={Quantum Large Language Model Fine-Tuning},
  author={Hyub Kim, Sang and Mei, Jonathan and Girotto, Claudio and Yamada, Masako and Roetteler, Martin},
  journal={arXiv e-prints},
  pages={arXiv--2504},
  year={2025}}

Future Work
=====================

**Future Work** includes:

* thorough comparison of the performances with respect to the number of parameters;
* analysis of the effect of the number of photons
* experiments on SetFit using more than 2 classes for more complex classification


.. note:: End of document.

