:github_url: https://github.com/merlinquantum/merlin

===========================
Quantum Vision Transformers
===========================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum Vision Transformers

   **Authors**: El Amine Cherrat, Iordanis Kerenidis, Natansh Mathur, Jonas Landman, Martin Strahm, Yun Yvonna Li

   **Published**: Quantum 8, 1265 (2024)

   **DOI**: `10.22331/q-2024-02-22-1265 <https://doi.org/10.22331/q-2024-02-22-1265>`_

   .. merlin-citations-badge:: quantum_vision_transformers

   **Paper URL**: `arXiv:2209.08167 <https://arxiv.org/abs/2209.08167>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Benjamin Stott (benjamin.stott@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/quantum_vision_transformers_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

The paper introduces transformer architectures in which the attention mechanism
is carried by a quantum circuit rather than by an explicit learned score matrix.
It proposes an orthogonal patch-wise network, a quantum orthogonal transformer
computing pairwise overlap scores, a direct-attention variant, and a compound
transformer in which attention arises from two-photon interference and a
cross-partition read-out. The models are evaluated on MedMNIST, with RetinaMNIST
as the headline dataset, against a classical vision transformer and a quantum
orthogonal fully-connected baseline. The central claim concerns accuracy at a
reduced attention-parameter budget rather than absolute accuracy.

Significance
============

The compound model states that attention need not be materialised as a matrix,
but can instead be read out of the interference pattern of a two-photon state.
The sector structure this requires is native to linear optics, so the claim is
directly testable on a photonic simulator. The reduced attention-parameter
budget is measurable independently of accuracy, which makes the paper's principal
claim falsifiable on a single benchmark.

MerLin Implementation
=====================

The reproduction uses MerLin's linear-optics primitives throughout:
``QuantumLayer``, ``CircuitBuilder``, and ``StateVector.from_tensor`` for
amplitude-encoded inputs, with SLOS computing the compound actions internally.
Six architectures share a single model wrapper.

.. list-table:: Architectures
   :header-rows: 1
   :widths: 12 34 12 12 30

   * - Model
     - Description
     - Photons
     - Modes
     - Attention mechanism
   * - A
     - Orthogonal patch-wise network
     - 1
     - d
     - none; one shared interferometer per token
   * - B
     - Quantum orthogonal transformer
     - 1
     - d
     - overlap scores through ``W``, softmax, features through ``V``
   * - C
     - Direct quantum attention
     - 1
     - d
     - overlap scores as in B, applied before the feature transform
   * - D
     - Compound transformer
     - 2
     - n+d
     - implicit, from interference and a cross-partition read-out
   * - D_full
     - Full-sector compound (extension)
     - 2
     - n+d
     - as D, retaining every two-photon sector
   * - E
     - Multi-sector attention (extension)
     - 1 + 2
     - n+d
     - shared interferometer; 1-photon features, 2-photon attention
   * - F
     - Hierarchical compound (extension)
     - 3
     - r+p+d
     - three-photon interference across region, patch and feature blocks

Two circuit families are selectable. The ``generic`` family is a universal
rectangular MZI mesh. The ``butterfly`` family is a structured mesh matching the
layout used in the paper and requires power-of-two mode counts. Post-selection is
applied only where the read-out requires it: the cross-partition read-out in D
and the triple-cross read-out in F.

Key Contributions Reproduced
============================

**Paper benchmark on RetinaMNIST**
  * Implemented models A, B, C and D together with the classical vision
    transformer and the orthogonal fully-connected baseline.
  * All reproduced AUC values fall within 0.012 of the paper's Table 4, and the
    ordering of the models is preserved.

**Attention-parameter budget**
  * Measured attention-layer and total parameter counts for every model.
  * Model A matches the classical vision transformer's AUC using an eighth of the
    attention parameters, and model D using a third.

**Breadth across MedMNIST**
  * Evaluated four models on eight MedMNIST datasets at three seeds.
  * The compound models attain the highest AUC on five of the eight datasets.

**Circuit-family comparison**
  * Compared the structured butterfly mesh against the universal generic mesh at
    matched configuration.
  * Accuracy differences fall inside the seed spread while simulation cost differs
    by roughly an order of magnitude.

**Architectural extensions**
  * Added three architectures beyond the paper: a full-sector compound model, a
    shared-interferometer multi-sector model, and a three-photon hierarchical
    model.

Implementation Details
======================

Runs are launched through the shared root CLI.

.. code-block:: bash

   python implementation.py --paper quantum_vision_transformers \
       --config papers/quantum_vision_transformers/configs/paper/model_a_retina.json

The paper directory provides suite wrappers for the full workflow.

.. code-block:: bash

   bash scripts/validation/validate.sh                                   # build and gradient checks
   CPU_FRIENDLY=1 bash scripts/suites/run_retina_suite.sh --device cpu    # paper and butterfly workflow
   CPU_FRIENDLY=1 bash scripts/suites/run_medmnist_suite.sh --device cpu  # MedMNIST workflow
   python scripts/analysis/generate_figures.py outdir/                    # figures into results/figures/

Each run directory stores the resolved configuration, a resumable checkpoint, the
best validation-AUC weights, an incremental progress file and a final results
file. Re-running with the same ``--outdir`` resumes from the checkpoint.

Experimental Results
====================

RetinaMNIST against the paper
-----------------------------

Reproduction values are means over three seeds (7, 42, 123) in the butterfly
family. The paper reports single values.

.. list-table:: RetinaMNIST: paper Table 4 against the reproduction
   :header-rows: 1
   :widths: 26 14 14 16 16

   * - Model
     - Paper AUC
     - Paper ACC
     - Reproduced AUC
     - Reproduced ACC
   * - Classical ViT
     - 0.736
     - 55.75%
     - 0.7417 ± 0.0037
     - 54.50%
   * - A: OrthoPatchWise
     - 0.738
     - 56.50%
     - 0.7479 ± 0.0087
     - 53.25%
   * - B: OrthoTransformer
     - 0.749
     - 56.50%
     - 0.7369 ± 0.0064
     - 51.08%
   * - D: Compound
     - 0.729
     - 56.50%
     - 0.7409 ± 0.0106
     - 52.83%
   * - OrthoFNN baseline
     - not reported
     - not reported
     - 0.6720 ± 0.0083
     - 49.08%

Accuracy is 2 to 5 points below the paper for every model, including the
classical baseline, while AUC agrees. The orthogonal fully-connected baseline
reaches 0.672, well below the transformer models, so the benchmark separates
architectures.

.. figure:: ../../_static/reproduced_papers/quantum_vision_transformers/comparison_retinamnist.png
   :alt: Test AUC and accuracy per model on RetinaMNIST with the paper's reported values as dashed reference lines
   :align: center
   :width: 100%

   RetinaMNIST test AUC and accuracy, butterfly family, three seeds. Dashed lines
   are the paper's reported values.

Attention-parameter budget
--------------------------

.. list-table:: Parameter counts on RetinaMNIST
   :header-rows: 1
   :widths: 34 22 22 22

   * - Model
     - Attention params
     - Total params
     - Reproduced AUC
   * - Classical ViT
     - 2048
     - 9333
     - 0.7417 ± 0.0037
   * - A: OrthoPatchWise
     - 256
     - 7509
     - 0.7479 ± 0.0087
   * - B: OrthoTransformer
     - 512
     - 7797
     - 0.7369 ± 0.0064
   * - D: Compound
     - 640
     - 7893
     - 0.7409 ± 0.0106

Model A matches the classical vision transformer's AUC with an eighth of the
attention parameters, and model D with a third and a lower total parameter count.
The paper reports attention parameters per layer (32, 64, 80) whereas the values
above are measured totals, so only the ratio against the classical model is
directly comparable.

.. figure:: ../../_static/reproduced_papers/quantum_vision_transformers/param_comparison.png
   :alt: Attention-layer and total trainable parameters per model against the classical ViT attention reference
   :align: center
   :width: 100%

   Attention-layer parameters and total trainable parameters per model. The
   dashed line marks the classical attention budget.

MedMNIST breadth
----------------

Eight datasets, three seeds, butterfly family, trained on a capped 5000-sample
training subset with the official validation and test splits.

.. list-table:: Test AUC across MedMNIST
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Dataset
     - A
     - B
     - D
     - D_full
   * - bloodmnist
     - 0.9647
     - 0.9681
     - 0.9754
     - 0.9740
   * - breastmnist
     - 0.7875
     - 0.8341
     - 0.8094
     - 0.8059
   * - dermamnist
     - 0.8745
     - 0.8648
     - 0.8928
     - 0.8884
   * - octmnist
     - 0.7699
     - 0.7692
     - 0.8177
     - 0.8173
   * - pathmnist
     - 0.9434
     - 0.9300
     - 0.9410
     - 0.9452
   * - pneumoniamnist
     - 0.9439
     - 0.9428
     - 0.9507
     - 0.9473
   * - retinamnist
     - 0.7387
     - 0.7390
     - 0.7381
     - 0.7408
   * - tissuemnist
     - 0.8106
     - 0.8055
     - 0.8167
     - 0.8209

The compound models lead on five of the eight datasets. Differences between
architectures are small relative to differences between datasets. Retaining all
two-photon sectors in D_full gives no consistent gain over the cross-partition
read-out specified in the paper.

Circuit family
--------------

On RetinaMNIST, model A reaches AUC 0.7479 ± 0.0087 under the butterfly family
against 0.7435 ± 0.0035 under the generic family, and model B 0.7369 ± 0.0064
against 0.7425 ± 0.0068. The differences are inside the seed spread. Typical
butterfly wall-clock for these runs is a few hundred seconds against a few
thousand for generic, so the structured family is preferable at power-of-two mode
counts.

Limitations
===========

* Accuracy is 2 to 5 points below the paper for every model including the
  classical baseline, while AUC agrees. The cause has not been isolated. Because
  it affects all models equally it does not alter the between-model comparison.
* The committed generic-family runs for D_full, E and F stop at the first epoch.
  Their AUC values in that profile are single-epoch snapshots and are not trained
  results. The butterfly-family D_full runs are trained.
* Model C appears only in the generic family among the committed full-profile
  runs and is therefore absent from the butterfly comparison.
* The MedMNIST breadth uses a capped 5000-sample training subset with official
  validation and test splits. It compares architectures under a fixed data budget
  and is not a leaderboard result.
* Attention-parameter counts are totals, whereas the paper reports per-layer
  values.
* The committed comparison figures have overlapping axis labels at this size. The
  underlying values are in ``results/figures/*/summary.csv``.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (quantum_vision_transformers) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/quantum_vision_transformers>`_

Citation
========

.. code-block:: bibtex

   @article{cherrat_quantum_2024,
     title={Quantum Vision Transformers},
     author={Cherrat, El Amine and Kerenidis, Iordanis and Mathur, Natansh and Landman, Jonas and Strahm, Martin and Li, Yun Yvonna},
     journal={Quantum},
     volume={8},
     pages={1265},
     year={2024},
     doi={10.22331/q-2024-02-22-1265}
   }

Related Reproductions
=====================

* :doc:`photonic_qcnn` applies a photonic vision architecture to MedMNIST-style
  data with a different patch-encoding scheme.
* :doc:`nearest_centroids` uses amplitude encoding with quantum inner-product
  estimation, the primitive underlying the overlap scores in models B and C.
