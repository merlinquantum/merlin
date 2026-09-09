:github_url: https://github.com/merlinquantum/merlin

=================================================================
A Binary Optimisation Algorithm for Near-Term Photonic Processors
=================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: A Binary Optimisation Algorithm for Near-Term Photonic Quantum Processors

   **Authors**: Alexander Makarovskiy, Mateusz Slysz, Łukasz Grodzki, Dawid Siera, Thorin Farnsworth, William R. Clements, Piotr Rydlichowski, Krzysztof Kurowski

   **Published**: preprint, submitted 9 October 2025 (ORCA Computing; Poznań Supercomputing and Networking Center; Poznań University of Technology)

   **DOI**: `10.48550/arXiv.2510.08274 <https://doi.org/10.48550/arXiv.2510.08274>`_

   **Paper URL**: `arXiv:2510.08274 <https://arxiv.org/abs/2510.08274>`_

   **Reproduction Status**: ⚠️ Partial (knapsack and TSP rows of Table I reproduced; tactical deconfliction and the hardware table out of scope)

   **Reproducer**: Jean Senellart (jean.senellart@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/bosonic_binary_solver_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

A train of :math:`m` optical pulses ("time bins") carrying the state
:math:`|1,0,1,0,\ldots\rangle` passes through programmable fibre delay lines of
lengths 1, 3 and 9. Threshold detectors report which bins contain at least one
photon, giving one bit string per shot. A trainable classical layer then flips
bit :math:`i` independently with probability :math:`p_i`, producing a candidate
solution :math:`X`. Training minimises :math:`\mathbb{E}[C(X)]` for the problem's
cost function :math:`C` — the beamsplitter angles by a photonic parameter-shift
rule, the flip probabilities by an analytic gradient, both under plain SGD.

The paper claims this solves knapsack, tactical-deconfliction and travelling-salesman
instances of up to 30 binary variables while evaluating a small fraction of the
solution space, that it is competitive with simulated annealing and hill climbing,
and that it runs on ORCA's PT-1 hardware.

The problem class is the usual one for quantum optimisation — unconstrained
minimisation over :math:`\{0,1\}^m`, which covers QUBO once a constraint is folded
into a penalty term. What differs from the QUBO solvers usually reproduced here is
where the quantum device sits. QAOA, CVaR-VQE and the photonic ObliQ family all
encode the cost matrix *into* the circuit, so the circuit changes with the instance.
Here the circuit never sees the problem at all: it is a fixed-topology sampler whose
angles are trained against :math:`\mathbb{E}[C(X)]`, and the cost function enters
only through that scalar. Any :math:`C` that can be evaluated on a bit string is
therefore in scope, whether or not it is quadratic.

Significance
============

The architecture is unusually clean to instrument. The circuit takes no data
input, so the only thing the quantum device supplies is a distribution over bit
strings, and the classical post-processing is a single independent bit-flip layer.
That makes it possible to hold the circuit, the trained angles, the flip layer and
the candidate budget fixed and vary one component at a time — including the click
source itself, which this reproduction adds as a control alongside the paper's own
ablation.

MerLin Implementation
=====================

.. figure:: /_static/reproduced_papers/bosonic_binary_solver.png
   :alt: The algorithm as a pipeline: an alternating input state enters a delay-line interferometer, threshold detectors produce a click pattern, a trainable bit-flip layer turns it into a candidate solution, and a training loop feeds the cost back to the angles and the flip probabilities.
   :width: 100%

   The algorithm as this reproduction implements it. The interferometer is drawn
   unrolled, one beamsplitter per pair of time bins :math:`(t, t+\ell)`, which is
   the topology the paper's own parameter count implies.

The reproduction lives in the
`reproduced_papers repository <https://github.com/merlinquantum/reproduced_papers>`_
under ``papers/bosonic_binary_solver`` and is launched with the common runtime:

.. code-block:: bash

   python implementation.py --paper bosonic_binary_solver --config papers/bosonic_binary_solver/configs/knapsack_m20_original.json

Three execution paths share one physics definition in ``lib/tbi.py``:

* **Reproduction path** — Perceval ``CliffordClifford2017`` sampling with the
  paper's photonic parameter-shift rule, which is what the paper describes.
* **MerLin path** — the same interferometer as a MerLin ``QuantumLayer`` with
  ``MeasurementStrategy.probs(ComputationSpace.FOCK)``, giving the exact click
  distribution and an autograd gradient instead of a shot-based estimator.
* **GPU path** — a batched sampler that runs every instance and every shifted
  angle of one update in a single call, which is what makes the size extension
  below affordable.

Key Contributions Reproduced
============================

**Table I, knapsack and TSP**
  * :math:`m=20`: **99%** of instances solved to optimality against the paper's 98%.
  * :math:`m=25`: **96%** against 99%.
  * :math:`m=30`: **73%** against 93% — an open gap, discussed below.
  * TSP :math:`m=29`: **7%** optimal and 5.59% mean error against the paper's 2% and 6.58%.

**Appendix B candidate bound**
  * The candidate count is reproduced term for term:
    :math:`N \cdot S \cdot (2\sum_i (m - l_i) + 2m + 1) = 2{,}150{,}000` at
    :math:`m=30` with 77 trainable beamsplitters.

**The paper's own ablation (C7)**
  * Freezing the interferometer at its random initialisation and training only the
    flip layer collapses performance: 96% → **21%** at :math:`m=25`
    (McNemar :math:`z=8.54`) and 73% → **3%** at :math:`m=30` (:math:`z=8.25`).
  * The paper's ablation therefore reproduces: with the interferometer frozen, the
    flip layer alone recovers almost nothing.

Implementation Details
======================

.. code-block:: python

   import perceval as pcvl
   import torch
   from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer

   # One beamsplitter per pair of time bins (t, t + l), for each delay l.
   # The paper's own parameter count, sum_i (m - l_i), requires this topology.
   circuit = pcvl.Circuit(m)
   for delay in (1, 3, 9):
       for t in range(m - delay):
           circuit.add((t, t + delay), pcvl.BS.Ry(theta=pcvl.P(f"theta_{delay}_{t}")))

   layer = QuantumLayer(
       circuit=circuit,
       input_state=[1, 0] * (m // 2),          # |1,0,1,0,...>
       trainable_parameters=["theta"],
       measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
       dtype=torch.float64,
   )

Two conventions matter and are easy to get silently wrong. ``pcvl.BS.Ry`` is the
real rotation; the default ``pcvl.BS.Rx`` carries complex phases that change the
interference pattern. And MerLin and Perceval use :math:`R=\cos^2(\theta/2)`
whereas the reference ORCA toolkit uses :math:`R=\cos^2(\theta)` — under which the
paper's printed parameter-shift denominator is exactly twice the true derivative,
a constant factor equivalent to doubling the learning rate.

Experimental Results
====================

**Varying the click source**

Same circuit, same trained angles, same flip layer, same candidate budget; only the
click source changes. Paired by instance, :math:`m=30` knapsack.

.. list-table:: Click source at fixed budget (:math:`m=30`)
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - Click source
     - Clicks / mode
     - % optimal
     - Mean error
     - McNemar :math:`z` vs boson
   * - boson (the paper's)
     - 38.0%
     - 73.0%
     - 0.193%
     - —
   * - shuffled boson (marginals kept, joint destroyed)
     - 37.2%
     - 79.0%
     - 0.166%
     - −0.19
   * - distinguishable photons
     - 42.8%
     - 86.5%
     - 0.100%
     - +1.86
   * - independent Bernoulli clicks
     - 42.5%
     - 89.0%
     - 0.081%
     - **+2.69**

The shuffled arm preserves every per-mode marginal of the sampler's output exactly
and destroys the joint distribution. The Bernoulli arm replaces the sampler
entirely with independent per-mode clicks. Both are drawn against the same trained
circuit, the same flip layer and the same budget, and paired by instance.

**Varying the click density**

``bernoulli@r`` scales the per-mode click rate by :math:`r` and changes nothing else.

.. list-table:: Click-density ladder (:math:`m=30`, 200 instances per arm)
   :header-rows: 1
   :widths: 34 22 22 22

   * - Arm
     - Clicks / mode
     - % optimal
     - Mean error
   * - ``bernoulli@0.87``
     - 36.9%
     - 73.5%
     - 0.219%
   * - boson
     - 38.0%
     - 73.0%
     - 0.193%
   * - ``bernoulli@1.0``
     - 42.5%
     - 89.0%
     - 0.081%
   * - ``bernoulli@1.15``
     - 48.0%
     - **98.5%**
     - 0.006%
   * - ``bernoulli@1.5``
     - 54.4%
     - 95.0%
     - 0.031%

The ladder rises to a peak near 48% density and turns back down
(0.87 → 1.5: :math:`z = 5.66`). At matched density the boson sampler and
independent Bernoulli clicks are not separated by these runs: boson (38.0%)
against ``bernoulli@0.87`` (36.9%) gives :math:`z = 0.00`, :math:`t = 0.38`.

Bunching places the boson source near 38% clicks per mode, at the low end of the
range covered by the ladder. Raising the flip learning rate tenfold moves it by
about one point, so the flip layer does not compensate for the difference at these
sizes.

**The same comparison on TSP**

On TSP :math:`m=29` a candidate is an integer that is Lehmer-decoded into a
permutation, so the number of ones in the string does not correspond to anything in
the solution. There, every source comparison is null (:math:`|z| \le 0.8`,
:math:`|t| \le 1.5`) and the density ladder is flat — the knapsack observation does
not carry across to this encoding.

**Fair baselines at the same budget**

.. list-table:: :math:`m=20` knapsack, all methods at the Appendix B budget
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - % optimal
     - Paper
   * - Simulated annealing
     - 100%
     - 100%
   * - Hill climbing
     - 100%
     - 84%
   * - Uniform random search
     - 81%
     - not run
   * - Bosonic Binary Solver
     - 99%
     - 98%

The Appendix B budget at :math:`m=20` is 1.29× the entire solution space, which is
worth keeping in view when reading this row: uniform sampling alone reaches 81%
under it. The paper's reported hill climbing (84%) sits below uniform random search
at that budget. The paper does not state what budget its baselines received.

Extensions and Future Work
==========================

**Beyond the paper's largest simulated size**

The batched GPU sampler takes the density comparison from :math:`m=30` (where the
budget still covers 0.2% of the space) out to :math:`m=42` with a boson arm and
:math:`m=60` without one, to test whether the effect was an artifact of the regime
where the budget is appreciable. Scaling only the Bernoulli click rate from 1.0 to
1.15, everything else fixed:

.. list-table:: Paired :math:`t` for click density alone, by problem size
   :header-rows: 1
   :widths: 30 12 12 12 12 12 12

   * - :math:`m`
     - 34
     - 38
     - 42
     - 48
     - 54
     - 60
   * - Solution-space coverage
     - 1.4e-4
     - 1.0e-5
     - 7.1e-7
     - 1.3e-8
     - 2.3e-10
     - 3.9e-12
   * - Paired :math:`t`
     - −2.08
     - −2.98
     - −4.01
     - −5.23
     - −6.26
     - **−8.24**

The separation grows monotonically across five orders of magnitude of coverage. At
:math:`m=60`, where the sparser arm no longer finds any optimum, a 15% higher click
rate still halves the mean error (2.978% → 1.580%).

**Open**
  * :math:`m=30` reaches 73% against the paper's 93%. The leading suspect is the
    unpublished instance generator — knapsack difficulty at fixed :math:`m` is very
    sensitive to the capacity fraction and to value/weight correlation. Recorded as
    unresolved.
  * Tactical deconfliction (5 of Table I's 13 rows): the cost function is given but
    the instance distribution is not.
  * Table II (hardware): needs a PT-1 and a loss budget the paper does not state.
    The density result makes a sharp prediction for it — loss lowers click density,
    so it should move the boson arm monotonically down the same ladder.

Interactive Exploration
=======================

**Jupyter Notebook**: :doc:`../../notebooks/reproduced_papers/bosonic_binary_solver`

.. toctree::
   :maxdepth: 1
   :hidden:

   ../../notebooks/reproduced_papers/bosonic_binary_solver

The notebook builds the delay-line interferometer with Perceval primitives, wraps
it in a MerLin ``QuantumLayer``, and trains it end to end on a small knapsack
instance with exact gradients — no shot noise, no GPU, about thirty seconds.

The full reproduction package ships its own notebook, which adds the shot-based
paper path, the fair baselines and the click-source control:
`reproduced_papers notebook <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/bosonic_binary_solver/notebook.ipynb>`_.

Code Access and Documentation
=============================

**Reproduction package**:
`reproduced_papers/papers/bosonic_binary_solver <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/bosonic_binary_solver>`_

The package includes:

* ``lib/tbi.py`` — the interferometer, validated against ORCA's released PT-Series
  simulator to a total-variation distance below 1e-12
* ``lib/bbs.py`` — the paper's shot-based solver; ``lib/bbs_merlin.py`` — the MerLin
  exact-gradient variant; ``lib/bbs_gpu.py`` — the batched GPU path
* ``lib/baselines.py`` — simulated annealing, hill climbing and uniform random
  search at the solver's own budget
* 65 configs, 39 tests, and ``results/summary.json`` over 4,340 runs, from which
  every number on this page is drawn

Citation
========

.. code-block:: bibtex

   @misc{makarovskiy2025binary,
     title={A Binary Optimisation Algorithm for Near-Term Photonic Quantum Processors},
     author={Makarovskiy, Alexander and Slysz, Mateusz and Grodzki, {\L}ukasz and Siera, Dawid and Farnsworth, Thorin and Clements, William R. and Rydlichowski, Piotr and Kurowski, Krzysztof},
     year={2025},
     eprint={2510.08274},
     archivePrefix={arXiv},
     primaryClass={quant-ph},
     doi={10.48550/arXiv.2510.08274},
     url={https://arxiv.org/abs/2510.08274}
   }

Related Reproductions
=====================

* :doc:`fock_state_expressivity` — the other reproduction where photon statistics,
  rather than a trained classical head, is the object under test.
* `ObliQ: solving QUBO problems on photonic quantum machines
  <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/ObliQ_photonic_QUBO>`_
  — the instance-encoding counterpart in the reproduced-papers repository, with QAOA,
  CVaR-VQE and D-Wave baselines on the same problem class.
* `Quantum latent distributions
  <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/quantum_latent_distributions>`_
  — shares four authors with this paper and asks the same question, what a
  boson-sampled distribution contributes over a classical one, in a generative
  setting rather than an optimisation one.

Impact and Applications
=======================

* **Combinatorial optimisation**: a shallow, encoding-free photonic sampler used as
  the proposal distribution for a classical search loop.
* **A reusable control**: the source swap is cheap and general — hold the circuit,
  the trained parameters and the budget fixed, replace the sampled distribution
  with a classical one that shares its marginals, and measure whether anything
  moves.
* **Hardware**: loss and detector efficiency act on click density directly, so the
  ladder above gives the hardware table a quantity to be read against.
