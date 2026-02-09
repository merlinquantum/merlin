merlin.core.probability_distribution
====================================

.. currentmodule:: merlin.core.probability_distribution

ProbabilityDistribution
-----------------------

.. autoclass:: ProbabilityDistribution
   :members:
   :member-order: bysource
   :show-inheritance:


Notes and Examples
------------------


Constructors
^^^^^^^^^^^^

**from_tensor** — wrap a probability tensor with Fock metadata.
The last dimension must match the basis size for the chosen computation space.
The distribution is normalized on construction:

.. code-block:: python

   import torch
   from merlin.core.probability_distribution import ProbabilityDistribution

   probs = torch.tensor([0.2, 0.3, 0.5])
   pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=2)

The optional ``computation_space`` parameter selects the basis ordering
(defaults to ``FOCK``):

.. code-block:: python

   from merlin.core.computation_space import ComputationSpace

   pd = ProbabilityDistribution.from_tensor(
       probs, n_modes=2, n_photons=2,
       computation_space=ComputationSpace.FOCK,
   )

Batched inputs are supported — leading dimensions are treated as batch axes:

.. code-block:: python

   batch = torch.rand(16, 3)    # 16 samples, basis_size = 3
   pd = ProbabilityDistribution.from_tensor(batch, n_modes=2, n_photons=2)
   assert pd.shape == (16, 3)

**from_state_vector** — compute :math:`|a_i|^2` from a
:class:`~merlin.core.state_vector.StateVector`:

.. code-block:: python

   from merlin.core.state_vector import StateVector

   sv = StateVector.from_basic_state([1, 0, 1, 0], sparse=False)
   pd = ProbabilityDistribution.from_state_vector(sv)
   assert pd.n_modes == 4 and pd.n_photons == 2

**from_perceval** — convert from a Perceval ``BSDistribution``:

.. code-block:: python

   import perceval as pcvl

   dist = pcvl.BSDistribution()
   dist[pcvl.BasicState([1, 0])] = 0.8
   dist[pcvl.BasicState([0, 1])] = 0.2

   pd = ProbabilityDistribution.from_perceval(dist)


Properties and metadata
^^^^^^^^^^^^^^^^^^^^^^^

``n_modes`` and ``n_photons`` are set at construction and immutable.
``shape``, ``device``, ``dtype``, and ``requires_grad`` are delegated to the
underlying tensor:

.. code-block:: python

   pd.n_modes        # 2
   pd.n_photons      # 2
   pd.shape          # torch.Size([3])
   pd.device         # device(type='cpu')

``basis`` returns the Fock ordering for the current computation space (or a
filtered basis after ``filter()``).  ``basis_size`` is ``len(basis)``:

.. code-block:: python

   pd.basis_size            # 3 for (2 modes, 2 photons, FOCK)
   list(pd.basis)           # [(2, 0), (1, 1), (0, 2)]

``computation_space`` records which basis scheme was used:

.. code-block:: python

   pd.computation_space     # ComputationSpace.FOCK

``is_normalized`` is always ``True`` — distributions are normalized on
construction and after every ``filter`` call.


Accessing probabilities
^^^^^^^^^^^^^^^^^^^^^^^

``probabilities()`` and ``to_dense()`` both return a dense, normalized tensor:

.. code-block:: python

   dense = pd.probabilities()   # shape: (3,)
   dense = pd.to_dense()        # equivalent

Use bracket syntax for a single Fock state's probability:

.. code-block:: python

   import perceval as pcvl

   p = pd[[1, 1]]                        # scalar tensor
   p = pd[pcvl.BasicState([1, 1])]       # equivalent

For batched distributions, the returned tensor matches the batch shape:

.. code-block:: python

   batch_pd = ProbabilityDistribution.from_tensor(
       torch.rand(8, 3), n_modes=2, n_photons=2,
   )
   p = batch_pd[[1, 1]]   # shape: (8,)


Filtering and post-selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``filter()`` applies post-selection and returns a **new**, renormalized
distribution.  The ``logical_performance`` attribute records the fraction of
probability mass kept per batch element.

**Filter by computation space:**

.. code-block:: python

   from merlin.core.computation_space import ComputationSpace

   pd = ProbabilityDistribution.from_tensor(
       torch.tensor([0.5, 0.25, 0.25]), n_modes=2, n_photons=2,
   )
   filtered = pd.filter(ComputationSpace.UNBUNCHED)
   filtered.basis_size               # 1 — only |1,1⟩ survives
   filtered.logical_performance      # tensor(0.25) — 25% of mass kept

String aliases also work: ``"fock"``, ``"unbunched"``, ``"dual_rail"``.

**Filter by predicate:**

.. code-block:: python

   # Keep only states where mode 0 has at least 1 photon
   filtered = pd.filter(lambda state: state[0] >= 1)

**Filter by explicit allowed states:**

.. code-block:: python

   filtered = pd.filter([(1, 1), (2, 0)])

**Combined space + predicate** — pass a tuple ``(space, predicate)``:

.. code-block:: python

   # Unbunched states where mode 0 is occupied
   filtered = pd.filter((ComputationSpace.UNBUNCHED, lambda s: s[0] == 1))

``logical_performance`` is ``None`` on unfiltered distributions and is set by
``filter()`` to a tensor of kept / total mass per batch element.


PyTorch-like helpers
^^^^^^^^^^^^^^^^^^^^

``to``, ``clone``, ``detach``, and ``requires_grad_`` mirror the standard
``torch.Tensor`` API while preserving metadata and ``logical_performance``:

.. code-block:: python

   pd_cuda = pd.to("cuda")               # moves tensor + logical_performance
   pd_copy = pd.clone()                   # independent copy
   pd_det  = pd.detach()                  # shares data, no gradient graph
   pd.requires_grad_(True)               # enable gradients in-place


QuantumLayer integration
^^^^^^^^^^^^^^^^^^^^^^^^

With ``return_object=True`` and a probability measurement strategy, the layer
returns a ``ProbabilityDistribution`` instead of a bare tensor:

.. code-block:: python

   import merlin as ML
   from merlin.core.computation_space import ComputationSpace

   layer = ML.QuantumLayer(
       builder=builder,
       input_state=[1, 0, 1, 0],
       n_photons=2,
       measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
       return_object=True,
   )

   pd = layer(x)                                       # ProbabilityDistribution
   pd.probabilities()                                   # dense tensor
   pd_ub = pd.filter(ComputationSpace.UNBUNCHED)        # post-select
   pd_ub.logical_performance                             # fraction of mass kept


Perceval interoperability
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import perceval as pcvl
   from merlin.core.probability_distribution import ProbabilityDistribution

   # Perceval → Merlin
   dist = pcvl.BSDistribution()
   dist[pcvl.BasicState([1, 0])] = 0.8
   dist[pcvl.BasicState([0, 1])] = 0.2
   pd = ProbabilityDistribution.from_perceval(dist)

   # Merlin → Perceval
   pcvl_back = pd.to_perceval()
   assert pcvl_back[pcvl.BasicState([1, 0])] == 0.8

For batched distributions, ``to_perceval()`` returns a list of
``pcvl.BSDistribution`` objects.