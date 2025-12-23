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

Creation from tensors
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from merlin.core.probability_distribution import ProbabilityDistribution

   probs = torch.tensor([0.2, 0.3, 0.5])
   pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=2)
   dense = pd.to_dense()

Filtering and performance
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from merlin.core.computation_space import ComputationSpace

   pd = ProbabilityDistribution.from_tensor(torch.tensor([0.5, 0.25, 0.25]), n_modes=2, n_photons=2)
   filtered = pd.filter(ComputationSpace.UNBUNCHED)
   assert filtered.basis_size == 1
   assert (filtered.logical_performance > 0).all()
   assert torch.isclose(filtered.to_dense().sum(), torch.tensor(1.0))

The ``logical_performance`` attribute is filled by ``filter`` to record the kept
mass per batch (kept / total) and remains ``None`` on distributions that have not
been filtered yet.

Perceval interoperability
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import perceval as pcvl

   dist = pcvl.BSDistribution()
   dist[pcvl.BasicState([1, 0])] = 0.8
   dist[pcvl.BasicState([0, 1])] = 0.2

   pd = ProbabilityDistribution.from_perceval(dist)
   pcvl_back = pd.to_perceval()
   assert pcvl_back[pcvl.BasicState([1, 0])] == 0.8
