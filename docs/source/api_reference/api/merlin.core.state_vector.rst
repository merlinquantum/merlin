merlin.core.state_vector
========================

.. currentmodule:: merlin.core.state_vector

StateVector
-----------

.. autoclass:: StateVector
   :members: from_basic_state, from_tensor, from_perceval, tensor_product, to_perceval, index, memory_bytes, __getitem__, __add__, __mul__, to_dense, basis, basis_size, is_sparse, is_sparse
   :member-order: bysource
   :show-inheritance:

Notes and Examples
------------------

One-hot creation (dense or sparse)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from merlin.core.state_vector import StateVector
   import perceval as pcvl

   # Default is sparse one-hot
   sv = StateVector.from_basic_state([1, 0, 0])
   assert sv.is_sparse
   amp = sv[pcvl.BasicState([1, 0, 0])]  # amplitude lookup

   # Dense one-hot
   sv_dense = StateVector.from_basic_state([0, 1, 0], sparse=False)

Tensor product with metadata propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   left = StateVector.from_basic_state([1, 0], sparse=False)
   right = StateVector.from_basic_state([0, 1], sparse=True)
   combined = left.tensor_product(right)  # dense because one operand is dense
   assert combined.n_modes == 4 and combined.n_photons == 2

Addition and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   a = StateVector.from_basic_state([1, 0], sparse=False)
   b = StateVector.from_basic_state([0, 1], sparse=False)
   superposed = (a + b).normalize()  # explicit in-place normalization

   scaled = 0.5 * a    # scaling is meaningful when you later add terms

``normalize`` acts in-place and returns the same ``StateVector`` instance.

Conversion to/from Perceval
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import perceval as pcvl

   # From Perceval
   pv = pcvl.StateVector(pcvl.BasicState([1, 0]))
   sv = StateVector.from_perceval(pv)

   # To Perceval (supports single non-zero occupation per state/batch)
   pv_back = sv.to_perceval()

Batch lookup
^^^^^^^^^^^^

.. code-block:: python

   import torch
   import perceval as pcvl

   batch = torch.tensor([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=torch.complex64)
   sv_batch = StateVector.from_tensor(batch, n_modes=2, n_photons=1)
   amps = sv_batch[pcvl.BasicState([1, 0])]  # shape matches batch prefix
