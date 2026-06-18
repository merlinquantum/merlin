merlin.core.state_vector
========================

.. automodule:: merlin.core.state_vector
   :no-members:

.. currentmodule:: merlin.core.state_vector

StateVector
-----------

.. autoclass:: StateVector
   :members: from_basic_state, from_tensor, from_perceval, tensor_product, to_perceval, index, logical_to_fock_map, memory_bytes, __getitem__, __add__, __sub__, __mul__, to_dense, to, clone, detach, requires_grad_, normalize, basis, basis_size, is_sparse, is_normalized
   :member-order: bysource
   :show-inheritance:


Notes and Examples
------------------


Constructors
^^^^^^^^^^^^

**from_basic_state** — one-hot Fock state (sparse by default):

.. code-block:: python

   from merlin.core.state_vector import StateVector

   sv = StateVector.from_basic_state([1, 0, 1, 0])
   assert sv.is_sparse
   assert sv.n_modes == 4 and sv.n_photons == 2

   # Dense variant
   sv_dense = StateVector.from_basic_state([1, 0, 1, 0], sparse=False)
   assert not sv_dense.is_sparse

**from_tensor** — wrap a real or complex tensor with Fock metadata.
Real data is auto-promoted to complex. By default, the last dimension must match
the Fock basis size :math:`\binom{n\_modes + n\_photons - 1}{n\_photons}`:

.. code-block:: python

   import torch
   from merlin.core.state_vector import StateVector

   # Single sample (1-D)
   features = torch.randn(10)
   sv = StateVector.from_tensor(features, n_modes=4, n_photons=2)

   # Batched (2-D) — leading dimensions are batch axes
   batch = torch.randn(32, 10)
   sv_batch = StateVector.from_tensor(batch, n_modes=4, n_photons=2)
   assert sv_batch.shape == (32, 10)

Pass ``encoding=...`` to provide compact logical amplitudes and embed them into
the Fock basis immediately. Structured encodings can infer their physical
metadata from the encoding contract. Explicit ``n_modes`` and ``n_photons`` are
also accepted, but they validate the contract rather than stretching it:

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace
   from merlin.core.state_vector import StateVector

   logical = torch.zeros(4, dtype=torch.complex64)
   logical[0] = 1.0
   sv = StateVector.from_tensor(
       logical,
       encoding=EncodingSpace.DUAL_RAIL,
   )
   assert sv.n_modes == 4
   assert sv.n_photons == 2
   assert sv.tensor.shape[-1] == 10
   assert sv.encoding is EncodingSpace.DUAL_RAIL

   # QLOQ dimensions are inferred from the qubit groups.
   qloq = EncodingSpace.qloq(qubit_groups=[2, 1])
   sv_qloq = StateVector.from_tensor(torch.zeros(8), encoding=qloq)
   assert sv_qloq.n_modes == 6
   assert sv_qloq.n_photons == 2

   # Add auxiliary vacuum modes after constructing the encoded state.
   padded = sv @ [0]
   assert padded.n_modes == 5
   assert padded.n_photons == 2

**from_perceval** — convert from a Perceval ``StateVector``:

.. code-block:: python

   import perceval as pcvl
   from merlin.core.state_vector import StateVector

   pv = pcvl.StateVector(pcvl.BasicState([1, 0]))
   sv = StateVector.from_perceval(pv)

   # Round-trip
   pv_back = sv.to_perceval()


Properties and metadata
^^^^^^^^^^^^^^^^^^^^^^^

``n_modes`` and ``n_photons`` are set at construction and immutable:

.. code-block:: python

   sv = StateVector.from_basic_state([1, 0, 1, 0])
   sv.n_modes    # 4
   sv.n_photons  # 2
   sv.n_modes = 5  # raises AttributeError

``shape``, ``device``, ``dtype``, and ``requires_grad`` are delegated to the
underlying tensor:

.. code-block:: python

   sv.shape          # torch.Size([10])  — basis_size for (4, 2)
   sv.device         # device(type='cpu')
   sv.dtype          # torch.complex64

``basis`` returns the combinadics Fock ordering for ``(n_modes, n_photons)``.
``basis_size`` is equivalent to ``len(basis)``:

.. code-block:: python

   sv.basis_size     # 10 for (4 modes, 2 photons)
   list(sv.basis)[:3]  # [(2,0,0,0), (1,1,0,0), (1,0,1,0)]

``logical_to_fock_map()`` exposes the logical-to-Fock index assignment used by
the stored encoding:

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace
   from merlin.core.state_vector import StateVector

   sv = StateVector.from_tensor(
       torch.zeros(4, dtype=torch.complex64),
       encoding=EncodingSpace.DUAL_RAIL,
   )
   sv.logical_to_fock_map()  # {(0, 0): 2, (0, 1): 3, (1, 0): 5, (1, 1): 6}


Amplitude lookup
^^^^^^^^^^^^^^^^

Use bracket syntax with an occupation list or ``pcvl.BasicState``:

.. code-block:: python

   import perceval as pcvl

   sv = StateVector.from_basic_state([1, 0, 1, 0], sparse=False)
   amp = sv[[1, 0, 1, 0]]                    # complex scalar
   amp = sv[pcvl.BasicState([1, 0, 1, 0])]   # equivalent

For batched states, the returned tensor matches the batch shape:

.. code-block:: python

   import torch

   batch = torch.randn(8, 10, dtype=torch.complex64)
   sv = StateVector.from_tensor(batch, n_modes=4, n_photons=2)
   amps = sv[[1, 0, 1, 0]]   # shape: (8,)

``index(state)`` returns the integer basis index (or ``None`` if the state
is absent in a sparse tensor):

.. code-block:: python

   sv.index([1, 0, 1, 0])  # e.g. 2


Superpositions and arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Addition and subtraction require matching ``n_modes`` and ``n_photons``.
Results are **not** automatically normalized — call ``.normalize()`` explicitly:

.. code-block:: python

   a = StateVector.from_basic_state([1, 0], sparse=False)
   b = StateVector.from_basic_state([0, 1], sparse=False)

   superposed = (a + b).normalize()       # (|1,0⟩ + |0,1⟩) / √2
   diff       = (a - b).normalize()       # (|1,0⟩ - |0,1⟩) / √2
   scaled     = 0.5 * a                   # unnormalized until .normalize()

``normalize()`` acts **in-place** and returns ``self``.

Tensor product (``tensor_product`` or ``@``) combines two sub-systems:

.. code-block:: python

   left  = StateVector.from_basic_state([1, 0], sparse=False)
   right = StateVector.from_basic_state([0, 1], sparse=True)
   combined = left.tensor_product(right)  # or: left @ right
   assert combined.n_modes == 4 and combined.n_photons == 2


Dense tensor access
^^^^^^^^^^^^^^^^^^^

``to_dense()`` returns a **normalized**, dense ``torch.Tensor``.  Sparse
states are materialized; already-dense states are returned directly:

.. code-block:: python

   sv = StateVector.from_basic_state([1, 0, 1, 0])
   dense = sv.to_dense()   # shape: (10,), complex, sum of |amplitudes|^2 == 1


PyTorch-like helpers
^^^^^^^^^^^^^^^^^^^^

``to``, ``clone``, ``detach``, and ``requires_grad_`` mirror the standard
``torch.Tensor`` API while preserving Fock metadata:

.. code-block:: python

   sv = StateVector.from_basic_state([1, 0], sparse=False)

   sv_cuda = sv.to("cuda")               # moves tensor, preserves n_modes/n_photons
   sv_copy = sv.clone()                   # independent copy with same metadata
   sv_det  = sv.detach()                  # shares data, no gradient graph
   sv.requires_grad_(True)               # enable gradients in-place; returns self


QuantumLayer integration
^^^^^^^^^^^^^^^^^^^^^^^^

**As** ``input_state`` — sets the initial photon configuration:

.. code-block:: python

   import merlin as ML
   from merlin.core.state_vector import StateVector

   layer = ML.QuantumLayer(
       builder=builder,
       input_state=StateVector.from_basic_state([1, 0, 1, 0]),
       n_photons=2,
       measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
   )

**As input to** ``forward()`` — activates amplitude encoding with classical data:

.. code-block:: python

   import torch
   from merlin.core.state_vector import StateVector

   features = torch.randn(32, len(layer.output_keys))
   sv = StateVector.from_tensor(features, n_modes=4, n_photons=2)
   output = layer(sv)   # shape: (32, output_size)

**As output from** ``forward()`` — with ``MeasurementStrategy.amplitudes(ComputationSpace.FOCK)``
and ``return_object=True``:

.. code-block:: python

   layer = ML.QuantumLayer(
       builder=builder,
       n_photons=2,
       measurement_strategy=ML.MeasurementStrategy.amplitudes(ML.ComputationSpace.FOCK),
       return_object=True,
   )
   sv_out = layer(sv)               # StateVector
   sv_out[[1, 0, 1, 0]]            # amplitude lookup on the output


Perceval interoperability
^^^^^^^^^^^^^^^^^^^^^^^^^

Round-trip between Merlin and Perceval representations:

.. code-block:: python

   import perceval as pcvl
   from merlin.core.state_vector import StateVector

   # Perceval → Merlin
   pcvl_sv = (
       pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0]))
       + pcvl.StateVector(pcvl.BasicState([0, 1, 0, 1]))
   )
   sv = StateVector.from_perceval(pcvl_sv)

   # Merlin → Perceval
   pv_back = sv.to_perceval()
