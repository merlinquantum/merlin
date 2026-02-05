===============================
Quantum Bridge (Qubit ↔ Merlin)
===============================

Overview
--------
The Quantum Bridge lets you plug a qubit state-preparation module into a Merlin ``QuantumLayer`` by mapping computational basis states into photonic Fock states using a one-photon-per-group encoding. Any PyTorch-compatible noisy or noiseless simulator that outputs a complex statevector of size :math:`2^n` can be placed upstream—PennyLane is a common choice but not a requirement.

Key ideas:
- You provide a preconfigured ``QuantumLayer`` (the bridge is parameter-free).
- You insert the bridge between a qubit-state module (e.g., a PennyLane ``QNode`` with ``interface="torch"``) that outputs a complex statevector of size :math:`2^n` and the target ``QuantumLayer``.
- You partition the ``n`` qubits into groups via ``qubit_groups``; ``[2, 1]`` means a two-qubit block encoded in four modes plus a dual-rail qubit, generalising dual-rail into the full QLOQ family.
- ``n_modes`` must equal :math:`\sum_i 2^{k_i}` and ``n_photons`` equals ``len(qubit_groups)``.
- Select the target computation space (``fock``, ``unbunched``, or ``dual_rail``); the bridge builds the corresponding transition matrix and emits amplitudes already aligned with the photonic key ordering.
- ``QuantumLayer`` can now consume the emitted tensor directly—no metadata plumbing or manual indexing required.

Minimal example
---------------
.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin import MeasurementStrategy, QuantumLayer
    from merlin.bridge.quantum_bridge import ComputationSpace, QuantumBridge

    # Build a simple identity photonic circuit with m = sum(2**g) modes
    qubit_groups = [1, 1]  # two groups of one qubit each → 2 photons, 4 modes
    m = sum(2**g for g in qubit_groups)

    circuit = pcvl.Circuit(m)
    layer = QuantumLayer(
        input_size=0,
        circuit=circuit,
        n_photons=len(qubit_groups),
        measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    class QubitStatePrep(torch.nn.Module):
        """Return a qubit statevector |01> (similar to what a PennyLane QNode would emit)."""

        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            psi = torch.zeros(4, dtype=torch.complex64)
            psi[1] = 1 + 0j  # |01>
            return psi

    state_prep = QubitStatePrep()
    bridge = QuantumBridge(
        qubit_groups=qubit_groups,
        n_modes=m,
        n_photons=len(qubit_groups),
        wires_order="little",  # or "big"
        computation_space=ComputationSpace.UNBUNCHED,
        normalize=True,        # L2-normalize the input state
    )

    model = torch.nn.Sequential(state_prep, bridge, layer)

    x = torch.zeros(1, 1)  # dummy input; state prep ignores it
    y = model(x)           # probability distribution over photonic outcomes

The bridge left-multiplies the statevector by its precomputed transition matrix and emits
amplitudes ordered exactly like ``QuantumLayer``'s ``mapped_keys``.  Gradients then flow
from ``y`` back through the photonic layer into the upstream qubit state preparation.
Because ``nn.Sequential`` modules exchange a single argument, the bridge does not forward
additional positional inputs; wrap bridge and layer in a custom module if you need to
thread extra data alongside the statevector.

.. note::
   The :meth:`~merlin.bridge.quantum_bridge.QuantumBridge.qubit_to_fock_state` helper exposes the
   exact bitstring → Fock-state mapping used internally. This is handy for spot-checking amplitudes
   or visualising how logical qubits populate photonic modes before running a full simulation.

Choosing the computation space
------------------------------
``computation_space`` controls both the size and ordering of the emitted tensor:

- ``ComputationSpace.DUAL_RAIL`` expects every ``qubit_groups`` entry to be 1 and produces
  :math:`2^n` outcomes—one per logical basis state.
- ``ComputationSpace.UNBUNCHED`` enumerates all configurations with at most one photon per
  mode (:math:`\binom{m}{n_{\text{photons}}}` outcomes).  The bridge populates only the
  subset consistent with its qubit groups; the remaining entries are zero.
- ``ComputationSpace.FOCK`` generates the full Fock space with bunching allowed
  (:math:`\binom{m + n_{\text{photons}} - 1}{n_{\text{photons}}}` outcomes).  As with the
  unbunched space, amplitudes outside the logical subspace are zero but remain available to
  downstream Merlin components.

Convenience properties ``bridge.n_modes`` and ``bridge.n_photons`` expose the photonic
settings that should be used when instantiating the receiving ``QuantumLayer``.

Inspecting the QLOQ transformation
----------------------------------
The bridge conceptually applies a transition matrix that maps the computational basis to
the photonic QLOQ basis.  Calling :meth:`~merlin.bridge.quantum_bridge.QuantumBridge.transition_matrix`
returns this sparse operator immediately after bridge construction. The matrix has shape
``(output_dim, 2**n_qubits)`` and contains exactly one non-zero entry per computational
basis column.

Devices and dtypes
------------------
- The bridge output device follows the `QuantumLayer` device. Ensure the layer and bridge use the same device (CPU/CUDA).
- The bridge converts input states to complex64 for float32 and to complex128 for float64. The emitted probability distribution keeps the `QuantumLayer` real dtype (float32/float64).

Constraints
-----------
- No ancilla or postselected modes; total modes m must equal sum(2**group_size).
- Number of photons equals the number of groups.
- `QuantumLayer` must be provided; the bridge does not create circuits or perform validation against it.

API
---
.. automethod:: merlin.bridge.QuantumBridge.qubit_to_fock_state

.. automodule:: merlin.bridge.QuantumBridge
   :members: QuantumBridge
   :undoc-members:
   :show-inheritance:
