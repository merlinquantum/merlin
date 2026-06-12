==================================
Kernels: Advanced Guide and Theory
==================================

This page dives into the implementation of MerLin's photonic kernel stack,
backing the API documented in :mod:`merlin.algorithms.kernels`.

Mathematical definition
-----------------------

For a photonic feature map that embeds a datapoint :math:`x` as a unitary
matrix :math:`U(x)` and a chosen input Fock state :math:`|s\rangle`, the
fidelity kernel is

.. math::

		k(x_1, x_2) = \big| \langle s | U^{\dagger}(x_2)\, U(x_1) | s \rangle \big|^2.

MerLin evaluates this quantity by constructing the composite circuit
:math:`U^{\dagger}(x_2)U(x_1)` and computing the transition probability from
the input state to itself under that circuit. When an experiment provides noise
and detectors, the raw probabilities are transformed accordingly before reading
the scalar result.

Architecture overview
---------------------

Four components cooperate to build and evaluate kernels:

1. :class:`~merlin.algorithms.kernels.FeatureMap` – a descriptor that stores
	 the photonic circuit and its parameter layout. It accepts:

	 - a :class:`pcvl.Circuit` (manual construction),
	 - a :class:`~merlin.builder.circuit_builder.CircuitBuilder` (declarative), or
	 - a :class:`pcvl.Experiment` (unitary circuit + measurement semantics).

	 ``FeatureMap`` is a pure descriptor. ``FidelityKernel`` reads its fields
	 (experiment, parameter prefixes, input size, dtype, device) to configure
	 the computation backend and does **not** call any method on it for
	 encoding or unitary construction.

2. ``_CCInvQuantumLayer`` – the internal computation backend, a
	 :class:`~merlin.algorithms.layer.QuantumLayer` subclass constructed by
	 ``FidelityKernel``. It owns encoding, unitary computation, SLOS
	 simulation, pairwise kernel-matrix assembly, and the photon-loss/detector
	 pipeline. All circuit-level work is delegated to this object.

3. :class:`~merlin.algorithms.kernels.FidelityKernel` – validates the feature
	 map, builds ``_CCInvQuantumLayer``, normalizes public inputs, and delegates
	 kernel-matrix construction to the backend.

4. :class:`~merlin.builder.circuit_builder.CircuitBuilder` – declarative
	 helper to build circuits for ``FeatureMap(builder=...)`` (and therefore ``FidelityKernel``).

Data encoding pipeline
----------------------

``FidelityKernel`` encodes datapoints through ``_CCInvQuantumLayer._encode_single``,
which maps a flat feature tensor to the parameter shape the circuit expects.
The supported encoding contract has two cases:

1. If the feature map was created from a
	 :class:`~merlin.builder.circuit_builder.CircuitBuilder`, use its
	 angle‑encoding metadata (``combinations`` and per‑index ``scales``) to
	 compute linear forms of the input vector via
	 ``_prepare_input_encoding``. This guarantees the encoded vector length
	 matches the converter specification for the declared input prefix.
2. Otherwise (plain :class:`pcvl.Circuit` or :class:`pcvl.Experiment`
	 construction), ``FeatureMap.input_size`` must exactly match the number of
	 circuit input parameters selected by ``FeatureMap.input_parameters``. The
	 input is passed through with that parameter dimension.

	 For compatibility, direct circuit and experiment feature maps may still use
	 the legacy ``FeatureMap.encoder`` callable or the legacy subset/truncation
	 behavior when ``FeatureMap.input_size`` differs from the circuit input
	 parameter count. This compatibility path emits a ``DeprecationWarning`` and
	 will be removed in a future release.

.. deprecated:: 0.4

	 The ``encoder`` callable accepted by :class:`~merlin.algorithms.kernels.FeatureMap`
	 is **only** consulted by the legacy :meth:`~merlin.algorithms.kernels.FeatureMap.compute_unitary`
	 path (``FeatureMap._encode_x``) and by ``FidelityKernel`` only for
	 deprecated compatibility with older direct-circuit feature maps. Pass
	 encoding logic through a :class:`~merlin.builder.circuit_builder.CircuitBuilder`
	 or pass inputs with the circuit parameter dimension instead.

Unitary construction
--------------------

The :class:`~merlin.pcvl_pytorch.locirc_to_tensor.CircuitConverter` holds a compiled representation of the photonic
model (unitary compute graph) and exposes ``to_tensor(...)`` to produce a
complex matrix on the configured ``device``/``dtype``. When the feature map is
trainable, the trainable ``torch.nn.Parameter`` objects are registered on
``_CCInvQuantumLayer`` (accessible via ``QuantumLayer.thetas``) and
concatenated before the encoded inputs when calling the converter.

.. note::

	 ``FeatureMap._training_dict`` still holds a copy of the trainable
	 parameters for the deprecated :meth:`~merlin.algorithms.kernels.FeatureMap.compute_unitary`
	 path. ``FidelityKernel`` does not read this dictionary.

Pairwise circuit evaluation and vectorization
---------------------------------------------

Given batches ``X1`` of size :math:`N` and (optionally) ``X2`` of size
:math:`M`, ``_CCInvQuantumLayer`` evaluates the transition probabilities for
all pairs by constructing the set of composite circuits
``U_forward @ U_adjoint`` where ``U_forward = U(x1)`` and
``U_adjoint = U(x2)^{\dagger}``:

* For train Gram matrices (``x2 is None``), only the upper triangular pairs are
	simulated; results are mirrored and the diagonal is filled with ones.
* For test Gram matrices (``x2`` provided), the full :math:`N\times M` set is
	simulated.

The resulting batch of composite unitaries is forwarded to the SLOS compute
graph.

SLOS compute graph
------------------

The kernel builds a SLOS distribution graph via
``build_slos_distribution_computegraph`` with parameters:

* number of modes :math:`m`, total photons :math:`n`,
* ``computation_space`` class, and ``keep_keys=True``.

The graph exposes the list of Fock states (``final_keys``) and a
``compute_probs(unitaries, input_state)`` method that returns transition
probabilities from the given input state to every output state for each
unitary. Internally, the implementation is vectorised (TorchScript‑friendly)
and reuses pre‑computed sparse transitions per layer.

Photon loss and detectors
-------------------------

If the :class:`~merlin.algorithms.kernels.FeatureMap` comes from an experiment (or if the kernel creates
one from its circuit), two transforms may be applied to raw probabilities:

* :class:`~merlin.measurement.photon_loss.PhotonLossTransform` – composes the
	experiment's :class:`pcvl.NoiseModel` into survival probabilities. This
	returns a new probability vector and a new set of output keys.
* :class:`~merlin.measurement.detectors.DetectorTransform` – projects (or maps)
	the post‑loss probabilities to the detector outcome basis (threshold, PNR,
	etc.).

The scalar fidelity value is then read either at the unique index that matches
the (surviving) input detection event or as a weighted sum across the detection
vector when several detection outcomes are compatible with the input pattern.

.. note::

	 Only ``ComputationSpace.FOCK`` can be combined with experiments that define
	 detectors. The kernel raises a ``RuntimeError`` if at least one
	 :class:`pcvl.Detector` is present in the experiment and any non-FOCK
	 computation space (e.g. ``UNBUNCHED`` or ``DUAL_RAIL``) is requested.

Sampling and autodiff
---------------------

If ``shots > 0``, the kernel converts exact detection probabilities to sampled
counts via the configured pseudo‑sampler (multinomial/binomial/gaussian) from
the :class:`~merlin.measurement.autodiff.AutoDiffProcess`. This enables
benchmarking robustness to shot noise. For gradient‑based learning of trainable
feature maps, keep ``shots=0`` to work with exact probabilities.

PSD projection and numerical safeguards
---------------------------------------

With ``force_psd=True`` (default), the symmetric train Gram matrix is
projected to the closest positive semi‑definite matrix by zeroing negative
eigenvalues in an eigendecomposition. This prevents downstream solvers from
failing due to small numerical inconsistencies. For test matrices, PSD
projection is applied only when inputs are equal (``X2 is None`` or
``X2 == X1``).

Shapes, devices and dtypes
--------------------------

* Inputs are reshaped to ``[N, input_size]`` (and ``[M, input_size]`` when
	``x2`` is provided). Scalars and 1D vectors are validated by
	:meth:`~merlin.algorithms.kernels.FeatureMap.is_datapoint` for single‑pair evaluations.
* All intermediate tensors are created on the feature map's device/dtype unless
	explicit overrides are passed to the kernel.
* The SLOS graph internally operates on complex dtypes that match the chosen
	float precision.

Complexity and performance tips
-------------------------------

* Reduce ``m`` (modes) or ``n`` (photons) to shrink the Fock space; use the 
    ``ComputationSpace.UNBUNCHED`` computation space instead of ``ComputationSpace.FOCK``
	when your circuit forbids multi‑occupancy per mode.
* Reuse feature maps and kernels across batches to amortize converter/setup
	costs.
* Keep inputs contiguous and on the same device to minimise transfers.
* Avoid sampling during model selection; add ``shots`` when stress‑testing.

Limitations
-----------

* The kernel API encodes classical inputs via angle encoding; amplitude‑encoded
	state vectors are not part of this kernel stack.
* Experiments passed to the kernel must be unitary and without post‑selection
	or heralding. Non‑unitary experiments are rejected.

----------------

For class/method signatures and basic usage examples, see the API reference:
:mod:`merlin.algorithms.kernels`.
