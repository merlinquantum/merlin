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

Three components cooperate to build and evaluate kernels:

1. :class:`~merlin.algorithms.kernels.FeatureMap` – embeds classical data into
	 a photonic circuit and returns unitaries ``U(x)``. It accepts:

	 - a :class:`perceval.Circuit` (manual construction),
	 - a :class:`~merlin.builder.circuit_builder.CircuitBuilder` (declarative), or
	 - a :class:`perceval.Experiment` (unitary circuit + measurement semantics).

2. :class:`~merlin.algorithms.kernels.FidelityKernel` – computes kernel values
	 from a feature map and an input Fock state using SLOS.

3. :class:`~merlin.algorithms.kernels.KernelCircuitBuilder` – convenience
	 helper to produce a standard feature map and fidelity kernel.

Data encoding pipeline (FeatureMap)
-----------------------------------

``FeatureMap`` takes a datapoint and maps it to the exact parameter vector that
its circuit expects before calling the Torch converter
(:class:`~merlin.pcvl_pytorch.locirc_to_tensor.CircuitConverter`) to obtain
``U(x)``. The encoding logic follows a strict preference order:

1. If the feature map was created from a :class:`CircuitBuilder`, use its
	 angle‑encoding metadata (``combinations`` and per‑index ``scales``) to
	 compute linear forms of the input vector. This guarantees the encoded vector
	 length matches the converter specification for the declared input prefix.
2. Otherwise, if the user provided a callable ``encoder(x)``, use it. The
	 output is validated against the expected length; if invalid or failing, the
	 code falls back to the deterministic expansion below.
3. As a final fallback, the deterministic subset‑sum expansion enumerates and
	 sums non‑empty feature subsets in lexicographic order until the expected
	 parameter length is reached. This matches the legacy behaviour of older
	 feature maps.

Unitary construction
--------------------

The :class:`CircuitConverter` holds a compiled representation of the photonic
model (unitary compute graph) and exposes ``to_tensor(...)`` to produce a
complex matrix on the configured ``device``/``dtype``. When the feature map is
trainable, extra trainable torch parameters (``torch.nn.Parameter``) are
registered on the feature map and concatenated after the encoded inputs.

Pairwise circuit evaluation and vectorization
---------------------------------------------

Given batches ``X1`` of size :math:`N` and (optionally) ``X2`` of size
:math:`M`, the kernel evaluates the transition probabilities for all pairs by
constructing the set of composite circuits
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

If the :class:`FeatureMap` comes from an experiment (or if the kernel creates
one from its circuit), two transforms may be applied to raw probabilities:

* :class:`~merlin.measurement.photon_loss.PhotonLossTransform` – composes the
	experiment's :class:`perceval.NoiseModel` into survival probabilities. This
	returns a new probability vector and a new set of output keys.
* :class:`~merlin.measurement.detectors.DetectorTransform` – projects (or maps)
	the post‑loss probabilities to the detector outcome basis (threshold, PNR,
	etc.).

The scalar fidelity value is then read either at the unique index that matches
the (surviving) input detection event or as a weighted sum across the detection
vector when several detection outcomes are compatible with the input pattern.

.. note::

	 ``ComputationSpace.UNBUNCHED`` cannot be combined with experiments that define
	 detectors. The kernel raises if detectors are present and ``ComputationSpace.UNBUNCHED`` or ``ComputationSpace.FOCK`` is
	 requested.

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
	:meth:`FeatureMap.is_datapoint` for single‑pair evaluations.
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
* ``KernelCircuitBuilder.bandwidth_tuning`` is a placeholder in the current
	release.

Cross‑references
----------------

For class/method signatures and basic usage examples, see the API reference:
:mod:`merlin.algorithms.kernels`.

