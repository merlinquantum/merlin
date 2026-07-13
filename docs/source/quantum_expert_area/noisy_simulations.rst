:github_url: https://github.com/merlinquantum/merlin

==============================================
Noisy Simulations
==============================================

This page describes how Merlin implements noisy SLOS simulations. For a
practical introduction to the available noise parameters and a first
:class:`~merlin.algorithms.layer.QuantumLayer` example, see
:doc:`/user_guide/noisy_simulations`.

----------------------------------------------
Noisy Simulation Implementation
----------------------------------------------

SLOS normally propagates amplitudes. Active source noise and stochastic phase
error are represented as probability mixtures, so Merlin computes and combines
probability distributions for those cases. The implementation details below
describe where each noise family enters the computation.

Brightness and Transmittance
----------------------------------------------

Brightness and transmittance are implemented in the same workflow. Merlin first
computes the ideal probability tensor for the fixed ``n``-photon, ``m``-mode
Fock space, then applies photon survival as a transition matrix over the output
probabilities. The survival probability for each photon is the product of
brightness and transmittance.

**Algorithm**

1. Compute

   .. math::

      l = \sum_{i=0}^{n} \dim(\mathcal{F}_{m,i})

   where :math:`\mathcal{F}_{m,i}` denotes the Fock space of
   ``m`` modes and ``i`` photons.

2. Initialize the transition matrix as a tensor of zeros with shape

   .. math::

      (\dim(\mathcal{F}_{m,n}), l)

3. For each basis state in the ``(m, n)`` Fock space:

   * For each possible output state:

     1. Compute the probability

        .. math::

           \binom{n}{n_{\mathrm{survived}}}
           (b t)^{n_{\mathrm{survived}}}
           (1-b t)^{n-n_{\mathrm{survived}}}

        where:

        * :math:`n_{\mathrm{survived}}` is the number of photons that
          survive,
        * :math:`b` is the source brightness,
        * :math:`t` is the transmittance.

     2. Assign the probability to the appropriate column of the
        transition matrix using the output state's corresponding index
        and key.

4. Return the completed transition matrix.

The possible output states are all of the possible combinations of losing photons in the basis state.

For g2 simulations, the photon loss algorithm is applied per n-photon sector at the output of the simulation. Indeed, the transition matrix is different for each sector. After this noise is applied, the probabilities are reclassified and returned as a large tensor.

Phase Error and Imprecision
----------------------------------------------

Circuit phase noise is applied while Merlin builds the differentiable circuit
unitary.

``phase_imprecision`` is deterministic. Each phase shifter value is quantized to
the nearest multiple of ``phase_imprecision`` during the forward pass:

.. math::

   \phi_\text{quantized}
   =
   \operatorname{round}\left(\frac{\phi}{\Delta \phi}\right)\Delta \phi

where :math:`\Delta \phi` is ``phase_imprecision``. This is nearest-grid
rounding, not truncation. Merlin uses ``torch.round`` for this operation, so
exact half-step ties follow PyTorch's rounding behavior. For example, if the
commanded phase is :math:`\pi/8` and ``phase_imprecision`` is :math:`\pi/4`,
then :math:`\phi / \Delta \phi = 0.5` and the quantized phase is ``0``. Values
slightly above :math:`\pi/8` quantize to :math:`\pi/4`. Merlin uses a
straight-through estimator, so gradients still flow through the commanded phase
value even though the forward pass uses the quantized phase.

``phase_error`` is stochastic. For each Monte Carlo sample, Merlin draws a fresh
Torch random perturbation from ``Uniform(-phase_error, phase_error)`` for every
phase shifter, builds one unitary, computes probabilities, then averages the
probability distributions. Amplitudes are not averaged. Use
``torch.manual_seed(...)`` to make the sampled perturbations reproducible.

Black-box unitaries under circuit phase noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pcvl.Unitary`` components contain no explicit phase shifters, but on hardware
they are implemented by a programmable interferometer whose phases are just as
imprecise as standalone phase shifters. When ``phase_imprecision`` or
``phase_error`` is active, Merlin therefore decomposes every ``pcvl.Unitary``
component into a rectangular (Clements) mesh of Mach-Zehnder interferometers
with fixed numeric phases plus an output phase layer, so the configured phase
noise applies to every physical phase. ``pcvl.PERM`` components are waveguide
crossings without programmable phases and are never decomposed.

The decomposition runs once, when the layer is built (about 0.2 s for a
10-mode unitary), and only when circuit phase noise is configured; without
noise, each unitary stays precomputed as a constant tensor, exactly as before.
The numerical fit reproduces the target unitary with a fidelity error of about
``1e-6``, which corresponds to matrix entries agreeing to about ``1e-3``. The
fit's arbitrary global phase is compensated on the mesh's output phase layer,
so unitaries acting on a subset of modes keep their relative phase with the
rest of the circuit. If the fit does not converge, construction raises an
error instead of silently changing the mesh topology.

Coherent and incoherent error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A coherent error is represented by a unitary transformation. It preserves the
phase relations between components of a quantum state, so amplitudes can still
interfere before they are converted to probabilities. In Merlin, deterministic
``phase_imprecision`` is coherent within a forward pass: it changes the phase
values used to build the unitary, but the circuit is still evaluated as one
unitary.

``phase_error`` is a stochastic coherent error at the sample level. Each Monte
Carlo sample draws one perturbed unitary and evaluates the quantum evolution
coherently for that unitary. Merlin then converts that sample's output
amplitudes to probabilities and averages the probabilities over all sampled
unitaries:

.. math::

   \frac{1}{K}\sum_{k=1}^{K} p(U_k, \psi)
   =
   \frac{1}{K}\sum_{k=1}^{K} \left|U_k\psi\right|^2

This is different from averaging amplitudes or unitaries first:

.. math::

   \left|\frac{1}{K}\sum_{k=1}^{K} U_k\psi\right|^2

Merlin does not use this second expression for ``phase_error``.

For example, consider two sampled output states:

.. math::

   \psi_+
   =
   \frac{1}{\sqrt{2}}\left(|10\rangle + |01\rangle\right),
   \qquad
   \psi_-
   =
   \frac{1}{\sqrt{2}}\left(|10\rangle - |01\rangle\right)

Each sampled state has probability :math:`1/2` on :math:`|10\rangle` and
probability :math:`1/2` on :math:`|01\rangle`, so averaging probabilities keeps
the 50/50 distribution. Averaging amplitudes first cancels the
:math:`|01\rangle` component. If that averaged amplitude vector is then
renormalized, it becomes :math:`|10\rangle`, giving probability 1 on
:math:`|10\rangle` and probability 0 on :math:`|01\rangle`. This is not the
Monte Carlo probability mixture used by Merlin.

An incoherent error is represented as a classical mixture of alternatives. The
relative phases between alternatives are not used for interference; Merlin
combines probabilities, not amplitudes. Source noise is handled this way. For a
tensor input state interpreted as a superposition, source-noise simulations
propagate each active input basis state independently and combine the resulting
probability distributions with weights :math:`|c_i|^2`.

The practical consequence is:

- with circuit phase noise only, a tensor input superposition remains coherent
  inside each sampled unitary;
- with source noise, tensor input components are treated as an incoherent
  mixture over basis states;
- with ``phase_error``, the final reported distribution is an incoherent Monte
  Carlo average of probability distributions, even though each sampled unitary
  is evaluated coherently.

When both circuit phase noises are active, Merlin first quantizes the phase and
then samples the stochastic perturbation around the quantized value:

.. math::

   \phi_\text{effective}
   =
   \operatorname{round}\left(\frac{\phi}{\Delta \phi}\right)\Delta \phi
   +
   \epsilon,
   \qquad
   \epsilon \sim \operatorname{Uniform}(-e, e)

where :math:`e` is ``phase_error``. If ``phase_imprecision`` is inactive, Merlin
uses :math:`\phi + \epsilon`. If ``phase_error`` is inactive, Merlin uses only
the deterministic quantized phase.

The ``n_phase_error_samples`` constructor parameter controls how many sampled
unitaries are averaged when active ``phase_error`` is present. If omitted,
Merlin uses 1 sample. Runtime scales roughly linearly with this value when
``phase_error > 0``. When source noise or ``g2`` is also active, the cost is
multiplicative: each phase-error sample runs the full source-noise mixture, so
the worst-case runtime is roughly
``n_phase_error_samples * n_active_input_states * SLOS``.

Suggested values:

- ``1`` for Perceval-like stochastic circuit sampling.
- ``5`` to ``10`` for quick expected-noise estimates.
- ``50`` to ``100`` for validation studies.
- ``200`` or more for production or publication results.

The parameter is ignored when ``phase_error`` is ``None`` or ``0.0``.

.. code-block:: python

    import math

    import perceval as pcvl
    import torch
    import merlin as ML

    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.BS.H())

    layer = ML.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 0],
        n_photons=1,
        trainable_parameters=["phi"],
        noise=pcvl.NoiseModel(
            phase_imprecision=math.pi / 4,
            phase_error=0.1,
        ),
        n_phase_error_samples=50,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
    )

    torch.manual_seed(42)
    probs_1 = layer()
    torch.manual_seed(42)
    probs_2 = layer()

    assert torch.allclose(probs_1, probs_2)

In this example, a commanded trainable phase equal to ``math.pi / 8`` would be
quantized to ``0`` before the stochastic perturbation is added, because it lies
exactly halfway between the ``0`` and ``math.pi / 4`` grid points and
``torch.round(0.5)`` returns ``0``.

Indistinguishability
----------------------------------------------

This noise is implemented using the one-bad-bit (OBB) principle first introduced in Merlin in this `reproduced paper <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/photonic_quantum_enhanced_kernels/utils/noise.py>`_. Indeed, since distinguishable photons can be tracked independently, each of those independent photons can be simulated separately. This implementation is done in the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph` class. Here are the main steps:

**Algorithm**

1. Create a
   :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
   object for ``m`` modes and photon numbers ranging from ``1`` to ``n``,
   where ``n`` is the number of photons in the input state.

2. For each possible configuration of distinguishable photons:

   * Run a single-photon SLOS simulation for each distinguishable
     photon.

   * Run a SLOS simulation for all remaining photons, treating them as
     indistinguishable.

   * Convolve the resulting output distributions.

   * Multiply the convolved distribution by

     .. math::

        p^{\,n-n_{\mathrm{dist}}}
        (1-p)^{\,n_{\mathrm{dist}}}

     where:

     * :math:`p` is the photon indistinguishability,
     * :math:`n` is the number of photons in the input state,
     * :math:`n_{\mathrm{dist}}` is the number of distinguishable
       photons.

3. Sum the weighted distributions into the output tensor.

4. Return the resulting output tensor.

The combinations of possible distinguishable photons are simply the ways of choosing 0 to n photons from the input state.


g2 and g2 distinguishable
----------------------------------------------

These noises build on the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph` class to create the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisyG2SLOSComputeGraph` object. Indeed, the noisy simulation must be performed for multiple input states with photon duplication. Here are the main implementation steps.

The ``g2`` value is converted to the probability :math:`p` that one source
emits an extra photon:

.. math::

   p =
   \frac{1-g^{(2)}-\sqrt{1-2g^{(2)}}}{2g^{(2)}}

Only the no-extra-photon and one-extra-photon cases are modeled for each input
photon. Higher-order emissions from the same source are not included.

**Algorithm**

1. Create a :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph`
   object for ``m`` modes and ``i`` photons, with ``i`` ranging from ``n``
   to ``2n``.

2. If ``g2_distinguishable`` is ``True``, create a
   :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
   object for ``m`` modes and a single photon.

3. Generate all possible photon-addition vectors. These are the vectors for all possible sequences of duplicated photons.

   For example, for the input state ``[1,1,0]``:

   * ``[[0,0,0]]`` — no duplicated photons.
   * ``[[1,0,0], [0,1,0]]`` — one duplicated photon.
   * ``[[1,1,0]]`` — both photons duplicated.

   Group the vectors according to the number of added photons.

4. For each group corresponding to ``i`` added photons:

   * For each photon-addition vector:

     * If ``g2_distinguishable`` is ``True``:

       1. Run the original input state on the
          :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
          with ``n`` photons.

       2. Run the single-photon SLOS computation for each added photon.

       3. Convolve the resulting output distributions.

     * Otherwise:

       1. Run the augmented input state on the
          :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
          with ``n+i`` photons.

     * Multiply the resulting distribution by

       .. math::

          p^{i}(1-p)^{n-i}

       where ``p`` is the probability that two photons are emitted and
       ``n`` is the photon number of the desired input state.

   * Combine all distributions into the tensor representing the
     ``n_i`` photon sector.

5. Return the probability distribution for each photon sector.

----------------------------------------------
Noisy Simulations Limitations
----------------------------------------------

Noisy simulations are significantly less efficient than ideal ones. You can profile the memory requirements of noisy simulations with source noise using the benchmark script: :file:`../../benchmarks/benchmark_noisy_slos_cache_memory.py`.

Memory and computational complexity grow significantly with the number of modes and photons. For example, a 5-photon 2-mode circuit requires around 200 MB, while a 20-mode 3-photon experiment requires around 3 GB. This is only for indistinguishable photons. Memory requirements are even greater for g2 simulations. To profile memory consumption in your specific use case, run the benchmark script with:

.. code-block:: bash

    python benchmarks/benchmark_noisy_slos_cache_memory.py --modes 6 7 8 9 --photons 1 2 3 4 5 --backward

Here is an example of the output graph of this run.

.. figure:: images/benchmark_noisy_slos_cache_memory.png
   :align: center
   :width: 600px
   :alt: Memory need for the QuantumLayer with distinguishable photons per output size

The public API constraints are listed in :doc:`/user_guide/noisy_simulations`.
The implementation reason is that Merlin does not currently use a density
matrix representation for noisy SLOS. Noise paths that produce classical
mixtures therefore return probabilities and cannot expose a single coherent
output amplitude vector.
