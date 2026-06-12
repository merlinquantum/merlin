===============================
MerlinProcessor API Reference
===============================

.. module:: merlin.core.merlin_processor

Overview
========
:class:`MerlinProcessor` is an RPC-style bridge that offloads **quantum leaves**
(e.g., layers exposing ``export_config()``) to a Perceval backend, while keeping
classical layers local. It supports these backend entry points:

* **Perceval** ``perceval.runtime.AProcessor`` — the primary constructor
  argument. Local, non-remote processors execute through synchronous Perceval
  local jobs.
* **Perceval** :class:`~pcvl.RemoteProcessor` — the original
  Quandela Cloud path. Passing a RemoteProcessor through ``processor=`` routes
  to this same path.
* **Perceval** `pcvl.runtime.session.ISession <https://perceval.quandela.net/docs/v1.2/providers.html#scaleway>`_ — the preferred path
  for Scaleway-hosted platforms (and any future session-based providers).

All execution paths support batched execution, per-call/global timeouts,
cooperative cancellation, and a Torch-friendly async interface returning
:class:`torch.futures.Future`. Remote backends additionally support chunking
with limited intra-leaf concurrency.

Key Capabilities
----------------
* Automatic traversal of a PyTorch module; offloads only **quantum leaves**.
* Remote batch **chunking** (``microbatch_size``) and **parallel** submission
  per leaf (``chunk_concurrency``).
* **Synchronous** (``forward``) and **asynchronous** (``forward_async``) APIs.
* **Cancellation** of a single call or **all** calls in flight.
* **Timeouts** that cancel in-flight cloud jobs for remote backends and check
  local jobs before and after synchronous execution.
* Isolated backend objects: local processors are rebuilt from copied
  non-circuit experiment state, while :class:`~pcvl.RemoteProcessor` objects
  are cloned from the original (RemoteProcessor path) or built from the session
  (ISession path).
* Stable, descriptive cloud job names (capped to 50 chars) for remote jobs.

.. note::
   Execution is supported both with exact probabilities (if the backend exposes
   the ``"probs"`` command) and with sampling (``"sample_count"`` or
   ``"samples"``). Shots are *user-controlled* via ``nsample``; there is no
   hidden auto-shot selection.

Key current limitation
-----------------------
* No gradient can flow through the MerLin processor. When calling backwards directly on a MerlinProcessor's
  forward pass, this error will be shown::

      element 0 of tensors does not require grad and does not have a grad_fn

* If a MerLin processor is part of a bigger ``Pytorch`` module, the quantum layer's parameter gradient will be None.
  This also means that every upstream models' (models called before the :class:`~merlin.algorithms.layer.QuantumLayer` in the model's forward) gradient will be None.

* The gradient can not pass through right now as pytorch can not access
  directly the computation on the QPU.

* Differentiation through the MerLin processor will be implemented in v0.5.

Class Reference
===============

BackendCapabilities
-------------------
.. class:: BackendCapabilities(name, available_commands)

   Immutable dataclass encapsulating capabilities extracted from a Perceval
   backend.

   **Attributes**

   .. attribute:: name

      ``str`` — Backend platform name (e.g., ``"sim:slos"``,
      ``"perceval-qpu:scaleway"``).

   .. attribute:: available_commands

      ``list[str]`` — Commands supported by the backend (e.g., ``"probs"``,
      ``"sample_count"``, ``"samples"``).

   **Example**

   .. code-block:: python

      proc = MerlinProcessor(processor=rp)
      caps = proc.backend_capabilities
      print(f"Platform: {caps.name}")
      print(f"Supports: {caps.available_commands}")

MerlinProcessor
---------------
.. class:: MerlinProcessor(remote_processor=None, session=None, microbatch_size=32, timeout=3600.0, max_shots_per_call=None, chunk_concurrency=1, token=None, *, processor=None)

   Create a processor that offloads quantum leaves to a Perceval backend.
   Exactly **one** of ``processor``, ``remote_processor``, or ``session`` must
   be provided.

   .. warning:: *Deprecated since version 0.4:*
      The ``remote_processor`` constructor argument is deprecated and will be
      removed in a future release. Pass the same
      :class:`~pcvl.RemoteProcessor` through ``processor=`` instead.

   :param remote_processor: Deprecated authenticated Perceval
      :class:`~pcvl.RemoteProcessor` entry point. Pass the same object through
      ``processor=`` instead. Merlin clones it per chunk so concurrent jobs
      have independent state. Type: ``RemoteProcessor | None``.
   :param session: A Perceval `pcvl.runtime.session.ISession <https://perceval.quandela.net/docs/v1.2/providers.html#scaleway>`_
      object — e.g. from ``pcvl.providers.scaleway.Session``. Merlin calls
      ``session.build_remote_processor()`` per chunk, giving each chunk
      an independent RP. Type: ``ISession | None``.
   :param int microbatch_size: Maximum **rows per remote backend chunk**.
      Ignored for local processors.
   :param float timeout: Default wall-time limit (seconds) per call. Per-call
      override via ``timeout=...`` on API methods.
   :param max_shots_per_call: Hard cap on **shots per backend call**.
      If ``None``, a safe default is used internally. If ``nsample`` exceeds
      this cap, Merlin clamps the submitted sample count to this value.
      Type: ``int | None``.
   :param int chunk_concurrency: Max number of remote chunk jobs in flight
      **per quantum leaf** during a single call. Ignored for local processors.
      ``>=1`` (default: 1, i.e., serial).
   :param token: Optional authentication token forwarded to cloned remote
      processors. If omitted, Merlin extracts the token from the
      RemoteProcessor's RPC handler. Ignored for local and session paths.
      Type: ``str | None``.
   :param processor: Keyword-only Perceval ``perceval.runtime.AProcessor``.
      Local processors execute through synchronous local Perceval jobs and do
      not require remote token extraction. RemoteProcessor instances passed here
      are normalized to the existing remote backend. Type:
      ``AProcessor | None``.

   :raises TypeError: If exactly one backend is not provided, if the provided
      argument is not the expected type, or if ``processor`` is an unsupported
      remote AProcessor subclass.
   :raises ValueError: If no authentication token can be resolved on the
      RemoteProcessor path (neither provided explicitly nor extractable from
      the processor's RPC handler).

   **Attributes**

   .. attribute:: backend_capabilities

      :class:`BackendCapabilities` — Encapsulates backend name and available
      commands extracted at initialization time. This is the primary way to
      access backend capabilities.

   .. attribute:: backend_kind

      ``str`` — Active backend route: ``"local_processor"``,
      ``"remote_processor"``, or ``"session"``.

   .. attribute:: backend_name

      ``str`` — Backward-compatibility property. Equivalent to
      ``backend_capabilities.name``. Best-effort backend name from the remote
      processor or session (e.g., ``"sim:slos"``).

   .. attribute:: available_commands

      ``list[str]`` — Backward-compatibility property. Equivalent to
      ``backend_capabilities.available_commands``. Commands exposed by the
      backend (e.g., ``"probs"``, ``"sample_count"``, ``"samples"``).

   .. attribute:: processor

      ``AProcessor | None`` — set when constructed with a local, non-remote
      ``processor``; ``None`` for remote and session paths.

   .. attribute:: remote_processor

      ``RemoteProcessor | None`` — set when constructed with
      ``remote_processor`` or a RemoteProcessor passed through ``processor``;
      ``None`` for local and session paths.

   .. attribute:: session

      ``ISession | None`` — set when constructed with ``session``;
      ``None`` for local and RemoteProcessor paths.

   .. attribute:: microbatch_size
                  default_timeout
                  max_shots_per_call
                  chunk_concurrency

      Constructor options reflected on the instance.

   .. attribute:: DEFAULT_MAX_SHOTS
                  DEFAULT_SHOTS_PER_CALL

      Library constants used when computing defaults for sampling paths.

Context Management
------------------
.. method:: __enter__()
.. method:: __exit__(exc_type, exc, tb)

   Entering returns the processor. Exiting triggers a best-effort
   :meth:`cancel_all` to ensure no stray jobs remain.

Execution APIs
--------------
.. method:: forward(module, input, *, nsample=None, timeout=None) -> torch.Tensor

   Synchronous convenience around :meth:`forward_async`.

   :param torch.nn.Module module: A Torch module/tree. Leaves exposing
      ``export_config()`` (and not ``force_local=True``) are offloaded.
   :param torch.Tensor input: 2D batch ``[B, D]`` or shape required by the
      first leaf. Tensors are moved to CPU for backend execution if needed; the
      result is moved back to the input's original device/dtype.
   :param int | None nsample: Shots per input when sampling. Ignored if the
      backend supports exact probabilities (``"probs"``).
   :param float | None timeout: Per-call override. ``None``/``0`` == unlimited.
   :returns: Output tensor with batch dimension ``B`` and leaf-determined
      distribution dimension.
   :rtype: torch.Tensor
   :raises RuntimeError: If ``module`` is in training mode.
   :raises TimeoutError: On global per-call timeout. Remote jobs are cancelled
      best-effort; local synchronous jobs are checked before and after execution.
   :raises concurrent.futures.CancelledError: If the call is cooperatively
      cancelled via the async API.

.. method:: forward_async(module, input, *, nsample=None, timeout=None) -> torch.futures.Future

   Asynchronous execution. Returns a :class:`torch.futures.Future` with extra
   helpers attached:

   **Future extensions**

   * ``future.job_ids: list[str]`` — accumulates remote job IDs across chunk jobs.
   * ``future.status() -> dict`` — current state/progress/message plus chunk
     counters: ``{"chunks_total", "chunks_done", "active_chunks"}``.
   * ``future.cancel_remote() -> None`` — cooperative cancel; in-flight jobs are
     best-effort cancelled and ``future.wait()`` raises
     ``CancelledError``.

   :param module: See :meth:`forward`.
   :param input: See :meth:`forward`.
   :param nsample: See :meth:`forward`.
   :param timeout: See :meth:`forward`.
   :returns: Future that resolves to the same tensor as :meth:`forward`.

Job & Lifecycle Utilities
-------------------------
.. method:: cancel_all() -> None

   Best-effort cancellation of **all** active jobs across outstanding calls.

.. method:: get_job_history()

   :returns: List of remote job handles.
   :rtype: list[``perceval.runtime.RemoteJob``]


   Returns a list of all jobs observed/submitted by this instance during the
   process lifetime (useful for diagnostics).

.. method:: clear_job_history() -> None

   Clears the internal job history list.

Shot Estimation (No Submission)
-------------------------------
.. method:: estimate_required_shots_per_input(layer, input, desired_samples_per_input) -> list[int]

   Ask the platform estimator how many shots are required **per input row** to
   reach a target number of *useful* samples. This helper is available only
   for remote processor and session backends.

   :param torch.nn.Module layer: A quantum leaf (must implement
      ``export_config()``).
   :param torch.Tensor input: ``[B, D]`` or a single vector ``[D]``. Values are
      mapped to the circuit parameters as they would be during execution.
   :param int desired_samples_per_input: Target **useful** samples per input.
   :returns: ``list[int]`` of length ``B`` (``0`` indicates "not viable" under
      current settings).
   :rtype: list[int]
   :raises RuntimeError: If called on a local ``processor=`` backend.
   :raises TypeError: If ``layer`` does not expose ``export_config()``.
   :raises ValueError: If ``input`` is not 1D or 2D.

Execution Semantics
-------------------
Traversal & Offload
^^^^^^^^^^^^^^^^^^^
* Leaves with ``export_config()`` are treated as **quantum leaves** and are
  offloaded unless they expose a ``should_offload()`` method that returns
  ``False``, or they set ``force_local=True``.
* Non-quantum leaves run locally under ``torch.no_grad()``.

Batching & Chunking
^^^^^^^^^^^^^^^^^^^
* Local processors keep the full PyTorch input as one Merlin-level batch, then
  represent its rows as Perceval sampler iterations. Perceval does not execute
  a native vectorized batch. ``microbatch_size`` and ``chunk_concurrency`` do
  not affect local processor execution; users control local batch size through
  the input batch they pass to PyTorch.
* RemoteProcessor and ISession backends split batches larger than
  ``microbatch_size`` into chunks of size ``<= microbatch_size``. Up to
  ``chunk_concurrency`` chunk jobs per quantum leaf are submitted in parallel.
* Remote chunks are retried up to 3 times with exponential backoff. Local
  synchronous execution propagates errors directly. Cancellation and timeout
  errors propagate immediately without retry.

Backends & Commands
^^^^^^^^^^^^^^^^^^^
* Backend capabilities (name and available commands) are extracted once at
  initialization and stored in ``backend_capabilities``. Local processors use
  their own ``name`` and ``available_commands`` directly. The RemoteProcessor
  path uses the original RP, and the ISession path uses the first processor
  built from the session.
* If the backend exposes ``"probs"`` and  ``nsample`` is None or 0, the processor queries exact probabilities.
* Otherwise it uses ``"sample_count"`` or ``"samples"`` with
  ``nsample`` or ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)``.
* The backward-compatibility properties ``backend_name`` and
  ``available_commands`` provide direct access to the capabilities.

Timeouts & Cancellation
^^^^^^^^^^^^^^^^^^^^^^^
* Per-call timeouts are enforced as **global deadlines**. On expiry,
  in-flight remote jobs are cancelled and a :class:`TimeoutError` is raised.
  Local synchronous jobs cannot be interrupted mid-call; the deadline is checked
  before submission and again before results are returned.
* ``future.cancel_remote()`` performs cooperative cancellation; awaiting the
  future raises :class:`concurrent.futures.CancelledError`.

Job Naming & Traceability
^^^^^^^^^^^^^^^^^^^^^^^^^
* Each chunk job receives a descriptive name of the form
  ``"mer:{layer}:{call_id}:{idx}/{total}:{cmd}"``, sanitized and
  truncated to 50 characters with a stable hash suffix when necessary. Local
  jobs do not use cloud job names.

Threading & Isolated Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* For each chunk attempt, the processor uses an isolated backend object:

  * **Local processor path**: rebuilds a fresh ``perceval.runtime.Processor``
    from copied non-circuit experiment state and a fresh local backend. The
    caller processor's previous circuit, input, and mode count are cleared
    before the layer circuit is set.
  * **RemoteProcessor path**: clones the original RP (independent RPC handler).
  * **ISession path**: calls ``session.build_remote_processor()`` (independent
    RP per chunk). The session's lifecycle is managed by the context manager
    (``__enter__`` and ``__exit__`` delegate to the session if provided).

  This prevents concurrent calls, chunks, and retries from sharing mutable
  backend state, and session resources are properly initialized and cleaned up.

Return Shapes & Mapping
^^^^^^^^^^^^^^^^^^^^^^^
* Distribution size is inferred from the leaf graph or from
  ``(n_modes, n_photons)`` and the computation space chosen (``UNBUNCHED``
  or ``FOCK``). Probability vectors are normalized if needed.

Examples
========
Synchronous execution (local AProcessor)
----------------------------------------
.. code-block:: python

   import perceval as pcvl

   local_processor = pcvl.Processor("SLOS")
   proc = MerlinProcessor(processor=local_processor)
   y = proc.forward(model, X)

Synchronous execution (RemoteProcessor)
----------------------------------------
.. code-block:: python

   proc = MerlinProcessor(processor=pcvl.RemoteProcessor("sim:slos"))
   y = proc.forward(model, X, nsample=20_000)

Synchronous execution (ISession)
---------------------------------
.. code-block:: python

   import perceval.providers.scaleway as scw

   with scw.Session("sim:ascella", project_id=..., token=...) as session:
       proc = MerlinProcessor(session=session, timeout=300.0)
       y = proc.forward(model, X, nsample=5_000)

Asynchronous with status and cancellation
-----------------------------------------
.. code-block:: python

   fut = proc.forward_async(model, X, nsample=5_000, timeout=None)
   print(fut.status())        # {'state': ..., 'progress': ..., ...}
   # If needed:
   fut.cancel_remote()        # cooperative cancel
   try:
       y = fut.wait()
   except Exception as e:
       print("Cancelled:", type(e).__name__)

High-throughput chunking
------------------------
.. code-block:: python

   proc = MerlinProcessor(rp, microbatch_size=8, chunk_concurrency=2)
   y = proc.forward(q_layer, X, nsample=3_000)

Version Notes
=============
* Only ``remote_processor`` and ``session`` paths use chunking and
  ``chunk_concurrency``. The local ``processor`` path keeps the full PyTorch
  input together at the Merlin level, maps rows to Perceval sampler iterations,
  and uses a rebuilt processor.
* Default ``chunk_concurrency`` is **1** (serial).
* The constructor ``timeout`` must be a **float**; use per-call ``timeout=None``
  for an unlimited call.
* ``max_shots_per_call`` caps the submitted sample count when needed.
* Shots are **user-controlled** (no auto-shot chooser); use the estimator helper
  to plan values ahead of time.
