===============================
MerlinProcessor API Reference
===============================

.. module:: merlin.core.merlin_processor

Overview
========
:class:`MerlinProcessor` is an RPC-style bridge that offloads **quantum leaves**
(e.g., layers exposing ``export_config()``) to a remote backend, while keeping
classical layers local. It supports two backend paths:

* **Perceval** :class:`~pcvl.RemoteProcessor` — the original
  Quandela Cloud path.
* **Perceval** `pcvl.runtime.session.ISession <https://perceval.quandela.net/docs/v1.2/providers.html#scaleway>`_ — the preferred path
  for Scaleway-hosted platforms (and any future session-based providers).

Both paths support batched execution with chunking, limited intra-leaf
concurrency, per-call/global timeouts, cooperative cancellation, and a
Torch-friendly async interface returning :class:`torch.futures.Future`.

Key Capabilities
----------------
* Automatic traversal of a PyTorch module; offloads only **quantum leaves**.
* Batch **chunking** (``microbatch_size``) and **parallel** submission per leaf
  (``chunk_concurrency``). Works identically for both backend paths.
* **Synchronous** (``forward``) and **asynchronous** (``forward_async``) APIs.
* **Cancellation** of a single call or **all** calls in flight.
* **Timeouts** that cancel in-flight cloud jobs.
* Per-chunk fresh :class:`~pcvl.RemoteProcessor` objects — cloned
  from the original (RemoteProcessor path) or built from the session (ISession
  path) — to avoid cross-thread handler sharing.
* Stable, descriptive cloud job names (capped to 50 chars).

.. note::
   Execution is supported both with exact probabilities (if the backend exposes
   the ``"probs"`` command) and with sampling (``"sample_count"`` or
   ``"samples"``). Shots are *user-controlled* via ``nsample``; there is no
   hidden auto-shot selection.

Class Reference
===============

BackendCapabilities
-------------------
.. class:: BackendCapabilities(name, available_commands)

   Immutable dataclass encapsulating backend capabilities extracted from a
   RemoteProcessor or ISession.

   **Attributes**

   .. attribute:: name

      ``str`` — Backend platform name (e.g., ``"sim:slos"``,
      ``"perceval-qpu:scaleway"``).

   .. attribute:: available_commands

      ``list[str]`` — Commands supported by the backend (e.g., ``"probs"``,
      ``"sample_count"``, ``"samples"``).

   **Example**

   .. code-block:: python

      proc = MerlinProcessor(remote_processor=rp)
      caps = proc.backend_capabilities
      print(f"Platform: {caps.name}")
      print(f"Supports: {caps.available_commands}")

MerlinProcessor
---------------
.. class:: MerlinProcessor(remote_processor=None, session=None, microbatch_size=32, timeout=3600.0, max_shots_per_call=None, chunk_concurrency=1)

   Create a processor that offloads quantum leaves to a remote backend.
   Exactly **one** of ``remote_processor`` or ``session`` must be provided.

   :param remote_processor: Authenticated Perceval
      :class:`~pcvl.RemoteProcessor` (simulator or QPU-backed).
      Merlin clones it per chunk so concurrent jobs have independent state.
      Type: ``RemoteProcessor | None``.
   :param session: A Perceval `pcvl.runtime.session.ISession <https://perceval.quandela.net/docs/v1.2/providers.html#scaleway>`_
      object — e.g. from ``pcvl.providers.scaleway.Session``. Merlin calls
      ``session.build_remote_processor()`` per chunk, giving each chunk
      an independent RP. Type: ``ISession | None``.
   :param int microbatch_size: Maximum **rows per cloud job** (chunk size).
   :param float timeout: Default wall-time limit (seconds) per call. Per-call
      override via ``timeout=...`` on API methods.
   :param max_shots_per_call: Hard cap on **shots per cloud call**.
      If ``None``, a safe default is used internally. If ``nsample`` exceeds
      this cap, Merlin automatically raises it to match.
      Type: ``int | None``.
   :param int chunk_concurrency: Max number of chunk jobs in flight **per
      quantum leaf** during a single call. ``>=1`` (default: 1, i.e., serial).

   :raises TypeError: If both or neither of ``remote_processor`` and ``session``
      are provided, or if the provided argument is not the expected type.

   **Attributes**

   .. attribute:: backend_capabilities

      :class:`BackendCapabilities` — Encapsulates backend name and available
      commands extracted at initialization time. This is the primary way to
      access backend capabilities.

   .. attribute:: backend_name

      ``str`` — Backward-compatibility property. Equivalent to
      ``backend_capabilities.name``. Best-effort backend name from the remote
      processor or session (e.g., ``"sim:slos"``).

   .. attribute:: available_commands

      ``list[str]`` — Backward-compatibility property. Equivalent to
      ``backend_capabilities.available_commands``. Commands exposed by the
      backend (e.g., ``"probs"``, ``"sample_count"``, ``"samples"``).

   .. attribute:: remote_processor

      ``RemoteProcessor | None`` — set when constructed with
      ``remote_processor``; ``None`` for the session path.

   .. attribute:: session

      ``ISession | None`` — set when constructed with ``session``;
      ``None`` for the RemoteProcessor path.

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
      first leaf. Tensors are moved to CPU for remote execution if needed; the
      result is moved back to the input's original device/dtype.
   :param int | None nsample: Shots per input when sampling. Ignored if the
      backend supports exact probabilities (``"probs"``).
   :param float | None timeout: Per-call override. ``None``/``0`` == unlimited.
   :returns: Output tensor with batch dimension ``B`` and leaf-determined
      distribution dimension.
   :rtype: torch.Tensor
   :raises RuntimeError: If ``module`` is in training mode.
   :raises TimeoutError: On global per-call timeout (remote cancel is issued).
   :raises concurrent.futures.CancelledError: If the call is cooperatively
      cancelled via the async API.

.. method:: forward_async(module, input, *, nsample=None, timeout=None) -> torch.futures.Future

   Asynchronous execution. Returns a :class:`torch.futures.Future` with extra
   helpers attached:

   **Future extensions**

   * ``future.job_ids: list[str]`` — accumulates job IDs across all chunk jobs.
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
   reach a target number of *useful* samples.

   :param torch.nn.Module layer: A quantum leaf (must implement
      ``export_config()``).
   :param torch.Tensor input: ``[B, D]`` or a single vector ``[D]``. Values are
      mapped to the circuit parameters as they would be during execution.
   :param int desired_samples_per_input: Target **useful** samples per input.
   :returns: ``list[int]`` of length ``B`` (``0`` indicates "not viable" under
      current settings).
   :rtype: list[int]
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
* If ``B > microbatch_size``, the batch is split into chunks of size
  ``<= microbatch_size``. Up to ``chunk_concurrency`` chunk jobs per quantum
  leaf are submitted in parallel. This applies to both the RemoteProcessor
  and ISession paths.
* Failed chunks are retried up to 3 times with exponential backoff.
  Cancellation and timeout errors propagate immediately without retry.

Backends & Commands
^^^^^^^^^^^^^^^^^^^
* Backend capabilities (name and available commands) are extracted once at
  initialization and stored in ``backend_capabilities``. Both the
  RemoteProcessor path (via the original RP) and ISession path (via the
  first processor built from the session) use this snapshot.
* If the backend exposes ``"probs"`` and  ``nsample`` is None or 0, the processor queries exact probabilities.
* Otherwise it uses ``"sample_count"`` or ``"samples"`` with
  ``nsample or DEFAULT_SHOTS_PER_CALL``.
* The backward-compatibility properties ``backend_name`` and
  ``available_commands`` provide direct access to the capabilities.

Timeouts & Cancellation
^^^^^^^^^^^^^^^^^^^^^^^
* Per-call timeouts are enforced as **global deadlines**. On expiry,
  in-flight jobs are cancelled and a :class:`TimeoutError` is raised.
* ``future.cancel_remote()`` performs cooperative cancellation; awaiting the
  future raises :class:`concurrent.futures.CancelledError`.

Job Naming & Traceability
^^^^^^^^^^^^^^^^^^^^^^^^^
* Each chunk job receives a descriptive name of the form
  ``"mer:{layer}:{call_id}:{idx}/{total}:{cmd}"``, sanitized and
  truncated to 50 characters with a stable hash suffix when necessary.

Threading & Fresh RPs
^^^^^^^^^^^^^^^^^^^^^
* For each chunk attempt, the processor builds a **fresh**
  :class:`~pcvl.RemoteProcessor`:

  * **RemoteProcessor path**: clones the original RP (independent RPC handler).
  * **ISession path**: calls ``session.build_remote_processor()`` (independent
    RP per chunk). The session's lifecycle is managed by the context manager
    (``__enter__`` and ``__exit__`` delegate to the session if provided).

  This ensures concurrent chunks and retries never share mutable RP state, and
  session resources are properly initialized and cleaned up.

Return Shapes & Mapping
^^^^^^^^^^^^^^^^^^^^^^^
* Distribution size is inferred from the leaf graph or from
  ``(n_modes, n_photons)`` and the computation space chosen (``UNBUNCHED``
  or ``FOCK``). Probability vectors are normalized if needed.

Examples
========
Synchronous execution (RemoteProcessor)
----------------------------------------
.. code-block:: python

   proc = MerlinProcessor(pcvl.RemoteProcessor("sim:slos"))
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
* Both ``remote_processor`` and ``session`` paths now support chunking and
  ``chunk_concurrency``. Each chunk gets an independent ``RemoteProcessor``.
* Default ``chunk_concurrency`` is **1** (serial).
* The constructor ``timeout`` must be a **float**; use per-call ``timeout=None``
  for an unlimited call.
* ``max_shots_per_call`` is automatically raised to match ``nsample`` when
  needed.
* Shots are **user-controlled** (no auto-shot chooser); use the estimator helper
  to plan values ahead of time.
