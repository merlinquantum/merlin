===============================
MerlinProcessor API Reference
===============================

.. module:: merlin.core.merlin_processor

Overview
========
:class:`MerlinProcessor` is an RPC-style bridge that offloads **quantum leaves**
(e.g., layers exposing ``export_config()``) to a Perceval
:class:`~perceval.runtime.RemoteProcessor`, while keeping classical layers local.
It supports batched execution with chunking, limited intra-leaf concurrency,
per-call/global timeouts, cooperative cancellation, and a Torch-friendly async
interface returning :class:`torch.futures.Future`.

Key Capabilities
----------------
* Automatic traversal of a PyTorch module; offloads only **quantum leaves**.
* Batch **chunking** (``microbatch_size``) and **parallel** submission per leaf
  (``chunk_concurrency``).
* **Synchronous** (``forward``) and **asynchronous** (``forward_async``) APIs.
* **Cancellation** of a single call or **all** calls in flight.
* **Timeouts** that cancel in-flight cloud jobs.
* Per-call pool of cloned :class:`~perceval.runtime.RemoteProcessor` objects
  (avoid cross-thread handler sharing).
* Stable, descriptive cloud job names (capped to 50 chars).

.. note::
   Execution is supported both with exact probabilities (if the backend exposes
   the ``"probs"`` command) and with sampling (``"sample_count"`` or
   ``"samples"``). Shots are *user-controlled* via ``nsample``; there is no
   hidden auto-shot selection.

Class Reference
===============

MerlinProcessor
---------------
.. class:: MerlinProcessor(remote_processor, microbatch_size=32, timeout=3600.0, max_shots_per_call=None, chunk_concurrency=1)

   Create a processor that offloads quantum leaves to the given Perceval
   :class:`~perceval.runtime.RemoteProcessor`.

   :param perceval.runtime.RemoteProcessor remote_processor:
      Authenticated Perceval remote processor (simulator or QPU-backed).
   :param int microbatch_size: Maximum **rows per cloud job** (chunk size).
      Values > 32 are accepted but effectively capped by platform/job strategy.
   :param float timeout: Default wall-time limit (seconds) per call. Must be a
      finite float. Per-call override via ``timeout=...`` on API methods.
   :param int | None max_shots_per_call: Hard cap on **shots per cloud call**.
      If ``None``, a safe default is used internally.
   :param int chunk_concurrency: Max number of chunk jobs in flight **per
      quantum leaf** during a single call. ``>=1`` (default: 1, i.e., serial).

   **Attributes**

   .. attribute:: backend_name

      ``str`` - Best-effort backend name of ``remote_processor``.

   .. attribute:: available_commands

      ``list[str]`` - Commands exposed by the backend (e.g., ``"probs"``,
      ``"sample_count"``, ``"samples"``). Empty if unknown.

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

   * ``future.job_ids: list[str]`` - Accumulates job IDs across all chunk jobs.
   * ``future.status() -> dict`` - Current state/progress/message plus chunk
     counters: ``{"chunks_total", "chunks_done", "active_chunks"}``.
   * ``future.cancel_remote() -> None`` - Cooperative cancel; in-flight jobs are
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

.. method:: get_job_history() -> list[perceval.runtime.RemoteJob]

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
  offloaded unless they expose a ``should_offload(remote_processor, nsample)``
  method that returns ``False``, or they set ``force_local=True``.
* Non-quantum leaves run locally under ``torch.no_grad()``.

Batching & Chunking
^^^^^^^^^^^^^^^^^^^
* If ``B > microbatch_size``, the batch is split into chunks of size
  ``<= microbatch_size``. Up to ``chunk_concurrency`` chunk jobs per quantum
  leaf are submitted in parallel.

Backends & Commands
^^^^^^^^^^^^^^^^^^^
* If the backend exposes ``"probs"``, the processor queries exact probabilities
  and ignores ``nsample``.
* Otherwise it uses ``"sample_count"`` or ``"samples"`` with
  ``nsample or DEFAULT_SHOTS_PER_CALL``.

Timeouts & Cancellation
^^^^^^^^^^^^^^^^^^^^^^^
* Per-call timeouts are enforced as **global deadlines**. On expiry,
  in-flight jobs are cancelled and a :class:`TimeoutError` is raised.
* ``future.cancel_remote()`` performs cooperative cancellation; awaiting the
  future raises :class:`concurrent.futures.CancelledError`.

Job Naming & Traceability
^^^^^^^^^^^^^^^^^^^^^^^^^
* Each chunk job receives a descriptive name of the form
  ``"mer:{layer}:{call_id}:{idx}/{total}:{pool_slot}:{cmd}"``, sanitized and
  truncated to 50 characters with a stable hash suffix when necessary.

Threading & Pools
^^^^^^^^^^^^^^^^^
* For each call and for each quantum leaf, the processor creates a **pool** of
  cloned :class:`~perceval.runtime.RemoteProcessor` objects sized to
  ``chunk_concurrency``. Each clone has its own RPC handler to avoid
  cross-thread sharing.

Return Shapes & Mapping
^^^^^^^^^^^^^^^^^^^^^^^
* Distribution size is inferred from the leaf graph or from
  ``(n_modes, n_photons)`` and the computation space chosen. Probability
  vectors are normalized if needed.

Examples
========
Synchronous execution
---------------------
.. code-block:: python

   y = proc.forward(model, X, nsample=20_000)

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
* Default ``chunk_concurrency`` is **1** (serial).
* The constructor ``timeout`` must be a **float**; use per-call ``timeout=None``
  for an unlimited call.
* Shots are **user-controlled** (no auto-shot chooser); use the estimator helper
  to plan values ahead of time.


