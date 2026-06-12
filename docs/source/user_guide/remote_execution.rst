==========================
MerlinProcessor User Guide
==========================

Overview
--------

``MerlinProcessor`` is a lightweight RPC-style bridge between your PyTorch
models and remote cloud QPU/simulator backends. It supports two backend paths:

* **Perceval ``RemoteProcessor``** — the original Quandela Cloud path.
* **Perceval ``ISession``** — the preferred path for Scaleway-hosted platforms
  (and any future session-based providers).

With either backend you can:

* Offload **quantum leaves** (e.g. ``QuantumLayer``) to the cloud while keeping
  **classical layers** local.
* Submit batched inputs; when remote batches are large, Merlin will **chunk**
  them and (optionally) **run chunks in parallel**.
* Drive execution **synchronously** (``forward``) or **asynchronously**
  (``forward_async`` returning a ``torch.futures.Future``).
* Monitor status, collect **job IDs**, **cancel** jobs, and enforce **timeouts**.
* Estimate **required shot counts per input** ahead of time.

Merlin deliberately avoids hidden "auto-shots": **you control sampling**. The
optional estimator is provided to help you choose appropriate values.

Prerequisites
-------------

You need **one** of the following backends configured:

**Option A — Quandela Cloud (RemoteProcessor)**

* ``perceval-quandela`` configured with a valid cloud token (via
  ``pcvl.RemoteConfig`` cache or environment).
* A Perceval ``RemoteProcessor`` instance (e.g. a simulator like
  ``"sim:slos"`` or a QPU-backed platform).

.. code-block:: python

   import perceval as pcvl
   from merlin.measurement.strategies import MeasurementStrategy

   # Configure your Quandela Cloud token (one of the following):
   RemoteConfig.set_token("YOUR_TOKEN")                # option 1: global config
   rp = pcvl.RemoteProcessor("sim:slos")

   rp = pcvl.RemoteProcessor("sim:slos", "YOUR_TOKEN") # option 2: inline token

**Option B — Scaleway (ISession)**

* The ``perceval.providers.scaleway`` module installed.
* A Scaleway project ID and API secret key (typically set via
  ``SCW_PROJECT_ID`` and ``SCW_SECRET_KEY`` environment variables).

**Both paths require:**

* A Merlin **quantum layer** that provides ``export_config()`` (e.g.
  ``merlin.algorithms.QuantumLayer``).

Quick Start — Quandela Cloud
-----------------------------

.. code-block:: python

    import perceval as pcvl
    import torch
    import torch.nn as nn

    from merlin.algorithms import QuantumLayer
    from merlin.builder.circuit_builder import CircuitBuilder
    from merlin.core.computation_space import ComputationSpace
    from merlin.core.merlin_processor import MerlinProcessor
    from merlin.measurement.strategies import MeasurementStrategy

    # 1) Create the Perceval RemoteProcessor (token must already be configured)
    rp = pcvl.RemoteProcessor("sim:slos")

    # 2) Wrap it with MerlinProcessor
    proc = MerlinProcessor(
        rp,
        microbatch_size=32,        # batch chunk size per cloud call
        timeout=3600.0,            # default wall-time per forward (seconds)
        max_shots_per_call=None,   # optional cap per cloud call (see below)
        chunk_concurrency=1,       # parallel chunk jobs within a quantum leaf
    )

    # 3) Build a QuantumLayer and a small model
    b = CircuitBuilder(n_modes=6)
    b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=[0, 1], name="px")
    b.add_entangling_layer()

    q = QuantumLayer(
        input_size=2,
        builder=b,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED,
        ),
    ).eval()

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        q,
        nn.Linear(15, 4, bias=False),   # 15 = C(6,2) unbunched outputs
        nn.Softmax(dim=-1),
    ).eval()

    # 4) Run remotely with sampling
    X = torch.rand(8, 3)
    y = proc.forward(model, X, nsample=5000)
    print(y.shape)  # (8, 4)


Quick Start — Scaleway Session
-------------------------------

.. code-block:: python

    import perceval.providers.scaleway as scw
    import torch

    from merlin.algorithms import QuantumLayer
    from merlin.builder.circuit_builder import CircuitBuilder
    from merlin.core.computation_space import ComputationSpace
    from merlin.core.merlin_processor import MerlinProcessor
    from merlin.measurement.strategies import MeasurementStrategy

    # 1) Open a Scaleway session (context manager handles cleanup)
    with scw.Session(
        "EMU-ASCELLA-6PQ",                         # platform name
        project_id="YOUR_SCW_PROJECT_ID",      # or read from env
        token="YOUR_SCW_SECRET_KEY",           # or read from env
        deduplication_id="merlin-guide",       # reuse session if still alive
        max_idle_duration_s=300,
        max_duration_s=600,
    ) as session:

        # 2) Wrap the session with MerlinProcessor
        proc = MerlinProcessor(
            session=session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=5000,
        )

        # 3) Build a quantum layer
        b = CircuitBuilder(n_modes=6)
        b.add_rotations(trainable=True, name="theta")
        b.add_angle_encoding(modes=[0, 1], name="px")
        b.add_entangling_layer()

        q = QuantumLayer(
            input_size=2,
            builder=b,
            n_photons=2,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED,
            ),
        ).eval()

        # 4) Run remotely
        X = torch.rand(8, 2)
        y = proc.forward(q, X, nsample=1000)
        print(y.shape)  # (8, 15)


Instantiation & Options
-----------------------

.. code-block:: text

    MerlinProcessor(
        remote_processor=None,       # RemoteProcessor — deprecated legacy path
        session=None,                # ISession — preferred path
        microbatch_size=32,
        timeout=3600.0,
        max_shots_per_call=None,
        chunk_concurrency=1,
        token=None,
        *,
        processor=None,              # AProcessor — keyword-only local/normalized path
    )

Exactly **one** of ``remote_processor``, ``session``, or ``processor`` must be
provided.

.. warning:: *Deprecated since version 0.4:*
   The ``remote_processor`` constructor argument is deprecated and will be
   removed in a future release. Pass the same ``RemoteProcessor`` through
   ``processor=`` instead.

* **remote_processor (RemoteProcessor | None)**: Deprecated Quandela Cloud
  backend entry point. Pass the same object through ``processor=`` instead.
  Merlin clones it internally per chunk so multiple jobs can run safely in
  parallel without altering your original instance.

* **session (ISession | None)**: A Perceval session object — e.g. from
  ``perceval.providers.scaleway.Session``. Merlin builds a fresh
  ``RemoteProcessor`` from the session for each chunk, so chunking and
  concurrency work identically to the ``RemoteProcessor`` path.

* **microbatch_size (int)**: maximum number of input rows per **cloud job**.
  If your input batch ``B`` is larger, the batch is split into chunks of size
  ``<= microbatch_size``. Applies to both the ``RemoteProcessor`` and
  ``ISession`` paths.

* **timeout (float)**: default wall-clock limit (in seconds) for each
  ``forward`` / ``forward_async`` call. Use per-call override (see below).

* **max_shots_per_call (int | None)**: cap for **each** cloud call's
  ``max_shots_per_call`` parameter on the Perceval ``Sampler``. If ``None``,
  Merlin uses an internal cap default (100 000). If the requested ``nsample``
  for a call exceeds this cap, Merlin clamps the submitted sample count to the
  cap.

* **chunk_concurrency (int)**: maximum number of **chunks** submitted in
  parallel **per quantum leaf**. Default ``1`` (serial). Increase for higher
  throughput when the backend allows it.

* **token (str | None)**: authentication token forwarded to cloned remote
  processors. If omitted, Merlin extracts it from the ``RemoteProcessor`` RPC
  handler. Ignored for local and session paths.

* **processor (AProcessor | None)**: keyword-only Perceval processor entry
  point. Local processors use the local backend path. ``RemoteProcessor``
  instances passed here are normalized to the remote processor path.

Computation Spaces
------------------

The computation space controls which output Fock states are included in the
probability vector. It is specified via ``MeasurementStrategy``:

.. code-block:: python

    from merlin.measurement.strategies import MeasurementStrategy

    # UNBUNCHED — at most one photon per mode. Output dim = C(m, n).
    MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED)

    # FOCK — arbitrary photon occupation (bunching allowed). Output dim = C(m + n − 1, n).
    MeasurementStrategy.probs(computation_space=ComputationSpace.FOCK)

``MerlinProcessor`` automatically detects the computation space of each quantum
leaf and arranges the returned probability tensor to match the state ordering
used by the local SLOS backend. This ensures that index *i* of the cloud result
maps to the same Fock state as index *i* of a local ``layer(X)`` call.

Execution API
-------------

Synchronous
^^^^^^^^^^^

.. code-block:: python

    y = proc.forward(layer_or_model, X, nsample=20000, timeout=15.0)

* **nsample (int | None)**:
  If the backend exposes ``"probs"`` in ``remote_processor.available_commands``,
  Merlin uses exact probabilities and ignores ``nsample``. Otherwise, Merlin
  uses sampling; ``nsample`` controls the shots per input.

* **timeout (float | None)**: overrides the constructor default for this call.
  ``None`` or ``0`` means no time limit.

Asynchronous
^^^^^^^^^^^^

.. code-block:: python

    fut = proc.forward_async(layer_or_model, X, nsample=3000, timeout=None)

    # Helpers injected on the Future:
    fut.job_ids         # list[str]: job ids across all chunks/leaves
    fut.status()        # dict: {state, progress, message, chunks_*}
    fut.cancel_remote() # request cancellation; .wait() -> CancelledError

    y = fut.wait()

* **Cancellation**:
  ``fut.cancel_remote()`` signals the worker to cancel and issues remote job
  cancellation (best effort). ``fut.wait()`` then raises
  ``concurrent.futures.CancelledError``.
  ``proc.cancel_all()`` cancels **all** active jobs across all futures.

* **Context manager**:
  Exiting a ``with MerlinProcessor(...) as proc:`` block triggers
  ``cancel_all()``, ensuring stray jobs are stopped.

Batching & Chunking
-------------------

* If ``len(X) > microbatch_size``, Merlin splits remote execution into chunks
  of size ``<= microbatch_size`` and submits up to ``chunk_concurrency``
  chunk-jobs in parallel **for that quantum leaf**. This applies to both the
  ``RemoteProcessor`` and ``ISession`` paths.
* Local ``processor=`` execution does not use remote chunking. Merlin keeps the
  full PyTorch input together, maps rows to Perceval sampler iterations, and
  returns a batched tensor. Perceval does not execute a native vectorized batch,
  so local users control batch size from their PyTorch input batch rather than
  ``microbatch_size``.
* The Future aggregates **all job IDs** across leaves in
  ``future.job_ids``. It also exposes chunk counters via ``future.status()``:

  .. code-block:: text

      {"state": "...", "progress": ..., "message": "...",
       "chunks_total": N, "chunks_done": k, "active_chunks": c}

* If a chunk fails, Merlin retries up to 3 times with exponential backoff.
  Cancellation and timeout errors are propagated immediately without retry.

Device & dtype round-trip
-------------------------

Inputs are moved to CPU for remote execution when needed, and the final tensor
is returned on the **original device and dtype** of your input (e.g., preserve
CUDA when possible for downstream ops).

Offload Policy & Local Overrides
--------------------------------

* By default, modules that provide ``export_config()`` are treated as
  **quantum leaves** and offloaded.
* Set ``layer.force_local = True`` to force **local** execution
  (useful for debugging and A/B comparisons).

Estimating Required Shots
-------------------------

Merlin includes a helper that proxies Perceval's built-in estimator and **does
not** submit jobs. The estimator is available only for remote processor and
session backends; local ``processor=`` backends raise ``RuntimeError``:

.. code-block:: python

    estimates = proc.estimate_required_shots_per_input(
        layer=q,
        input=X,                          # shape [B, D] or [D]
        desired_samples_per_input=2_000,
    )
    # -> list[int] of length B (or 1 for a single vector).
    #    0 means "not viable" under current platform/filters.

This is a **planner** only; it doesn't modify processor state or job history.

Timeouts & Errors
-----------------

* **Timeout**: if a per-call or default timeout elapses, Merlin issues remote
  cancellation and raises ``TimeoutError``.
* **Cancellation**: ``fut.cancel_remote()`` or ``proc.cancel_all()`` -->
  pending chunk workers raise ``CancelledError``; completed chunks are
  discarded for the call.
* **Remote failures**: if the backend marks a job as failed, Merlin raises a
  ``RuntimeError`` with the platform message. If the message indicates an
  explicit remote cancel, Merlin maps it to ``CancelledError``.
* **Retries**: transient failures (non-cancel, non-timeout) trigger up to 3
  automatic retries per chunk with exponential backoff.

Multiple Quantum Layers
-----------------------

Sequential models with multiple quantum leaves are supported:

* Each quantum leaf is processed in order; remote leaves may chunk and run
  those chunks with their own intra-leaf concurrency (``chunk_concurrency``).
* ``future.job_ids`` will include all job IDs across all leaves.

Workflow Recipes
----------------

Mixed classical → quantum → classical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Works with both computation spaces — just adjust the output dimension:

.. code-block:: python

    from math import comb
    from merlin.measurement.strategies import MeasurementStrategy

    # UNBUNCHED: output dim = C(m, n)
    q = QuantumLayer(
        input_size=2,
        builder=b,  # your CircuitBuilder with n_modes=6
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED,
        ),
    ).eval()
    dist = comb(6, 2)  # 15

    # Or FOCK (bunched): output dim = C(m + n - 1, n)
    # q = QuantumLayer(
    #     input_size=2, builder=b, n_photons=2,
    #     measurement_strategy=MeasurementStrategy.probs(
    #         computation_space=ComputationSpace.FOCK,
    #     ),
    # ).eval()
    # dist = comb(6 + 2 - 1, 2)  # 21

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        q,
        nn.Linear(dist, 4, bias=False),
        nn.Softmax(dim=-1),
    ).eval()

    proc = MerlinProcessor(processor=pcvl.RemoteProcessor("sim:slos"))
    X = torch.rand(6, 3)
    y = proc.forward(model, X, nsample=5000)

Gradient-free fine-tuning with COBYLA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No autograd through the quantum layer — optimise circuit parameters directly
using SciPy:

.. code-block:: python

    from scipy.optimize import minimize
    from merlin.measurement.strategies import MeasurementStrategy

    q = QuantumLayer(
        input_size=2,
        builder=b,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED,
        ),
    ).eval()
    dist = q(torch.rand(2, 2)).shape[1]

    readout = nn.Linear(dist, 1, bias=False).eval()
    pre = nn.Linear(3, 2, bias=False).eval()
    model = nn.Sequential(pre, q, readout).eval()

    # Flatten quantum params we will tune (keep classical layers fixed)
    q_params = [(n, p) for n, p in q.named_parameters() if p.requires_grad]
    shapes = [p.shape for _, p in q_params]
    sizes = [p.numel() for _, p in q_params]

    def get_flat():
        return torch.cat([p.detach().flatten().cpu() for _, p in q_params], dim=0)

    def set_from_flat(vec):
        off = 0
        with torch.no_grad():
            for (_, p), sz, shp in zip(q_params, sizes, shapes, strict=False):
                chunk = vec[off : off + sz].view(shp).to(p.dtype)
                p.data.copy_(chunk.to(p.device))
                off += sz

    x0 = get_flat().double().numpy()
    proc = MerlinProcessor(processor=pcvl.RemoteProcessor("sim:slos"))
    X = torch.rand(8, 3)

    # Objective: maximise mean scalar output → minimise negative
    def objective(v_np):
        v = torch.from_numpy(v_np).to(torch.float64)
        set_from_flat(v.to(torch.float32))
        with torch.no_grad():
            y = proc.forward(model, X, nsample=5000)
            return -float(y.mean().item())

    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": len(x0) + 6, "rhobeg": 0.5})
    print("final objective:", res.fun)

Local vs remote A/B (force simulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    q = QuantumLayer(...).eval()
    X = torch.rand(4, q.input_size)
    proc = MerlinProcessor(processor=pcvl.RemoteProcessor("sim:slos"))

    # Remote path (offloaded)
    y_remote = proc.forward(q, X, nsample=5000)

    # Local path (force simulation)
    q.force_local = True
    y_local = proc.forward(q, X, nsample=5000)

    # Compare distributions (allowing some sampling noise)
    print((y_local - y_remote).abs().mean())

Monitoring status & safe cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    fut = proc.forward_async(q, torch.rand(16, 2), nsample=40000, timeout=None)

    # Poll status (state/progress/message + chunk counters)
    print(fut.status())

    # If needed, cancel cooperatively
    fut.cancel_remote()
    try:
        _ = fut.wait()
    except Exception as e:
        print("Cancelled:", type(e).__name__)

High-throughput batching with chunking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    proc = MerlinProcessor(
        pcvl.RemoteProcessor("sim:slos"),
        microbatch_size=8,
        chunk_concurrency=2,
    )
    X = torch.rand(64, 2)
    fut = proc.forward_async(q, X, nsample=3000)
    Y = fut.wait()
    print("chunks:", fut.status())

Scaleway session with context manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import os
    import perceval.providers.scaleway as scw

    with scw.Session(
        "sim:ascella",
        project_id=os.environ["SCW_PROJECT_ID"],
        token=os.environ["SCW_SECRET_KEY"],
        deduplication_id="my-training-run",
        max_idle_duration_s=300,
        max_duration_s=1800,
    ) as session:

        with MerlinProcessor(session=session, timeout=300.0) as proc:
            q = QuantumLayer(...).eval()
            y = proc.forward(q, X, nsample=1000)
            # ...

        # MerlinProcessor context manager cancels any stray jobs on exit.
    # Scaleway session is closed on exit.

Gradient Propagation
---------------------
* No gradient can flow through the MerLin processor. When calling backwards directly on a MerlinProcessor's
  forward pass, this error will be shown::

      element 0 of tensors does not require grad and does not have a grad_fn

* If a MerLin processor is part of a bigger ``Pytorch`` module, the quantum layer's parameter gradient will be None.
  This also means that every upstream models' (models called before the :class:`~merlin.algorithms.layer.QuantumLayer` in the model's forward) gradient will be None.

* The gradient can not pass through right now as pytorch can not access
  directly the computation on the QPU.

* Differentiation through the MerLin processor will be implemented in v0.5.

Troubleshooting
---------------

* **No job IDs appear**:
  Your backend may be very fast, or your layer ran locally (e.g.,
  ``force_local=True``).
* **"Lowered max_samples" warning from Perceval**:
  This means ``nsample`` exceeded ``max_shots_per_call``. Merlin clamps the
  submitted sample count, but if you see this with an older version, set
  ``max_shots_per_call`` >= your ``nsample``.
* **Timeouts in CI**:
  Backends vary. Make tests resilient to fast or slow responses by polling
  ``future.done()`` before asserting on timeout exceptions.

API Reference (Summary)
-----------------------

**Constructor**

* ``MerlinProcessor(remote_processor=None, session=None, microbatch_size=32, timeout=3600.0, max_shots_per_call=None, chunk_concurrency=1, token=None, *, processor=None)``
  ``remote_processor`` is deprecated; pass remote processors through
  ``processor=``.

**Execution**

* ``forward(module, input, *, nsample=None, timeout=None) -> torch.Tensor``
* ``forward_async(module, input, *, nsample=None, timeout=None) -> Future``

  * ``future.job_ids: list[str]``
  * ``future.status() -> dict``
  * ``future.cancel_remote() -> None``

**Lifecycle**

* ``cancel_all() -> None``
* Context manager (``with MerlinProcessor(...) as proc:``)

**Estimation**

* ``estimate_required_shots_per_input(layer, input, desired_samples_per_input) -> list[int]``

**History**

* ``get_job_history() -> list[RemoteJob]``
* ``clear_job_history() -> None``

Version Notes
-------------

* ``session`` parameter added for ``ISession``-based backends (Scaleway).
  Exactly one of ``processor``, ``remote_processor``, or ``session`` must be
  provided. ``remote_processor`` is deprecated and remains supported for
  compatibility. A
  given session then uses its remote processor from the ``session.build_remote_processor()``
  method to define the available commands. Both remote paths support chunking and ``chunk_concurrency`` — each chunk
  gets an independent ``RemoteProcessor`` via ``session.build_remote_processor()``.

  To see the available commands and backend name, call the ``backend_capabilities`` attribute of the ``MerlinProcessor``.
* If the probs command is available and nsample is None or 0, the processor will run a probabilities simulations.
  Otherwise, sampling simulations or quantum executions will be done.
* ``MeasurementStrategy.probs(computation_space=...)`` replaces the older
  ``no_bunching`` flag and bare ``computation_space`` parameter on
  ``QuantumLayer``. Both ``ComputationSpace.FOCK`` (bunched) and
  ``ComputationSpace.UNBUNCHED`` are fully supported for cloud execution.
* Default ``chunk_concurrency`` is **1** (serial intra-leaf).
* Failed chunks are retried up to 3 times with exponential backoff.
  Cancellation and timeout errors propagate immediately.
* ``max_shots_per_call`` sets the maximum number of samples sent in a
   single call. When ``nsample > max_shots_per_call``, the call is limited
   to ``max_shots_per_call`` samples, which prevents Perceval from silently
   clamping the requested sample count. The default ``max_shots_per_call`` is 100 000.
