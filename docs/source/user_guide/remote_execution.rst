==========================
MerlinProcessor User Guide
==========================

Overview
--------

``MerlinProcessor`` is a lightweight RPC-style bridge between your PyTorch
models and Remote Cloud provider (for instance Quandela Cloud) via Perceval's ``RemoteProcessor``. It lets you:

* Offload **quantum leaves** (e.g. ``QuantumLayer``) to the cloud while keeping
  **classical layers** local.
* Submit batched inputs; when batches are large, Merlin will **chunk** them and
  (optionally) **run chunks in parallel**.
* Drive execution **synchronously** (``forward``) or **asynchronously**
  (``forward_async`` returning a ``torch.futures.Future``).
* Monitor status, collect **job IDs**, **cancel** jobs, and enforce **timeouts**.
* Estimate **required shot counts per input** ahead of time.

Merlin deliberately avoids hidden "auto-shots": **you control sampling**. The
optional estimator is provided to help you choose appropriate values.

Prerequisites
-------------

* `perceval-quandela` configured with a valid cloud token (via
  ``pcvl.RemoteConfig`` cache or environment).
* A Perceval ``RemoteProcessor`` instance (e.g. a simulator like
  ``"sim:slos"`` or a QPU-backed platform).
* A Merlin **quantum layer** that provides ``export_config()`` (e.g.
  ``merlin.algorithms.QuantumLayer``).

Quick Start
-----------

.. code-block:: python

    import perceval as pcvl
    import torch
    import torch.nn as nn

    from merlin.algorithms import QuantumLayer
    from merlin.builder.circuit_builder import CircuitBuilder
    from merlin.core.merlin_processor import MerlinProcessor
    from merlin.measurement.strategies import MeasurementStrategy

    # 1) Create the Perceval RemoteProcessor (token must already be configured)
    rp = pcvl.RemoteProcessor("sim:slos")

    # 2) Wrap it with MerlinProcessor
    proc = MerlinProcessor(
        rp,
        microbatch_size=32,        # batch chunk size per cloud call (<=32)
        timeout=3600.0,           # default wall-time per forward (seconds)
        max_shots_per_call=None,  # optional cap per cloud call (see below)
        chunk_concurrency=1       # parallel chunk jobs within a quantum leaf
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
        measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),  # raw probability vector
    ).eval()

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        q,
        nn.Linear(15, 4, bias=False),   # 15 = C(6,2) from the chosen circuit
        nn.Softmax(dim=-1)
    ).eval()

    # 4) Run remotely with sampling (nsample) or exact probs if available
    X = torch.rand(8, 3)
    y = proc.forward(model, X, nsample=5000)   # synchronous
    print(y.shape)

Instantiation & Options
-----------------------

``MerlinProcessor(remote_processor, *, max_batch_size=32, timeout=3600.0,
max_shots_per_call=None, chunk_concurrency=1)``

* **remote_processor (pcvl.RemoteProcessor)**: your authenticated platform.
  Merlin clones it internally per quantum leaf so multiple jobs can run safely
  in parallel without altering your original instance.

* **max_batch_size (int)**: maximum number of input rows per **cloud job**.
  If your input batch ``B`` is larger, the batch is split into chunks of size
  ``<= microbatch_size``. Hard-capped by Merlin at 32.

* **timeout (float)**: default wall-clock limit (in seconds) for each
  ``forward/forward_async`` call. Use per-call override (see below). This must
  be a real number (not ``None``).

* **max_shots_per_call (int | None)**: cap for **each** cloud call. If
  ``None``, Merlin passes a safe default internally for Perceval. If you want a
  stricter cap, set this explicitly. (This is **not** an auto-shot chooser.)

* **chunk_concurrency (int)**: maximum number of **chunks** submitted in
  parallel **per quantum leaf**. Default ``1`` (serial). Increase for higher
  throughput when the backend allows it.

Execution API
-------------

Synchronous
^^^^^^^^^^^

.. code-block:: python

    y = proc.forward(layer_or_model, X, nsample=20000, timeout=15.0)

* **nsample (int | None)**:
  * If the backend exposes ``"probs"`` in ``remote_processor.available_commands``,
    Merlin uses exact probabilities and ignores ``nsample``.
  * Otherwise, Merlin uses sampling; ``nsample`` controls the shots per input.
    (Subject to your platform limits and ``max_shots_per_call``.)

* **timeout (float | None)**: overrides the constructor default for this call.
  * ``None`` --> no time limit for this call.
  * ``0`` or falsy is treated as "no limit".
  * Otherwise --> seconds until a **global timeout** cancels all in-flight jobs
    launched for this call and raises ``TimeoutError``.

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
  * ``fut.cancel_remote()`` signals the worker to cancel and issues remote job
    cancellation (best effort). ``fut.wait()`` then raises
    ``concurrent.futures.CancelledError``.
  * ``proc.cancel_all()`` cancels **all** active jobs across all futures.

* **Context manager**:
  Exiting a ``with MerlinProcessor(...) as proc:`` block triggers
  ``cancel_all()``, ensuring stray jobs are stopped.

Batching & Chunking
-------------------

* If ``len(X) > microbatch_size``, Merlin splits into chunks of size
  ``<= microbatch_size`` and submits up to ``chunk_concurrency`` chunk-jobs in
  parallel **for that quantum leaf**.
* The Future aggregates **all job IDs** across leaves in
  ``future.job_ids``. It also exposes chunk counters via ``future.status()``:

  .. code-block:: text

      {"state": "...", "progress": ..., "message": "...",
       "chunks_total": N, "chunks_done": k, "active_chunks": c}

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
* You can also use a context helper ``with layer.as_simulation():``
  to temporarily force local-mode (if your layer provides it).

Estimating Required Shots (No Auto-Execute)
-------------------------------------------

Merlin includes a helper that proxies Perceval's built-in estimator and **does
not** submit jobs:

.. code-block:: python

    estimates = proc.estimate_required_shots_per_input(
        layer=q,
        input=X,                          # shape [B, D] or [D]
        desired_samples_per_input=2_000
    )
    # -> list[int] length B (or 1 for a single vector).
    #    0 means "not viable" under current platform/perfs/filters).

Behavior:

* For each input row, Merlin maps your feature vector to the circuit parameter
  values (same mapping used during remote execution), then calls
  ``remote_processor.estimate_required_shots(...)``.
* It mirrors the layer's exported **circuit** and **input state** (including
  detected-photon filters) so the estimate aligns with actual execution.
* This is a **planner** only; it doesn't modify processor/job history.

Timeouts & Errors
-----------------

* **Timeout**: if a per-call or default timeout elapses, Merlin issues remote
  cancellation and raises ``TimeoutError``.
* **Cancellation**:
  * ``fut.cancel_remote()`` or ``proc.cancel_all()`` --> pending chunk workers
    raise ``CancelledError``; completed chunks are discarded for the call.
* **Remote failures**:
  * If the backend marks a job as failed, Merlin raises a ``RuntimeError`` with
    the platform message. If the message indicates an explicit remote cancel,
    Merlin maps it to ``CancelledError``.

Multiple Quantum Layers
-----------------------

Sequential models with multiple quantum leaves are supported:

* Each quantum leaf is processed in order; each may chunk and run those chunks
  with its own intra-leaf concurrency (``chunk_concurrency``).
* ``future.job_ids`` will include all job IDs across both leaves.

Controlling Shots Explicitly
----------------------------

* Sampling backends respect the **per-call** ``nsample`` you pass to
  ``forward/forward_async``; Merlin does not auto-derive or override it.
* Use ``estimate_required_shots_per_input`` ahead of time to pick good values.
* ``max_shots_per_call`` lets you enforce a **hard cap** for each cloud job.

Workflow Recipes (End-to-End Examples)
--------------------------------------

The following examples mirror tested workflows (see
``tests/core/cloud/test_userguide_examples.py``).

Mixed classical --> quantum --> classical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Build the quantum layer and probe its output size
    q = QuantumLayer(...).eval()  # see Quick Start builder pattern
    dist = q(torch.rand(2, q.input_size)).shape[1]

    model = nn.Sequential(
        nn.Linear(3, q.input_size, bias=False),
        q,                          # offloaded by Merlin
        nn.Linear(dist, 4, bias=False),
        nn.Softmax(dim=-1),
    ).eval()

    proc = MerlinProcessor(pcvl.RemoteProcessor("sim:slos"))
    # Prefer exact probabilities if supported; else sample.
    use_probs = "probs" in getattr(proc, "available_commands", [])
    nsamp = None if use_probs else 20_000

    X = torch.rand(6, 3)
    fut = proc.forward_async(model, X, nsample=nsamp)
    Y = fut.wait()
    print("shape:", Y.shape, "job_ids:", len(fut.job_ids))  # expect >= 1

Gradient-free fine-tuning with COBYLA (no autograd on quantum layer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Optional dependency: SciPy
    from scipy.optimize import minimize

    # Small model: Linear -> Quantum -> Linear(scalar)
    q = QuantumLayer(...).eval()
    dist = q(torch.rand(2, q.input_size)).shape[1]
    readout = nn.Linear(dist, 1, bias=False).eval()
    pre = nn.Linear(3, q.input_size, bias=False).eval()
    model = nn.Sequential(pre, q, readout).eval()

    # Flatten quantum params we will tune (keep classical layers fixed)
    q_params = [(n, p) for n, p in q.named_parameters() if p.requires_grad]
    shapes = [p.shape for _, p in q_params]
    sizes = [p.numel() for _, p in q_params]

    def get_flat():
        import torch
        return torch.cat([p.detach().flatten().cpu() for _, p in q_params], dim=0)

    def set_from_flat(vec):
        import torch
        off = 0
        with torch.no_grad():
            for (_, p), sz, shp in zip(q_params, sizes, shapes, strict=False):
                chunk = vec[off:off+sz].view(shp).to(p.dtype)
                p.data.copy_(chunk.to(p.device))
                off += sz

    x0 = get_flat().double().numpy()
    proc = MerlinProcessor(pcvl.RemoteProcessor("sim:slos"))
    nsamp = None if "probs" in getattr(proc, "available_commands", []) else 20_000
    X = torch.rand(8, 3)

    # Objective: maximize mean scalar output -> minimize negative
    def objective(v_np):
        v = torch.from_numpy(v_np).to(torch.float64)
        set_from_flat(v.to(torch.float32))
        with torch.no_grad():
            y = proc.forward(model, X, nsample=nsamp)
            return -float(y.mean().item())

    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": 12, "rhobeg": 0.5})
    print("final objective:", res.fun)

Local vs remote A/B (force simulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    q = QuantumLayer(...).eval()
    X = torch.rand(4, q.input_size)
    proc = MerlinProcessor(pcvl.RemoteProcessor("sim:slos"))

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

    fut = proc.forward_async(q, torch.rand(16, q.input_size), nsample=40000, timeout=None)
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
        microbatch_size=8,          # split big batches into <=8 rows per job
        chunk_concurrency=2        # up to 2 chunk-jobs in flight per quantum leaf
    )
    X = torch.rand(64, q.input_size)  # big batch
    fut = proc.forward_async(q, X, nsample=3000)
    Y = fut.wait()
    print("chunks_total/done/active:", fut.status())

Troubleshooting
---------------

* **No job IDs appear**:
  * Your backend may be very fast, or your layer ran locally (e.g.,
    ``force_local=True``).
* **Perceval requires ``max_shots_per_call``**:
  * Merlin passes a safe default when you leave it ``None``. If your org policy
    requires explicit bounds, set it at construction.
* **Timeouts in CI**:
  * Backends vary. Make tests resilient to fast or slow responses by polling
    ``future.done()`` before asserting on timeout exceptions.

API Reference (Summary)
-----------------------

* ``forward(module, input, *, nsample=None, timeout=None) -> torch.Tensor``
* ``forward_async(module, input, *, nsample=None, timeout=None) -> Future``
  * Future helpers:
    * ``future.job_ids: list[str]``
    * ``future.status() -> dict``
    * ``future.cancel_remote() -> None``
* ``cancel_all() -> None``
* ``estimate_required_shots_per_input(layer, input, desired_samples_per_input) -> list[int]``
* ``get_job_history() -> list[RemoteJob]``
* ``clear_job_history() -> None``

Version Notes
-------------

* Default ``chunk_concurrency`` is **1** (serial intra-leaf). Opt in to
  parallelism by setting it > 1.
* Constructor ``timeout`` must be a **float**. Use per-call ``timeout=None`` for
  an unlimited call.
* Estimation helper added to keep **shot selection user-driven** without
  auto-submitting jobs.


