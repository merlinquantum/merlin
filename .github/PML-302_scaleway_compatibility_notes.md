# PML-302 Scaleway Compatibility Notes

Branch under review: `PML-302-MerLinProcessor-export_config-and-typing`

This note explains the local fixes applied while probing the Scaleway failures.
Nothing has been pushed.

## Summary

Two local fixes were applied:

1. `MerlinProcessor(session=...)` no longer requires a token extracted from a
   `RemoteProcessor`.
2. Perceval 1.2 sampler iteration payloads are normalized before remote
   submission so Scaleway can JSON-serialize them.

These address two different causes:

- The Perceval iterator serialization issue affects `main` with
  `perceval-quandela==1.2.1`.
- The session-token failure is caused by PML-302.

## General Fix: Perceval 1.2 Iterator Compatibility

### Problem

Merlin batches remote inputs through Perceval `Sampler.add_iteration()`:

```python
sampler.clear_iterations()
for params in iteration_params:
    sampler.add_iteration(circuit_params=params)
```

With Perceval 1.1, sampler iterations were stored as a plain list, so the
Scaleway payload was JSON-serializable.

With Perceval 1.2.1, the sampler stores iterations in a `ParameterIterator`
object. Perceval then places that object in the remote job payload, but the
Scaleway handler still calls:

```python
json.dumps(payload.get("payload", {}))
```

That fails with:

```text
TypeError: Object of type ParameterIterator is not JSON serializable
```

### Local Fix

Added `_ensure_serializable_sampler_iterator()` in
`merlin/core/merlin_processor.py`.

The helper checks for the Perceval 1.2 shape:

```python
iterator = getattr(sampler, "_iterator", None)
iterations = getattr(iterator, "iterations", None)
```

If present, it replaces the private remote-job payload iterator object with a
plain list:

```python
payload["iterator"] = list(iterations)
```

This is backwards compatible with Perceval 1.1 because `sampler._iterator` is
already a list there and has no `.iterations` attribute, so the helper returns
without changing anything.

### Test Added

Added:

```text
tests/core/test_merlin_processor_unit.py::test_submit_job_serializes_perceval_12_parameter_iterator_payload
```

It uses a fake Perceval 1.2-style sampler payload and confirms `_submit_job()`
converts the iterator before `execute_async()`.

## PML-302 Fix: Session Path Should Not Require Token Extraction

### Problem

PML-302 unified the `remote_processor=` and `session=` constructor paths and
then always attempted to extract a token:

```python
if self._token is None:
    self._token = self._extract_rp_token(remote_processor)

if self._token is None:
    raise ValueError(...)
```

That is valid for `remote_processor=...`, because Merlin later clones the
processor with:

```python
RemoteProcessor(name=rp.name, token=self._token, ...)
```

It is not valid for `session=...`. In the session path, authentication is owned
by the `ISession`, and Merlin creates fresh processors with:

```python
self.session.build_remote_processor()
```

The session path should not require Merlin to extract a token from the generated
remote processor.

### Local Fix

Token extraction is now only performed when `self.session is None`.

### Test Added

Added:

```text
tests/core/test_merlin_processor_unit.py::test_session_path_does_not_require_remote_processor_token
```

It verifies `MerlinProcessor(session=session)` succeeds even when
`_extract_rp_token()` would return `None`, and that token extraction is not
called in the session path.

## Remaining PML-302 Issues

After the two fixes above, Scaleway tests on current Perceval no longer fail on
constructor auth or `ParameterIterator` serialization. The remaining failures are
PML-302 review/test-expectation issues.

### 1. Context Manager Double-Starts The Session

PML-302 changes `MerlinProcessor.__enter__()` to call:

```python
self.session.__enter__()
```

The Scaleway pytest fixture already creates an active session with:

```python
with scw.Session(...) as session:
    yield session
```

So `with MerlinProcessor(session=scaleway_session)` tries to start an already
attached session and fails with:

```text
Exception: A session is already attached to this RPC handler
```

This appears branch-caused. The safer ownership rule is that callers own the
session lifecycle when they pass an existing `ISession` into `MerlinProcessor`.

### 2. Tests Assume `probs` Exists On The Real Backend

The real Scaleway backend currently reports:

```text
('sample_count', 'samples')
```

Several PML-302 tests assert `"probs" in proc.available_commands`, so they fail
before exercising forward execution. This is a test assumption unless the remote
backend is guaranteed to expose `probs`.

### 3. Sample Cap Warning Is Treated As A Failure

One test calls:

```python
proc = MerlinProcessor(..., max_shots_per_call=100)
y = proc.forward(q, X, nsample=1000)
```

PML-302 emits a `UserWarning` and caps `nsample` to `100`. Since `pytest.ini`
treats warnings as errors, the test fails. The test should either request a value
within the cap or assert the warning explicitly.

## Test Results

### Before Local Fixes On PML-302 With Perceval 1.2.1

```text
tests/core/cloud/test_scaleway_session.py
15 failed
```

All failures occurred at construction:

```text
ValueError: Could not extract auth token from RemoteProcessor.
```

### After Local Fixes On PML-302 With Perceval 1.2.1

```text
tests/core/test_merlin_processor_unit.py
53 passed in 4.87s
```

```text
tests/core/cloud/test_scaleway_session.py
10 passed, 5 failed in 200.58s
```

The remaining 5 failures are the context-manager/test-expectation issues listed
above, not the Perceval iterator compatibility issue.

### Main Branch Comparison

On `origin/main` with `perceval-quandela==1.2.1`:

```text
tests/core/cloud/test_scaleway_session.py
8 failed, 3 passed, 1 skipped
```

The failures were all:

```text
TypeError: Object of type ParameterIterator is not JSON serializable
```

On `origin/main` with Perceval 1.1.0 forced through `PYTHONPATH`:

```text
tests/core/cloud/test_scaleway_session.py
12 passed in 232.70s
```

That confirms the iterator serialization failure is a Perceval 1.2
compatibility issue, not a PML-302-only issue.
