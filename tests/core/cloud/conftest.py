# tests/core/cloud/conftest.py
from __future__ import annotations

import os

import perceval as pcvl
import pytest
from perceval.runtime import RemoteConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Scaleway test option (--run-cloud-tests is defined in parent conftest)."""
    try:
        parser.addoption(
            "--run-scaleway-tests",
            action="store_true",
            default=False,
            help="Run tests that require Scaleway credentials (SCW_PROJECT_ID, SCW_SECRET_KEY)",
        )
    except ValueError:
        pass  # Already defined elsewhere


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests that require cloud or Scaleway credentials unless opted in.

    - If --run-cloud-tests is NOT passed: skip tests that use the
      'remote_processor' fixture.
    - If --run-cloud-tests is passed: all cloud tests run.  The token is
      resolved from ``RemoteConfig`` or the environment.
    - If --run-scaleway-tests is NOT passed: skip tests that use the
      'scaleway_session' fixture.
    """
    run_cloud = config.getoption("--run-cloud-tests", default=False)
    run_scaleway = config.getoption("--run-scaleway-tests", default=False)

    cloud_skip = pytest.mark.skip(
        reason="requires Quandela Cloud token; run with --run-cloud-tests"
    )
    scaleway_skip = pytest.mark.skip(
        reason="requires Scaleway credentials; run with --run-scaleway-tests"
    )

    for item in items:
        fixturenames = getattr(item, "fixturenames", ())

        # Skip cloud tests unless --run-cloud-tests
        if not run_cloud and "remote_processor" in fixturenames:
            item.add_marker(cloud_skip)

        # Skip scaleway tests unless --run-scaleway-tests
        if not run_scaleway and (
            "scaleway_session" in fixturenames
            or "scaleway_session_probs" in fixturenames
        ):
            item.add_marker(scaleway_skip)


# ---------------------------------------------------------------------------
# Quandela Cloud fixtures
# ---------------------------------------------------------------------------


def _resolve_cloud_token() -> str | None:
    """Return the cloud token from RemoteConfig or the environment.

    ``RemoteConfig.get_token()`` checks (in order): its in-memory cache,
    the ``PCVL_CLOUD_TOKEN`` env var, and Perceval's persistent config file.
    Returns ``None`` if no token is available.
    """
    rc = RemoteConfig()
    token = (rc.get_token() or "").strip()
    return token or None


@pytest.fixture
def remote_processor():
    """Provide a RemoteProcessor with an inline token.

    The token is resolved from ``RemoteConfig`` (global cache, env var,
    or persistent config).  If no token is available the test is skipped.

    Passing the token inline ensures ``MerlinProcessor`` can always
    extract it for cloning.

    Collection-level skipping is handled by ``pytest_collection_modifyitems``
    via the ``--run-cloud-tests`` flag.
    """
    token = _resolve_cloud_token()
    if token is None:
        pytest.skip(
            "Quandela Cloud token not configured: call RemoteConfig.set_token(), "
            "set PCVL_CLOUD_TOKEN, or configure Perceval persistent storage."
        )
    return pcvl.RemoteProcessor("sim:slos", token)


# ---------------------------------------------------------------------------
# Scaleway fixtures
# ---------------------------------------------------------------------------


def _has_scaleway_credentials() -> bool:
    """Check if Scaleway credentials are available in environment."""
    return bool(os.environ.get("SCW_PROJECT_ID") and os.environ.get("SCW_SECRET_KEY"))


@pytest.fixture(scope="session")
def has_scaleway_credentials() -> bool:
    """Whether Scaleway credentials are configured."""
    return _has_scaleway_credentials()


@pytest.fixture(scope="module")
def scaleway_credentials(has_scaleway_credentials: bool):
    """Provide Scaleway credentials or skip if not available."""
    if not has_scaleway_credentials:
        pytest.skip(
            "Scaleway credentials not configured: set SCW_PROJECT_ID and SCW_SECRET_KEY"
        )
    return {
        "project_id": os.environ["SCW_PROJECT_ID"],
        "token": os.environ["SCW_SECRET_KEY"],
    }


@pytest.fixture(scope="module")
def scaleway_session(scaleway_credentials):
    """
    Provide a Scaleway Session for testing.

    Uses EMU-ASCELLA-6PQ platform with reasonable timeouts for testing.
    The session is shared across all tests in a module to avoid
    repeatedly creating/destroying sessions.
    """
    scw = pytest.importorskip("perceval.providers.scaleway")

    with scw.Session(
        "EMU-ASCELLA-6PQ",
        project_id=scaleway_credentials["project_id"],
        token=scaleway_credentials["token"],
        deduplication_id="merlin-test-session",
        max_idle_duration_s=300,
        max_duration_s=600,
    ) as session:
        yield session


# Uncomment when there is a probs backend
@pytest.fixture(scope="module")
def scaleway_session_probs(scaleway_credentials):
    """
    Provide a Scaleway Session connected to a platform that supports the ``probs``
    command.  The fixture queries the Scaleway platform list at runtime and picks
    the first Quandela backend that exposes ``probs``; the test is skipped if none
    is available.
    """
    requests = pytest.importorskip("requests")
    scw = pytest.importorskip("perceval.providers.scaleway")

    token = scaleway_credentials["token"]
    project_id = scaleway_credentials["project_id"]

    r = requests.get(
        "https://api.scaleway.com/qaas/v1alpha1/platforms",
        headers={"X-Auth-Token": token, "Content-Type": "application/json"},
    )
    r.raise_for_status()

    probs_platform: str | None = None
    for p in r.json().get("platforms", []):
        if p.get("provider_name") != "quandela":
            continue
        try:
            with scw.Session(
                p["name"],
                project_id=project_id,
                token=token,
                deduplication_id="merlin-probs-probe",
            ) as probe_session:
                proc = probe_session.build_remote_processor()
                if "probs" in proc.available_commands:
                    probs_platform = p["name"]
                    break
        except Exception:
            continue

    if probs_platform is None:
        pytest.skip(
            "No Scaleway Quandela backend with 'probs' support found; "
            "skipping probs-specific tests."
        )

    with scw.Session(
        probs_platform,
        project_id=project_id,
        token=token,
        deduplication_id="merlin-test-probs-session",
        max_idle_duration_s=300,
        max_duration_s=600,
    ) as session:
        yield session
