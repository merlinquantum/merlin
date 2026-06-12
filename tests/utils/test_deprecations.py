import warnings

from merlin.utils import deprecations
from merlin.utils.deprecations import sanitize_parameters


class NestedDeprecatedCalls:
    @sanitize_parameters
    def outer(self, n_modes: int | None = None) -> int | None:
        return self.inner(n_modes=n_modes)

    @sanitize_parameters
    def inner(self, n_modes: int | None = None) -> int | None:
        return n_modes


def test_nested_sanitize_parameters_suppresses_duplicate_warning(monkeypatch):
    message = "Use the default mode count instead."
    monkeypatch.setitem(
        deprecations.DEPRECATION_REGISTRY,
        f"{NestedDeprecatedCalls.outer.__qualname__}.n_modes",
        (message, False, None),
    )
    monkeypatch.setitem(
        deprecations.DEPRECATION_REGISTRY,
        f"{NestedDeprecatedCalls.inner.__qualname__}.n_modes",
        (message, False, None),
    )

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", DeprecationWarning)
        result = NestedDeprecatedCalls().outer(n_modes=4)

    assert result == 4
    assert len(warning_list) == 1
    assert "Parameter 'n_modes' is deprecated" in str(warning_list[0].message)
