from __future__ import annotations

from merlin.core.circuit import Circuit


class DummyComponent:
    def __init__(self, depth: int | None = None, params: dict | None = None):
        if depth is not None:
            self.depth = depth
        self._params = params or {}

    def get_params(self):
        return self._params


def test_add_returns_self_and_tracks_components():
    circuit = Circuit(n_modes=2)
    component = object()
    returned = circuit.add(component)

    assert returned is circuit
    assert circuit.num_components == 1
    assert circuit.components[-1] is component


def test_depth_accumulates_declared_depth_and_defaults_to_one():
    circuit = Circuit(n_modes=3)
    circuit.add(DummyComponent(depth=2))
    circuit.add(DummyComponent(depth=1))
    circuit.add(DummyComponent())  # lacks depth attribute -> counts as 1

    assert circuit.depth == 4


def test_get_parameters_merges_component_mappings():
    circuit = Circuit(n_modes=1)
    circuit.add(DummyComponent(params={"theta": 0.1}))
    circuit.add(DummyComponent(params={"phi": None}))

    assert circuit.get_parameters() == {"theta": 0.1, "phi": None}


def test_clear_resets_components_and_metadata():
    circuit = Circuit(n_modes=1)
    circuit.add(DummyComponent())
    circuit.metadata["tag"] = "value"

    circuit.clear()

    assert circuit.components == []
    assert circuit.metadata == {}
