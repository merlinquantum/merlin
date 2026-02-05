import math

import perceval as pcvl
import pytest

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.utils.combinadics import Combinadics


def _collect_states(combo: Combinadics):
    total = combo.compute_space_size()
    states_iter = list(combo.iter_states())
    assert len(states_iter) == total
    states_enum = combo.enumerate_states()
    assert states_enum == states_iter
    states_rank = [combo.index_to_fock(i) for i in range(total)]
    assert states_rank == states_iter
    # ensure descending lexicographic order
    for prev, curr in zip(states_iter, states_iter[1:], strict=False):
        assert prev >= curr
    # check bidirectional consistency
    for idx, state in enumerate(states_iter):
        assert combo.fock_to_index(state) == idx
    return states_iter


def test_getitem_roundtrip_both_directions():
    combo = Combinadics("unbunched", n=2, m=4)
    states = list(combo.iter_states())
    for idx, state in enumerate(states):
        assert combo[idx] == state
        assert combo[state] == idx
        assert combo[list(state)] == idx

    with pytest.raises(TypeError):
        combo["0101"]

    bs = pcvl.BasicState("|1,0,0,1>")
    expected_idx = states.index(tuple(bs))
    assert combo[bs] == expected_idx


def test_len_and_index_match_compute_space_size():
    combo = Combinadics("fock", n=3, m=4)
    assert len(combo) == combo.compute_space_size()
    for idx, state in enumerate(combo):
        assert combo.index(state) == idx


def test_dual_rail_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("dual_rail", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == 2**n
    states = _collect_states(combo)
    assert len(states) == expected_total
    # each pair must contain exactly one photon
    for state in states:
        for i in range(0, m, 2):
            a, b = state[i], state[i + 1]
            assert a in (0, 1) and b in (0, 1) and a + b == 1


def test_unbunched_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("unbunched", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(m, n)
    states = _collect_states(combo)
    assert len(states) == expected_total
    # unbunched: collision free
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ in (0, 1)


def test_fock_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("fock", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(n + m - 1, m - 1)
    states = _collect_states(combo)
    assert len(states) == expected_total
    # fock states: non-negative integers summing to n
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ >= 0


def test_dual_rail_requires_twice_as_many_modes():
    with pytest.raises(ValueError):
        Combinadics("dual_rail", n=4, m=6)


@pytest.mark.parametrize(
    ("scheme"),
    [
        "unbunched",
        "dual_rail",
        "fock",
    ],
)
def test_iteration_order_matches_quantum_layer(scheme: str):
    print("Testing scheme:", scheme)
    n, m = 4, 8
    if scheme == "unbunched":
        ql = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(m),
            n_photons=n,
        )
    else:
        computation_space = (
            ComputationSpace.DUAL_RAIL
            if scheme == "dual_rail"
            else ComputationSpace.FOCK
        )
        ql = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(m),
            n_photons=n,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=computation_space
            ),
        )

    mapped_keys = [
        tuple(state) for state in ql.computation_process.simulation_graph.mapped_keys
    ]

    combo = Combinadics(scheme, n=n, m=m)
    states = list(combo.iter_states())

    assert mapped_keys == states
