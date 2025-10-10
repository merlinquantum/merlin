from __future__ import annotations

import pytest

import merlin.core.observables as observables_mod

PauliObservable = observables_mod.PauliObservable
NumberOperator = observables_mod.NumberOperator
CompositeObservable = observables_mod.CompositeObservable
parse_observable = observables_mod.parse_observable


def test_parse_simple_pauli_observable():
    observable = parse_observable("XYZ", n_modes=3)
    assert isinstance(observable, PauliObservable)
    assert observable.pauli_string == "XYZ"
    assert observable.coefficient == 1.0


def test_parse_composite_observable_splits_terms():
    composite = parse_observable("0.5*ZZ + 1.5*IZ", n_modes=2)
    assert isinstance(composite, CompositeObservable)

    terms = list(composite)
    assert len(terms) == 2
    assert isinstance(terms[0], PauliObservable)
    assert terms[0].coefficient == pytest.approx(0.5)
    assert isinstance(terms[1], PauliObservable)
    assert terms[1].coefficient == pytest.approx(1.5)


def test_parse_number_operator_with_optional_coefficient():
    observable = parse_observable("2.5*n_1")
    assert isinstance(observable, NumberOperator)
    assert observable.mode_index == 1
    assert observable.coefficient == pytest.approx(2.5)


def test_parse_observable_validates_length():
    with pytest.raises(ValueError):
        parse_observable("ZZ", n_modes=3)


def test_pauli_observable_rejects_unknown_symbols():
    with pytest.raises(ValueError):
        PauliObservable("ZAZ")
