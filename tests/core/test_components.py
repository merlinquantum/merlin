from __future__ import annotations

import math

import pytest

import merlin.core.components as components_mod

Rotation = components_mod.Rotation
BeamSplitter = components_mod.BeamSplitter
EntanglingBlock = components_mod.EntanglingBlock
GenericInterferometer = components_mod.GenericInterferometer
ParameterRole = components_mod.ParameterRole


def test_rotation_get_params_for_trainable_custom_name():
    rotation = Rotation(target=0, role=ParameterRole.TRAINABLE, custom_name="theta")
    assert rotation.get_params() == {"theta": None}


def test_rotation_get_params_for_fixed_returns_empty():
    rotation = Rotation(
        target=1, role=ParameterRole.FIXED, custom_name="phi", value=0.5
    )
    assert rotation.get_params() == {}


def test_beam_splitter_get_params_exposes_non_fixed_names():
    beam_splitter = BeamSplitter(
        targets=(0, 1),
        theta_role=ParameterRole.TRAINABLE,
        theta_name="theta",
        phi_role=ParameterRole.INPUT,
        phi_name="phi",
    )
    assert beam_splitter.get_params() == {"theta": None, "phi": None}


def test_entangling_block_exposes_no_parameters():
    block = EntanglingBlock(targets=[0, 1], depth=2, trainable=True)
    assert block.get_params() == {}


def test_generic_interferometer_non_trainable_gets_random_phases():
    """Non-trainable phase shifters must draw random phases, not default to 0.0."""
    gi = GenericInterferometer(start_mode=0, span=4, trainable=False)

    assert gi.trainable is False
    count = gi.span * (gi.span - 1) // 2
    assert len(gi.fixed_inner_values) == count
    assert len(gi.fixed_outer_values) == count
    assert any(v != 0.0 for v in gi.fixed_inner_values)
    assert any(v != 0.0 for v in gi.fixed_outer_values)
    for value in gi.fixed_inner_values + gi.fixed_outer_values:
        assert 0.0 <= value < 2 * math.pi


def test_generic_interferometer_trainable_gets_no_fixed_phases():
    gi = GenericInterferometer(start_mode=0, span=4, trainable=True)
    assert gi.fixed_inner_values == []
    assert gi.fixed_outer_values == []


def test_generic_interferometer_seed_is_reproducible():
    gi_a = GenericInterferometer(start_mode=0, span=3, trainable=False, seed=99)
    gi_b = GenericInterferometer(start_mode=0, span=3, trainable=False, seed=99)
    assert gi_a.fixed_inner_values == gi_b.fixed_inner_values
    assert gi_a.fixed_outer_values == gi_b.fixed_outer_values


def test_generic_interferometer_no_seed_varies_between_instances():
    gi_a = GenericInterferometer(start_mode=0, span=3, trainable=False)
    gi_b = GenericInterferometer(start_mode=0, span=3, trainable=False)
    assert gi_a.fixed_inner_values != gi_b.fixed_inner_values


@pytest.mark.parametrize("field_name", ["fixed_inner_values", "fixed_outer_values"])
def test_generic_interferometer_rejects_short_fixed_values(field_name):
    """A provided fixed-values list shorter than the shifter count must raise.

    Without this check, the circuit builder silently substitutes 0.0 for the
    missing indices, reintroducing the zero-phase bug that random
    initialization of non-trainable shifters is meant to prevent.
    """
    # span=4 -> 6 phase shifters; provide only 3 values
    with pytest.raises(ValueError, match=f"{field_name} must contain exactly 6"):
        GenericInterferometer(
            start_mode=0,
            span=4,
            trainable=False,
            **{field_name: [0.1, 0.2, 0.3]},
        )


@pytest.mark.parametrize("field_name", ["fixed_inner_values", "fixed_outer_values"])
def test_generic_interferometer_rejects_overlong_fixed_values(field_name):
    with pytest.raises(ValueError, match=f"{field_name} must contain exactly 1"):
        GenericInterferometer(
            start_mode=0,
            span=2,
            trainable=False,
            **{field_name: [0.1, 0.2]},
        )


def test_generic_interferometer_accepts_complete_fixed_values():
    """Explicit lists covering every shifter are preserved as-is."""
    inner = [0.1, 0.2, 0.3]
    outer = [0.4, 0.5, 0.6]
    gi = GenericInterferometer(
        start_mode=0,
        span=3,
        trainable=False,
        fixed_inner_values=inner,
        fixed_outer_values=outer,
    )
    assert gi.fixed_inner_values == inner
    assert gi.fixed_outer_values == outer