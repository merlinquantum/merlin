from __future__ import annotations

import math

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
