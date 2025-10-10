from __future__ import annotations

import merlin.core.components as components_mod

Rotation = components_mod.Rotation
BeamSplitter = components_mod.BeamSplitter
EntanglingBlock = components_mod.EntanglingBlock
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
