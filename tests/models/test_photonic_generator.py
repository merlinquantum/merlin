# MIT License
#
# Copyright (c) 2026 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Tests for the PhotonicGenerator model."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from torch import nn

import merlin as ML
from merlin.core.partial_measurement import PartialMeasurement
from merlin.models import GeneratorMeasurements


def _make_layer(
    *,
    input_size: int = 2,
    measurement_strategy: ML.MeasurementStrategy | None = None,
) -> ML.QuantumLayer:
    """Build a small trainable layer for generator tests."""
    builder = ML.CircuitBuilder(n_modes=max(3, input_size + 1))
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(input_size)), name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    input_state = [0] * builder.n_modes
    input_state[0] = 1
    strategy = measurement_strategy or ML.MeasurementStrategy.probs(
        computation_space=ML.ComputationSpace.FOCK
    )
    return ML.QuantumLayer(
        input_size=input_size,
        builder=builder,
        input_state=input_state,
        measurement_strategy=strategy,
    )


class SumAdapter(nn.Module):
    """Adapter used to prove custom nn.Module adapters are accepted."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        tensors = []
        for output in measurements.outputs:
            if not isinstance(output, torch.Tensor):
                raise TypeError("SumAdapter only supports tensor outputs.")
            tensors.append(output.reshape(output.shape[0], -1))
        return torch.cat(tensors, dim=1).sum(dim=1, keepdim=True)


class ConstantLatent(ML.LatentDistribution):
    """Latent sampler used to prove custom latent distributions are accepted."""

    def __init__(self, dim: int, value: float) -> None:
        super().__init__(dim)
        self.value = value

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
        return torch.full(
            (batch_size, self.dim),
            self.value,
            device=device,
            dtype=resolved_dtype,
        )


class FirstFeatureAdapter(ML.OutputAdapter):
    """OutputAdapter subclass used to prove the extension contract works."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        output = measurements.outputs[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("FirstFeatureAdapter only supports tensor outputs.")
        return output.reshape(output.shape[0], -1)[:, :1]


class PartialProbabilityAdapter(ML.OutputAdapter):
    """Adapter used to prove custom adapters can consume partial measurements."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        output = measurements.outputs[0]
        if not isinstance(output, PartialMeasurement):
            raise TypeError("PartialProbabilityAdapter expects PartialMeasurement.")
        return output.tensor[:, :1]


def _reference_state_to_int(key: tuple[int, ...]) -> int:
    """Return the reproduced QGAN integer code for an output key."""
    mode_count = len(key)
    result = 0
    for index, count in enumerate(key):
        if count != 0:
            result += 2 ** (mode_count - index)
    return result


def _reference_center_crop_or_pad(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop or zero-pad a batched matrix to a target width."""
    current_size = x.shape[1]
    if current_size == size:
        return x
    if current_size > size:
        left = (current_size - size) // 2
        return x[:, left : left + size]

    left = (size - current_size) // 2
    right = size - current_size - left
    return torch.nn.functional.pad(x, (left, right))


def _reference_occupancy_outputs(
    output_keys: list[tuple[int, ...]],
    raw_results_list: list[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[int, ...], ...]]:
    """Return the reproduced QGAN occupancy regrouping for tensor outputs."""
    rev_map: dict[int, list[tuple[int, ...]]] = {}
    possible_outputs: list[int] = []
    for key in output_keys:
        int_state = _reference_state_to_int(key)
        rev_map.setdefault(int_state, []).append(key)
        if int_state not in possible_outputs:
            possible_outputs.append(int_state)

    output_map: dict[tuple[int, ...], int] = {}
    grouped_keys = []
    for index, int_state in enumerate(sorted(possible_outputs)):
        keys = rev_map[int_state]
        grouped_keys.append(tuple(1 if count > 0 else 0 for count in keys[0]))
        for key in keys:
            output_map[key] = index

    bin_count = len(grouped_keys)
    group_indices = [output_map[key] for key in output_keys]

    grouped_outputs = []
    for result in raw_results_list:
        idx = torch.tensor(group_indices, dtype=torch.long, device=result.device)
        grouped = torch.zeros(
            result.shape[0],
            bin_count,
            dtype=result.dtype,
            device=result.device,
        )
        grouped.index_add_(1, idx, result)
        total_count = grouped.sum(dim=1, keepdim=True)
        grouped = torch.where(
            total_count > 0,
            grouped / total_count.clamp_min(torch.finfo(grouped.dtype).eps),
            grouped,
        )
        grouped_outputs.append(grouped)

    return tuple(grouped_outputs), tuple(grouped_keys)


def _reference_dist_to_image(
    output_keys: list[tuple[int, ...]],
    raw_results_list: list[torch.Tensor],
    image_size: int,
) -> torch.Tensor:
    """Test-only proxy for reproduced photonic_QGAN output conversion.

    The live comparison against the reproduced API is handled by the local
    uncommitted script under ``docs``. This helper keeps only the deterministic
    regroup/crop/normalize contract in the unit suite.
    """
    grouped_outputs, _ = _reference_occupancy_outputs(output_keys, raw_results_list)
    expected_size = image_size * image_size // len(raw_results_list)

    patches = []
    for grouped in grouped_outputs:
        patch = _reference_center_crop_or_pad(grouped, expected_size)
        max_values = patch.max(dim=1, keepdim=True).values
        patches.append(patch / (max_values + 1e-8))

    return torch.cat(patches, dim=1)


def test_latent_distribution_is_abstract():
    with pytest.raises(TypeError, match="abstract class"):
        ML.LatentDistribution(dim=2)


def test_output_adapter_is_abstract():
    with pytest.raises(TypeError, match="abstract class"):
        ML.OutputAdapter()


def test_generator_rejects_empty_layers():
    with pytest.raises(ValueError, match="at least one QuantumLayer"):
        ML.PhotonicGenerator(layers=[], output_adapter=ML.VectorAdapter(size=4))


def test_generator_rejects_non_quantum_layer_objects():
    bad_layer = cast(ML.QuantumLayer, nn.Linear(2, 2))

    with pytest.raises(TypeError, match="QuantumLayer"):
        ML.PhotonicGenerator(
            layers=[bad_layer], output_adapter=ML.VectorAdapter(size=4)
        )

def test_raises_when_number_of_heads_exceeds_data_size_image():
    layer = _make_layer(input_size=2)
    adapter = ML.ImageAdapter(shape=(1, 4, 4))

    with pytest.raises(
        ValueError,
        match=r"Number of heads \(.*\) must not exceed data size",
    ):
        ML.PhotonicGenerator(
            layers=layer,
            count=18,
            output_adapter=adapter,
        )

def test_raises_when_number_of_heads_exceeds_data_size_vector():
    layer = _make_layer(input_size=2)
    adapter = ML.VectorAdapter(size=4)

    with pytest.raises(
        ValueError,
        match=r"Number of heads \(.*\) must not exceed data size",
    ):
        ML.PhotonicGenerator(
            layers=layer,
            count=5,
            output_adapter=adapter,
        )

def test_warns_when_data_size_exceeds_generated_output_space():
    layer = _make_layer(input_size=2)
    adapter = ML.ImageAdapter(shape=(1, 28, 28))

    with pytest.warns(
        UserWarning,
        match=r"Size of data \(.*\) exceeds size of generated output space",
    ):
        ML.PhotonicGenerator(
            layers=layer,
            output_adapter=adapter,
            count=2,
        )


def test_generator_accepts_single_layer():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=4)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 1
    assert generator.latent_dim == 4


def test_generator_accepts_single_layer_object():
    layer = _make_layer(input_size=2)

    generator = ML.PhotonicGenerator(
        layers=layer,
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 1
    assert generator[0] is layer


def test_generator_count_creates_independent_layer_copies():
    torch.manual_seed(10)
    template = _make_layer(input_size=2)

    generator = ML.PhotonicGenerator(
        layers=template,
        count=3,
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 3
    assert all(generator[index] is not template for index in range(len(generator)))
    first_head_param_ids = {id(param) for param in generator[0].parameters()}
    for head_index in range(1, len(generator)):
        other_head_param_ids = {
            id(param) for param in generator[head_index].parameters()
        }
        assert first_head_param_ids.isdisjoint(other_head_param_ids)

    for head_index in range(1, len(generator)):
        assert any(
            not torch.equal(first_param, other_param)
            for first_param, other_param in zip(
                generator[0].parameters(),
                generator[head_index].parameters(),
                strict=True,
            )
        )


def test_generator_count_matches_repeated_fresh_layer_initialization():
    torch.manual_seed(25)
    template = _make_layer(input_size=2)
    counted_generator = ML.PhotonicGenerator(
        layers=template,
        count=3,
        output_adapter=ML.VectorAdapter(size=4),
    )

    torch.manual_seed(25)
    explicit_layers = [_make_layer(input_size=2) for _ in range(3)]
    explicit_generator = ML.PhotonicGenerator(
        layers=explicit_layers,
        output_adapter=ML.VectorAdapter(size=4),
    )

    for counted_layer, explicit_layer in zip(
        counted_generator.layers,
        explicit_generator.layers,
        strict=True,
    ):
        counted_state = counted_layer.state_dict()
        explicit_state = explicit_layer.state_dict()
        assert counted_state.keys() == explicit_state.keys()
        for key in counted_state:
            assert torch.equal(counted_state[key], explicit_state[key])


def test_generator_rejects_invalid_count_usage():
    layer = _make_layer(input_size=2)

    with pytest.raises(ValueError, match="count can only"):
        ML.PhotonicGenerator(
            layers=[layer],
            count=2,
            output_adapter=ML.VectorAdapter(size=4),
        )
    with pytest.raises(TypeError, match="count"):
        ML.PhotonicGenerator(
            layers=layer,
            count=cast(int, 1.5),
            output_adapter=ML.VectorAdapter(size=4),
        )
    with pytest.raises(ValueError, match="count"):
        ML.PhotonicGenerator(
            layers=layer,
            count=0,
            output_adapter=ML.VectorAdapter(size=4),
        )


def test_generator_rejects_duplicate_layer_objects():
    layer = _make_layer(input_size=2)

    with pytest.raises(ValueError, match="duplicate"):
        ML.PhotonicGenerator(
            layers=[layer, layer],
            output_adapter=ML.VectorAdapter(size=4),
        )


def test_generator_rejects_inconsistent_latent_dims():
    layers = [_make_layer(input_size=2), _make_layer(input_size=3)]

    with pytest.raises(ValueError, match="same input_size"):
        ML.PhotonicGenerator(layers=layers, output_adapter=ML.VectorAdapter(size=4))


def test_generator_rejects_amplitude_measurement_strategy():
    layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.amplitudes(
            computation_space=ML.ComputationSpace.FOCK
        )
    )

    with pytest.raises(ValueError, match="does not support amplitude"):
        ML.PhotonicGenerator(layers=[layer], output_adapter=ML.VectorAdapter(size=4))


def test_generator_accepts_non_amplitude_measurement_strategies():
    mode_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.mode_expectations(
            computation_space=ML.ComputationSpace.FOCK
        )
    )
    partial_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.partial(
            modes=[0],
            computation_space=ML.ComputationSpace.FOCK,
        )
    )

    assert (
        ML.PhotonicGenerator(
            layers=[mode_layer], output_adapter=SumAdapter()
        ).latent_dim
        == 2
    )
    assert (
        ML.PhotonicGenerator(
            layers=[partial_layer], output_adapter=SumAdapter()
        ).latent_dim
        == 2
    )


def test_latent_dim_is_inferred_from_layers():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert generator.latent_dim == 2


def test_sample_latent_shape():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    z = generator.sample_latent(batch_size=5)

    assert z.shape == (5, 3)


def test_default_generator_latent_uses_original_qgan_scale():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert isinstance(generator.latent, ML.NormalLatent)
    assert generator.latent.std == pytest.approx(2 * math.pi)


def test_sample_latent_respects_explicit_dtype():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    z = generator.sample_latent(batch_size=5, dtype=torch.float64)

    assert z.dtype == torch.float64


def test_custom_latent_distribution_is_supported():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
        latent=ConstantLatent(dim=3, value=0.25),
    )

    z = generator.sample_latent(batch_size=3, dtype=torch.float64)

    assert z.dtype == torch.float64
    assert torch.allclose(z, torch.full((3, 2), 0.25, dtype=torch.float64))


def test_custom_latent_distribution_dimension_must_match_layers():
    with pytest.raises(ValueError, match="Latent dimension"):
        ML.PhotonicGenerator(
            layers=[_make_layer(input_size=3)],
            output_adapter=ML.VectorAdapter(size=4),
            latent=ConstantLatent(dim=4, value=0.25),
        )


def test_normal_latent_validates_configuration_and_batch_size():
    with pytest.raises(ValueError, match="dim"):
        ML.NormalLatent(dim=0)
    with pytest.raises(ValueError, match="std"):
        ML.NormalLatent(dim=2, std=0.0)

    latent = ML.NormalLatent(dim=2)
    with pytest.raises(TypeError, match="batch_size"):
        latent.sample(cast(int, 1.0))
    with pytest.raises(ValueError, match="batch_size"):
        latent.sample(batch_size=0)


def test_measure_returns_one_output_per_layer():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=8),
    )
    z = torch.randn(3, generator.latent_dim)

    measurements = generator.measure(z)

    assert len(measurements.outputs) == 2
    assert len(measurements.output_keys) == 2
    assert measurements.output_keys[0] == tuple(layers[0].output_keys)
    assert measurements.outputs[0].shape[0] == 3


def test_forward_uses_output_adapter():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=SumAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_output_adapter_subclass_is_supported():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=FirstFeatureAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_partial_measurement_output_can_use_custom_adapter():
    partial_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.partial(
            modes=[0],
            computation_space=ML.ComputationSpace.FOCK,
        )
    )
    generator = ML.PhotonicGenerator(
        layers=[partial_layer],
        output_adapter=PartialProbabilityAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_generate_samples_latent_and_forwards():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=4)],
        output_adapter=ML.VectorAdapter(size=5),
    )

    output = generator.generate(batch_size=6)

    assert output.shape == (6, 5)


def test_getitem_returns_quantum_layer():
    layer = _make_layer(input_size=3)
    generator = ML.PhotonicGenerator(
        layers=[layer],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert generator[0] is layer


def test_len_returns_number_of_layers():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 2


def test_vector_adapter_output_shape():
    adapter = ML.VectorAdapter(size=5)
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(2, 2), torch.ones(2, 4)),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    assert output.shape == (2, 5)


def test_vector_adapter_center_crops_and_zero_pads():
    measurements = GeneratorMeasurements(
        outputs=(torch.arange(6, dtype=torch.float32).reshape(1, 6),),
        output_keys=((),),
    )

    cropped = ML.VectorAdapter(size=4)(measurements)
    padded = ML.VectorAdapter(size=8)(measurements)

    assert torch.equal(cropped, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(padded, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]]))


def test_image_adapter_output_shape_grayscale():
    adapter = ML.ImageAdapter(shape=(2, 3))
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(4, 2), torch.ones(4, 4)),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    assert output.shape == (4, 1, 2, 3)


def test_image_adapter_output_shape_multichannel():
    adapter = ML.ImageAdapter(shape=(2, 2, 3))
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(4, 12),),
        output_keys=((),),
    )

    output = adapter(measurements)

    assert output.shape == (4, 2, 2, 3)


def test_image_adapter_headwise_crops_and_pads_each_head():
    adapter = ML.ImageAdapter(shape=(2, 3), headwise=True)
    measurements = GeneratorMeasurements(
        outputs=(
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            torch.tensor([[10.0, 11.0]]),
        ),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    expected = torch.tensor([[[[0.0, 1.0, 2.0], [10.0, 11.0, 0.0]]]])
    assert torch.equal(output, expected)


def test_image_adapter_headwise_normalizes_each_patch():
    adapter = ML.ImageAdapter(shape=(2, 3), headwise=True, normalize_patches=True)
    measurements = GeneratorMeasurements(
        outputs=(
            torch.tensor([[0.0, 2.0, 4.0]]),
            torch.tensor([[0.0, 3.0, 0.0]]),
        ),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    expected = torch.tensor([[[[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]]]])
    assert torch.allclose(output, expected)


def test_image_adapter_headwise_requires_even_patch_split():
    adapter = ML.ImageAdapter(shape=(1, 5), headwise=True)
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(1, 2), torch.ones(1, 2)),
        output_keys=((), ()),
    )

    with pytest.raises(ValueError, match="divisible"):
        adapter(measurements)


def test_image_adapter_rejects_patch_normalization_without_headwise():
    with pytest.raises(ValueError, match="requires headwise"):
        ML.ImageAdapter(shape=(2, 3), normalize_patches=True)


def test_generator_with_occupancy_readout_uses_reduced_measurement_keys():
    strategy = ML.MeasurementStrategy.probs(
        computation_space=ML.ComputationSpace.FOCK,
        occupancy_readout=True,
    )
    layer = _make_layer(input_size=4, measurement_strategy=strategy)
    generator = ML.PhotonicGenerator(
        layers=layer,
        count=4,
        output_adapter=ML.ImageAdapter(shape=(1, 4, 4), headwise=True),
    )
    z = torch.randn(3, generator.latent_dim)

    measurements = generator.measure(z)
    output = generator(z)

    assert output.shape == (3, 1, 4, 4)
    assert len(measurements.outputs) == 4
    assert len(measurements.output_keys) == 4
    for head_index, head_output in enumerate(measurements.outputs):
        assert isinstance(head_output, torch.Tensor)
        assert head_output.shape[1] == generator[head_index].output_size
        assert len(measurements.output_keys[head_index]) == head_output.shape[1]


def test_occupancy_readout_image_pipeline_matches_reference_dist_to_image():
    output_keys = [(2, 0), (1, 0), (0, 2), (0, 1), (1, 1), (0, 0)]
    raw_results_list = [
        torch.tensor([
            [0.30, 0.15, 0.10, 0.20, 0.20, 0.05],
            [0.00, 0.25, 0.35, 0.10, 0.20, 0.10],
        ]),
        torch.tensor([
            [0.05, 0.25, 0.15, 0.30, 0.20, 0.05],
            [0.40, 0.05, 0.10, 0.20, 0.15, 0.10],
        ]),
    ]
    reference = _reference_dist_to_image(
        output_keys,
        raw_results_list,
        image_size=4,
    )
    grouped_outputs, grouped_keys = _reference_occupancy_outputs(
        output_keys,
        raw_results_list,
    )
    adapter = ML.ImageAdapter(
        shape=(1, 4, 4),
        headwise=True,
        normalize_patches=True,
    )

    output = adapter(
        GeneratorMeasurements(
            outputs=grouped_outputs,
            output_keys=(grouped_keys, grouped_keys),
        )
    ).reshape(reference.shape)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference)


def test_parameters_include_all_layer_parameters():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=4),
    )

    generator_param_ids = {id(param) for param in generator.parameters()}
    for layer in layers:
        for param in layer.parameters():
            assert id(param) in generator_param_ids


def test_state_dict_includes_all_layer_state():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=4),
    )

    generator_state = generator.state_dict()

    for layer_index, layer in enumerate(layers):
        for key in layer.state_dict():
            assert f"layers.{layer_index}.{key}" in generator_state


def test_gradients_flow_through_photonic_generator():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)
    loss = output[:, 0].sum()
    loss.backward()

    grads = [param.grad for param in generator.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)


def test_forward_rejects_wrong_latent_shape():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )

    with pytest.raises(ValueError, match="shape"):
        generator(torch.randn(2, 3))


def test_forward_rejects_non_batched_latent_input():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )

    with pytest.raises(ValueError, match="rank"):
        generator(torch.randn(2))
